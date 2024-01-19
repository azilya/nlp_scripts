# coding=utf-8
# Copyright 2018 The Google AI Language Team Authors and The HuggingFace Inc. team.
# Copyright (c) 2018, NVIDIA CORPORATION.  All rights reserved.
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
#     http://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.
"""
CLM training with distillation.

This is the slightly tweaked script
from `transformers/examples/language-modeling/run_clm_no_trainer.py` (v4.34.1)
with an additional step of distillation and MLflow tracking.
"""

import argparse
import glob
import math
import os

import datasets
import dotenv
import mlflow
import torch
from torch import nn
from torch.utils.data import DataLoader
from tqdm import tqdm
from transformers import (
    CONFIG_MAPPING,
    MODEL_MAPPING,
    AutoConfig,
    AutoModelForCausalLM,
    AutoTokenizer,
    default_data_collator,
    get_scheduler,
    set_seed,
)
from transformers.utils import logging

from utils import check_if_save, load_examples, parse_args

logger = logging.get_logger(__name__)


MODEL_CONFIG_CLASSES = list(MODEL_MAPPING.keys())
MODEL_TYPES = tuple(conf.model_type for conf in MODEL_CONFIG_CLASSES)


def train_eval_loop(
    args: argparse.Namespace,
    train_dataloader: DataLoader,
    eval_dataloader: DataLoader,
    model: nn.Module,
    optimizer: torch.optim.Optimizer,
    scheduler,
    teacher=None,
):
    # Train!
    total_batch_size = (
        args.per_device_train_batch_size * args.gradient_accumulation_steps
    )

    logger.info("***** Running training *****")
    logger.info("  Num examples = %d" % len(train_dataloader.dataset))
    logger.info("  Num Epochs = %d" % args.num_train_epochs)
    logger.info(
        "  Instantaneous batch size per device = %d" % args.per_device_train_batch_size
    )
    logger.info(
        "  Total train batch size (w. parallel, distributed & accumulation) = %d"
        % total_batch_size
    )
    logger.info("  Gradient Accumulation steps = %d" % args.gradient_accumulation_steps)
    logger.info("  Total optimization steps = %d" % args.max_train_steps)
    # Only show the progress bar once on each machine.
    progress_bar = tqdm(
        range(args.max_train_steps),  # disable=not accelerator.is_local_main_process
    )
    completed_steps = 0
    starting_epoch = 0

    # Check if continuing training from a checkpoint
    if args.resume_from_checkpoint:
        if args.resume_from_checkpoint is not None or args.resume_from_checkpoint != "":
            checkpoint_path = args.resume_from_checkpoint
            path = os.path.basename(args.resume_from_checkpoint)
        else:
            # Get the most recent checkpoint
            dirs = [f.name for f in os.scandir(os.getcwd()) if f.is_dir()]
            dirs.sort(key=os.path.getctime)
            path = dirs[
                -1
            ]  # Sorts folders by date modified, most recent checkpoint is the last
            checkpoint_path = path
            path = os.path.basename(checkpoint_path)

        logger.info("Resumed from checkpoint: %s" % checkpoint_path)

        # Extract `epoch_{i}` or `step_{i}`
        training_difference = os.path.splitext(path)[0]

        if "epoch" in training_difference:
            starting_epoch = int(training_difference.replace("epoch_", "")) + 1
            resume_step = None
            completed_steps = starting_epoch * args.num_update_steps_per_epoch
        else:
            # need to multiply `gradient_accumulation_steps` to reflect real steps
            resume_step = (
                int(training_difference.replace("step_", ""))
                * args.gradient_accumulation_steps
            )
            starting_epoch = resume_step // len(train_dataloader)
            completed_steps = resume_step // args.gradient_accumulation_steps
            resume_step -= starting_epoch * len(train_dataloader)
    # update the progress_bar if load from checkpoint
    progress_bar.update(completed_steps)

    distil_loss_fct = nn.KLDivLoss(reduction="batchmean")

    def evaluate():
        losses = []
        for _, batch in tqdm(enumerate(eval_dataloader), total=len(eval_dataloader)):
            model_batch = {k: v.to(model.device) for k, v in batch.items()}
            teach_batch = {k: v.to(teacher.device) for k, v in batch.items()}
            with torch.no_grad():
                outputs = model(**model_batch)
                teacher_outputs = teacher(**teach_batch)
            assert outputs.logits.size() == teacher_outputs.logits.size()
            loss = outputs.loss
            # Distillation loss
            loss_teach = distil_loss_fct(
                torch.log_softmax(outputs.logits.cpu() / args.temperature, dim=-1),
                torch.softmax(teacher_outputs.logits.cpu() / args.temperature, dim=-1),
            ) * (args.temperature**2)
            loss = args.alpha_kl * loss_teach + args.alpha_ce * loss

            losses.append(loss.detach().cpu())
        losses = torch.cat(losses)
        try:
            eval_loss = torch.mean(losses)
            perplexity = math.exp(eval_loss)
        except OverflowError:
            perplexity = float("inf")
            eval_loss = float("inf")

        logger.info(
            "epoch %d: perplexity: %0.3f eval_loss: %0.3f"
            % (epoch, perplexity, eval_loss)
        )

        mlflow.log_metrics(
            {
                "perplexity": perplexity,
                "eval_loss": float(eval_loss),
                "train_loss": total_loss / len(train_dataloader),
                "epoch": epoch,
                "step": completed_steps,
            },
            step=completed_steps,
        )

    model.cuda(0)
    teacher.cuda(1)
    teacher.eval()

    for epoch in range(starting_epoch, args.num_train_epochs):
        total_loss = 0

        if (
            args.resume_from_checkpoint
            and epoch == starting_epoch
            and resume_step is not None
        ):
            # Skip first `n` batches when resuming from a checkpoint
            for _ in range(resume_step):
                next(train_dataloader._get_iterator())
        active_dataloader = train_dataloader

        for batch in active_dataloader:
            model_batch = {k: v.to(model.device) for k, v in batch.items()}
            teach_batch = {k: v.to(teacher.device) for k, v in batch.items()}
            # Distillation loss; arXiv:1503.02531
            with torch.no_grad():
                teacher_outputs = teacher(**teach_batch)
            with torch.autocast(device_type="cuda"):
                outputs = model(**model_batch)
                assert outputs.logits.size() == teacher_outputs.logits.size()
                teach_loss = distil_loss_fct(
                    nn.functional.log_softmax(
                        outputs.logits / args.temperature, dim=-1
                    ),
                    nn.functional.softmax(
                        teacher_outputs.logits.to(model.device) / args.temperature,
                        dim=-1,
                    ),
                ) * (args.temperature**2)
                loss = args.alpha_kl * teach_loss + args.alpha_ce * outputs.loss
            total_loss += loss.detach().cpu().float()

            loss.backward()
            # nn.utils.clip_grad_norm_(model.parameters(), max_norm=2)
            optimizer.step()
            scheduler.step()
            optimizer.zero_grad()

            completed_steps += 1
            progress_bar.update()

            if isinstance(args.checkpointing_steps, int):
                if (
                    completed_steps > 0
                    and completed_steps % args.checkpointing_steps == 0
                ):
                    evaluate()
                    check_if_save(
                        args, model, optimizer, output_dir=f"step_{completed_steps}"
                    )
            if completed_steps >= args.max_train_steps:
                break

        if args.checkpointing_steps == "epoch":
            evaluate()
            check_if_save(args, model, optimizer, output_dir=f"epoch_{epoch}")


def reduce_layers(model):
    keep_layers = list(range(1, model.config.num_hidden_layers, 6))
    # try to take at least a part of different architectures into account
    if "decoder" in {m for m, _ in model.model.named_children()}:
        new_layers = nn.ModuleList(
            [
                layer
                for i, layer in enumerate(model.model.decoder.layers)
                if i in keep_layers
            ]
        )
        model.model.decoder.layers = new_layers
    else:
        new_layers = nn.ModuleList(
            [layer for i, layer in enumerate(model.model.layers) if i in keep_layers]
        )
        model.model.layers = new_layers
    model.config.num_hidden_layers = len(keep_layers)


def main(args: argparse.Namespace):
    datasets.utils.logging.set_verbosity_warning()
    logging.set_verbosity_info()

    # If passed along, set the training seed now.
    if args.seed is not None:
        set_seed(args.seed)

    if args.output_dir is not None:
        os.makedirs(args.output_dir, exist_ok=True)

    if args.config_name:
        config = AutoConfig.from_pretrained(
            args.config_name,
        )
    elif args.model_name_or_path:
        config = AutoConfig.from_pretrained(
            args.model_name_or_path,
        )
    else:
        config = CONFIG_MAPPING[args.model_type]()
        logger.warning("You are instantiating a new config instance from scratch.")

    if args.tokenizer_name:
        tokenizer = AutoTokenizer.from_pretrained(
            args.tokenizer_name,
            use_fast=not args.use_slow_tokenizer,
        )
    elif args.model_name_or_path:
        tokenizer = AutoTokenizer.from_pretrained(
            args.model_name_or_path,
            use_fast=not args.use_slow_tokenizer,
        )
    else:
        raise ValueError(
            "You are instantiating a new tokenizer from scratch."
            "This is not supported by this script."
            "You can do it from another script, save it,"
            " and load it from here, using --tokenizer_name."
        )

    if args.model_name_or_path:
        model = AutoModelForCausalLM.from_pretrained(
            args.model_name_or_path,
            config=config,
            low_cpu_mem_usage=args.low_cpu_mem_usage,
        )
    else:
        logger.info("Training new model from scratch")
        model = AutoModelForCausalLM.from_config(
            config,
        )

    teacher_config = AutoConfig.from_pretrained(
        args.teacher_name_or_path,
    )
    teacher = AutoModelForCausalLM.from_pretrained(
        args.teacher_name_or_path,
        config=teacher_config,
        low_cpu_mem_usage=args.low_cpu_mem_usage,
    )

    if args.use_layer_reduction:
        reduce_layers(model)

    train_dataset, eval_dataset = load_examples(args, tokenizer, logger)

    train_dataloader = DataLoader(
        train_dataset,
        shuffle=True,
        collate_fn=default_data_collator,
        batch_size=args.per_device_train_batch_size,
    )
    eval_dataloader = DataLoader(
        eval_dataset,
        collate_fn=default_data_collator,
        batch_size=args.per_device_eval_batch_size,
    )

    # Scheduler and math around the number of training steps.
    overrode_max_train_steps = False
    num_update_steps_per_epoch = math.ceil(
        len(train_dataloader) / args.gradient_accumulation_steps
    )
    if args.max_train_steps is None:
        args.max_train_steps = args.num_train_epochs * num_update_steps_per_epoch
        overrode_max_train_steps = True

    if args.warmup_ratio is not None:
        args.num_warmup_steps = args.max_train_steps * args.warmup_ratio

    # Prepare optimizer and schedule
    no_decay = ["bias", "LayerNorm.weight"]
    optimizer_grouped_parameters = [
        {
            "params": [
                p
                for n, p in model.named_parameters()
                if not any(nd in n for nd in no_decay)
            ],
            "weight_decay": args.weight_decay,
        },
        {
            "params": [
                p
                for n, p in model.named_parameters()
                if any(nd in n for nd in no_decay)
            ],
            "weight_decay": 0.0,
        },
    ]
    optimizer = torch.optim.AdamW(optimizer_grouped_parameters, lr=args.learning_rate)
    scheduler = get_scheduler(
        args.lr_scheduler_type,
        optimizer,
        num_warmup_steps=args.num_warmup_steps * args.gradient_accumulation_steps,
        num_training_steps=args.max_train_steps * args.gradient_accumulation_steps,
    )

    # We need to recalculate our total training steps as the size
    # of the training dataloader may have changed.
    args.num_update_steps_per_epoch = math.ceil(
        len(train_dataloader) / args.gradient_accumulation_steps
    )
    if overrode_max_train_steps:
        args.max_train_steps = args.num_train_epochs * args.num_update_steps_per_epoch
    # Afterwards we recalculate our number of training epochs
    args.num_train_epochs = math.ceil(
        args.max_train_steps / args.num_update_steps_per_epoch
    )

    # Figure out how many steps we should save the Accelerator states
    checkpointing_steps = args.checkpointing_steps
    if checkpointing_steps is not None and checkpointing_steps.isdigit():
        args.checkpointing_steps = int(checkpointing_steps)

    mlflow.log_params(
        {
            "student": args.model_name_or_path,
            "teacher": args.teacher_name_or_path,
            "hidden_layers": model.config.num_hidden_layers,
            "weight_decay": args.weight_decay,
            "learning_rate": args.learning_rate,
            "lr_scheduler": args.lr_scheduler_type,
            "num_epochs": args.num_train_epochs,
            "train_batch_size": args.per_device_train_batch_size,
            "gradient_accumulation_steps": args.gradient_accumulation_steps,
            "num_warmup_steps": args.num_warmup_steps,
            "seed": args.seed,
        }
    )

    # Training
    train_eval_loop(
        args,
        train_dataloader,
        eval_dataloader,
        model,
        optimizer,
        scheduler,
        teacher,
    )

    if args.output_dir is not None:
        model.save_pretrained(args.output_dir)
        tokenizer.save_pretrained(args.output_dir)
        mlflow.pytorch.log_model(
            model,
            artifact_path=args.output_dir,
            registered_model_name="distil-clm",
            pip_requirements=["torch~=2.0.1", "transformers~=4.34.1"],
        )
        for token_file in glob.glob1(args.output_dir, "*token*"):
            mlflow.log_artifact(token_file, args.output_dir)


if __name__ == "__main__":
    args = parse_args()
    dotenv.load_dotenv()

    mlflow.set_tracking_uri(os.environ["MLFLOW_TRACKING_URL"])
    mlflow.set_experiment(os.environ["EXP_NAME"])
    with mlflow.start_run():
        main(args)
