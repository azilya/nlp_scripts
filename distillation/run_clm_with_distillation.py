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
import logging
import math
import os

import datasets
import dotenv
import mlflow
import torch
import transformers
from accelerate import Accelerator
from accelerate.logging import get_logger
from accelerate.utils import ProjectConfiguration, set_seed
from torch import nn
from torch.utils.data import DataLoader
from tqdm import tqdm
from transformers import (
    MODEL_MAPPING,
    AutoConfig,
    AutoModelForCausalLM,
    AutoTokenizer,
    default_data_collator,
    get_scheduler,
)

from utils import checkpoint, load_examples, parse_args

logger = get_logger(__name__)


MODEL_CONFIG_CLASSES = list(MODEL_MAPPING.keys())
MODEL_TYPES = tuple(conf.model_type for conf in MODEL_CONFIG_CLASSES)


def train_eval_loop(
    args: argparse.Namespace,
    train_dataloader: DataLoader,
    eval_dataloader: DataLoader,
    accelerator: Accelerator,
    model: nn.Module,
    optimizer: torch.optim.Optimizer,
    scheduler,
    teacher=None,
):
    """
    Train-and-evaluate function that does the bulk of the actual work:
    - places models on GPUs,
    - creates train loop,
    - runs eval loop.

    :param args: script launch arguments
    :type args: argparse.Namespace
    :param train_dataloader: dataloader for the train dataset
    :type train_dataloader: DataLoader
    :param eval_dataloader: dataloader for the eval dataset
    :type eval_dataloader: DataLoader
    :param accelerator: accelerator created for the loop
    :type accelerator: Accelerator
    :param model: model to train
    :type model: nn.Module
    :param optimizer: optimizer created for the loop
    :type optimizer: torch.optim.Optimizer
    :param scheduler: scheduler created for the loop
    :type scheduler: _type_
    :param teacher: teacher model, defaults to None
    :type teacher: _type_, optional
    """
    # Train!
    total_batch_size = (
        args.per_device_train_batch_size
        * accelerator.num_processes
        * args.gradient_accumulation_steps
    )

    logger.info("***** Running training *****")
    logger.info(f"  Num examples = {len(train_dataloader.dataset)}")
    logger.info(f"  Num Epochs = {args.num_train_epochs}")
    logger.info(
        f"  Instantaneous batch size per device = {args.per_device_train_batch_size}"
    )
    logger.info(
        f"  Total train batch size (w. parallel, distributed & accumulation) = {total_batch_size}"
    )
    logger.info(f"  Gradient Accumulation steps = {args.gradient_accumulation_steps}")
    logger.info(f"  Total optimization steps = {args.max_train_steps}")
    # Only show the progress bar once on each machine.
    progress_bar = tqdm(
        range(args.max_train_steps), disable=not accelerator.is_local_main_process
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

        accelerator.print(f"Resumed from checkpoint: {checkpoint_path}")
        accelerator.load_state(path)
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
    teacher.eval()

    def evaluate():
        model.eval()
        losses = []
        for _, batch in tqdm(enumerate(eval_dataloader), total=len(eval_dataloader)):
            with torch.no_grad():
                outputs = model(**batch)
                teacher_outputs = teacher(**batch)
            loss = outputs.loss
            # Distillation loss
            assert outputs.logits.size() == teacher_outputs.logits.size()
            loss_teach = distil_loss_fct(
                torch.log_softmax(outputs.logits / args.temperature, dim=-1),
                torch.softmax(teacher_outputs.logits / args.temperature, dim=-1),
            ) * (args.temperature**2)
            loss = args.alpha_kl * loss_teach + args.alpha_ce * loss

            losses.append(
                accelerator.gather_for_metrics(
                    loss.detach().repeat(args.per_device_eval_batch_size)
                )
            )
            # Free up space on GPU
            # del outputs, teacher_outputs, loss, loss_teach
        losses = torch.cat(losses)
        try:
            eval_loss = torch.mean(losses)
            perplexity = math.exp(eval_loss)
        except OverflowError:
            perplexity = float("inf")
            eval_loss = float("inf")

        logger.info(f"epoch {epoch}: perplexity: {perplexity} eval_loss: {eval_loss}")

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
        model.train()

    for epoch in range(starting_epoch, args.num_train_epochs):
        model.train()

        total_loss = 0

        if (
            args.resume_from_checkpoint
            and epoch == starting_epoch
            and resume_step is not None
        ):
            # We skip the first `n` batches in the dataloader when resuming from a checkpoint
            active_dataloader = accelerator.skip_first_batches(
                train_dataloader, resume_step
            )
        else:
            active_dataloader = train_dataloader

        for batch in active_dataloader:
            with accelerator.accumulate(model):
                outputs = model(**batch)
                loss = outputs.loss
                # Distillation loss; arXiv:1503.02531
                with torch.no_grad():
                    teacher_outputs = teacher(**batch)
                assert outputs.logits.size() == teacher_outputs.logits.size()
                loss_teach = distil_loss_fct(
                    torch.log_softmax(outputs.logits / args.temperature, dim=-1),
                    torch.softmax(teacher_outputs.logits / args.temperature, dim=-1),
                ) * (args.temperature**2)
                loss = args.alpha_kl * loss_teach + args.alpha_ce * loss
                total_loss += loss.detach().float()

                accelerator.backward(loss)
                optimizer.step()
                scheduler.step()
                optimizer.zero_grad()

            if accelerator.sync_gradients:
                progress_bar.update(1)
                completed_steps += 1

            if isinstance(args.checkpointing_steps, int):
                if (
                    completed_steps > 0
                    and completed_steps % args.checkpointing_steps == 0
                ):
                    evaluate()
                    checkpoint(
                        args,
                        model,
                        optimizer,
                        scheduler,
                        output_dir=f"step_{completed_steps}",
                    )
            if completed_steps >= args.max_train_steps:
                break

        if args.checkpointing_steps == "epoch":
            evaluate()
            checkpoint(args, model, optimizer, scheduler, output_dir=f"epoch_{epoch}")


def reduce_layers(model, reduce_parameter=6):
    """
    Reducing layer number in the trained model

    :param model: the input model to train later
    :type model: _type_
    :param reduce_parameter: degree to which the layers are reduced, defaults to 6
    :type reduce_parameter: int, optional
    """
    keep_layers = list(range(1, model.config.num_hidden_layers, reduce_parameter))
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
    """
    Main function to init models, start logging and training and output the results

    Args:
        args (argparse.Namespace): script launch parameters

    Raises:
        ValueError: if no existing tokenizer is given in parameters,
        an error is raised
    """
    acc_config = ProjectConfiguration(total_limit=args.checkpoint_limit)
    accelerator = Accelerator(
        gradient_accumulation_steps=args.gradient_accumulation_steps,
        mixed_precision=args.mixed_precision,
        project_config=acc_config,
        project_dir=args.output_dir,
    )

    # Make one log on every process with the configuration for debugging.
    logging.basicConfig(
        format="{asctime} - {levelname} - {name} - {message}",
        datefmt="%m/%d/%Y %H:%M:%S",
        level=logging.INFO,
        style="{",
    )
    logger.info(accelerator.state, main_process_only=False)
    if accelerator.is_local_main_process:
        datasets.utils.logging.set_verbosity_warning()
        transformers.utils.logging.set_verbosity_info()
    else:
        datasets.utils.logging.set_verbosity_error()
        transformers.utils.logging.set_verbosity_error()

    # If passed along, set the training seed now.
    if args.seed is not None:
        set_seed(args.seed)

    if accelerator.is_main_process:
        if args.output_dir is not None:
            os.makedirs(args.output_dir, exist_ok=True)
    accelerator.wait_for_everyone()

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
            "You are instantiating a new tokenizer from scratch. This is not supported by this script."
            "You can do it from another script, save it, and load it from here, using --tokenizer_name."
        )

    config = AutoConfig.from_pretrained(
        args.model_name_or_path,
    )

    student = AutoModelForCausalLM.from_pretrained(
        args.model_name_or_path,
        config=config,
        low_cpu_mem_usage=args.low_cpu_mem_usage,
    )
    teacher = AutoModelForCausalLM.from_pretrained(
        args.model_name_or_path,
        config=config,
        low_cpu_mem_usage=args.low_cpu_mem_usage,
    )

    reduce_layers(student)

    train_dataset, eval_dataset = load_examples(args, accelerator, tokenizer)

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
                for n, p in student.named_parameters()
                if not any(nd in n for nd in no_decay)
            ],
            "weight_decay": args.weight_decay,
        },
        {
            "params": [
                p
                for n, p in student.named_parameters()
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

    # We need to recalculate our total training steps as the size of the training dataloader may have changed.
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
            "model": args.model_name_or_path,
            "hidden_layers": student.config.num_hidden_layers,
            "weight_decay": args.weight_decay,
            "learning_rate": args.learning_rate,
            "lr_scheduler": args.lr_scheduler_type,
            "num_epochs": args.num_train_epochs,
            "train_batch_size": args.per_device_train_batch_size,
            "gradient_accumulation_steps": args.gradient_accumulation_steps,
            "num_warmup_steps": args.num_warmup_steps,
            "mixed_precision": args.mixed_precision,
            "seed": args.seed,
        }
    )

    (
        teacher,
        student,
        optimizer,
        scheduler,
        train_dataloader,
        eval_dataloader,
    ) = accelerator.prepare(
        teacher,
        student,
        optimizer,
        scheduler,
        train_dataloader,
        eval_dataloader,
    )

    # Training
    train_eval_loop(
        args,
        train_dataloader,
        eval_dataloader,
        accelerator,
        student,
        optimizer,
        scheduler,
        teacher,
    )

    accelerator.end_training()

    if args.output_dir is not None:
        accelerator.wait_for_everyone()
        unwrapped_model = accelerator.unwrap_model(student)
        unwrapped_model.save_pretrained(
            args.output_dir,
            is_main_process=accelerator.is_main_process,
            save_function=accelerator.save,
        )
        if accelerator.is_main_process:
            tokenizer.save_pretrained(args.output_dir)
        mlflow.pytorch.log_model(
            unwrapped_model,
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
