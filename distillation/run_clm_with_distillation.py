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
import random
from itertools import chain

import datasets
import dotenv
import mlflow
import torch
import transformers
from accelerate import Accelerator
from accelerate.logging import get_logger
from accelerate.utils import ProjectConfiguration, set_seed
from datasets import load_dataset
from mlflow.data.huggingface_dataset import from_huggingface
from torch import nn
from torch.utils.data import DataLoader
from tqdm import tqdm
from transformers import (
    CONFIG_MAPPING,
    MODEL_MAPPING,
    AutoConfig,
    AutoModelForCausalLM,
    AutoTokenizer,
    SchedulerType,
    default_data_collator,
    get_scheduler,
)

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
            loss = outputs.loss
            # Distillation loss
            teacher_outputs = teacher(**batch)
            assert outputs.logits.size() == teacher_outputs.logits.size()
            loss_teach = distil_loss_fct(
                torch.log_softmax(outputs.logits / args.temperature, dim=-1),
                torch.softmax(teacher_outputs.logits / args.temperature, dim=-1),
            ) * (args.temperature**2)
            loss = args.alpha_kl * loss_teach + args.alpha_ce * loss

            losses.append(
                accelerator.gather_for_metrics(
                    loss.detach().cpu().repeat(args.per_device_eval_batch_size)
                )
            )
            # Free up space on GPU
            del outputs, teacher_outputs, loss, loss_teach
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

    def check_if_save(output_dir):
        if args.output_dir is not None:
            dirs = [f.name for f in os.scandir(args.output_dir) if f.is_dir()]
            dirs.sort(key=os.path.getctime)

            if args.checkpoint_limit is not None and len(dirs) >= args.checkpoint_limit:
                chckp_diff = 1 + (len(dirs) - args.checkpoint_limit)
                for i in range(chckp_diff):
                    os.rmdir(dirs[i])
            output_dir = os.path.join(args.output_dir, output_dir)
            accelerator.save_state(output_dir)

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
                    check_if_save(output_dir=f"step_{completed_steps}")
            if completed_steps >= args.max_train_steps:
                break

        if args.checkpointing_steps == "epoch":
            evaluate()
            check_if_save(output_dir=f"step_{completed_steps}")


def load_examples(args, accelerator, tokenizer):
    if args.dataset_name is not None:
        raw_datasets = load_dataset(args.dataset_name, args.dataset_config_name)
        if "validation" not in raw_datasets.keys():
            raw_datasets["validation"] = load_dataset(
                args.dataset_name,
                args.dataset_config_name,
                split=f"train[:{args.validation_split_percentage}%]",
            )
            raw_datasets["train"] = load_dataset(
                args.dataset_name,
                args.dataset_config_name,
                split=f"train[{args.validation_split_percentage}%:]",
            )
    else:
        data_files = {}
        dataset_args = {}
        if args.train_file is not None:
            data_files["train"] = args.train_file
        if args.validation_file is not None:
            data_files["validation"] = args.validation_file
        extension = args.train_file.split(".")[-1]
        if extension == "txt":
            extension = "text"
            dataset_args["keep_linebreaks"] = not args.no_keep_linebreaks
        raw_datasets = load_dataset(extension, data_files=data_files, **dataset_args)
        # If no validation data is there, validation_split_percentage will be used to divide the dataset.
        if "validation" not in raw_datasets.keys():
            raw_datasets["validation"] = load_dataset(
                extension,
                data_files=data_files,
                split=f"train[:{args.validation_split_percentage}%]",
                **dataset_args,
            )
            raw_datasets["train"] = load_dataset(
                extension,
                data_files=data_files,
                split=f"train[{args.validation_split_percentage}%:]",
                **dataset_args,
            )

    # Preprocessing the datasets.
    # First we tokenize all the texts.
    column_names = raw_datasets["train"].column_names
    text_column_name = "text" if "text" in column_names else column_names[0]

    def tokenize_function(examples):
        return tokenizer(examples[text_column_name])

    with accelerator.main_process_first():
        tokenized_datasets = raw_datasets.map(
            tokenize_function,
            batched=True,
            num_proc=args.preprocessing_num_workers,
            remove_columns=column_names,
            load_from_cache_file=not args.overwrite_cache,
            desc="Running tokenizer on dataset",
        )

    if args.block_size is None:
        block_size = tokenizer.model_max_length
        if block_size > 1024:
            logger.warning(
                "The chosen tokenizer supports a `model_max_length` that is longer than the default `block_size` value"
                " of 1024. If you would like to use a longer `block_size` up to `tokenizer.model_max_length` you can"
                " override this default with `--block_size xxx`."
            )
        block_size = 1024
    else:
        if args.block_size > tokenizer.model_max_length:
            logger.warning(
                f"The block_size passed ({args.block_size}) is larger than the maximum length for the model"
                f"({tokenizer.model_max_length}). Using block_size={tokenizer.model_max_length}."
            )
        block_size = min(args.block_size, tokenizer.model_max_length)

    # Main data processing function that will concatenate all texts from our dataset and generate chunks of block_size.
    def group_texts(examples):
        # Concatenate all texts.
        concatenated_examples = {k: list(chain(*examples[k])) for k in examples.keys()}
        total_length = len(concatenated_examples[list(examples.keys())[0]])
        # We drop the small remainder, and if the total_length < block_size  we exclude this batch and return an empty dict.
        # We could add padding if the model supported it instead of this drop, you can customize this part to your needs.
        total_length = (total_length // block_size) * block_size
        # Split by chunks of max_len.
        result = {
            k: [t[i : i + block_size] for i in range(0, total_length, block_size)]
            for k, t in concatenated_examples.items()
        }
        result["labels"] = result["input_ids"].copy()
        return result

    # Note that with `batched=True`, this map processes 1,000 texts together, so group_texts throws away a remainder
    # for each of those groups of 1,000 texts. You can adjust that batch_size here but a higher value might be slower
    # to preprocess.
    #
    # To speed up this part, we use multiprocessing. See the documentation of the map method for more information:
    # https://huggingface.co/docs/datasets/process#map

    with accelerator.main_process_first():
        lm_datasets = tokenized_datasets.map(
            group_texts,
            batched=True,
            num_proc=args.preprocessing_num_workers,
            load_from_cache_file=not args.overwrite_cache,
            desc=f"Grouping texts in chunks of {block_size}",
        )

    train_dataset = lm_datasets["train"]
    eval_dataset = lm_datasets["validation"]

    # Log a few random samples from the training set:
    for index in random.sample(range(len(train_dataset)), 3):
        logger.info(f"Sample {index} of the training set: {train_dataset[index]}.")

    train_ds = (
        args.dataset_name + "[train]"
        if args.dataset_name is not None
        else args.train_file
    )
    eval_ds = (
        args.dataset_name + "[validation]"
        if args.dataset_name is not None
        else args.validation_file
    )
    mlflow.log_input(
        from_huggingface(train_dataset, path=train_ds, name=train_ds),
        context="training",
    )
    mlflow.log_input(
        from_huggingface(eval_dataset, path=eval_ds, name=eval_ds),
        context="evaluation",
    )

    return (train_dataset, eval_dataset)


def init_student_layers(student, teacher):
    keep_layers = list(range(1, teacher.config.num_hidden_layers, 3))
    # try to take at least a part of different architectures into account
    if "decoder" in {m for m, _ in teacher.model.named_children()}:
        new_layers = nn.ModuleList(
            [
                layer
                for i, layer in enumerate(teacher.model.decoder.layers)
                if i in keep_layers
            ]
        )
        student.model.decoder.layers = new_layers
    else:
        new_layers = nn.ModuleList(
            [layer for i, layer in enumerate(teacher.model.layers) if i in keep_layers]
        )
        student.model.layers = new_layers
    student.config.num_hidden_layers = len(keep_layers)


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Finetune and distillate a transformers model on a causal language modeling task"
    )
    parser.add_argument(
        "--dataset_name",
        type=str,
        default=None,
        help="The name of the dataset to use (via the datasets library).",
    )
    parser.add_argument(
        "--dataset_config_name",
        type=str,
        default=None,
        help="The configuration name of the dataset to use (via the datasets library).",
    )
    parser.add_argument(
        "--train_file",
        type=str,
        default=None,
        help="A csv, txt or a json file containing the training data.",
    )
    parser.add_argument(
        "--validation_file",
        type=str,
        default=None,
        help="A csv, txt or a json file containing the validation data.",
    )
    parser.add_argument(
        "--validation_split_percentage",
        default=5,
        help="The percentage of the train set used as validation set in case there's no validation split",
    )
    parser.add_argument(
        "--model_name_or_path",
        type=str,
        help="Path to pretrained model or model identifier from huggingface.co/models.",
        required=False,
    )
    parser.add_argument(
        "--config_name",
        type=str,
        default=None,
        help="Pretrained config name or path if not the same as model_name",
    )
    parser.add_argument(
        "--tokenizer_name",
        type=str,
        default=None,
        help="Pretrained tokenizer name or path if not the same as model_name",
    )
    parser.add_argument(
        "--use_slow_tokenizer",
        action="store_true",
        help="If passed, will use a slow tokenizer (not backed by the 🤗 Tokenizers library).",
    )
    parser.add_argument(
        "--per_device_train_batch_size",
        type=int,
        default=8,
        help="Batch size (per device) for the training dataloader.",
    )
    parser.add_argument(
        "--per_device_eval_batch_size",
        type=int,
        default=8,
        help="Batch size (per device) for the evaluation dataloader.",
    )
    parser.add_argument(
        "--learning_rate",
        type=float,
        default=5e-5,
        help="Initial learning rate (after the potential warmup period) to use.",
    )
    parser.add_argument(
        "--weight_decay", type=float, default=0.0, help="Weight decay to use."
    )
    parser.add_argument(
        "--num_train_epochs",
        type=int,
        default=3,
        help="Total number of training epochs to perform.",
    )
    parser.add_argument(
        "--max_train_steps",
        type=int,
        default=None,
        help="Total number of training steps to perform. If provided, overrides num_train_epochs.",
    )
    parser.add_argument(
        "--gradient_accumulation_steps",
        type=int,
        default=1,
        help="Number of updates steps to accumulate before performing a backward/update pass.",
    )
    parser.add_argument(
        "--lr_scheduler_type",
        type=SchedulerType,
        default="linear",
        help="The scheduler type to use.",
        choices=[
            "linear",
            "cosine",
            "cosine_with_restarts",
            "polynomial",
            "constant",
            "constant_with_warmup",
        ],
    )
    parser.add_argument(
        "--num_warmup_steps",
        type=int,
        default=0,
        help="Number of steps for the warmup in the lr scheduler.",
    )
    parser.add_argument(
        "--warmup_ratio",
        type=float,
        default=0,
        help="Ratio of training steps to be used for the warmup in the lr scheduler.",
    )
    parser.add_argument(
        "--output_dir", type=str, default=None, help="Where to store the final model."
    )
    parser.add_argument(
        "--seed", type=int, default=None, help="A seed for reproducible training."
    )
    parser.add_argument(
        "--model_type",
        type=str,
        default=None,
        help="Model type to use if training from scratch.",
        choices=MODEL_TYPES,
    )
    parser.add_argument(
        "--block_size",
        type=int,
        default=None,
        help=(
            "Optional input sequence length after tokenization. The training dataset will be truncated in block of"
            " this size for training. Default to the model max input length for single sentence inputs (take into"
            " account special tokens)."
        ),
    )
    parser.add_argument(
        "--preprocessing_num_workers",
        type=int,
        default=None,
        help="The number of processes to use for the preprocessing.",
    )
    parser.add_argument(
        "--overwrite_cache",
        action="store_true",
        help="Overwrite the cached training and evaluation sets",
    )
    parser.add_argument(
        "--no_keep_linebreaks",
        action="store_true",
        help="Do not keep line breaks when using TXT files.",
    )
    parser.add_argument(
        "--checkpointing_steps",
        type=str,
        default=None,
        help="Whether the various states should be saved at the end of every n steps, or 'epoch' for each epoch.",
    )
    parser.add_argument(
        "--checkpoint_limit",
        default=None,
        type=int,
        help="Max. number of checkpoints to store.",
    )
    parser.add_argument(
        "--resume_from_checkpoint",
        type=str,
        default=None,
        help="If the training should continue from a checkpoint folder.",
    )
    parser.add_argument(
        "--low_cpu_mem_usage",
        action="store_true",
        help=(
            "It is an option to create the model as an empty shell, then only materialize its parameters when the pretrained weights are loaded."
            "If passed, LLM loading time and RAM consumption will be benefited."
        ),
    )
    parser.add_argument(
        "--mixed_precision",
        default=None,
        type=str,
        choices=["no", "fp16", "bf16", "fp8"],
        help="Mixed precision setting to use with Accelerator.",
    )
    parser.add_argument(
        "--teacher_name_or_path",
        type=str,
        required=True,
        help="Path to the teacher model.",
    )
    parser.add_argument(
        "--alpha_kl", default=0.5, type=float, help="Distillation loss linear weight."
    )
    parser.add_argument(
        "--alpha_ce", default=0.5, type=float, help="Task loss linear weight."
    )
    parser.add_argument(
        "--temperature",
        default=2.0,
        type=float,
        help="Distillation temperature. Only for distillation.",
    )
    parser.add_argument(
        "--use_layer_reduction",
        action="store_true",
        help="Init the student model with hidden layers from the teacher.",
    )
    args = parser.parse_args()

    # Sanity checks
    if (
        args.dataset_name is None
        and args.train_file is None
        and args.validation_file is None
    ):
        raise ValueError("Need either a dataset name or a training/validation file.")
    else:
        if args.train_file is not None:
            extension = args.train_file.split(".")[-1]
            if extension not in ["csv", "json", "txt"]:
                raise ValueError("`train_file` should be a csv, json or txt file.")
        if args.validation_file is not None:
            extension = args.validation_file.split(".")[-1]
            if extension not in ["csv", "json", "txt"]:
                raise ValueError("`validation_file` should be a csv, json or txt file.")
    if args.use_layer_reduction:
        assert (
            args.model_name_or_path == args.teacher_name_or_path
        ), "To init weights from teacher layers models should be of the same type."

    assert args.alpha_kl > 0
    assert (
        args.alpha_kl + args.alpha_ce == 1.0
    ), "Weights of training losses should sum up to 1.0"

    if args.num_warmup_steps > 0 and args.warmup_ratio > 0:
        raise ValueError(
            "You can't set num_warmup_steps and warmup_ratio at the same time."
            "Use one or the other"
        )

    return args


def main(args: argparse.Namespace):
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
            "You are instantiating a new tokenizer from scratch. This is not supported by this script."
            "You can do it from another script, save it, and load it from here, using --tokenizer_name."
        )

    if args.model_name_or_path:
        model = AutoModelForCausalLM.from_pretrained(
            args.model_name_or_path,
            from_tf=bool(".ckpt" in args.model_name_or_path),
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
        init_student_layers(model, teacher)

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

    (
        teacher,
        model,
        optimizer,
        scheduler,
        train_dataloader,
        eval_dataloader,
    ) = accelerator.prepare(
        teacher,
        model,
        optimizer,
        scheduler,
        train_dataloader,
        eval_dataloader,
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
        dict(
            student=args.model_name_or_path,
            teacher=args.teacher_name_or_path,
            hidden_layers=model.config.num_hidden_layers,
            weight_decay=args.weight_decay,
            learning_rate=args.learning_rate,
            lr_scheduler=args.lr_scheduler_type,
            num_epochs=args.num_train_epochs,
            train_batch_size=args.per_device_train_batch_size,
            gradient_accumulation_steps=args.gradient_accumulation_steps,
            num_warmup_steps=args.num_warmup_steps,
            mixed_precision=args.mixed_precision,
            seed=args.seed,
        )
    )

    # Training
    train_eval_loop(
        args,
        train_dataloader,
        eval_dataloader,
        accelerator,
        model,
        optimizer,
        scheduler,
        teacher,
    )

    accelerator.end_training()

    if args.output_dir is not None:
        accelerator.wait_for_everyone()
        unwrapped_model = accelerator.unwrap_model(model)
        unwrapped_model.save_pretrained(
            args.output_dir,
            is_main_process=accelerator.is_main_process,
            save_function=accelerator.save,
        )
        if accelerator.is_main_process:
            tokenizer.save_pretrained(args.output_dir)
        mlflow.transformers.log_model(
            unwrapped_model,
            args.output_dir,
            task="text-generation",
            registered_model_name="distil-clm",
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
