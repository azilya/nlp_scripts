import argparse
import os
import random
from itertools import chain

import mlflow
import torch
from datasets import load_dataset
from mlflow.data.huggingface_dataset import from_huggingface
from transformers import MODEL_MAPPING, SchedulerType

MODEL_CONFIG_CLASSES = list(MODEL_MAPPING.keys())
MODEL_TYPES = tuple(conf.model_type for conf in MODEL_CONFIG_CLASSES)


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description=(
            "Finetune and distillate a transformers model"
            "on a causal language modeling task"
        )
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
        help=(
            "The percentage of the train set used as validation set"
            "in case there's no validation split"
        ),
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
        help=(
            "If passed, will use a slow tokenizer"
            "(not backed by the 🤗 Tokenizers library)."
        ),
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
        help=(
            "Total number of training steps to perform."
            " If provided, overrides num_train_epochs."
        ),
    )
    parser.add_argument(
        "--gradient_accumulation_steps",
        type=int,
        default=1,
        help=(
            "Number of updates steps to accumulate before "
            "performing a backward/update pass."
        ),
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
            "Optional input sequence length after tokenization. The training dataset"
            " will be truncated in block of this size for training. Default to the"
            " model max input length for single sentence inputs"
            " (take into account special tokens)."
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
        help="Whether the various states should be saved at the end of every n steps, "
        "or 'epoch' for each epoch.",
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
            "It is an option to create the model as an empty shell, then only"
            " materialize its parameters when the pretrained weights are loaded."
            " If passed, LLM loading time and RAM consumption will be benefited."
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


def check_if_save(args, model, optimizer, output_dir):
    if args.output_dir is not None:
        dirs = [
            os.path.join(args.output_dir, f.name)
            for f in os.scandir(args.output_dir)
            if f.is_dir()
        ]
        dirs.sort(key=os.path.getctime)

        if args.checkpoint_limit is not None and len(dirs) >= args.checkpoint_limit:
            chckp_diff = 1 + (len(dirs) - args.checkpoint_limit)
            for i in range(chckp_diff):
                os.rmdir(dirs[i])
        output_dir = os.path.join(args.output_dir, output_dir)
        torch.save(model.state_dict(), output_dir + "/state_dict.bin")
        torch.save(optimizer, output_dir + "/optimizer.bin")


def load_examples(args, tokenizer, logger):
    if args.dataset_name is not None:
        raw_datasets = load_dataset(args.dataset_name, args.dataset_config_name)
        if "validation" not in raw_datasets.keys():
            raw_datasets["validation"] = load_dataset(
                args.dataset_name,
                args.dataset_config_name,
                # split=f"train[:{args.validation_split_percentage}%]",
                split="test_gen",
            )
            raw_datasets["train"] = load_dataset(
                args.dataset_name,
                args.dataset_config_name,
                # split=f"train[{args.validation_split_percentage}%:]",
                split="train_gen",
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

    def group_texts(examples):
        """
        Main data processing function that will concatenate all texts
        from our dataset and generate chunks of block_size.
        """
        # Concatenate all texts.
        concatenated_examples = {k: list(chain(*examples[k])) for k in examples.keys()}
        total_length = len(concatenated_examples[list(examples.keys())[0]])
        # We drop the small remainder, and if the total_length < block_size
        #  we exclude this batch and return an empty dict.
        # We could add padding if the model supported it instead of this drop,
        # you can customize this part to your needs.
        total_length = (total_length // block_size) * block_size
        # Split by chunks of max_len.
        result = {
            k: [t[i : i + block_size] for i in range(0, total_length, block_size)]
            for k, t in concatenated_examples.items()
        }
        result["labels"] = result["input_ids"].copy()
        return result

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
