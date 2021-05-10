import logging
import argparse
from transformers import (
HfArgumentParser,
TrainingArguments
)

import dataclasses
import logging
import os
import sys
from dataclasses import dataclass, field
from typing import Dict, List, Optional

logger = logging.getLogger(__name__)


@dataclass
class ModelArguments:
    """
    Arguments pertaining to which model/config/tokenizer we are going to fine-tune from.
    """

    model_name_or_path: str = field(
        default="allenai/longformer-base-4096", metadata={"help": "Path to pretrained model or model identifier from hugginface.co/models"}
    )
    tokenizer_name: Optional[str] = field(
        default=None, metadata={"help": "Pretrained tokenizer name or path if not the same as model_name"}
    )
    cache_dir: Optional[str] = field(
        default=None, metadata={"help": "Where do you want to store the pretrained models downloaded from s3"}
    )


@dataclass
class DataTrainingArguments:
    """
    Arguments pertaining to what data we are going to input our model for training and eval.
    """
    train_file_path: Optional[str] = field(
        default='dataset/train_data_4096.pt',
        metadata={"help": "Path for cached train dataset"},
    )
    valid_file_path: Optional[str] = field(
        default='dataset/valid_data_4096.pt',
        metadata={"help": "Path for cached valid dataset"},
    )
    max_len: Optional[int] = field(
        default=4096,
        metadata={"help": "Max input length for the source text"},
    )


@dataclass
class TrainingArguments(TrainingArguments):
    """
    Training Arguments that we are extending from default transformer TrainingArguments
    """
    run_name: Optional[str] = field(
        default='LongFormer-optimal-search',
        metadata={"help": "Train Name which we use to identify the training. Need not be unique."},
    )
    evaluation_strategy: Optional[str] = field(
        default="epoch",
        metadata={"help": "The evaluation strategy to adopt during training. Possible values are: no, steps and epoch"},
    )
    #eval_steps: Optional[int] = field(
    #    default=1000,
    #    metadata={
    #        "help": "Number of update steps between two evaluations if evaluation_strategy='steps'. Will default to the same value as logging_steps if not set."},
    #)
    num_epochs: Optional[int] = field(
        default=7,
        metadata={"help": "Number of epochs for which to train. Negative means forever."},
    )
    output_dir: Optional[str] = field(
        default="./save/train/longFormer-optimal-search",
        metadata={
            "help": "Directory to store the model in (If already exists, use --overwrite_output_dir)."},
    )
    overwrite_output_dir: Optional[bool] = field(
        default=True,
        metadata={"help": "Overwrite existing model directory."},
    )
    per_device_train_batch_size: Optional[int] = field(
        default=128,
    )
    per_device_eval_batch_size: Optional[int] = field(
        default=128,
    )
    gradient_accumulation_steps: Optional[int] = field(
        default=1,
    )
    learning_rate: Optional[float] = field(
        default=1e-4,
        metadata={"help": "Learning rate for the model."},
    )
    num_train_epochs: Optional[int] = field(
        default=7,
        metadata={"help": "Number of epochs for which to train. Negative means forever."},
    )
    save_strategy: Optional[str] = field(
        default="epoch",
        metadata={"help": "Saving model strategy"}
    )
    do_train: Optional[bool] = field(
        default=True,
        metadata={
            "help": "Whether to train the model."},
    )
    do_eval: Optional[bool] = field(
        default=True,
        metadata={
            "help": "Whether to eval the model."},
    )
    prediction_loss_only: Optional[bool] = field(
        default=True,
        metadata={
            "help": "When performing evaluation and generating predictions, only returns the loss."},
    )
    seed: Optional[int] = field(
        default=48,
        metadata={"help": "Seed to control randomization."},
    )
    local_rank: Optional[int] = field(
        default=-1,
        metadata={"help": "Rank of the process during distributed training."},
    )
    fp16: Optional[bool] = field(
        default=True,
    )
    #fp16_backend: Optional[str] = field(
    #    default='apex',
    #)
    load_best_model_at_end: Optional[bool] = field(
        default=True,
    )
    report_to: Optional[str] = field(
        default='wandb',
    )






    # parser.add_argument('--sharded_ddp',
    #                    type=str,
    #                    default="zero_dp_3 auto_wrap")
    # parser.add_argument('--deepspeed',
    #                    type=str,
    #                    default="ds_config.json")

