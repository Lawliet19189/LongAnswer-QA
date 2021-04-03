import logging
import argparse
from transformers import (
HfArgumentParser,
TrainingArguments
)
logger = logging.getLogger(__name__)


def model_args(parser):
    """
    model/config/tokenizer arguments
    """
    parser.add_argument(
        '--model_name_or_path',
        type=str,
        default="allenai/longformer-base-4096",
        help='Path to pretrained model or model identifier from hugginface.co/models'
    )
    parser.add_argument(
        '--tokenizer_name',
        type=str,
        default=None,
        help="Pretrained tokenizer name or path"
    )
    parser.add_argument(
        '--cache_dir',
        type=str,
        default=None,
        help="Where do you want to store the pretrained models downloaded from s3"
    )


def setup_args(parser):
    """
    model/config/tokenizer arguments
    """
    parser.add_argument(
        '--max_len',
        type=int,
        default=4096,
        help='PMax input length for the source text'
    )


def get_train_args():
    parser = argparse.ArgumentParser('Train spanBert model on NQ')

    model_args(parser)

    parser.add_argument('--name',
                        type=str,
                        default='SpanBert',
                        help='Train Name which we use to identify the training. Need not be unique.')
    parser.add_argument('--train_file_path',
                        type=str,
                        default='data/train_data.pt',
                        help='Path for cached train dataset')
    parser.add_argument('--valid_file_path',
                        type=str,
                        default='data/valid_data.pt',
                        help='Path for cached valid dataset')
    parser.add_argument('--max_len',
                        type=str,
                        default=4096,
                        help='Max input length for the source text.')
    parser.add_argument('--evaluation_strategy',
                        type=str,
                        default="steps")
    parser.add_argument('--eval_steps',
                        type=int,
                        default=100)
    parser.add_argument('--num_epochs',
                        type=int,
                        default=3,
                        help='Number of epochs for which to train. Negative means forever.')
    parser.add_argument('--n_gpu',
                        type=int,
                        default=2,
                        help='Number of GPU to train the model on.')
    parser.add_argument('--output_dir',
                        type=str,
                        default='./save/train/longFormer',
                        help='Directory to store the model in (If already exists, use --overwrite_output_dir).')
    parser.add_argument('--overwrite_output_dir',
                        type=bool,
                        default=True,
                        help='Overwrite existing model directory')
    parser.add_argument('--per_device_train_batch_size',
                        type=int,
                        default=1)
    parser.add_argument('--per_device_eval_batch_size',
                        type=int,
                        default=1)
    parser.add_argument('--gradient_accumulation_steps',
                        type=int,
                        default=1)
    parser.add_argument('--learning_rate',
                        type=float,
                        default=1e-4,
                        help='Learning rate for the model')
    parser.add_argument('--num_train_epochs',
                        type=int,
                        default=3,
                        help='Number of epochs for which to train. Negative means forever.')
    parser.add_argument('--do_train',
                        type=bool,
                        default=True,
                        help='Whether to train the model.')
    parser.add_argument('--seed',
                        type=int,
                        default=42,
                        help='Seed to reproduce.')
    parser.add_argument('--do_eval',
                        type=bool,
                        default=True,
                        help='Whether to evaluate the model after training.')
    parser.add_argument('--prediction_loss_only',
                        type=bool,
                        default=True)
    parser.add_argument('--local_rank',
                        type=int,
                        default=-1)
    parser.add_argument('--nproc_per_node',
                        type=int,
                        default=2)
    parser.add_argument('--fp16',
                        type=bool,
                        default=True)
    # parser.add_argument('--sharded_ddp',
    #                    type=str,
    #                    default="zero_dp_3 auto_wrap")
    parser.add_argument('--deepspeed',
                       type=str,
                       default="ds_config.json")
    args = parser.parse_args()
    return args

