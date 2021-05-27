import torch
import os
import util

import transformers
from transformers import LongformerForQuestionAnswering, LongformerTokenizerFast
from transformers import (
    HfArgumentParser,
    Trainer,
    set_seed,
)
from transformers.trainer_utils import is_main_process
from args import ModelArguments, DataTrainingArguments, TrainingArguments
import os
from fairscale.nn.data_parallel import FullyShardedDataParallel as FullyShardedDDP
import datasets

os.environ['WANDB_PROJECT'] = "NaturalQuestions-LongFormer"
os.environ['WANDB_LOG_MODEL'] = "true"
os.environ['WANDB_WATCH'] = "all"


def main():

    parser = HfArgumentParser((ModelArguments, DataTrainingArguments, TrainingArguments))
    model_args, data_args, training_args = parser.parse_args_into_dataclasses()
    print("+++++++++++++++++++++++++++++++++++++++++++++++++++++++++++")
    print(model_args, "\n\n", data_args, "\n\n", training_args, "\n\n")
    print("+++++++++++++++++++++++++++++++++++++++++++++++++++++++++++")
    logger = util.setup_logger(training_args)
    logger.warning(
        f"Process rank: {training_args.local_rank}, device: {training_args.device}, n_gpu: {training_args.n_gpu}"
        + f"distributed training: {bool(training_args.local_rank != -1)}, 16-bits training: {training_args.fp16}"
    )
    # Set the verbosity to info of the Transformers logger (on main process only):
    if is_main_process(training_args.local_rank):
        transformers.utils.logging.set_verbosity_info()
    logger.info(f"Training/evaluation parameters {training_args}")

    # Set seed
    set_seed(training_args.seed)

    # load model
    tokenizer = LongformerTokenizerFast.from_pretrained(
        model_args.tokenizer_name if model_args.tokenizer_name else model_args.model_name_or_path,
        cache_dir=model_args.cache_dir,
    )
    model = LongformerForQuestionAnswering.from_pretrained(
        model_args.model_name_or_path,
        #gradient_checkpointing=True,
        cache_dir=model_args.cache_dir
    ).to("cpu")
    #model = FullyShardedDDP(model)
    #train_dataset = torch.load(data_args.train_file_path)
    #valid_dataset = torch.load(data_args.valid_file_path)
    train_dataset = datasets.load_from_disk(data_args.train_file_path)
    #train_dataset = train_dataset.map(util.convert_to_tensors)
    train_dataset.set_format(type='torch', columns=['input_ids', 'attention_mask', 'end_positions', 'start_positions'])
    valid_dataset = datasets.load_from_disk(data_args.valid_file_path)
    valid_dataset.set_format(type='torch', columns=['input_ids', 'attention_mask', 'end_positions', 'start_positions'])
    #valid_dataset = valid_dataset.map(util.convert_to_tensors)
    print("+++++++++++++++++++++++++++++++++++++++++++++++++++++++++++")
    print(train_dataset.shape, valid_dataset.shape)
    print("+++++++++++++++++++++++++++++++++++++++++++++++++++++++++++")
    trainer = Trainer(
        model=model,
        args=training_args,
        train_dataset=train_dataset,
        eval_dataset=valid_dataset,
        data_collator=util.DummyDataCollator(),
        #       prediction_loss_only=True,
    )

    # Training
    if training_args.do_train:
        trainer.train(
            #model_path=training_args.output_dir
        )
        trainer.save_model()

        if trainer.is_world_process_zero():
            tokenizer.save_pretrained(training_args.output_dir)

    # Evaluation
    results = {}
    if training_args.do_eval and training_args.local_rank in [-1, 0]:
        logger.info("*** Evaluate ***")

        eval_output = trainer.evaluate()

        output_eval_file = os.path.join(training_args.output_dir, "eval_results.txt")
        with open(output_eval_file, "w") as writer:
            logger.info("***** Eval results *****")
            for key in sorted(eval_output.keys()):
                logger.info("  %s = %s", key, str(eval_output[key]))
                writer.write("%s = %s\n" % (key, str(eval_output[key])))

        results.update(eval_output)

    return results


def _mp_fn(index):
    # For xla_spawn (TPUs)
    main()


if __name__ == "__main__":
    main()
