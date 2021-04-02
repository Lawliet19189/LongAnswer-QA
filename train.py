import numpy as np
import torch
import os
from args import get_train_args
import util

from transformers import LongformerForQuestionAnswering, LongformerTokenizerFast, EvalPrediction
from transformers import (
    HfArgumentParser,
    DataCollator,
    Trainer,
    TrainingArguments,
    set_seed,
)


def main(args):
    # Set up logging and devices
    args.output_dir = util.get_save_dir(args.output_dir, args.name, training=True)
    log = util.get_logger(args.output_dir, args.name)

    Hparser = HfArgumentParser((TrainingArguments))

    args_dict = vars(args)

    training_args = Hparser.parse_dict(args_dict)[0]

    # Set seed
    set_seed(args.seed)

    tokenizer = LongformerTokenizerFast.from_pretrained(
        args.tokenizer_name if args.tokenizer_name else args.model_name_or_path,
        cache_dir=args.cache_dir,
    )
    model = LongformerForQuestionAnswering.from_pretrained(
        args.model_name_or_path,
        cache_dir=args.cache_dir,
    )

    # Get datasets
    #log.info('loading data from: ', args.train_file_path, ' and ', args.valid_file_path)
    train_dataset = torch.load(args.train_file_path)
    valid_dataset = torch.load(args.valid_file_path)
    #log.info('Data loading done')
    #print(vars(training_args))
    #print(training_args.parallel_mode)
    # Initialize our Trainer
    trainer = Trainer(
        model=model,
        args=training_args,
        train_dataset=train_dataset,
        eval_dataset=valid_dataset,
        data_collator=util.DummyDataCollator(),
        #       prediction_loss_only=True,
    )

    # Training
    if args.do_train:
        trainer.train(
            model_path=args.output_dir
        )
        trainer.save_model()
        # For convenience, we also re-save the tokenizer to the same directory,
        # so that you can share your model easily on huggingface.co/models =)
        if trainer.is_world_master():
            tokenizer.save_pretrained(args.output_dir)

    # Evaluation
    results = {}
    if args.do_eval and args.local_rank in [-1, 0]:
        log.info("*** Evaluate ***")

        eval_output = trainer.evaluate()

        output_eval_file = os.path.join(args.output_dir, "eval_results.txt")
        with open(output_eval_file, "w") as writer:
            log.info("***** Eval results *****")
            for key in sorted(eval_output.keys()):
                log.info("  %s = %s", key, str(eval_output[key]))
                writer.write("%s = %s\n" % (key, str(eval_output[key])))

        results.update(eval_output)

    return results


def _mp_fn(index):
    # For xla_spawn (TPUs)
    main(get_train_args())


if __name__=="__main__":
    main(get_train_args())
