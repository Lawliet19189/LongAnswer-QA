from args import get_train_args
from transformers import (
HfArgumentParser,
TrainingArguments,
)
from args import model_args

import argparse
#parser = get_train_args()
#print(HfArgumentParser(TrainingArguments))
#parserb = argparse.ArgumentParser(parents=[parser, HfArgumentParser(TrainingArguments)])
parser = argparse.ArgumentParser('Train spanBert model on NQ')
#parsera = argparse.ArgumentParser(get_train_args())
#parserb = argparse.ArgumentParser(TrainingArguments)
#parser = argparse.ArgumentParser(parents=[parsera, parserb], conflict_handler='resolve')
#or arg in dict(TrainingArguments):
#    print(arg)
#HfArgumentParser(TrainingArguments)
#args = HfArgumentParser.parse_args()
#print(parser)

# parser = HfArgumentParser(TrainingArguments)
#
# model_args(parser)
# print(parser.parse_args())

Hparser = HfArgumentParser((TrainingArguments))

args = get_train_args()
args_dict = vars(args)
#with open('args.json', "r") as fh:

#training_args = Hparser.parse_json_file(json_file=os.path.abspath('args.json'))
training_args = Hparser.parse_dict(args_dict)[0]
print(training_args.do_train)