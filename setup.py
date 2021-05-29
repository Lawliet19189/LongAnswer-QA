import torch
from transformers import LongformerTokenizerFast

from datasets import load_dataset

tokenizer = LongformerTokenizerFast.from_pretrained('allenai/longformer-base-4096')

dataset = load_dataset("natural_questions")


def convert_to_features(example, max_length=4096):
    try:
        start_idx, end_idx = example['annotations']['long_answer'][0]['start_token'], \
                             example['annotations']['long_answer'][0]['end_token']
        answer = " ".join([itm for idx, itm in enumerate(example['document']['tokens']['token'][start_idx:end_idx]) if
                           not example['document']['tokens']['is_html'][start_idx + idx]])
        context = " ".join([itm for idx, itm in enumerate(example['document']['tokens']['token'][:]) if
                            not example['document']['tokens']['is_html'][idx]])
        query = example['question']['text']
        input_pairs = [query, context]
        # encodings = tokenizer.encode_plus(input_pairs, pad_to_max_length=True, truncation=True, max_length=max_length)
        encodings = tokenizer.encode_plus(input_pairs, padding='max_length', truncation=True, max_length=max_length)
        context_encodings = tokenizer.encode_plus(context)

        start_byte_idx = context.find(answer)
        end_byte_idx = start_byte_idx + len(answer) - 1

        if start_idx == -1:
            encodings.update({'start_positions': 0,
                              'end_positions': 0,
                              'attention_mask': encodings['attention_mask']})

            return encodings
        start_positions_context = context_encodings.char_to_token(start_byte_idx)
        end_positions_context = context_encodings.char_to_token(end_byte_idx)

        sep_idx = encodings['input_ids'].index(tokenizer.sep_token_id)

        start_positions = start_positions_context + sep_idx + 1
        end_positions = end_positions_context + sep_idx + 1

        if end_positions > 4096:
            start_positions, end_positions = 0, 0

        encodings.update({'start_positions': start_positions,
                          'end_positions': end_positions,
                          'attention_mask': encodings['attention_mask']})
    except Exception as e:
        print(e, example['id'])
    return encodings

train_dataset = dataset['train']
train_dataset = train_dataset.map(convert_to_features, num_proc=20)
train_dataset.set_format(type='numpy', columns=['input_ids', 'attention_mask', 'end_positions', 'start_positions'])
train_dataset.flatten_indices().save_to_disk("datasets/train_data_4096")

valid_dataset = dataset['validation']
valid_dataset = valid_dataset.map(convert_to_features, num_proc=20, load_from_cache_file=False)
valid_dataset.set_format(type='numpy', columns=['input_ids', 'attention_mask', 'end_positions', 'start_positions'])
valid_dataset.flatten_indices().save_to_disk("datasets/validation_data_4096")
