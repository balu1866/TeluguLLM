from datasets import load_dataset, Dataset
from transformers import AutoTokenizer
from dotenv import load_dotenv
import os

load_dotenv()

tokenizer = AutoTokenizer.from_pretrained(os.getenv("MODEL_NAME"), use_fast=True)
tokenizer.pad_token = tokenizer.eos_token

dataset = load_dataset("json", data_files="dataset.json")

split_dataset = dataset["train"].train_test_split(test_size=0.2)

train_dataset = split_dataset["train"]
temp_dataset = split_dataset["test"].train_test_split(test_size=0.5)

val_dataset = temp_dataset["train"]
test_dataset = temp_dataset["test"]


def tokenize(record):
  full_prompt = record['prompt'] + "\n" + record['completion']
  tokenized = tokenizer(
      full_prompt,
      max_length=512,
      padding="max_length",
      truncation=True
  )
  prompt_ids = tokenizer(record['prompt'], truncation=True, max_length=512)['input_ids']
  prompt_ids_len = len(prompt_ids)

  input_ids = tokenized['input_ids']
  # if not isinstance(input_ids,  list):
  #   input_ids = input_ids.tolist()
  labels = input_ids.copy()  # make a copy of input_ids for labels
  labels[:prompt_ids_len] = [-100] * prompt_ids_len
  tokenized['labels'] = labels
  return tokenized

def write_tokenized_dataset(tokenized_dataset, filename):
    tokenized_dataset.save_to_disk(filename)
    print(f"Tokenized dataset saved to {filename}")

def read_tokenized_dataset(filename):
    return Dataset.load_from_disk(filename)

tokenized_train = train_dataset.map(tokenize, remove_columns=train_dataset.column_names)
tokenized_val = val_dataset.map(tokenize, remove_columns=val_dataset.column_names)
tokenized_test = test_dataset.map(tokenize, remove_columns=test_dataset.column_names)

write_tokenized_dataset(tokenized_train, "tokenized_data/tokenized_train_dataset")
write_tokenized_dataset(tokenized_val, "tokenized_data/tokenized_val_dataset")
write_tokenized_dataset(tokenized_test, "tokenized_data/tokenized_test_dataset")
