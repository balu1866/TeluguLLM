import pandas as pd
import random
import json


df = pd.read_csv("hf://datasets/Telugu-LLM-Labs/telugu_alpaca_yahma_cleaned_filtered_romanized/yahma_alpaca_cleaned_telugu_filtered_and_romanized.csv")


df = df[['instruction', 'output', 'telugu_instruction', 'telugu_output', 'telugu_transliterated_instruction', 'telugu_transliterated_output']]

eng_tel_map = {}
eng_tel_list =[]

eng_tel_trans_map = {}
eng_tel_trans_list =[]

tel_trans_eng_map = {}
tel_trans_eng_list =[]

tel_trans_tel_map = {}
tel_trans_tel_list =[]

tel_trans_tel_trans_map = {}
tel_trans_tel_trans_list =[]

tel_tel_trans_map = {}
tel_tel_trans_list =[]

tel_eng_map = {}
tel_eng_list =[]

tel_tel_map = {}
tel_tel_list =[]


for index, row in df.iterrows():
  # Access individual columns using their names
  instruction = row['instruction']
  output = row['output']
  telugu_instruction = row['telugu_instruction']
  telugu_output = row['telugu_output']
  telugu_transliterated_instruction = row['telugu_transliterated_instruction']
  telugu_transliterated_output = row['telugu_transliterated_output']

  # Telugu output
  eng_tel_list.append({
      'english_instruction': instruction,
      'telugu_output': telugu_output,
  })

  tel_tel_list.append({
      'telugu_instruction': telugu_instruction,
      'telugu_output': telugu_output,
  })

  tel_trans_tel_list.append({
      'telugu_transliterated_instruction': telugu_transliterated_instruction,
      'telugu_output': telugu_output,
  })

  # Telugu transliterated output
  eng_tel_trans_list.append({
      'english_instruction': instruction,
      'telugu_transliterated_output': telugu_transliterated_output,
  })

  tel_tel_trans_list.append({
      'telugu_instruction': telugu_instruction,
      'telugu_transliterated_output': telugu_transliterated_output,
  })

  tel_trans_tel_trans_list.append({
      'telugu_transliterated_instruction': telugu_transliterated_instruction,
      'telugu_transliterated_output': telugu_transliterated_output,
  })

  # English output
  tel_eng_list.append({
      'telugu_instruction': telugu_instruction,
      'english_output': output,
  })

  tel_trans_eng_list.append({
      'telugu_transliterated_instruction': telugu_transliterated_instruction,
      'english_output': output,
  })


dataset_list = []
dataset_list.extend(eng_tel_list)
dataset_list.extend(tel_tel_list)
dataset_list.extend(tel_trans_tel_list)

dataset_list.extend(eng_tel_trans_list)
dataset_list.extend(tel_tel_trans_list)
dataset_list.extend(tel_trans_tel_trans_list)

dataset_list.extend(tel_eng_list)
dataset_list.extend(tel_trans_eng_list)

random.shuffle(dataset_list)


def format_record(example):
    if 'english_instruction' in example and 'telugu_output' in example:
        return {
            "prompt": f"Translate the following instruction to Telugu:\n{example['english_instruction']}",
            "completion": example["telugu_output"]
        }

    elif 'telugu_instruction' in example and 'telugu_output' in example:
        return {
            "prompt": f"Rewrite the following in Telugu:\n{example['telugu_instruction']}",
            "completion": example["telugu_output"]
        }

    elif 'telugu_transliterated_instruction' in example and 'telugu_output' in example:
        return {
            "prompt": f"Translate this transliterated instruction to Telugu:\n{example['telugu_transliterated_instruction']}",
            "completion": example["telugu_output"]
        }

    elif 'english_instruction' in example and 'telugu_transliterated_output' in example:
        return {
            "prompt": f"Translate to Telugu (transliterated):\n{example['english_instruction']}",
            "completion": example["telugu_transliterated_output"]
        }

    elif 'telugu_instruction' in example and 'telugu_transliterated_output' in example:
        return {
            "prompt": f"Transliterate the following Telugu instruction:\n{example['telugu_instruction']}",
            "completion": example["telugu_transliterated_output"]
        }


    elif 'telugu_transliterated_instruction' in example and 'telugu_transliterated_output' in example:
        return {
            "prompt": f"Complete this transliterated Telugu instruction:\n{example['telugu_transliterated_instruction']}",
            "completion": example["telugu_transliterated_output"]
        }

    elif 'telugu_instruction' in example and 'english_output' in example:
        return {
            "prompt": f"Translate this Telugu instruction to English:\n{example['telugu_instruction']}",
            "completion": example["english_output"]
        }

    elif 'telugu_transliterated_instruction' in example and 'english_output' in example:
        return {
            "prompt": f"Translate this transliterated Telugu instruction to English:\n{example['telugu_transliterated_instruction']}",
            "completion": example["english_output"]
        }

    else:
        return None

formatted_data = []

for item in dataset_list:
    formatted = format_record(item)
    if formatted:  # skip malformed
        formatted_data.append(formatted)

with open('dataset.json', 'w', encoding='utf-8') as file:
  for item in formatted_data:
    file.write(json.dumps(item) + '\n')

print(f"Total records in the dataset: {len(formatted_data)}")

