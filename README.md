
# Telugu LLM Data Preparation

## 🧠 Project Overview

This project prepares instruction-based training data for fine-tuning or training a language model (LLM) to understand and generate responses in **Telugu**. It supports various combinations of:
- English to Telugu translation
- Telugu to English translation
- Transliteration tasks (Telugu <-> Romanized Telugu)

The processed output is saved in a `dataset.json` file in **prompt-completion** format, suitable for supervised fine-tuning of LLMs.

---

## 📦 Dependencies

The following Python packages are used:
- `langchain`
- `datasets`
- `bitsandbytes`
- `pandas`
- `torch`

Install them via:
```bash
pip install langchain datasets bitsandbytes pandas torch
```

---

## 🧾 Dataset

The raw dataset is loaded from Hugging Face:
```
hf://datasets/Telugu-LLM-Labs/telugu_alpaca_yahma_cleaned_filtered_romanized/yahma_alpaca_cleaned_telugu_filtered_and_romanized.csv
```

It includes:
- English instructions & outputs
- Native Telugu translations
- Transliterated Telugu (Roman script)

---

## ⚙️ Processing Steps

1. **Load and Filter Columns**  
   Extracts relevant columns for various instruction/output pairs.

2. **Create Instruction Variants**  
   Builds multiple task types:
   - English → Telugu
   - Telugu → Telugu (rewrites)
   - Transliteration tasks
   - Telugu → English

3. **Format as Prompt/Completion Pairs**  
   Each example is converted into a structure like:
   ```json
   {
     "prompt": "Translate the following instruction to Telugu:\nHow are you?",
     "completion": "నీవు ఎలా ఉన్నావు?"
   }
   ```

4. **Save as JSONL**  
   Writes each prompt-completion pair as a line in `dataset.json` — ready for LLM training.

---

## 📁 Output

The final file generated:
- `dataset.json` — A JSONL file with `{"prompt": "...", "completion": "..."}` on each line.

---

## 🚀 Use Cases

This dataset can be used for:
- Fine-tuning multilingual LLMs like Mistral, LLaMA, etc.
- Building chatbots that respond in Telugu
- Teaching LLMs transliteration tasks
