import os
import time

from datetime import timedelta
from dotenv import load_dotenv
from peft import LoraConfig, get_peft_model, TaskType, prepare_model_for_kbit_training
from transformers import AutoTokenizer, AutoModelForCausalLM
from transformers import TrainingArguments, Trainer, DataCollatorForLanguageModeling
from datasets import load_from_disk

load_dotenv()

model_name = os.getenv("MODEL_NAME")


def load_tokenized_data():
    """Load tokenized datasets from disk."""
    tokenized_train = load_from_disk("tokenized_data/tokenized_train_dataset")
    tokenized_val = load_from_disk("tokenized_data/tokenized_val_dataset")
    tokenized_test = load_from_disk("tokenized_data/tokenized_test_dataset")
    return tokenized_train, tokenized_val, tokenized_test


def load_tokenizer():
    """Load the tokenizer."""
    model_name = os.getenv("MODEL_NAME")
    tokenizer = AutoTokenizer.from_pretrained(model_name, use_fast=True)
    tokenizer.pad_token = tokenizer.eos_token
    print("Tokenizer loaded successfully")
    return tokenizer

def load_model():
    """Load the pre-trained model."""
    model_name = os.getenv("MODEL_NAME")
    model = AutoModelForCausalLM.from_pretrained(model_name, load_in_8bit=True, llm_int8_enable_fp32_cpu_offload=True)
    model = prepare_model_for_kbit_training(model)
    peft_config = LoraConfig(
        r=8,
        lora_alpha=32,
        lora_dropout=0.05,
        bias="none",
        task_type=TaskType.CAUSAL_LM,
    )
    model = get_peft_model(model, peft_config)
    print("Model loaded and PEFT configuration applied successfully")
    return model


def get_model_trainer(model, tokenizer, training_args, train_dataset, eval_dataset):
    
    data_collator = DataCollatorForLanguageModeling(
        tokenizer=tokenizer,
        mlm=False  
    )
    trainer = Trainer(
        model = model,
        tokenizer = tokenizer,
        args = training_args,
        train_dataset = train_dataset,
        eval_dataset = eval_dataset,
        data_collator = data_collator 
    )
    return trainer
    


if __name__ == "__main__":
    train, val, test = load_tokenized_data()
    tokenizer = load_tokenizer()
    model = load_model()
    print(model.hf_device_map)

    training_args = TrainingArguments(
        output_dir="./lora-mistral-telugu",  
        evaluation_strategy="epoch",   
        per_device_train_batch_size=4,
        per_device_eval_batch_size=4,
        gradient_accumulation_steps=4,
        num_train_epochs=2,
        save_strategy="epoch", 
        logging_steps=10,  
        learning_rate=2e-4, 
        weight_decay=0.01,  
        push_to_hub=False,
        report_to="none",
        label_names=["labels"],
        fp16=True,  
        load_best_model_at_end=True, 
    )


    trainer = get_model_trainer(model = model, 
                                tokenizer = tokenizer,
                                training_args = training_args, 
                                train_dataset = train, 
                                eval_dataset = val)
    
    start_time = time.time() 

    trainer.train() 

    end_time = time.time() 
    elapsed_time = end_time - start_time  
    formatted_time = str(timedelta(seconds=int(elapsed_time)))  
    print(f"Training time: {formatted_time}")  
    

    results = trainer.evaluate()
    print(results)  
    trainer.save_model("./lora-mistral-telugu") 
    tokenizer.save_pretrained("./lora-mistral-telugu")  

    print("Training complete. Model and tokenizer saved.")