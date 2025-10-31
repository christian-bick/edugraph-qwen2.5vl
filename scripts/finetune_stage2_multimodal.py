import os
import torch
from dataclasses import dataclass
from typing import Any, Dict, List, Union
from PIL import Image
from dotenv import load_dotenv
import numpy as np
import evaluate

from datasets import load_dataset
from transformers import (
    AutoProcessor,
    BitsAndBytesConfig,
    TrainingArguments,
    Qwen2_5_VLForConditionalGeneration,
    Trainer,
)
from peft import get_peft_model, PeftModel
from scripts.config import get_config

# --- Custom Data Collator ---
@dataclass
class DataCollatorForQwenVL:
    processor: AutoProcessor

    def __call__(self, features: List[Dict[str, Any]]) -> Dict[str, torch.Tensor]:
        texts = [feature["text"] for feature in features]
        image_paths = [feature["image"] for feature in features]

        try:
            # Ensure all images are converted to RGB format before processing
            images = [Image.open(path).convert("RGB") for path in image_paths]
        except Exception as e:
            print(f"Error loading image paths: {image_paths}")
            print(f"Error: {e}")
            raise e

        batch = self.processor(text=texts, images=images, return_tensors="pt", padding=True)
        batch["labels"] = batch["input_ids"].clone()
        return batch

def compute_metrics(eval_pred):
    logits, labels = eval_pred.predictions, eval_pred.label_ids

    # If logits is a tuple, extract the first element
    if isinstance(logits, tuple):
        logits = logits[0]

    predictions = np.argmax(logits, axis=-1)
    # Ignore padding tokens (-100)
    mask = labels != -100
    # Use the evaluate library's accuracy metric
    return evaluate.load("accuracy").compute(predictions=predictions[mask], references=labels[mask])


def main():
    # Load environment variables from .env file for local development
    load_dotenv()

    # --- Configuration ---
    run_mode = os.environ.get("RUN_MODE", "train")
    model_size = os.environ.get("MODEL_SIZE", "3b")
    
    # Get model and training configurations
    model_config = get_config(model_size)
    stage2_config = model_config.stage2
    base_model_id = f"Qwen/Qwen2.5-VL-{model_size.upper()}-Instruct"
    
    multimodal_dataset_path = "train_dataset.jsonl"
    knowledge_adapter_path = "out/adapters/knowledge_adapter" # Input from Stage 1
    final_adapter_path = "out/adapters/multimodal_adapter" # Final output
    os.makedirs("out/adapters", exist_ok=True)

    # --- Mode-specific Adjustments ---
    if run_mode == "test":
        print("--- Running in TEST mode ---")
        num_train_epochs = 1
        max_train_samples = 30
    else:
        print("--- Running in TRAIN mode ---")
        num_train_epochs = stage2_config.num_train_epochs
        max_train_samples = None

    print("--- Starting Stage 2: Multimodal Task Tuning ---")

    # Load processor
    processor = AutoProcessor.from_pretrained(base_model_id, trust_remote_code=True)

    # Configure QLoRA
    bnb_config = BitsAndBytesConfig(
        load_in_4bit=True,
        bnb_4bit_quant_type="nf4",
        bnb_4bit_compute_dtype=torch.bfloat16,
        bnb_4bit_use_double_quant=True
    )

    # Load base model
    model = Qwen2_5_VLForConditionalGeneration.from_pretrained(
        base_model_id,
        quantization_config=bnb_config,
        device_map="auto",
        trust_remote_code=True
    )

    # --- Load the knowledge adapter from Stage 1 (if it exists) ---
    if os.path.exists(knowledge_adapter_path):
        print(f"Loading and merging knowledge adapter from {knowledge_adapter_path}...")
        model = PeftModel.from_pretrained(model, knowledge_adapter_path)
        model = model.merge_and_unload()
        print("Knowledge adapter loaded and merged successfully.")
    else:
        print(f"Knowledge adapter not found at {knowledge_adapter_path}. Skipping.")
        print("Proceeding with the base model for Stage 2.")
    
    # --- Configure a NEW LoRA adapter for the multimodal task ---
    model = get_peft_model(model, stage2_config.lora_config)
    print("Trainable parameters for Stage 2:")
    model.print_trainable_parameters()

    # Load and process the dataset
    dataset = load_dataset("json", data_files=multimodal_dataset_path, split="train")

    # Split the dataset into training and evaluation sets (90% train, 10% eval)
    dataset_split = dataset.train_test_split(test_size=0.1)
    train_dataset = dataset_split["train"]
    eval_dataset = dataset_split["test"]
    if max_train_samples:
        train_dataset = train_dataset.select(range(max_train_samples))
        # Also limit the eval set for quick tests, ensuring it's not empty
        eval_dataset = eval_dataset.select(range(max(1, int(max_train_samples * 0.1))))

    # The 'image' column is automatically decoded by the datasets library

    # Load the detailed prompt from the file
    with open("prompts/classification_v2.txt", "r") as f:
        prompt_text = f.read()

    def create_chat_template(examples):
        prompts = []
        for i in range(len(examples['image'])):
            chat = [
                {"role": "system", "content": prompt_text},
                {"role": "user", "content": [{"type": "image"}]},
                {"role": "assistant", "content": examples['conversations'][i][1]['value']}
            ]
            prompts.append(processor.apply_chat_template(chat, tokenize=False, add_generation_prompt=False))
        return {"text": prompts}

    print("Processing training dataset...")
    processed_train_dataset = train_dataset.map(create_chat_template, batched=True, remove_columns=["id", "conversations"], num_proc=1)

    print("Processing evaluation dataset...")
    processed_eval_dataset = eval_dataset.map(create_chat_template, batched=True, remove_columns=["id", "conversations"], num_proc=1)

    # Instantiate the custom data collator
    data_collator = DataCollatorForQwenVL(processor)

    # Set up TrainingArguments
    training_args = TrainingArguments(
        output_dir="out/results/multimodal_results",
        num_train_epochs=num_train_epochs,
        per_device_train_batch_size=1,
        gradient_accumulation_steps=8,
        learning_rate=stage2_config.learning_rate, # Use learning rate from config
        logging_steps=10,
        save_strategy="epoch",
        eval_strategy="epoch",
        fp16=True,
        remove_unused_columns=False,
        max_grad_norm=1.0, # Add gradient clipping
    )

    # Initialize Trainer
    trainer = Trainer(
        model=model,
        args=training_args,
        train_dataset=processed_train_dataset,
        eval_dataset=processed_eval_dataset,
        compute_metrics=compute_metrics,
        data_collator=data_collator,
    )

    # Train the final adapter
    trainer.train()

    # Save the final adapter
    print(f"Saving final adapter to {final_adapter_path}")
    model.save_pretrained(final_adapter_path)

if __name__ == "__main__":
    main()
