import os
import torch
from dataclasses import dataclass
from typing import Any, Dict, List, Union
from PIL import Image

from datasets import load_dataset
from transformers import (
    AutoProcessor,
    BitsAndBytesConfig,
    TrainingArguments,
    Qwen2_5_VLForConditionalGeneration,
)
from peft import LoraConfig, get_peft_model, PeftModel # Import PeftModel
from trl import SFTTrainer

# --- Custom Data Collator ---
@dataclass
class DataCollatorForQwenVL:
    processor: AutoProcessor

    def __call__(self, features: List[Dict[str, Any]]) -> Dict[str, torch.Tensor]:
        texts = [feature["text"] for feature in features]
        
        # Explicitly load images from file paths
        image_paths = [feature["images"] for feature in features]
        try:
            images = [Image.open(path) for path in image_paths]
        except Exception as e:
            print(f"Error loading image paths: {image_paths}")
            print(f"Error: {e}")
            raise e
        
        batch = self.processor(text=texts, images=images, return_tensors="pt", padding=True)
        batch["labels"] = batch["input_ids"].clone()
        return batch

def main():
    # --- Run Mode Configuration ---
    run_mode = os.environ.get("RUN_MODE", "train")  # Default to "train"

    # --- Base Configuration ---
    base_model_id = "Qwen/Qwen2.5-VL-3B-Instruct"
    multimodal_dataset_path = "train_dataset.jsonl"
    knowledge_adapter_path = "out/adapters/knowledge_adapter" # Input from Stage 1
    final_adapter_path = "out/adapters/multimodal_adapter" # Final output
    os.makedirs("out/adapters", exist_ok=True)

    # --- Mode-specific Adjustments ---
    if run_mode == "test":
        print("--- Running in TEST mode ---")
        num_train_epochs = 0.1  # Run for a fraction of an epoch
        max_train_samples = 10   # Use only 10 samples
    else:
        print("--- Running in TRAIN mode ---")
        num_train_epochs = 3     # Original value
        max_train_samples = None # Use the full dataset

    print("--- Starting Stage 2: Multimodal Task Tuning ---")

    # Load processor
    processor = AutoProcessor.from_pretrained(base_model_id, trust_remote_code=True, local_files_only=True)

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
        device_map={"": 0},
        trust_remote_code=True,
        local_files_only=True
    )

    # --- Load the knowledge adapter from Stage 1 (if it exists) ---
    if os.path.exists(knowledge_adapter_path):
        print(f"Loading knowledge adapter from {knowledge_adapter_path}...")
        model = PeftModel.from_pretrained(model, knowledge_adapter_path)
        print("Knowledge adapter loaded successfully.")
    else:
        print(f"Knowledge adapter not found at {knowledge_adapter_path}. Skipping.")
        print("Proceeding with the base model for Stage 2.")
    
    # --- Configure a NEW LoRA adapter for the multimodal task ---
    lora_config_multimodal = LoraConfig(
        r=32,
        lora_alpha=64,
        lora_dropout=0.1,
        bias="none",
        target_modules=["q_proj", "v_proj"], # Use the safe, restricted modules
        task_type="CAUSAL_LM"
    )
    model = get_peft_model(model, lora_config_multimodal)
    print("Trainable parameters for Stage 2:")
    model.print_trainable_parameters()

    # Load and process the dataset
    dataset = load_dataset("json", data_files=multimodal_dataset_path, split="train")
    if max_train_samples:
        dataset = dataset.select(range(max_train_samples))
    dataset = dataset.rename_column("image", "images")

    # Load the detailed prompt from the file
    with open("prompts/classification_v2.txt", "r") as f:
        prompt_text = f.read()

    def create_chat_template(examples):
        prompts = []
        for i in range(len(examples['images'])):
            chat = [
                {"role": "system", "content": "You are a helpful assistant."},
                {"role": "user", "content": [{"type": "image"}, {"type": "text", "text": prompt_text}]},
                {"role": "assistant", "content": examples['conversations'][i][1]['value']}
            ]
            prompts.append(processor.apply_chat_template(chat, tokenize=False, add_generation_prompt=False))
        return {"text": prompts}

    processed_dataset = dataset.map(create_chat_template, batched=True, remove_columns=["id", "conversations"], num_proc=1)

    # Instantiate the custom data collator
    data_collator = DataCollatorForQwenVL(processor)

    # Set up TrainingArguments
    training_args = TrainingArguments(
        output_dir="out/results/multimodal_results",
        num_train_epochs=num_train_epochs,
        per_device_train_batch_size=1,
        gradient_accumulation_steps=8,
        learning_rate=1e-4,
        logging_steps=10,
        save_strategy="epoch",
        fp16=True,
        remove_unused_columns=False,
    )

    # Trainer for multimodal SFT
    trainer = SFTTrainer(
        model=model,
        args=training_args,
        train_dataset=processed_dataset,
        data_collator=data_collator,
    )

    # Train the final adapter
    trainer.train()

    # Save the final adapter
    print(f"Saving final adapter to {final_adapter_path}")
    model.save_pretrained(final_adapter_path)

if __name__ == "__main__":
    main()
