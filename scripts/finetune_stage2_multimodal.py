import os
import torch
from dataclasses import dataclass
from typing import Any, Dict, List, Union

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
        images = [feature["images"] for feature in features]
        
        batch = self.processor(text=texts, images=images, return_tensors="pt", padding=True)
        batch["labels"] = batch["input_ids"].clone()
        return batch

def main():
    # --- Configuration ---
    base_model_id = "Qwen/Qwen2.5-VL-3B-Instruct"
    multimodal_dataset_path = "train_dataset.jsonl"
    knowledge_adapter_path = "out/adapters/knowledge_adapter" # Input from Stage 1
    final_adapter_path = "out/adapters/multimodal_adapter" # Final output
    os.makedirs("out/adapters", exist_ok=True)

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

    # --- Load the knowledge adapter from Stage 1 ---
    print(f"Loading knowledge adapter from {knowledge_adapter_path}...")
    model = PeftModel.from_pretrained(model, knowledge_adapter_path)
    print("Knowledge adapter loaded successfully.")
    
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
    dataset = dataset.rename_column("image", "images")

    def create_chat_template(examples):
        prompts = []
        for i in range(len(examples['images'])):
            chat = [
                {"role": "system", "content": "You are a helpful assistant."},
                {"role": "user", "content": [{"type": "image"}, {"type": "text", "text": "Classify this learning material using the provided ontology."}]},
                {"role": "assistant", "content": examples['conversations'][i][1]['value']}
            ]
            prompts.append(processor.apply_chat_template(chat, tokenize=False, add_generation_prompt=False))
        return {"text": prompts}

    processed_dataset = dataset.map(create_chat_template, batched=True, remove_columns=["id", "conversations"])

    # Instantiate the custom data collator
    data_collator = DataCollatorForQwenVL(processor)

    # Set up TrainingArguments
    training_args = TrainingArguments(
        output_dir="out/results/multimodal_results",
        num_train_epochs=3,
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
