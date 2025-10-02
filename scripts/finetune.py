
import os
import torch
from torch.optim import AdamW
from datasets import load_dataset
from transformers import (
    Qwen2VLForConditionalGeneration as AutoModelForCausalLM,
    AutoProcessor,
    BitsAndBytesConfig,
    TrainingArguments,
)
from peft import LoraConfig, get_peft_model
from trl import SFTTrainer
from torch.utils.data import DataLoader, Dataset

# --- 1. Define the Dataset ---
class EduGraphDataset(Dataset):
    def __init__(self, dataset_path, processor):
        self.dataset = load_dataset("json", data_files=dataset_path, split="train")
        self.processor = processor

    def __len__(self):
        return len(self.dataset)

    def __getitem__(self, idx):
        item = self.dataset[idx]
        
        # Create the chat prompt structure
        chat = [
            {"role": "system", "content": "You are a helpful assistant."},
            {"role": "user", "content": [{"type": "image"}, {"type": "text", "text": "Classify this learning material using the provided ontology."}]},
            {"role": "assistant", "content": item['conversations'][1]['value']}
        ]
        
        # Apply the template to get the text prompt
        prompt = self.processor.apply_chat_template(chat, tokenize=False, add_generation_prompt=False)
        
        # Process image and text together
        # This returns input_ids, attention_mask, and pixel_values
        inputs = self.processor(text=prompt, images=[item['image']], return_tensors="pt")

        # Squeeze the batch dimension and set labels
        inputs = {k: v.squeeze(0) for k, v in inputs.items()}
        inputs["labels"] = inputs["input_ids"].clone()
        
        return inputs

# --- 2. Main Execution Block ---
def main():
    # --- Configuration ---
    base_model_id = "Qwen/Qwen2.5-VL-3B-Instruct"
    multimodal_dataset_path = "train_dataset.jsonl"
    final_adapter_path = "./final_edugraph_adapter"

    print("--- Starting Fine-Tuning (Corrected Data Pipeline) ---")

    # Load processor
    processor = AutoProcessor.from_pretrained(base_model_id, trust_remote_code=True)

    # Configure 4-bit quantization
    bnb_config = BitsAndBytesConfig(
        load_in_4bit=True,
        bnb_4bit_quant_type="nf4",
        bnb_4bit_compute_dtype=torch.bfloat16,
        bnb_4bit_use_double_quant=True
    )

    # Load base model
    model = AutoModelForCausalLM.from_pretrained(
        base_model_id,
        quantization_config=bnb_config,
        device_map="auto",
        trust_remote_code=True
    )
    
    # Configure LoRA
    lora_config = LoraConfig(
        r=32,
        lora_alpha=64,
        lora_dropout=0.1,
        bias="none",
        target_modules=["q_proj", "k_proj", "v_proj", "o_proj", "gate_proj", "up_proj", "down_proj"],
        task_type="CAUSAL_LM"
    )
    model = get_peft_model(model, lora_config)
    model.print_trainable_parameters()

    # Initialize the dataset
    train_dataset = EduGraphDataset(multimodal_dataset_path, processor)

    # Set up TrainingArguments
    training_args = TrainingArguments(
        output_dir="./results_multimodal",
        num_train_epochs=5,
        per_device_train_batch_size=1,
        gradient_accumulation_steps=8,
        learning_rate=1e-4,
        logging_steps=10,
        save_strategy="epoch",
        fp16=True,
        remove_unused_columns=False, # Keep this for safety
    )

    # Trainer for multimodal SFT
    trainer = SFTTrainer(
        model=model,
        train_dataset=train_dataset,
        # The trainer will automatically use the 'text' column and a default max_seq_length
        args=training_args,
    )

    # Train the final adapter
    trainer.train()

    # Save the final adapter
    print(f"Saving final adapter to {final_adapter_path}")
    model.save_pretrained(final_adapter_path)

if __name__ == "__main__":
    main()
