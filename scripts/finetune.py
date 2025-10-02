import os
import torch
from datasets import load_dataset
from transformers import (
    AutoModelForCausalLM,
    AutoProcessor,
    BitsAndBytesConfig,
    TrainingArguments,
    Qwen2VLForConditionalGeneration,
)
from peft import LoraConfig, get_peft_model
from trl import SFTTrainer

def main():
    # --- Configuration ---
    base_model_id = "Qwen/Qwen2.5-VL-3B-Instruct"
    multimodal_dataset_path = "train_dataset.jsonl"
    final_adapter_path = "./final_edugraph_adapter"

    print("--- Starting Fine-Tuning (Final Pipeline) ---")

    # Load processor
    processor = AutoProcessor.from_pretrained(base_model_id, trust_remote_code=True)

    # Configure 4-bit quantization
    bnb_config = BitsAndBytesConfig(
        load_in_4bit=True,
        bnb_4bit_quant_type="nf4",
        bnb_4bit_compute_dtype=torch.bfloat16,
        bnb_4bit_use_double_quant=True
    )

    # Load base model using the correct specific class
    model = Qwen2VLForConditionalGeneration.from_pretrained(
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

    # Load and process the dataset
    dataset = load_dataset("json", data_files=multimodal_dataset_path, split="train")
    dataset = dataset.rename_column("image", "images")

    def create_chat_template(examples):
        # The SFTTrainer will handle tokenization. We just need to format the text part.
        # The `image` column will be handled automatically by the trainer.
        prompts = []
        for i in range(len(examples['images'])):
            chat = [
                {"role": "system", "content": "You are a helpful assistant."},
                {"role": "user", "content": [{"type": "image"}, {"type": "text", "text": "Classify this learning material using the provided ontology."}]},
                {"role": "assistant", "content": examples['conversations'][i][1]['value']}
            ]
            prompts.append(processor.apply_chat_template(chat, tokenize=False, add_generation_prompt=False))
        return {"text": prompts}

    processed_dataset = dataset.map(create_chat_template, batched=True)

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
        remove_unused_columns=False, # Important for multimodal datasets
    )

    # Trainer for multimodal SFT
    trainer = SFTTrainer(
        model=model,
        args=training_args,
        train_dataset=processed_dataset,
        # The trainer will automatically use the `image` and `text` columns

        # The trainer will automatically use the `image` column and process it
    )

    # Train the final adapter
    trainer.train()

    # Save the final adapter
    print(f"Saving final adapter to {final_adapter_path}")
    model.save_pretrained(final_adapter_path)

if __name__ == "__main__":
    main()