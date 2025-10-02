
import os
import torch
from datasets import load_dataset
from transformers import (
    AutoProcessor,
    BitsAndBytesConfig,
    TrainingArguments,
    Qwen2_5_VLForConditionalGeneration,
)
from peft import LoraConfig, get_peft_model
from trl import SFTTrainer

def main():
    # --- Configuration ---
    base_model_id = "Qwen/Qwen2.5-VL-3B-Instruct"
    text_dataset_path = "ontology_qa.jsonl"
    knowledge_adapter_path = "./knowledge_adapter"

    print("--- Starting Stage 1: Knowledge Infusion ---")

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
    
    # Configure LoRA
    lora_config = LoraConfig(
        r=8,
        lora_alpha=16,
        lora_dropout=0.05,
        bias="none",
        target_modules=["q_proj", "v_proj"], # Use the safe, restricted modules
        task_type="CAUSAL_LM"
    )
    model = get_peft_model(model, lora_config)
    model.print_trainable_parameters()

    # Load and process the dataset
    dataset = load_dataset("json", data_files=text_dataset_path, split="train")

    def format_qa_dataset(examples):
        # For text-only SFT, format the instruction-output pair
        texts = []
        for i in range(len(examples['instruction'])):
            text = f"<|im_start|>user\n{examples['instruction'][i]}<|im_end|>\n<|im_start|>assistant\n{examples['output'][i]}<|im_end|>"
            texts.append(text)
        return {"text": texts}

    processed_dataset = dataset.map(format_qa_dataset, batched=True)

    # Set up TrainingArguments
    training_args = TrainingArguments(
        output_dir="./results_text_only",
        num_train_epochs=3,
        per_device_train_batch_size=2,
        gradient_accumulation_steps=4,
        learning_rate=2e-4,
        logging_steps=10,
        save_strategy="epoch",
        fp16=True,
        remove_unused_columns=False,
    )

    # Trainer for text-only SFT
    trainer = SFTTrainer(
        model=model,
        args=training_args,
        train_dataset=processed_dataset,
        dataset_text_field="text",
    )

    # Train the knowledge adapter
    trainer.train()

    # Save the final adapter
    print(f"Saving knowledge adapter to {knowledge_adapter_path}")
    model.save_pretrained(knowledge_adapter_path)

if __name__ == "__main__":
    main()
