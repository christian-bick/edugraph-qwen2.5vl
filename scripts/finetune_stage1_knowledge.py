
import os
import torch
from dotenv import load_dotenv
from datasets import load_dataset
from transformers import (
    AutoProcessor,
    BitsAndBytesConfig,
    TrainingArguments,
    Qwen2_5_VLForConditionalGeneration,
    DataCollatorForLanguageModeling,
)
from peft import get_peft_model
from trl import SFTTrainer
from scripts.config import get_config

def main():
    # Load environment variables from .env file for local development
    load_dotenv()

    # --- Configuration ---
    run_mode = os.environ.get("RUN_MODE", "train")
    model_size = os.environ.get("MODEL_SIZE", "3b")
    
    # Get model and training configurations
    model_config = get_config(model_size)
    stage1_config = model_config.stage1
    base_model_id = f"Qwen/Qwen2.5-VL-{model_size.upper()}-Instruct"
    
    text_dataset_path = "ontology_qa_v3.jsonl"
    knowledge_adapter_path = "out/adapters/knowledge_adapter"
    os.makedirs("out/adapters", exist_ok=True)

    # --- Mode-specific Adjustments ---
    if run_mode == "test":
        print("--- Running in TEST mode ---")
        num_train_epochs = 0.1
        max_train_samples = 10
    else:
        print("--- Running in TRAIN mode ---")
        num_train_epochs = stage1_config.num_train_epochs
        max_train_samples = None

    print("--- Starting Stage 1: Knowledge Infusion ---")

    # Load processor and tokenizer
    processor = AutoProcessor.from_pretrained(base_model_id, trust_remote_code=True, local_files_only=True)
    tokenizer = processor.tokenizer

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
    
    # Configure LoRA using the centralized config
    model = get_peft_model(model, stage1_config.lora_config)
    model.print_trainable_parameters()

    # Load and process the dataset
    dataset = load_dataset("json", data_files=text_dataset_path, split="train")
    if max_train_samples:
        dataset = dataset.select(range(max_train_samples))

    def format_qa_dataset(examples):
        # This function now handles the entire formatting and tokenization process.
        instructions = examples['instruction']
        outputs = examples['output']
        texts = []
        for instruction, output in zip(instructions, outputs):
            text = f"<|im_start|>user\n{instruction}<|im_end|>\n<|im_start|>assistant\n{output}<|im_end|>"
            texts.append(text)
        # Tokenize the formatted texts
        return tokenizer(texts, truncation=True, padding="max_length", max_length=512)

    processed_dataset = dataset.map(format_qa_dataset, batched=True, remove_columns=['instruction', 'output'], num_proc=1)

    # Instantiate a text-only data collator
    data_collator = DataCollatorForLanguageModeling(tokenizer=tokenizer, mlm=False)

    # Set up TrainingArguments
    training_args = TrainingArguments(
        output_dir="out/results/knowledge_results",
        num_train_epochs=num_train_epochs,
        per_device_train_batch_size=2,
        gradient_accumulation_steps=4,
        learning_rate=stage1_config.learning_rate, # Use learning rate from config
        logging_steps=10,
        save_strategy="epoch",
        fp16=True,
        remove_unused_columns=False,
    )

    # Trainer for text-only SFT
    print("Initializing SFTTrainer...")
    trainer = SFTTrainer(
        model=model,
        args=training_args,
        train_dataset=processed_dataset,
        data_collator=data_collator, # Use the text-only collator
    )
    print("SFTTrainer initialized.")

    # Train the knowledge adapter
    print("Starting training...")
    trainer.train()
    print("Training finished.")

    # Save the final adapter
    print(f"Saving knowledge adapter to {knowledge_adapter_path}")
    model.save_pretrained(knowledge_adapter_path)

if __name__ == "__main__":
    main()
