import os
import torch
import argparse
import json

from transformers import (
    AutoProcessor,
    Qwen2_5_VLForConditionalGeneration,
)
from peft import PeftModel

def main(args):
    # --- Configuration ---
    base_model_id = "Qwen/Qwen2.5-VL-3B-Instruct"
    adapter_path = "out/adapters/multimodal_adapter"

    print("--- Loading model and adapter for inference ---")

    # Load the processor
    processor = AutoProcessor.from_pretrained(base_model_id, trust_remote_code=True)

    # Load the base model without quantization for best quality
    model = Qwen2_5_VLForConditionalGeneration.from_pretrained(
        base_model_id,
        torch_dtype=torch.bfloat16,
        device_map="auto",
        trust_remote_code=True
    )

    # Load the LoRA adapter and merge it into the base model
    model = PeftModel.from_pretrained(model, adapter_path)
    model = model.merge_and_unload()
    print("Adapter merged successfully.")

    # --- Run Inference using model.generate() ---
    print(f"\n--- Running inference on {args.image_path} ---")
    
    # Create the conversational prompt
    conversation = [
        {"role": "system", "content": "You are a helpful assistant."}, 
        {"role": "user", "content": [{"type": "image"}, {"type": "text", "text": "Classify this learning material using the provided ontology."}]}
    ]
    
    # Apply the chat template and prepare inputs
    text_prompt = processor.apply_chat_template(conversation, tokenize=False, add_generation_prompt=True)
    inputs = processor(text=[text_prompt], images=[args.image_path], return_tensors="pt").to(model.device)

    # Generate the token IDs
    generated_ids = model.generate(**inputs, max_new_tokens=512)

    # Decode the generated tokens, skipping special tokens and the prompt
    response = processor.batch_decode(generated_ids, skip_special_tokens=True)[0]
    
    # The response includes the prompt, so we need to extract just the assistant's part.
    # This is a bit brittle and depends on the exact template format.
    assistant_response = response.split("assistant\n")[1].strip()

    print("\n--- Generated Classification (Raw String) ---")
    print(assistant_response)


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Run inference with the fine-tuned EduGraph model.")
    parser.add_argument("--image_path", type=str, required=True, help="Path to the image file to classify.")
    args = parser.parse_args()
    main(args)
