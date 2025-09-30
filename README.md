# Blueprint: Fine-Tuning Qwen2.5-VL for the EduGraph Ontology

## Introduction

This report is a hands-on technical guide for specializing the Qwen2.5-VL vision-language model. The goal is to create a powerful classifier for the EduGraph project, an initiative aimed at enabling "Smart Learning" by structuring the concepts of human learning into an open ontology.

The EduGraph system is designed to analyze diverse educational materials—including photos of worksheets, videos of learning sessions, and PDF documents—to track progress and generate personalized learning plans. A general-purpose AI cannot reliably perform this task, as it requires deep, domain-specific knowledge and the ability to output data that conforms to a strict ontological structure.

Qwen2.5-VL is the ideal foundation for this task due to its advanced capabilities in parsing complex documents, understanding long-form video, and generating structured JSON output natively. This blueprint provides a direct, action-oriented path to adapt Qwen2.5-VL for the EduGraph use case, focusing on efficient fine-tuning, practical data handling, and reliable, structured output generation.

## Setup

### Hardware Setup

Before fine-tuning, a solid foundation is required. This involves selecting the right hardware and preparing the software environment.

#### Hardware Selection: On-Premise vs. Cloud

The primary constraint for fine-tuning is GPU VRAM. Using memory-efficient techniques, it is feasible to fine-tune a 7B parameter model like Qwen2.5-VL on a high-end consumer GPU or a standard cloud instance.

| Metric | On-Premise: NVIDIA RTX 4090 | Cloud: NVIDIA A100 |
| :--- | :--- | :--- |
| **VRAM** | 24 GB | 80 GB |
| **Feasibility** | Feasible for 7B models using QLoRA (4-bit quantization). | Comfortable for 7B models; allows for larger batch sizes and faster training. |
| **Estimated Time** | \~12-24 hours per full training run. | \~4-8 hours per full training run. |
| **Estimated Cost** | Near-zero operational cost. | Operational cost of ~$1.30 - $2.40 per hour. |

Prototyping on an RTX 3060 or RTX 4060 is also feasible using QLoRA (4-bit quantization) using the 3B model.

### Environment and Dependencies

Set up your Python environment and install the required libraries. It is highly recommended to install `transformers` from the source to ensure the latest Qwen2.5-VL updates are included.

#### 1. Create and activate a virtual environment
```
python -m venv qwen-env
source qwen-env/bin/activate
```
#### 2. Install core dependencies
```
pip install torch torchvision torchaudio --index-url [https://download.pytorch.org/whl/cu121](https://download.pytorch.org/whl/cu121)
```
#### 3. Install Hugging Face ecosystem and Qwen utilities
```
pip install "transformers @ git+[https://github.com/huggingface/transformers.git](https://github.com/huggingface/transformers.git)"  
accelerate  
peft  
bitsandbytes  
"trl @ git+[https://github.com/huggingface/trl.git](https://github.com/huggingface/trl.git)"  
"qwen-vl-utils[decord]"
```
#### 4. For data processing and structured output
```
pip install datasets pdf2image opencv-python outlines
```

## Training

### Data Preparation

The model's performance is dictated by the quality of its training data. All input materials (videos, PDFs) must be converted into a consistent format (images) and structured correctly.

#### Preprocessing Multimodal Inputs

Create a unified dataset of images. Videos can be sampled into keyframes, and PDF pages can be rendered as images.

```python
import cv2
from pdf2image import convert_from_path
import os

def extract_frames_from_video(video_path, output_folder, frame_interval=150):
    """Extracts a frame every N frames (e.g., 1 every 5 seconds for a 30fps video)."""
    if not os.path.exists(output_folder):
        os.makedirs(output_folder)
    
    vidcap = cv2.VideoCapture(video_path)
    success, image = vidcap.read()
    count = 0
    while success:
        if count % frame_interval == 0:
            frame_path = os.path.join(output_folder, f"frame_{count}.jpg")
            cv2.imwrite(frame_path, image)
        success, image = vidcap.read()
        count += 1

def convert_pdf_to_images(pdf_path, output_folder):
    """Converts each page of a PDF to a PNG image."""
    if not os.path.exists(output_folder):
        os.makedirs(output_folder)
    
    images = convert_from_path(pdf_path)
    for i, image in enumerate(images):
        image_path = os.path.join(output_folder, f"page_{i + 1}.png")
        image.save(image_path, 'PNG')

# --- Example Usage ---
# extract_frames_from_video("lecture.mp4", "processed_data/lecture_frames")
# convert_pdf_to_images("worksheet.pdf", "processed_data/worksheet_pages")
````

#### Structuring the Dataset for Training

Fine-tuning requires a `jsonl` file where each line is a JSON object representing one training example. The data must be in a conversational format.

**Example `annotations.jsonl` entry:**

```json
{
  "id": "sample_001",
  "image": "path/to/processed_data/worksheet_pages/page_1.png",
  "conversations": {"Scopes": ["Cellular"], "Abilities": ["Memorize"]}
}
```

Use the following script to generate this file from your raw data:

```python
import json
import os

def create_training_jsonl(raw_data, output_file):
    """
    Converts a list of (image_path, json_label_string) tuples to a JSONL file.
    """
    with open(output_file, 'w') as f:
        for i, (image_path, label_str) in enumerate(raw_data):
            data_entry = {
                "id": f"sample_{i:06d}",
                "image": image_path,
                "conversations": [
                    {
                        "from": "human",
                        "value": "<image>\nClassify this learning material using the provided ontology."
                    },
                    {
                        "from": "gpt",
                        "value": label_str
                    }
                ]
            }
            f.write(json.dumps(data_entry) + '\n')

# --- Example Usage ---
# create_training_jsonl(my_dataset, "train_dataset.jsonl")
```

### Fine-Tuning Workflow

This section provides the core code for fine-tuning the model. The primary strategy is **QLoRA (Quantized Low-Rank Adaptation)**, which makes training feasible on consumer hardware by only updating a small fraction of the model's parameters. 

#### Loading the Model for QLoRA Training

First, configure `bitsandbytes` for 4-bit quantization and `peft` for LoRA. This setup loads the large base model into memory efficiently and prepares it for training.

```python
import torch
from transformers import AutoModelForCausalLM, AutoProcessor, BitsAndBytesConfig
from peft import LoraConfig, get_peft_model

model_id = "Qwen/Qwen2.5-VL-7B-Instruct"

# Configure 4-bit quantization
bnb_config = BitsAndBytesConfig(
    load_in_4bit=True,
    bnb_4bit_quant_type="nf4",
    bnb_4bit_compute_dtype=torch.bfloat16,
    bnb_4bit_use_double_quant=True
)

# Load the base model with quantization
model = AutoModelForCausalLM.from_pretrained(
    model_id,
    quantization_config=bnb_config,
    device_map="auto",
    trust_remote_code=True
)
processor = AutoProcessor.from_pretrained(model_id, trust_remote_code=True)

# Configure LoRA adapters
lora_config = LoraConfig(
    r=16,
    lora_alpha=32,
    lora_dropout=0.05,
    bias="none",
    target_modules=["q_proj", "k_proj", "v_proj", "o_proj", "gate_proj", "up_proj", "down_proj"],
    task_type="CAUSAL_LM"
)

# Add LoRA adapters to the model
model = get_peft_model(model, lora_config)
model.print_trainable_parameters()
# Expected output: trainable params: ~0.2-0.4% of total params
```

#### Advanced Strategy 1: Two-Stage Fine-Tuning

This method first teaches the model your ontology's terminology (Knowledge Infusion) and then teaches it the classification task (Task Tuning).

**Stage 1: Knowledge Infusion**
Create a text-only dataset of question-answer pairs from your ontology document. Fine-tune the model on this dataset and save the resulting adapter.

```python
# 1. Create a text-only dataset (e.g., 'ontology_qa.jsonl')
#    Format: {"instruction": "What is 'Cellular Biology'?", "output": "It is the study of..."}

# 2. Run a text-only SFT using LLaMA Factory or a custom script.
#    This produces an adapter in the specified output directory.
#    (LLaMA Factory is a popular tool for this: [https://github.com/hiyouga/LLaMA-Factory](https://github.com/hiyouga/LLaMA-Factory))

# 3. The output is a directory, e.g., './knowledge_adapter/', containing the LoRA weights.
```

**Stage 2: Task Tuning**
Load the adapter from Stage 1 onto the base model before starting the main classification fine-tuning. This gives the model a head start.

```python
from peft import PeftModel

# Load the base model (as shown in the QLoRA section)
base_model = AutoModelForCausalLM.from_pretrained(model_id, quantization_config=bnb_config, device_map="auto")

# Load the knowledge adapter from Stage 1
model_with_knowledge = PeftModel.from_pretrained(base_model, "path/to/your/knowledge_adapter")

# Now, use 'model_with_knowledge' as the model in the SFTTrainer for your multimodal classification task.
# The trainer will add a *new* set of LoRA adapters for the classification task on top of the knowledge-infused ones.
```

#### Advanced Strategy 2: Retrieval-Augmented Fine-Tuning (RAFT)

RAFT teaches the model to perform classification in an "open-book" setting. For each training example, you provide the correct definition from your ontology (the "oracle") along with several incorrect ones ("distractors"). This trains the model to identify and use relevant information while ignoring noise.

**Example RAFT Data Point:**

```json
{
  "id": "raft_sample_001",
  "image": "path/to/pendulum_diagram.png",
  "conversations": { "Scopes": ["Mechanics"], "Abilities": ["Analyze"] }
}
```

This requires a more complex data preparation pipeline but results in a model that is highly robust to noisy or irrelevant context during inference.

## Inference

### Getting Reliable, Structured Output

A fine-tuned model will not automatically produce perfect JSON. To guarantee the output conforms to your ontology's schema, you must use **constrained decoding**. This is done by manipulating the model's output probabilities (logits) at each step of generation. 

The best way to implement this is with a dedicated library like `outlines`, which can enforce a Pydantic schema or JSON schema directly. 

**Code Example: Constrained Generation with `outlines`**
This is the recommended, production-ready approach.

```python
import outlines
from pydantic import BaseModel, Field
from typing import List, Optional

# 1. Define your desired output structure using Pydantic
class OntologyLabels(BaseModel):
    Areas: List[str]
    Scopes: List[str]
    Abilities: List[str]

class ClassificationOutput(BaseModel):
    error: Optional[str] = Field(default=None)
    labels: OntologyLabels

# 2. Load your fine-tuned model (after merging the LoRA adapter)
# model.merge_and_unload() # Merge adapter into base model for deployment

# 3. Create a generator that enforces the Pydantic model's schema
generator = outlines.generate.json(model, ClassificationOutput)

# 4. Run inference
# The 'generator' will now only produce valid JSON that matches the ClassificationOutput schema.
result_json_str = generator(
    "Classify this learning material using the provided ontology.",
    # The image would be passed to the model through the processor, not directly here.
)

# The output is guaranteed to be a valid JSON string that can be parsed.
import json
parsed_output = json.loads(result_json_str)
print(parsed_output)
```

This method ensures every output from your model is programmatically usable and consistent with your ontology, eliminating the need for fragile post-processing and error handling.
