
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
# my_dataset = [('data/processed/image1.png', '{"Scopes": ["Cellular"], "Abilities": ["Memorize"]}')
#               ('data/processed/image2.png', '{"Scopes": ["Mechanics"], "Abilities": ["Analyze"]}')]
# create_training_jsonl(my_dataset, "train_dataset.jsonl")

