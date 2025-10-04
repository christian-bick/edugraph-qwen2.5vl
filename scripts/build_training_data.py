
import os
import json
import glob
from create_dataset import create_training_jsonl

def find_and_process_metadata(root_dir, output_jsonl_file):
    """
    Finds all 'meta.json' files in a directory, processes them, 
    and creates a consolidated training JSONL file.
    """
    master_dataset = []

    # Find all meta.json files recursively
    for meta_path in glob.glob(os.path.join(root_dir, '**', 'meta.json'), recursive=True):
        print(f"Processing: {meta_path}")
        meta_dir = os.path.dirname(meta_path)
        
        with open(meta_path, 'r', encoding='utf-8') as f:
            try:
                metadata_list = json.load(f)
            except json.JSONDecodeError as e:
                print(f"  Error decoding JSON from {meta_path}: {e}")
                continue

            # Each meta.json contains a list of entries
            for entry in metadata_list:
                if 'questionImage' not in entry or 'labels' not in entry:
                    print(f"  Skipping entry in {meta_path} due to missing 'questionImage' or 'labels' field.")
                    continue

                # Construct the path for the image relative to the project root
                relative_image_path_from_meta = entry['questionImage']
                image_path_relative_to_root = os.path.join(meta_dir, relative_image_path_from_meta)

                # Normalize the path to use forward slashes for consistency across platforms
                image_path_relative_to_root = image_path_relative_to_root.replace('\\', '/')

                if not os.path.exists(image_path_relative_to_root):
                    print(f"  Image not found: {image_path_relative_to_root}")
                    continue

                # The 'labels' field is already a JSON object, so we just need to dump it to a string
                label_str = json.dumps(entry['labels'])

                master_dataset.append((image_path_relative_to_root, label_str))

    if not master_dataset:
        print("No data found. The 'train_dataset.jsonl' file will not be created.")
        return

    # Use the imported function to create the final JSONL file
    print(f"\nFound {len(master_dataset)} total training examples.")
    print(f"Creating '{output_jsonl_file}'...")
    create_training_jsonl(master_dataset, output_jsonl_file)
    print("Done.")

if __name__ == '__main__':
    # Define the root directory for worksheets and the output file name
    worksheets_root = os.path.join('data', 'worksheets')
    output_file = 'train_dataset.jsonl'
    
    find_and_process_metadata(worksheets_root, output_file)
