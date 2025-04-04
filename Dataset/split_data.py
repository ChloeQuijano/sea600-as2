import pandas as pd
import os

load_path = 'full_data'
columns = ['text', 'label']

# Create output directories
for split in ['train', 'val', 'test']:
    os.makedirs(split, exist_ok=True)

# Walk through all files in full_data (recursively)
for root, dirs, files in os.walk(load_path):
    for filename in files:
        if filename.endswith('.csv'):
            full_file_path = os.path.join(root, filename)
            rel_path = os.path.relpath(full_file_path, load_path)
            lang_folder = os.path.dirname(rel_path)

            # Try reading dataset file (no header, 2 columns)
            try:
                try:
                    df = pd.read_csv(full_file_path, header=None, names=columns, encoding='utf-8')
                except UnicodeDecodeError:
                    df = pd.read_csv(full_file_path, header=None, names=columns, encoding='ISO-8859-1')
            except Exception as e:
                print(f"❌ Failed to read {full_file_path}: {e}")
                continue

            # Ensure output subfolders exist
            for split in ['train', 'val', 'test']:
                os.makedirs(os.path.join(split, lang_folder), exist_ok=True)

            # Process train/val/test splits
            for split in ['train', 'val', 'test']:
                id_path = os.path.join('ID Mapping', split, rel_path)

                try:
                    id_df = pd.read_csv(id_path)  # WITH header: must have column 'id'
                    if 'id' not in id_df.columns:
                        raise ValueError("Missing 'id' column in mapping file")
                    split_df = df.iloc[list(id_df['id'])]
                    output_path = os.path.join(split, rel_path)
                    split_df.to_csv(output_path, index=False)
                    print(f"✅ Wrote {split} → {output_path}")
                except FileNotFoundError:
                    print(f"⚠️ Missing ID file: {id_path}")
                except Exception as e:
                    print(f"❌ Error processing {id_path}: {e}")
