import pandas as pd
import os

# Folders
input_folder = 'full_data_raw'
output_folder = 'full_data'

os.makedirs(output_folder, exist_ok=True)

# Encodings to try
encodings_to_try = ['utf-8-sig', 'utf-8', 'ISO-8859-1', 'cp1252']

# Loop through all text/extensionless files
for filename in os.listdir(input_folder):
    if filename.endswith('.txt') or '.' not in filename:
        file_path = os.path.join(input_folder, filename)
        df = None

        # Try multiple encodings
        for encoding in encodings_to_try:
            try:
                df = pd.read_csv(file_path, sep='\t', encoding=encoding)
                print(f"ℹ️  Successfully read {filename} with encoding: {encoding}")
                break
            except Exception as e:
                last_exception = e

        if df is None:
            print(f"❌ Failed to convert {filename}: {last_exception}")
            continue

        # Optional: Rename columns (if needed)
        # df.columns = ['text', 'label']

        # Save to CSV
        new_filename = os.path.splitext(filename)[0] + '.csv'
        output_path = os.path.join(output_folder, new_filename)
        try:
            df.to_csv(output_path, index=False, encoding='utf-8-sig')
            print(f"✅ Converted: {filename} → {new_filename}")
        except Exception as e:
            print(f"❌ Failed to save {filename} as CSV: {e}")

