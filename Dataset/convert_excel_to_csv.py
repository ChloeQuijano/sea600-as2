import pandas as pd
import os
import shutil

input_folder = 'full_data_raw'     # Change as needed
output_folder = 'full_data'    # Output folder

os.makedirs(output_folder, exist_ok=True)

for filename in os.listdir(input_folder):
    input_path = os.path.join(input_folder, filename)

    # Handle Excel files (.xlsx)
    if filename.endswith('.xlsx'):
        try:
            df = pd.read_excel(input_path)
            base_name = os.path.splitext(filename)[0]
            output_path = os.path.join(output_folder, base_name + '.csv')
            df.to_csv(output_path, index=False, encoding='utf-8-sig')
            print(f"✅ Converted Excel: {filename} → {base_name}.csv")
        except Exception as e:
            print(f"❌ Failed to convert Excel file {filename}: {e}")

    # Handle CSV files (just copy)
    elif filename.endswith('.csv'):
        try:
            output_path = os.path.join(output_folder, filename)
            shutil.copy2(input_path, output_path)
            print(f"📋 Copied CSV: {filename}")
        except Exception as e:
            print(f"❌ Failed to copy CSV file {filename}: {e}")

    # Skip other file types
    else:
        print(f"⏭️ Skipped: {filename} (not .xlsx or .csv)")
