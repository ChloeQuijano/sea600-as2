import pandas as pd
from pathlib import Path

# Harmful keywords
harmful_keywords = ['abusive', 'offensive', 'hateful', 'disrespectful', 'fearful', 'hate']

# Label conversion logic
def is_harmful(label):
    if pd.isna(label):
        return None
    if isinstance(label, str):
        label = label.lower()
        if 'normal' in label:
            return 0
        if label in ['yes', 'hs']:
            return 1
        if label in ['no', 'non_hs']:
            return 0
        return int(any(keyword in label for keyword in harmful_keywords))
    try:
        return int(float(label) != 0)
    except:
        return 0

# Process and overwrite a single file
def binarize_file(csv_path):
    try:
        df = pd.read_csv(csv_path)
    except Exception as e:
        print(f"❌ Error reading {csv_path.name}: {e}")
        return

    if 'label' not in df.columns:
        print(f"⚠️ Skipping {csv_path.name} — no 'label' column.")
        return

    df['label'] = df['label'].apply(is_harmful)

    if df['label'].isnull().all():
        print(f"⚠️ Skipping {csv_path.name} — all labels are NaN after conversion.")
        return

    df.to_csv(csv_path, index=False)
    print(f"✅ Overwritten: {csv_path.name}")

# Walk through Dataset/train/val/test and process only *_full.csv files
def run_binarization(root_dir='Dataset'):
    root = Path(root_dir)
    for split in ['train', 'val', 'test']:
        split_path = root / split
        if not split_path.exists():
            continue
        for lang_dir in split_path.glob('*'):
            for csv_file in lang_dir.glob('*_full.csv'):
                binarize_file(csv_file)

if __name__ == "__main__":
    run_binarization()
