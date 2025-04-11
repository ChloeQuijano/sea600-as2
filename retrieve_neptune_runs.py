import os
from dotenv import load_dotenv
import neptune
import pandas as pd

load_dotenv()

project = neptune.init_project(
    project=os.getenv("PROJECT_NAME"),
    api_token=os.getenv("NEPTUNE_API_TOKEN"),
)

lang_map = {
    'Arabic': 'ar', 'French': 'fr', 'Portugese': 'pt', 'Spanish': 'es',
    'English': 'en', 'Indonesian': 'id', 'Italian': 'it', 'German': 'de', 'Polish': 'pl'
}

all_runs_df = pd.DataFrame()
runs_table = project.fetch_runs_table()
df = runs_table.to_pandas()

for lang in lang_map.keys():
    print(f"Fetching runs for language: {lang}")
    
    # Use model_name column to identify the language
    if "model_name" not in df.columns:
        print(f"⚠️ 'model_name' column missing, skipping {lang}")
        continue
    
    df_lang = df[df["model_name"].astype(str).str.contains(lang, case=False, na=False)]

    if df_lang.empty:
        print(f"No runs found for {lang}, skipping.")
        continue

    df_lang["language"] = lang
    all_runs_df = pd.concat([all_runs_df, df_lang], ignore_index=True)
    print(f"Found {len(df_lang)} run(s) for {lang}")

# Select relevant columns
columns_to_keep = [
    "sys/id", "sys/name", "language", "model_name",
    "params/learning_rate", "params/sample_ratio", "params/random_seed",
    "val_f1score", "val_accuracy", "test_f1score", "test_accuracy"
]

columns_available = [col for col in columns_to_keep if col in all_runs_df.columns]
summary_df = all_runs_df[columns_available]

summary_df.to_csv("neptune_run_summary.csv", index=False)
print("\nSaved run summary to neptune_run_summary.csv")

