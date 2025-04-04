import pandas as pd
import os
import emoji
from googletrans import Translator
from multiprocessing import Pool
import numpy as np
import glob

def translate_text(text, src_lang, dest_lang='en'):
    translator = Translator()
    result = translator.translate(text, src=src_lang, dest=dest_lang)
    return result.text

# Helper to extract language code from filename
def get_language_code(filename):
    lang_map = {
        'Arabic': 'ar',
        'English': 'en',
        'French': 'fr',
        'German': 'de',
        'Indonesian': 'id',
        'Portugese': 'pt'
    }
    for lang_prefix, code in lang_map.items():
        if filename.startswith(lang_prefix):
            return code
    return 'unk'  # unknown

def extract_emojis(str1):
    try:
        return [(c, i) for i, c in enumerate(str1) if c in emoji.UNICODE_EMOJI]
    except AttributeError:
        return []

def approximate_emoji_insert(string, index, char):
    if index < (len(string) - 1):
        while string[index] != ' ':
            if index + 1 == len(string):
                break
            index += 1
        return string[:index] + ' ' + char + ' ' + string[index:]
    else:
        return string + ' ' + char + ' '

def translate(text, lang):
    emoji_list = extract_emojis(text)
    try:
        translated_text = translate_text(text, lang, 'en')
    except:
        translated_text = text
    for emoji_char, pos in emoji_list:
        translated_text = approximate_emoji_insert(translated_text, pos, emoji_char)
    return translated_text

def add_features(df, lang):
    translated_text = []
    for _, row in df.iterrows():
        if lang in ['en', 'unk']:
            translated_text.append(row['text'])
        else:
            translated_text.append(translate(row['text'], lang))
    df["translated"] = translated_text
    return df

def parallelize_dataframe(df, lang, func, n_cores=4):
    df_split = np.array_split(df, n_cores)
    pool = Pool(n_cores)
    df = pd.concat(pool.starmap(func, [(chunk, lang) for chunk in df_split]))
    pool.close()
    pool.join()
    return df

# ==== MAIN ====
if __name__ == "__main__":
    input_folder = 'full_data'
    output_folder = 'full_data'
    files = glob.glob(os.path.join(input_folder, '*.csv'))

    for file in files:
        filename = os.path.basename(file)
        print(f"📄 Translating: {filename}")
        try:
            lang = get_language_code(filename)
            df = pd.read_csv(file, header=None, encoding='utf-8')

            if df.shape[1] < 2:
                print(f"⚠️ Skipping {filename} — file doesn't contain at least 2 columns.")
                continue

            df = df.iloc[:, :2]
            df.columns = ['text', 'label']

            # Translate in chunks
            chunk_size = 10
            list_df = []
            for i in range(0, len(df), chunk_size):
                df_chunk = df[i:i+chunk_size]
                df_translated = parallelize_dataframe(df_chunk, lang, add_features, n_cores=4)
                list_df.append(df_translated)

            result_df = pd.concat(list_df, ignore_index=True)

            base, ext = os.path.splitext(filename)
            out_file = os.path.join(output_folder, f"{base}_translated.csv")
            result_df.to_csv(out_file, index=False, encoding='utf-8-sig')
            print(f"✅ Saved: {out_file}")

        except Exception as e:
            print(f"❌ Error processing {filename}: {e}")
