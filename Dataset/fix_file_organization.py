import os
import shutil

langs = ['Arabic', 'French', 'Portugese', 'Spanish', 'English', 'Indonesian', 'Italian', 'German', 'Polish']

for split in ['train', 'val', 'test']:
    for lang in langs:
        lang_folder = os.path.join(split, lang)
        os.makedirs(lang_folder, exist_ok=True)

        for file in os.listdir(split):
            if file.startswith(lang) and file.endswith(".csv"):
                src = os.path.join(split, file)
                dst = os.path.join(lang_folder, file)
                shutil.move(src, dst)
