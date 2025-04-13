import pandas as pd
import numpy as np

# Sample the given dataframe df to select n_sample number of points. 
def stratified_sample_df(df, col, n_samples,sampled='stratified',random_state=1):
    if(sampled=='stratified'):
        df_=df.groupby(col, group_keys=False).apply(lambda x: x.sample(int(np.rint(n_samples*len(x)/len(df))))).sample(frac=1,random_state=random_state).reset_index(drop=True)
    
    elif(sampled=='equal'):
        df_=df.groupby(col, group_keys=False).apply(lambda x: x.sample(int(n_samples/2))).sample(frac=1,random_state=random_state).reset_index(drop=True)
    
    return df_

###### data collection taking all at a time

def data_collector(file_names, params, is_train):
    sample_ratio = params['sample_ratio']
    type_train = params['how_train']
    sampled = params['samp_strategy']
    take_ratio = params['take_ratio']
    language = params['language']

    df_test = []

    def extract_lang_from_filename(file):
        filename = file.replace('\\', '/').split('/')[-1]
        if filename.endswith("_full.csv"):
            return filename[:-len("_full.csv")]
        elif filename.endswith("_translated.csv"):
            return filename[:-len("_translated.csv")]
        else:
            raise ValueError(f"Unrecognized file format: {filename}")

    # Validation or test data — use all available
    if not is_train:
        for file in file_names:
            lang_temp = extract_lang_from_filename(file)
            if lang_temp.lower().startswith(language.lower()):
                df_test.append(pd.read_csv(file))
        if not df_test:
            raise ValueError(f"No matching validation files for language: {language}")
        return pd.concat(df_test, axis=0)

    # Training data
    if type_train == 'baseline':
        for file in file_names:
            lang_temp = extract_lang_from_filename(file)
            print(f"Matched files for {language} ({'train' if is_train else 'val/test'}): {len(df_test)}")
            if lang_temp.lower().startswith(language.lower()):
                temp = pd.read_csv(file)
                df_test.append(temp)

    elif type_train == 'zero_shot':
        for file in file_names:
            lang_temp = extract_lang_from_filename(file)
            if lang_temp == 'English':
                temp = pd.read_csv(file)
                df_test.append(temp)

    elif type_train == 'all_but_one':
        for file in file_names:
            lang_temp = extract_lang_from_filename(file)
            if lang_temp != language:
                temp = pd.read_csv(file)
                df_test.append(temp)

    if not df_test:
        raise ValueError(f"No matching training files found for strategy: {type_train}, language: {language}")

    df_test = pd.concat(df_test, axis=0)

    # Apply sampling
    if take_ratio:
        n_samples = int(len(df_test) * sample_ratio / 100)
    else:
        n_samples = sample_ratio

    if n_samples == 0:
        n_samples = 1

    df_test = stratified_sample_df(df_test, 'label', n_samples, sampled, params['random_seed'])
    return df_test

