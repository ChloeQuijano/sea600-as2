#Necessary imports

import transformers 
import torch
from dotenv import load_dotenv
import glob 
import re
import time
from transformers import get_linear_schedule_with_warmup, get_cosine_schedule_with_warmup
from transformers import BertTokenizer, CamembertTokenizer, AutoTokenizer, AutoModelForSequenceClassification
from transformers import BertForSequenceClassification, BertConfig
from transformers import AutoConfig, XLMRobertaForSequenceClassification, XLMRobertaTokenizer
from transformers import AdamW
import random
from transformers.models.bert.modeling_bert import *
from bert_codes.feature_generation import combine_features, return_dataloader
from bert_codes.data_extractor import data_collector
from bert_codes.utils import *
from sklearn.metrics import accuracy_score, f1_score
from tqdm import tqdm
from BERT_inference import *
from pathlib import Path
import os
import pandas as pd
try:
    from deep_translator import GoogleTranslator
except ImportError:
    print("deep-translator not installed. Run: pip install deep-translator")

load_dotenv()  # take environment variables

# If there's a GPU available...
if torch.cuda.is_available():
    device = torch.device("cuda")
    print(f"There are {torch.cuda.device_count()} GPU(s) available.")
    print(f"We will use the GPU: {torch.cuda.get_device_name(torch.cuda.current_device())}")
else:
    device = torch.device("cpu")
    print("No GPU available, using the CPU instead.")

# Language-specific models mapping
LANGUAGE_MODELS = {
    'English': 'bert-base-uncased',
    'French': 'camembert-base',
    'German': 'dbmdz/bert-base-german-cased',
    'Arabic': 'asafaya/bert-base-arabic',
    'Spanish': 'dccuchile/bert-base-spanish-wwm-cased',
    'Italian': 'dbmdz/bert-base-italian-cased',
    'Portugese': 'neuralmind/bert-base-portuguese-cased',
    'Indonesian': 'indobenchmark/indobert-base-p1',
    'Polish': 'dkleczek/bert-base-polish-cased'
}

# Language-specific tokenizers mapping
LANGUAGE_TOKENIZERS = {
    'English': AutoTokenizer,
    'French': AutoTokenizer, 
    'German': AutoTokenizer,
    'Arabic': AutoTokenizer,
    'Spanish': AutoTokenizer,
    'Italian': AutoTokenizer,
    'Portugese': AutoTokenizer,
    'Indonesian': AutoTokenizer,
    'Polish': AutoTokenizer
}

# Text preprocessing function
def preprocess_text(text):
    """Clean and normalize text"""
    if isinstance(text, str):
        # Convert to lowercase
        text = text.lower()
        # Remove extra whitespace
        text = re.sub(r'\s+', ' ', text).strip()
        # Remove URLs
        text = re.sub(r'https?://\S+|www\.\S+', '', text)
        # Remove HTML tags
        text = re.sub(r'<.*?>', '', text)
    return text

# Back translation function
def back_translate(text, source_lang, target_lang='en'):
    """Translate text to target language and back to source language"""
    try:
        if 'GoogleTranslator' not in globals():
            return text  # Return original if deep-translator not available
        
        # Translate to target language (e.g., English)
        translated = GoogleTranslator(source=source_lang, target=target_lang).translate(text)
        # Translate back to source language
        back_translated = GoogleTranslator(source=target_lang, target=source_lang).translate(translated)
        return back_translated
    except:
        return text  # Return original text if translation fails

# Modified model selection with dropout
def select_model_with_dropout(what_bert, path_files, weights, dropout_rate=0.3):
    try:
        # Use AutoConfig to automatically determine the correct configuration
        config = AutoConfig.from_pretrained(
            path_files,
            num_labels=2,
            hidden_dropout_prob=dropout_rate,
            attention_probs_dropout_prob=dropout_rate
        )
        
        # Always use AutoModelForSequenceClassification for better compatibility
        model = AutoModelForSequenceClassification.from_pretrained(path_files, config=config)
        
    except Exception as e:
        print(f"Error loading model: {e}")
        print("Falling back to base XLM-RoBERTa model")
        model = AutoModelForSequenceClassification.from_pretrained('xlm-roberta-base', 
                                                                 num_labels=2,
                                                                 hidden_dropout_prob=dropout_rate,
                                                                 attention_probs_dropout_prob=dropout_rate)
    
    return model
# Apply data augmentation with back translation
def augment_dataset(df, lang_code, text_column='text', augment_ratio=0.3):
    """Augment dataset with back-translated samples"""
    lang_map = {
        'Arabic': 'ar', 'French': 'fr', 'Portugese': 'pt', 'Spanish': 'es',
        'English': 'en', 'Indonesian': 'id', 'German': 'de', 'Italian': 'it',
        'Polish': 'pl'
    }
    
    # Determine source language code
    source_lang = lang_map.get(lang_code, 'en')
    
    # Select subset to augment
    minority_label = df['label'].value_counts().idxmin()
    minority_samples = df[df['label'] == minority_label]
    n_to_augment = int(len(minority_samples) * augment_ratio)
    
    if n_to_augment == 0:
        return df
    
    samples_to_augment = minority_samples.sample(n=min(n_to_augment, len(minority_samples)))
    
    # Apply back-translation
    augmented_texts = []
    augmented_labels = []
    
    for _, row in tqdm(samples_to_augment.iterrows(), desc="Augmenting data", total=len(samples_to_augment)):
        original_text = row[text_column]
        label = row['label']
        
        # Apply back-translation
        augmented_text = back_translate(original_text, source_lang)
        
        # Only add if translation is different and not empty
        if augmented_text != original_text and len(augmented_text) > 0:
            augmented_texts.append(augmented_text)
            augmented_labels.append(label)
    
    # Create DataFrame with augmented samples
    augment_df = pd.DataFrame({
        text_column: augmented_texts,
        'label': augmented_labels
    })
    
    # Combine with original DataFrame
    combined_df = pd.concat([df, augment_df], ignore_index=True)
    
    print(f"Added {len(augment_df)} augmented samples. New dataset size: {len(combined_df)}")
    return combined_df

# The main function that does the training
def train_model(params, best_val_fscore):
    # In case of english languages, translation is the origin data itself.
    lang = params['language']
    params['csv_file'] = f"{lang}_*_full.csv"
    
    # Use language-specific model if available
    if params.get('use_language_specific_models', True) and lang in LANGUAGE_MODELS:
        params['path_files'] = LANGUAGE_MODELS[lang]
        print(f"Using language-specific model: {params['path_files']}")
    
    train_path = Path(params['files']) / 'train' / lang / params['csv_file']
    val_path = Path(params['files']) / 'val' / lang / params['csv_file']

    train_files = glob.glob(str(train_path))
    val_files = glob.glob(str(val_path))

    print(f"Train path: {train_path}")
    print(f"Val path: {val_path}")
    print(f"Train files found: {train_files}")
    print(f"Validation files found: {val_files}")
    
    # Load the appropriate tokenizer
    print('Loading tokenizer...')
    tokenizer_class = LANGUAGE_TOKENIZERS.get(lang, AutoTokenizer)
    print(f"Using tokenizer class: {tokenizer_class.__name__}")
    try:
        tokenizer = tokenizer_class.from_pretrained(params['path_files'], do_lower_case=False)
    except Exception as e:
        print(f"Error loading tokenizer: {e}")
        print("Falling back to AutoTokenizer")
        tokenizer = AutoTokenizer.from_pretrained(params['path_files'], do_lower_case=False)
    
    df_train = data_collector(train_files, params, True)
    df_val = data_collector(val_files, params, False)
    
    # Apply text preprocessing if enabled
    if params.get('use_text_preprocessing', True):
        print("Applying text preprocessing...")
        if df_train.columns.__contains__('text'):
            df_train['text'] = df_train['text'].apply(preprocess_text)
            df_val['text'] = df_val['text'].apply(preprocess_text)
        elif df_train.columns.__contains__('translated'):
            df_train['translated'] = df_train['translated'].apply(preprocess_text)
            df_val['translated'] = df_val['translated'].apply(preprocess_text)
    
    # Apply data augmentation if enabled
    if params.get('use_data_augmentation', True) and 'GoogleTranslator' in globals():
        print("Applying data augmentation with back translation...")
        text_col = 'text' if df_train.columns.__contains__('text') else 'translated'
        df_train = augment_dataset(df_train, lang, text_column=text_col, augment_ratio=params.get('augment_ratio', 0.3))
    
    # Get the comment texts and corresponding labels
    if df_train.columns.__contains__('text'):
        sentences_train = df_train.text.values
        sentences_val = df_val.text.values
    elif df_train.columns.__contains__('translated'):
        sentences_train = df_train.translated.values
        sentences_val = df_val.translated.values
    else:
        raise ValueError("Expected 'text' or 'translated' column in the data.")
    
    # Convert string labels to numeric labels
    label_mapping = {label: idx for idx, label in enumerate(df_train['label'].unique())}
    print("Label mapping:", label_mapping)

    df_train['label'] = df_train['label'].map(label_mapping)
    df_val['label'] = df_val['label'].map(label_mapping)

    labels_train = df_train.label.values
    labels_val = df_val.label.values

    # Calculate class weights for weighted loss function
    label_counts = df_train['label'].value_counts()
    print(label_counts)
    
    # Calculate weights inversely proportional to class frequencies
    if params.get('use_weighted_loss', True):
        total_samples = len(df_train)
        weights = [total_samples/label_counts[i] for i in range(len(label_counts))]
        # Normalize weights
        sum_weights = sum(weights)
        weights = [w/sum_weights * len(weights) for w in weights]
        params['weights'] = weights
        print(f"Using weighted loss function with weights: {weights}")
    
    # Select the required bert model with dropout
    if params.get('use_dropout', True):
        model = select_model_with_dropout(
            params['what_bert'], 
            params['path_files'], 
            params['weights'], 
            dropout_rate=params.get('dropout_rate', 0.3)
        )
    else:
        model = select_model(params['what_bert'], params['path_files'], params['weights'])
    
    # Tell pytorch to run this model on the GPU.
    model.to(device)

    # Do the required encoding using the bert tokenizer
    input_train_ids, att_masks_train = combine_features(sentences_train, tokenizer, params['max_length'])
    input_val_ids, att_masks_val = combine_features(sentences_val, tokenizer, params['max_length'])

    # Create dataloaders for both the train and validation datasets.
    train_dataloader = return_dataloader(input_train_ids, labels_train, att_masks_train, batch_size=params['batch_size'], is_train=params['is_train'])
    validation_dataloader = return_dataloader(input_val_ids, labels_val, att_masks_val, batch_size=params['batch_size'], is_train=False)
    
    # Initialize AdamW optimizer.
    optimizer = AdamW(model.parameters(),
                  lr = params['learning_rate'],
                  eps = params['epsilon']
                 )
    
    # Number of training epochs (authors recommend between 2 and 4)
    # Total number of training steps is number of batches * number of epochs.
    total_steps = len(train_dataloader) * params['epochs']

    # Create the learning rate scheduler.
    if params.get('use_cosine_scheduler', True):
        scheduler = get_cosine_schedule_with_warmup(
            optimizer, 
            num_warmup_steps=int(total_steps/10),
            num_training_steps=total_steps
        )
    else:
        scheduler = get_linear_schedule_with_warmup(
            optimizer, 
            num_warmup_steps=int(total_steps/10),
            num_training_steps=total_steps
        )

    # Set the seed value all over the place to make this reproducible.
    fix_the_random(seed_val=params['random_seed'])
    # Store the average loss after each epoch so we can plot them.
    loss_values = []

    # Create a model name identifier
    bert_model = params['path_files']
    language = params['language']
    name_one = bert_model + "_" + language
    
    # The best val fscore obtained till now, for the purpose of hyper parameter finetuning.
    best_val_fscore = best_val_fscore

    # For each epoch...
    for epoch_i in range(0, params['epochs']):
        print("")
        print('======== Epoch {:} / {:} ========'.format(epoch_i + 1, params['epochs']))
        print('Training...')

        # Measure how long the training epoch takes.
        t0 = time.time()

        # Reset the total loss for this epoch.
        total_loss = 0
        model.train()

        # For each batch of training data...
        try:
            for step, batch in tqdm(enumerate(train_dataloader)):
                # Progress update every 40 batches.
                if step % 40 == 0 and not step == 0:
                    # Calculate elapsed time in minutes.
                    elapsed = format_time(time.time() - t0)
                    
                # `batch` contains three pytorch tensors:
                #   [0]: input ids 
                #   [1]: attention masks
                #   [2]: labels 
                b_input_ids = batch[0].to(device)
                b_input_mask = batch[1].to(device)
                b_labels = batch[2].to(device)
                # (source: https://stackoverflow.com/questions/48001598/why-do-we-need-to-call-zero-grad-in-pytorch)
                model.zero_grad()        

                # Get the model outputs for this batch.
                outputs = model(b_input_ids, 
                            token_type_ids=None, 
                            attention_mask=b_input_mask, 
                            labels=b_labels)

                # The call to `model` always returns a tuple, so we need to pull the 
                # loss value out of the tuple.
                loss = outputs[0]
                
                # Accumulate the training loss over all of the batches
                total_loss += loss.item()

                # Perform a backward pass to calculate the gradients.
                loss.backward()

                # Clip the norm of the gradients to 1.0.
                # This is to help prevent the "exploding gradients" problem.
                torch.nn.utils.clip_grad_norm_(model.parameters(), 1.0)
                # Update parameters and take a step using the computed gradient.
                optimizer.step()
                # Update the learning rate.
                scheduler.step()
                
            # Calculate the average loss over the training data.
            avg_train_loss = total_loss / len(train_dataloader)
            print(f"Average training loss: {avg_train_loss}")

            # Store the loss value for plotting the learning curve.
            loss_values.append(avg_train_loss)
            
            # Compute the metrics on the validation and test sets.
            try:
                val_fscore, val_accuracy = Eval_phase(params, 'val', model)        
                test_fscore, test_accuracy = Eval_phase(params, 'test', model)
                
                print(f"Validation F1: {val_fscore:.4f}, Accuracy: {val_accuracy:.4f}")
                print(f"Test F1: {test_fscore:.4f}, Accuracy: {test_accuracy:.4f}")

                # Save the model only if the validation fscore improves. After all epochs, the best model is the final saved one. 
                if val_fscore > best_val_fscore:
                    print(f"Validation F1 improved: {val_fscore:.4f} > {best_val_fscore:.4f}")
                    best_val_fscore = val_fscore
                    save_model(model, tokenizer, params)
            except Exception as e:
                print(f"Error during evaluation: {e}")
                # Continue to next epoch even if evaluation fails
                
        except Exception as e:
            print(f"Error during training epoch {epoch_i+1}: {e}")
            # Continue to next epoch even if current epoch fails
            continue

    del model
    torch.cuda.empty_cache()
    # Define or calculate fscore before returning
    fscore = val_fscore if 'val_fscore' in locals() else 0.0  # Use validation fscore if available, otherwise default to 0
    return fscore, best_val_fscore

# Explanation of all the params used below. 
params = {
    'logging': 'local',
    'language': 'English',
    'is_train': True,
    'is_model': True,
    'learning_rate': 2e-5,
    'files': '../Dataset',
    'csv_file': '*_full.csv',
    'samp_strategy': 'stratified',
    'epsilon': 1e-8,
    'path_files': 'bert-base-multilingual-cased',
    'take_ratio': False,
    'sample_ratio': 16,
    'how_train': 'baseline',
    'epochs': 5,
    'batch_size': 16,
    'to_save': True,
    'weights': [1.0, 1.0],
    'what_bert': 'normal',
    'save_only_bert': False,
    'max_length': 128,
    'random_seed': 42,
    # New parameters for improvements
    'use_language_specific_models': True,
    'use_text_preprocessing': True,
    'use_data_augmentation': True,
    'augment_ratio': 0.3,
    'use_weighted_loss': True,
    'use_dropout': True,
    'dropout_rate': 0.3,
    'use_cosine_scheduler': True
}

if __name__ == '__main__':
    lang_map = {
        'French': 'fr', 'Indonesian': 'id','English': 'en', 'Portugese': 'pt', 'Arabic': 'ar'
    }

    lang_list = list(lang_map.keys())

    # Check if deep-translator is available
    try:
        from deep_translator import GoogleTranslator
        print("deep-translator is available for data augmentation")
    except ImportError:
        print("deep-translator not installed. Data augmentation will be skipped.")
        params['use_data_augmentation'] = False

    # Ask user if they want to run the full hyperparameter search or just train with best settings
    run_hyperparameter_search = False
    user_input = input("Run full hyperparameter search? (y/n, default: n): ").lower()
    if user_input == 'y':
        run_hyperparameter_search = True

    # Initialize best_val_fscore before the loop
    best_val_fscore = 0

    for lang in lang_list:
        print(f"\n===== Starting training for {lang} =====\n")
        params['language'] = lang

        # Skip this language if model directory already exists
        if glob.glob(f"models_saved/multilingual_bert_{lang}_translated*"):
            skip = input(f"Model for {lang} already exists. Skip training? (y/n, default: y): ").lower()
            if skip != 'n':
                print(f"Skipping training for {lang}.")
                continue
        
        if run_hyperparameter_search:
            curr_best_val_fscore = 0
            for sample_ratio, take_ratio in [(16, False), (32, False), (64, False), (128, False), (256, False)]:
                for lr in [2e-5, 3e-5, 5e-5]:
                    for ss in ['stratified', 'equal']:
                        for seed in [2018, 2019, 2020, 2021, 2022]:
                            params['take_ratio'] = take_ratio
                            params['sample_ratio'] = sample_ratio
                            params['learning_rate'] = lr
                            params['samp_strategy'] = ss
                            params['random_seed'] = seed
                            
                            try:
                                _, curr_best_val_fscore = train_model(params, curr_best_val_fscore)
                            except Exception as e:
                                print(f"Error training model: {e}")
                                continue
            best_val_fscore = max(best_val_fscore, curr_best_val_fscore)
        else:
            # Train with best settings
            try:
                _, curr_best_val_fscore = train_model(params, best_val_fscore)
                best_val_fscore = max(best_val_fscore, curr_best_val_fscore)
            except Exception as e:
                print(f"Error training model: {e}")
                continue

        print(f'\n============================')
        print(f'Model for Language {lang} is trained')
        print('============================\n')