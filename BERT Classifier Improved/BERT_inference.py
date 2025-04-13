# Necessary imports
import transformers 
import torch
from dotenv import load_dotenv
import glob 
from transformers import get_linear_schedule_with_warmup
from transformers import BertTokenizer
from transformers import BertForSequenceClassification, AdamW, BertConfig
import random
from transformers import BertTokenizer
from bert_codes.feature_generation import combine_features,return_dataloader
from bert_codes.data_extractor import data_collector
from bert_codes.own_bert_models import *
from bert_codes.utils import *
from sklearn.metrics import accuracy_score,f1_score
from tqdm import tqdm
import os

load_dotenv()  # take environment variables

# If gpu is available
if torch.cuda.is_available():
    device = torch.device("cuda")
    print(f"There are {torch.cuda.device_count()} GPU(s) available.")
    print(f"We will use the GPU: {torch.cuda.get_device_name(torch.cuda.current_device())}")
else:
    device = torch.device("cpu")
    print("No GPU available, using the CPU instead.")

# The function for evaluating
# Params - see below for description
# which_files - what files to test on - {'train','val','test'}
# model - the model to use if passed. If model==None, the model is loaded based on the params passed.
def Eval_phase(params, which_files='test', model=None):
    try:
        # For english, there is no translation, hence use full dataset.
        #if(params['language']=='English'):
        params['csv_file']='*_full.csv'
        
        # Load the files to test on
        path = os.path.join(params['files'], which_files, params['language'], params['csv_file'])
        test_files = glob.glob(path)
        
        if not test_files:
            print(f"WARNING: No {which_files} files found at {path}")
            return 0.0, 0.0
        
        print(f"Evaluating on {which_files} files: {test_files}")
        
        '''Testing phase of the model'''
        print('Loading BERT tokenizer...')
        # Load bert tokenizer
        try:
            tokenizer = BertTokenizer.from_pretrained(params['path_files'], do_lower_case=False)
        except Exception as e:
            print(f"Error loading tokenizer: {e}")
            print("Falling back to AutoTokenizer")
            from transformers import AutoTokenizer
            tokenizer = AutoTokenizer.from_pretrained(params['path_files'], do_lower_case=False)

        # If model is passed, then use the given model. Else load the model from the saved location
        # Put the model in evaluation mode--the dropout layers behave differently
        # during evaluation.
        if(params['is_model']==True):
            print("model previously passed")
            model.eval()
        else:
            try:
                model=select_model(params['what_bert'],params['path_files'],params['weights'])
                #model.cuda()
                model.to(device)
                model.eval()
            except Exception as e:
                print(f"Error loading model: {e}")
                return 0.0, 0.0

        # Load the dataset
        try:
            df_test=data_collector(test_files,params,False)
            if(params['csv_file']=='*_translated.csv'):
                sentences_test = df_test.translated.values
            elif(params['csv_file']=='*_full.csv'):
                sentences_test = df_test.text.values
                
            labels_test = df_test.label.values
        except Exception as e:
            print(f"Error loading dataset: {e}")
            return 0.0, 0.0
            
        # Encode the dataset using the tokenizer
        try:
            input_test_ids,att_masks_test=combine_features(sentences_test,tokenizer,params['max_length'])
            test_dataloader=return_dataloader(input_test_ids,labels_test,att_masks_test,batch_size=params['batch_size'],is_train=False)
        except Exception as e:
            print(f"Error encoding dataset: {e}")
            return 0.0, 0.0
            
        print(f"Running eval on {which_files}...")
        t0 = time.time()

        # Tracking variables 
        eval_loss, eval_accuracy = 0, 0
        nb_eval_steps, nb_eval_examples = 0, 0
        true_labels=[]
        pred_labels=[]
        
        try:
            for batch in test_dataloader:
                # Add batch to GPU
                batch = tuple(t.to(device) for t in batch)
                # Unpack the inputs from our dataloader
                b_input_ids, b_input_mask, b_labels = batch
                # Telling the model not to compute or store gradients, saving memory and
                # speeding up validation
                with torch.no_grad():        
                    outputs = model(b_input_ids, 
                                    token_type_ids=None, 
                                    attention_mask=b_input_mask)

                logits = outputs[0]
                # Move logits and labels to CPU
                logits = logits.detach().cpu().numpy()
                label_ids = b_labels.to('cpu').numpy()
                # Calculate the accuracy for this batch of test sentences.
                tmp_eval_accuracy = flat_accuracy(logits, label_ids)
                # Accumulate the total accuracy.
                eval_accuracy += tmp_eval_accuracy
                nb_eval_steps += 1
                nb_eval_examples += len(b_labels)
                
                # Store predictions and true labels for F1 calculation
                pred_labels.extend(np.argmax(logits, axis=1).flatten())
                true_labels.extend(label_ids.flatten())
                
            # Report the final accuracy for this validation run.
            eval_accuracy = eval_accuracy / nb_eval_steps
            print(f"  Accuracy: {eval_accuracy}")
            
            # Calculate F1 score
            fscore = f1_score(true_labels, pred_labels, average='weighted')
            print(f"  F1 Score: {fscore}")
            
            return fscore, eval_accuracy
            
        except Exception as e:
            print(f"Error during evaluation: {e}")
            return 0.0, 0.0
            
    except Exception as e:
        print(f"Error in evaluation phase: {e}")
        return 0.0, 0.0  # Return defaults to prevent crashes

# Params used here 
params={
    'logging':'local',  # Changed from 'locals' to 'local' for consistency
    'language':'German',
    'is_train':False,
    'is_model':False,
    'learning_rate':2e-5,
    'epsilon':1e-8,
    'path_files':'models_saved/multilingual_bert_English_baseline_100/',
    'sample_ratio':0.1,
    'how_train':'baseline',
    'epochs':5,
    'batch_size':16,
    'to_save':False,
    'weights':[1.0,1.0],
    'what_bert':'weighted',
    'save_only_bert':True,
    'max_length':128,  # Added max_length which was missing
    'files':'../Dataset'  # Added files path which was missing
}


if __name__=='__main__':
    for lang in ['English','Polish','Portugese','German','Indonesian','Italian','Arabic']:
        params['language']=lang
        Eval_phase(params,'test')