#Necessary imports

import transformers 
import torch
import neptune
from dotenv import load_dotenv
import glob 
from transformers import get_linear_schedule_with_warmup
from transformers import BertTokenizer
from transformers import BertForSequenceClassification, BertConfig
from transformers import AdamW
import random
from transformers import BertTokenizer
from bert_codes.feature_generation import combine_features,return_dataloader
from bert_codes.data_extractor import data_collector
from transformers.models.bert.modeling_bert import *
from bert_codes.utils import *
from sklearn.metrics import accuracy_score,f1_score
from tqdm import tqdm
from BERT_inference import *
from pathlib import Path
import os

load_dotenv()  # take environment variables

# If there's a GPU available...
if torch.cuda.is_available():
    device = torch.device("cuda")
    print(f"There are {torch.cuda.device_count()} GPU(s) available.")
    print(f"We will use the GPU: {torch.cuda.get_device_name(torch.cuda.current_device())}")
else:
    device = torch.device("cpu")
    print("No GPU available, using the CPU instead.")


# Set the gpu device 
#print("current gpu device", torch.cuda.current_device())
#torch.cuda.set_device(0)
#print("current gpu device",torch.cuda.current_device())


# Initialize neptune for logging the parameters and metrics
run = neptune.init_run(
    project=os.getenv("PROJECT_NAME"),
    api_token=os.getenv("NEPTUNE_API_TOKEN"),
    proxies=os.getenv("PROXY")
)
  
# The main function that does the training

def train_model(params,best_val_fscore):
	
	# In case of english languages, translation is the origin data itself.
	#if(params['language']=='English'):
	lang = params['language']
	params['csv_file'] = f"{lang}_*_full.csv"

	train_path = Path(params['files']) / 'train' / lang / params['csv_file']
	val_path = Path(params['files']) / 'val' / lang / params['csv_file']

	train_files = glob.glob(str(train_path))
	val_files = glob.glob(str(val_path))

	print(f"Train path: {train_path}")
	print(f"Val path: {val_path}")
	print(f"Train files found: {train_files}")
	print(f"Validation files found: {val_files}")
	
	#Load the bert tokenizer
	print('Loading BERT tokenizer...')
	tokenizer = BertTokenizer.from_pretrained(params['path_files'], do_lower_case=False)
	df_train=data_collector(train_files,params,True)
	df_val=data_collector(val_files,params,False)
	
	# Get the comment texts and corresponding labels
	"""
	if(params['csv_file']=='*_full.csv'):
		sentences_train = df_train.text.values
		sentences_val = df_val.text.values
	elif(params['csv_file']=='*_translated.csv'):
		sentences_train = df_train.translated.values
		sentences_val = df_val.translated.values
    """
	if df_train.columns.__contains__('text'):
		sentences_train = df_train.text.values
		sentences_val = df_val.text.values
	elif df_train.columns.__contains__('translated'):
		sentences_train = df_train.translated.values
		sentences_val = df_val.translated.values
	else:
		raise ValueError("Expected 'text' or 'translated' column in the data.")
	
	#labels_train = df_train.label.values
	#labels_val = df_val.label.values
	# Convert string labels to numeric labels
	label_mapping = {label: idx for idx, label in enumerate(df_train['label'].unique())}
	print("Label mapping:", label_mapping)

	df_train['label'] = df_train['label'].map(label_mapping)
	df_val['label'] = df_val['label'].map(label_mapping)

	labels_train = df_train.label.values
	labels_val = df_val.label.values

	label_counts=df_train['label'].value_counts()
	print(label_counts)
	label_weights = [ (len(df_train))/label_counts[0],len(df_train)/label_counts[1] ]
	print(label_weights)
	
	# Select the required bert model. Refer below for explanation of the parameter values.
	model=select_model(params['what_bert'],params['path_files'],params['weights'])
	# Tell pytorch to run this model on the GPU.
	#model.cuda()
	model.to(device)

	# Do the required encoding using the bert tokenizer
	input_train_ids,att_masks_train=combine_features(sentences_train,tokenizer,params['max_length'])
	input_val_ids,att_masks_val=combine_features(sentences_val,tokenizer,params['max_length'])

	# Create dataloaders for both the train and validation datasets.
	train_dataloader = return_dataloader(input_train_ids,labels_train,att_masks_train,batch_size=params['batch_size'],is_train=params['is_train'])
	validation_dataloader=return_dataloader(input_val_ids,labels_val,att_masks_val,batch_size=params['batch_size'],is_train=False)
	
	# Initialize AdamW optimizer.
	optimizer = AdamW(model.parameters(),
				  lr = params['learning_rate'], # args.learning_rate - default is 5e-5, our notebook had 2e-5
				  eps = params['epsilon'] # args.adam_epsilon  - default is 1e-8.
				)
	
	# Number of training epochs (authors recommend between 2 and 4)
	# Total number of training steps is number of batches * number of epochs.
	total_steps = len(train_dataloader) * params['epochs']

	# Create the learning rate scheduler.
	scheduler = get_linear_schedule_with_warmup(optimizer, 
												num_warmup_steps = int(total_steps/10), # Default value in run_glue.py
												num_training_steps = total_steps)

	# Set the seed value all over the place to make this reproducible.
	fix_the_random(seed_val = params['random_seed'])
	# Store the average loss after each epoch so we can plot them.
	loss_values = []

	# Create a new experiment in neptune for this run. 
	bert_model = params['path_files']
	language  = params['language']
	name_one=bert_model+"_"+language
	"""
	if(params['logging']=='neptune'):
		neptune.create_experiment(name_one,params=params,send_hardware_metrics=False,run_monitoring_thread=False)
		neptune.append_tag(bert_model)
		neptune.append_tag(language)
	"""
	if params['logging'] == 'neptune':
		run = neptune.init_run(
			project=os.getenv("PROJECT_NAME"),
			api_token=os.getenv("NEPTUNE_API_TOKEN"),
			proxies=os.getenv("PROXY")
		)
		run["params"] = params  # optional: logs all config settings
		run["model_name"] = name_one
		
	# The best val fscore obtained till now, for the purpose of hyper parameter finetuning.
	best_val_fscore=best_val_fscore

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
			if(params['logging']=='neptune'):
				run["batch_loss"].log(loss)
			# Accumulate the training loss over all of the batches so that we can
			# calculate the average loss at the end. `loss` is a Tensor containing a
			# single value; the `.item()` function just returns the Python value 
			# from the tensor.
			total_loss += loss.item()

			# Perform a backward pass to calculate the gradients.
			loss.backward()

			# Clip the norm of the gradients to 1.0.
			# This is to help prevent the "exploding gradients" problem.
			torch.nn.utils.clip_grad_norm_(model.parameters(), 1.0)
			# Update parameters and take a step using the computed gradient.
			# The optimizer dictates the "update rule"--how the parameters are
			# modified based on their gradients, the learning rate, etc.
			optimizer.step()
			# Update the learning rate.
			scheduler.step()
		# Calculate the average loss over the training data.
		avg_train_loss = total_loss / len(train_dataloader)
		if(params['logging']=='neptune'):
			run["avg_train_loss"].log(avg_train_loss)
		

		# Store the loss value for plotting the learning curve.
		loss_values.append(avg_train_loss)
		# Compute the metrics on the validation and test sets.
		val_fscore,val_accuracy=Eval_phase(params,'val',model)		
		test_fscore,test_accuracy=Eval_phase(params,'test',model)

		#Report the final accuracy and fscore for this validation run.
		if(params['logging']=='neptune'):
			run["val_f1score"].log(val_fscore)
			run["val_accuracy"].log(val_accuracy)
			run["test_f1score"].log(test_fscore)
			run["test_accuracy"].log(test_accuracy)

		# Save the model only if the validation fscore improves. After all epochs, the best model is the final saved one. 
		if(val_fscore > best_val_fscore):
			print(val_fscore,best_val_fscore)
			best_val_fscore=val_fscore

			save_model(model,tokenizer,params) 		

	if params['logging'] == 'neptune':
		try:
			run.stop()
		except Exception as e:
			print("Neptune stop error:", e)
	del model
	torch.cuda.empty_cache()
	# Define or calculate fscore before returning
	fscore = val_fscore  # Assuming fscore is the validation fscore
	return fscore, best_val_fscore



# Explanation of all the params used below. 

# 'logging':where logging {'local','neptune'}
# 'language': language {'Arabic', 'English','German','Indonesian','Italian','Polish','Portugese','Spanish','French'}
# 'is_train': whether train dataset 
# 'is_model':is model 
# 'learning_rate':Adam parameter lr
# 'files': Path to the dataset folder ( containing the train, val and test subfolders)
# 'csv_file': The regex used by glob to load the datasets. {'*_full.csv','*_translated.csv'} for untranslated and translated datasets respectively
# 'samp_strategy': The way in which we sample the training data points. {'stratified'}
# 'epsilon': Adam parameter episilon
# 'path_files':bert path from where the bert model should be loaded,
# 'take_ratio': Whether the sample ratio is ratio of total points or absolute number of points needed.
# 'sample_ratio':ratio or the number of the training data points to take
# 'how_train':how the bert is trained possible option {'all','baseline','all_but_one'}
# 'epochs': number of epochs to train bert
# 'batch_size': batch size
# 'to_save': whether to save the model or not
# 'weights': weights for binary classifier
# 'what_bert': type of bert possible option {'normal','weighted'}
# 'save_only_bert': if only bert (without classifier) should be used 
# 'max_length': maximum length for input tokenization
# 'random_seed': seed value for reproducibility

params={
	'logging':'neptune',
	'language':'English',
	'is_train':True,
	'is_model':True,
	'learning_rate':2e-5,
	'files':'../Dataset',
	'csv_file':'*_full.csv',
	'samp_strategy':'stratified',
	'epsilon':1e-8,
	'path_files':'multilingual_bert',
	'take_ratio':False,
	'sample_ratio':16,
	'how_train':'baseline',
	'epochs':5,
	'batch_size':16,
	'to_save':True,
	'weights':[1.0,1.0],
	'what_bert':'normal',
	'save_only_bert':False,
	'max_length':128,
	'random_seed':42

}



if __name__ == '__main__':
    lang_map = {
        'Arabic': 'ar', 'French': 'fr', 'Portugese': 'pt', 'Spanish': 'es',
        'English': 'en', 'Indonesian': 'id', 'Italian': 'it', 'German': 'de', 'Polish': 'pl'
    }

    lang_list = list(lang_map.keys())

    for lang in lang_list:
        print(f"\n===== Starting training for {lang} =====\n")
        params['language'] = lang

        # Skip this language if model directory already exists
        if glob.glob(f"models_saved/multilingual_bert_{lang}_translated*"):
            print(f"Model for {lang} already exists. Skipping training.")
            continue
		
        for sample_ratio, take_ratio in [(16, False), (32, False), (64, False), (128, False), (256, False)]:
            for lr in [2e-5, 3e-5, 5e-5]:
                for ss in ['stratified', 'equal']:
                    for seed in [2018, 2019, 2020, 2021, 2022]:
                        params['take_ratio'] = take_ratio
                        params['sample_ratio'] = sample_ratio
                        params['learning_rate'] = lr
                        params['samp_strategy'] = ss
                        params['random_seed'] = seed
                        best_val_fscore = 0
                        _, best_val_fscore = train_model(params, best_val_fscore)

        print(f'\n============================')
        print(f'Model for Language {lang} is trained')
        print('============================\n')