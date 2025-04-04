import os
from transformers import BertTokenizer, BertModel

# Set save path
save_path = "BERT Classifier/multilingual_bert"

# Create the directory if it doesn't exist
os.makedirs(save_path, exist_ok=True)

# Load model and tokenizer
tokenizer = BertTokenizer.from_pretrained('bert-base-multilingual-cased')
model = BertModel.from_pretrained('bert-base-multilingual-cased')

# Save model and tokenizer
tokenizer.save_pretrained(save_path)
model.save_pretrained(save_path)
