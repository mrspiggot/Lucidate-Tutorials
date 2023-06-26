import torch
from torch.utils.data import random_split
from transformers import AutoTokenizer, AutoModelForCausalLM, Trainer, TrainingArguments, GPT2LMHeadModel
from datasets import load_dataset
import numpy as np
import matplotlib.pyplot as plt
from datasets import load_metric

# Load the perplexity metric
perplexity_metric = load_metric("perplexity")

def compute_metrics(eval_pred):
    logits, labels = eval_pred
    predictions = torch.argmax(logits, dim=-1)
    perplexity = perplexity_metric.compute(predictions, labels)
    return {"perplexity": perplexity}

# Download the dataset
dataset_name = "wikitext"  # Change this to your desired dataset name
dataset = load_dataset(dataset_name, "wikitext-103-raw-v1")  # Change the second argument to the specific dataset version

# Download the model and tokenizer
model_name = "distilgpt2"  # Change this to your desired model name
tokenizer = AutoTokenizer.from_pretrained(model_name)
model = GPT2LMHeadModel.from_pretrained(model_name)
if tokenizer.pad_token is None:
    tokenizer.pad_token = tokenizer.eos_token

# ... (your existing code)



def generate_text(prompt, max_length=50):
    input_ids = tokenizer.encode(prompt, return_tensors="pt")
    generated = model.generate(input_ids, max_length=max_length, num_return_sequences=1, no_repeat_ngram_size=2)
    return tokenizer.decode(generated[0], skip_special_tokens=True)

# Provide a prompt for text completion
prompts = [
    "Recent advances in the treatment of Alzheimer's disease include...",
    "The CRISPR-Cas9 gene editing technique has revolutionized the field of genetics by...",
    "In cancer immunotherapy, the role of checkpoint inhibitors is crucial because...",
    "Stem cell therapy has the potential to treat various degenerative diseases, such as...",
    "The primary factors contributing to antibiotic resistance in bacteria are...",
    "The development of mRNA vaccines has greatly impacted the fight against infectious diseases like...",
    "In the context of precision medicine, the importance of personalized drug therapy lies in...",
    "One of the major challenges in the field of gene therapy is the safe and efficient delivery of...",
    "Nanotechnology has opened up new possibilities in targeted drug delivery systems by enabling...",
    "The human microbiome plays a critical role in maintaining health by..."
]


for prompt in prompts:
    # Call the function with the prompt
    generated_text = generate_text(prompt)

    # Print the generated text
    print("Generated Text:")
    print(generated_text)



import pubmed_parser as pp
import pandas as pd

# Parse the PubMed XML file
input_file = "pubmed23n0001.xml"
dict_out = pp.parse_medline_xml(input_file)

# Extract the abstracts
data = [{"text": record["abstract"]} for record in dict_out if record["abstract"]]

# Convert to DataFrame and save as CSV
df = pd.DataFrame(data)
df.to_csv("pubmed_abstracts.csv", index=False)

from datasets import load_dataset

dataset = load_dataset("csv", data_files="pubmed_abstracts.csv")
train_dataset = dataset['train']

# Tokenize the dataset
def tokenize_function(examples):
    return tokenizer(examples['text'], truncation=True, padding='max_length', max_length=128)

def tokenize_function2(examples):
    tokenized_inputs = tokenizer(examples['text'], truncation=True, padding='max_length', max_length=128, return_tensors="pt")
    tokenized_inputs["labels"] = tokenized_inputs["input_ids"].detach().clone()
    return tokenized_inputs

tokenized_dataset = train_dataset.map(tokenize_function2, batched=True, remove_columns=['text'])

# Split the dataset into train and validation
train_size = int(0.9 * len(tokenized_dataset))
val_size = len(tokenized_dataset) - train_size
train_dataset, val_dataset = random_split(tokenized_dataset, [train_size, val_size])

# Define the training configuration
training_args = TrainingArguments(
    output_dir='./results',  # Output directory for the model and checkpoints
    num_train_epochs=1,  # Number of training epochs
    per_device_train_batch_size=2,  # Batch size per device during training
    per_device_eval_batch_size=2,  # Batch size for evaluation
    gradient_accumulation_steps=8,  # Accumulate gradients for 2 steps before updating the model weights
    warmup_steps=500,  # Number of warmup steps for learning rate scheduler
    weight_decay=0.01,  # Strength of weight decay
    logging_dir='./logs',  # Directory for storing logs
    evaluation_strategy="epoch",  # Evaluate at the end of each epoch
    save_strategy="epoch",  # Maximum number of training steps, regardless of the number of epochs
    # save_total_limit=3,  # Keep only the 3 most recent checkpoints
    # fp16=True,  # Enable mixed-precision training
    # fp16_backend="amp",  # Use the PyTorch "amp" backend for mixed-precision training
    load_best_model_at_end=True,  # Load the best model at the end of training
)

# Instantiate the custom Trainer
trainer = Trainer(
    model=model,                         # The pre-trained model
    args=training_args,                  # Training configuration
    train_dataset=train_dataset,         # Training dataset
    eval_dataset=val_dataset,            # Validation dataset
    compute_metrics=compute_metrics,  # Add the compute_metrics function
)

# Train the model
trainer.train()

# ... (your existing code for inference)

# Plot the metrics
train_losses = [float(loss) for loss in trainer.state.log_history if 'loss' in loss]
validation_losses = [float(loss) for loss in trainer.state.log_history if 'eval_loss' in loss]
validation_perplexities = [float(perplexity) for perplexity in trainer.state.log_history if 'eval_perplexity' in perplexity]

# Plot training and validation loss
plt.plot(train_losses, label='Training Loss')
plt.plot(validation_losses, label='Validation Loss')
plt.xlabel('Updates')
plt.ylabel('Loss')
plt.legend()
plt.title('Training and Validation Loss')
plt.show()

# Plot validation perplexity
plt.plot(validation_perplexities, label='Validation Perplexity')
plt.xlabel('Epochs')
plt.ylabel('Perplexity')
plt.legend()
plt.title('Validation Perplexity')
plt.show()

# Calculate gradient norms
grad_norms = []
for state in trainer.state.log_history:
    if 'total_grad_norm' in state:
        grad_norms.append(float(state['total_grad_norm']))

# Plot gradient norms
plt.plot(grad_norms, label='Gradient Norms')
plt.xlabel('Updates')
plt.ylabel('Gradient Norm')
plt.legend()
plt.title('Gradient Norms')
plt.show()
