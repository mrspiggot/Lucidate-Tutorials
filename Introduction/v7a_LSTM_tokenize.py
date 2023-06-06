import nltk
import numpy as np
from nltk.tokenize import word_tokenize
from sklearn.feature_extraction.text import CountVectorizer
from collections import defaultdict

import torch
import torch.nn as nn
import torch.optim as optim


def generate_text(model, start_seq, length):
    model.eval()
    words = start_seq.split()
    input_indices = [word2idx[word] for word in words]
    inputs = torch.tensor(input_indices).unsqueeze(0)

    for _ in range(length):
        outputs = model(inputs)
        _, predicted = torch.max(outputs[:, -1, :], dim=1)
        inputs = torch.cat((inputs, predicted.unsqueeze(0)), dim=1)
        words.append(idx2word[predicted.item()])

    return ' '.join(words)


nltk.download('punkt')  # Download the Punkt tokenizer

# Assuming you have the text of Alice in Wonderland in a text file
with open('alice_in_wonderland.txt', 'r') as file:
    text = file.read().replace('\n', ' ')

# Tokenize the text
tokens = word_tokenize(text)

# Remove non-alphabetic tokens and convert to lower case
# Also remove words that include non-ASCII characters
tokens = [token.lower() for token in tokens if token.isalpha() and all(ord(c) < 128 for c in token)]


vectorizer = CountVectorizer()
vectorizer.fit_transform(tokens)

# Create word to index mapping
word2idx = vectorizer.vocabulary_

# We also create a reverse mapping to use later
idx2word = {idx: word for word, idx in word2idx.items()}

input_sequences = []
target_sequences = []

# Assuming seq_length is the length of the sequence you want for your LSTM
seq_length = 20  # you can adjust this

for i in range(len(tokens) - seq_length):
    input_seq = tokens[i: i + seq_length]
    target_seq = tokens[i + 1: i + seq_length + 1]

    # Convert sequence of words to sequence of indices
    input_indices = [word2idx[word] for word in input_seq if word in word2idx]
    target_indices = [word2idx[word] for word in target_seq if word in word2idx]

    # Only add sequences where input and target length matches the seq_length
    if len(input_indices) == seq_length and len(target_indices) == seq_length:
        input_sequences.append(input_indices)
        target_sequences.append(target_indices)




# Define LSTM model
class LSTMModel(nn.Module):
    def __init__(self, vocab_size, embedding_dim, hidden_dim):
        super(LSTMModel, self).__init__()
        self.embedding = nn.Embedding(vocab_size, embedding_dim)
        self.lstm = nn.LSTM(embedding_dim, hidden_dim, batch_first=True)
        self.fc = nn.Linear(hidden_dim, vocab_size)

    def forward(self, x):
        x = self.embedding(x)
        output, (hidden, _) = self.lstm(x)
        output = self.fc(output)  # take all outputs not just the last one
        return output


# Assuming these variables are defined
vocab_size = len(word2idx)  # depends on your data
embedding_dim = 128
hidden_dim = 256
learning_rate = 0.001
epochs = 10

# Initialize model, loss function, and optimizer
model = LSTMModel(vocab_size, embedding_dim, hidden_dim)
criterion = nn.CrossEntropyLoss()
optimizer = optim.Adam(model.parameters(), lr=learning_rate)

# Assume input_sequences and target_sequences are your data
# input_sequences and target_sequences are lists of indices
# for example: input_sequences[0] might be [5, 12, 3] and target_sequences[0] might be [12, 3, 15]



for epoch in range(epochs):
    for i in range(len(input_sequences)):
        inputs = torch.tensor(input_sequences[i]).unsqueeze(0)  # unsqueeze to have batch dimension
        targets = torch.tensor(target_sequences[i]).unsqueeze(0)  # add batch dimension to targets too

        # forward pass
        outputs = model(inputs)
        loss = criterion(outputs.view(-1, vocab_size), targets.view(-1))  # flatten outputs and targets

        # backward pass and optimization
        optimizer.zero_grad()
        loss.backward()
        optimizer.step()

    print (f'Epoch {epoch+1}/{epochs}, Loss: {loss.item()}')



start_seq = 'alice was beginning'
length = 100  # length of the generated sequence
print(generate_text(model, start_seq, length))

