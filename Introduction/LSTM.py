import torch
from torch import nn
import torch.optim as optim
from torchtext.datasets import AG_NEWS
from torchtext.data.utils import get_tokenizer
from torchtext.vocab import build_vocab_from_iterator
from torch.utils.data import DataLoader, Dataset
from collections import Counter
from sklearn.model_selection import train_test_split
import matplotlib.pyplot as plt
import os
import gensim
import spacy

# Load spaCy's English tokenizer
spacy_en = spacy.load('en_core_web_sm')

# A function to tokenize text using spaCy
def tokenize_en(text):
    return [token.text for token in spacy_en.tokenizer(text)]

# Prepare dataset
tokenizer = tokenize_en

# Prepare dataset
# tokenizer = get_tokenizer('basic_english')

# Restricting to only 10000 sentences for the demonstration
train_iter = AG_NEWS(split='train')
sentences = [" ".join(tokenizer(item[1])) for item in list(train_iter)]
labels = [item[0] - 1 for item in list(train_iter)]


def build_vocab(sentences, tokenizer):
    counter = Counter()
    for sentence in sentences:
        counter.update(tokenizer(sentence))
    return counter

word2vec = gensim.models.KeyedVectors.load_word2vec_format('GoogleNews-vectors-negative300.bin', binary=True)

vocab = build_vocab(sentences, tokenizer)
# vocab = {word: idx for idx, (word, _) in enumerate(vocab.items()) if word in word2vec}
# vocab["<unk>"] = len(vocab)  # add <unk> token to the vocabulary
vocab = {word: idx for idx, (word, _) in enumerate(vocab.items()) if word in word2vec}
vocab["<unk>"] = len(vocab)  # add <unk> token to the vocabulary


# Convert the sentences to sequences of numbers
# sequences = [[vocab[word] for word in tokenizer(sentence)] for sentence in sentences]
sequences = [[vocab[word] if word in vocab else vocab["<unk>"] for word in tokenizer(sentence)] for sentence in sentences]


# Split the dataset into train, validation and test
train_sequences, test_sequences, train_labels, test_labels = train_test_split(sequences, labels, test_size=0.2)
train_sequences, val_sequences, train_labels, val_labels = train_test_split(train_sequences, train_labels,
                                                                            test_size=0.2)


# Build a simple RNN model
# class SimpleRNN(nn.Module):
#     def __init__(self, input_size, hidden_size, output_size):
#         super(SimpleRNN, self).__init__()
#         self.hidden_size = hidden_size
#         self.embedding = nn.Embedding(input_size, hidden_size)
#         self.rnn = nn.RNN(hidden_size, hidden_size, batch_first=True)
#         self.fc = nn.Linear(hidden_size, output_size)
#
#     def forward(self, x):
#         embedded = self.embedding(x)
#         output, hidden = self.rnn(embedded)
#         output = self.fc(output[:, -1, :])
#         return output



# Load Google's pre-trained Word2Vec model.
# word2vec = gensim.models.KeyedVectors.load_word2vec_format('GoogleNews-vectors-negative300.bin', binary=True)
# Hyperparameters

input_size = len(vocab)
hidden_size = 300
output_size = max(labels) + 1
num_epochs = 60
learning_rate = 0.001

# class SimpleRNN(nn.Module):
#     def __init__(self, input_size, hidden_size, output_size, pretrained_vectors):
#         super(SimpleRNN, self).__init__()
#         self.hidden_size = hidden_size
#
#         # Use pre-trained embeddings
#         self.embedding = nn.Embedding.from_pretrained(pretrained_vectors)
#         self.rnn = nn.RNN(hidden_size, hidden_size, batch_first=True)
#         self.fc = nn.Linear(hidden_size, output_size)
#
#     def forward(self, x):
#         embedded = self.embedding(x)
#         output, hidden = self.rnn(embedded)
#         output = self.fc(output[:, -1, :])
#         return output

class LSTMModel(nn.Module):
    def __init__(self, input_size, hidden_size, output_size, pretrained_vectors, num_layers=2, dropout=0.5):
        super(LSTMModel, self).__init__()
        self.hidden_size = hidden_size
        self.num_layers = num_layers

        # Use pre-trained embeddings
        self.embedding = nn.Embedding.from_pretrained(pretrained_vectors)
        self.lstm = nn.LSTM(hidden_size, hidden_size, num_layers, dropout=dropout, batch_first=True)
        self.fc = nn.Linear(hidden_size, output_size)

    def forward(self, x):
        h0 = torch.zeros(self.num_layers, x.size(0), self.hidden_size).to(x.device)
        c0 = torch.zeros(self.num_layers, x.size(0), self.hidden_size).to(x.device)

        embedded = self.embedding(x)
        out, _ = self.lstm(embedded, (h0, c0))  # out: tensor of shape (batch_size, seq_length, hidden_size)
        out = self.fc(out[:, -1, :])
        return out


# Create a tensor to hold the pre-trained embeddings
# pretrained_vectors = torch.randn(len(vocab), 300)  # 300 is the dimension of the Google Word2Vec model
# for i, word in enumerate(vocab.keys()):
#     if word in word2vec:
#         pretrained_vectors[i] = torch.from_numpy(word2vec[word])
# Create a tensor to hold the pre-trained embeddings
# pretrained_vectors = torch.randn(len(vocab), 300)  # 300 is the dimension of the Google Word2Vec model
# for i, word in enumerate(vocab.keys()):
#     if word in word2vec:
#         pretrained_vectors[i] = torch.from_numpy(word2vec[word])
#     else:
#         pretrained_vectors[i] = torch.randn(300) # create a random vector for words not in pretrained model


pretrained_vectors = torch.empty(len(vocab), 300)
for word, idx in vocab.items():
    if word in word2vec:
        pretrained_vectors[idx] = torch.from_numpy(word2vec[word])
    else:
        pretrained_vectors[idx] = torch.randn(300)

# # Increase hidden_size
# hidden_size = 128

# model = SimpleRNN(input_size, hidden_size, output_size, pretrained_vectors)
num_layers=3
model = LSTMModel(input_size, hidden_size, output_size, pretrained_vectors, num_layers=num_layers)

# Hyperparameters
# input_size = len(vocab)
# hidden_size = 64
# output_size = max(labels) + 1
# num_epochs = 100
# learning_rate = 0.001

model_filename = f'lstm_model_{num_epochs}.pth'

# model = SimpleRNN(input_size, hidden_size, output_size)
# model = SimpleRNN(input_size, hidden_size, output_size, pretrained_vectors)

# Loss and optimizer
criterion = nn.CrossEntropyLoss()
optimizer = torch.optim.Adam(model.parameters(), lr=learning_rate)

# For storing metrics
train_losses = []
val_accuracies = []

# Train the model
if os.path.isfile(model_filename):
    model.load_state_dict(torch.load(model_filename))
else:
    # Training loop here
    for epoch in range(num_epochs):
    # your training code here
        model.train()
        train_loss = 0
        for i, sequence in enumerate(train_sequences):
            sequence = torch.tensor(sequence).unsqueeze(0)
            label = torch.tensor([train_labels[i]])

            # Forward pass
            outputs = model(sequence)
            loss = criterion(outputs, label)
            train_loss += loss.item()

            # Backward and optimize
            optimizer.zero_grad()
            loss.backward()
            optimizer.step()

        train_loss /= len(train_sequences)
        train_losses.append(train_loss)

        model.eval()
        correct, total = 0, 0
        with torch.no_grad():
            for i, sequence in enumerate(val_sequences):
                sequence = torch.tensor(sequence).unsqueeze(0)
                label = torch.tensor([val_labels[i]])

                outputs = model(sequence)
                _, predicted = torch.max(outputs.data, 1)
                total += label.size(0)
                correct += (predicted == label).sum().item()

        val_accuracy = correct / total * 100
        val_accuracies.append(val_accuracy)

        print(f'Epoch {epoch + 1}/{num_epochs}, Loss: {train_loss:.4f}, Validation Accuracy: {val_accuracy:.2f}%')
    torch.save(model.state_dict(), model_filename)
# Plot training metrics
plt.figure(figsize=(10, 5))
plt.plot(train_losses, label='Training Loss')
plt.xlabel('Epochs')
plt.ylabel('Loss')
plt.title('Training Loss')
plt.legend()
plt.show()

plt.figure(figsize=(10, 5))
plt.plot(val_accuracies, label='Validation Accuracy')
plt.xlabel('Epochs')
plt.ylabel('Accuracy')
plt.title('Validation Accuracy')
plt.legend()
plt.show()


# Interactive Testing
# Interactive Testing
def predict_next_word(model, sentence):
    model.eval()
    with torch.no_grad():
        inputs = torch.tensor([vocab[word] for word in tokenizer(sentence)]).unsqueeze(0)
        outputs = model(inputs)
        _, predicted = torch.max(outputs.data, 1)
        predicted_word = list(vocab.keys())[predicted.item()]
        return predicted_word


# 1. List of Prompts
print("\nPredicting last word from a list of prompts")
prompts = ["I love to play ", "I am studying ", "She is very ", "It is a nice ", "The cat is ", "He went to ", "The weather is ", "I will travel to ", "Tomorrow will be ", "We had a "]
for prompt in prompts:
    words = prompt.split()
    incomplete_sentence = ' '.join(words[:-1])
    actual_word = words[-1]
    predicted_word = predict_next_word(model, incomplete_sentence)
    print(f"Prompt: {prompt}, Actual word: {actual_word}, Predicted word: {predicted_word}")

# 2. User Prompt
print("\nPredicting next word from a user prompt (type 'QUIT' to stop)")
while True:
    prompt = input("Enter a sentence: ")
    if prompt == 'QUIT':
        break
    predicted_word = predict_next_word(model, prompt)
    print(f"Next word: {predicted_word}")

# 3. Continual Prediction
print("\nContinual prediction (type 'QUIT' to stop)")
prompt = input("Enter a starting sentence: ")
for _ in range(10):
    predicted_word = predict_next_word(model, prompt)
    print(f"Next word: {predicted_word}")
    prompt += ' ' + predicted_word
