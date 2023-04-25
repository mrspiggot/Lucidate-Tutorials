import numpy as np
import torch
import torch.nn as nn
import torch.optim as optim
import matplotlib.pyplot as plt

# Create radian and sine value tensors
radians = np.linspace(0, 2 * np.pi, 100)
sine_values = np.sin(radians)

radians = torch.tensor(radians, dtype=torch.float32).view(-1, 1)
sine_values = torch.tensor(sine_values, dtype=torch.float32).view(-1, 1)

class SineNet2(nn.Module):
    def __init__(self):
        super(SineNet2, self).__init__()
        self.layer1 = nn.Linear(1, 10)
        self.relu1 = nn.ReLU()
        self.layer2 = nn.Linear(10, 10)
        self.relu2 = nn.ReLU()
        self.layer3 = nn.Linear(10, 1)

    def forward(self, x):
        x = self.layer1(x)
        x = self.relu1(x)
        x = self.layer2(x)
        x = self.relu2(x)
        x = self.layer3(x)
        return x

# Define the neural network
class SineNet(nn.Module):
    def __init__(self):
        super(SineNet, self).__init__()
        self.layer1 = nn.Linear(1, 10)
        self.relu = nn.ReLU()
        self.layer2 = nn.Linear(10, 1)

    def forward(self, x):
        x = self.layer1(x)
        x = self.relu(x)
        x = self.layer2(x)
        return x


# Instantiate network, loss, and optimizer
model = SineNet()
loss_function = nn.MSELoss()
optimizer = optim.SGD(model.parameters(), lr=0.01)

# Train the neural network
epochs = 200
avg_losses = []

for epoch in range(epochs):
    predictions = []
    epoch_loss = 0

    for radian, target in zip(radians, sine_values):
        optimizer.zero_grad()
        output = model(radian)
        loss = loss_function(output, target)
        loss.backward()
        optimizer.step()

        predictions.append(output.item())
        epoch_loss += loss.item()

    avg_losses.append(epoch_loss / len(radians))

# Plot results
plt.style.use('dark_background')
plt.figure()
plt.plot(radians, sine_values, label='Actual Sine Function', linewidth=2)
plt.plot(radians, predictions, label='Predicted Sine Function', linewidth=2)
plt.xlabel('Radians')
plt.ylabel('Sine Values')
plt.title('Actual vs Predicted Sine Function')
plt.legend()
plt.show()

plt.figure()
plt.plot(range(epochs), avg_losses)
plt.xlabel('Epochs')
plt.ylabel('Average Loss')
plt.title('Average Loss per Epoch')
plt.show()
