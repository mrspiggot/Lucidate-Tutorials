import torch
import torchvision
import torchvision.transforms as transforms
import torch.nn as nn
import torch.optim as optim
import torch.nn.functional as F
import matplotlib.pyplot as plt
import numpy as np
plt.style.use('dark_background')

if torch.backends.mps.is_available():
    print("GPU is available!")
    device = torch.device("mps")
else:
    print("GPU is not available. Using CPU.")
    device = torch.device("cpu")


# Function to display a grid of images
def show_images_grid(images, nrows, ncols, upscale_factor=1):
    fig, axes = plt.subplots(nrows, ncols, figsize=(ncols * upscale_factor, nrows * upscale_factor))
    for i, ax in enumerate(axes.flat):
        img = images[i].squeeze().numpy()
        img = (img * 0.5) + 0.5  # Unnormalize
        ax.imshow(img, cmap='gray')
        ax.axis('off')
    plt.show()

# Function to display images with histograms
def show_images_with_histograms(images, labels, predictions, nrows, ncols, upscale_factor=1):
    fig, axes = plt.subplots(nrows * 2, ncols, figsize=(ncols * upscale_factor, nrows * 2 * upscale_factor))
    for i in range(nrows * ncols):
        img_idx = i // ncols
        img_col = i % ncols

        # Show image
        ax = axes[img_idx * 2, img_col]
        img = images[i].cpu().squeeze().numpy()
        img = (img * 0.5) + 0.5  # Unnormalize
        ax.imshow(img, cmap='gray')
        ax.axis('off')

        # Show histogram
        ax = axes[img_idx * 2 + 1, img_col]
        ax.bar(range(10), predictions[i])
        ax.set_xticks(range(10))
        ax.set_yticks([])
        ax.set_title(f"True: {labels[i].item()}, Pred: {np.argmax(predictions[i])}")

    plt.show()


# Load MNIST dataset
transform = transforms.Compose([transforms.ToTensor(), transforms.Normalize((0.5,), (0.5,))])

trainset = torchvision.datasets.MNIST(root='./data', train=True, download=True, transform=transform)
trainloader = torch.utils.data.DataLoader(trainset, batch_size=64, shuffle=True, num_workers=0)

testset = torchvision.datasets.MNIST(root='./data', train=False, download=True, transform=transform)
testloader = torch.utils.data.DataLoader(testset, batch_size=128, shuffle=False, num_workers=0)

# Display a random sample of 64 training images
images, _ = next(iter(trainloader))
show_images_grid(images, nrows=8, ncols=8, upscale_factor=28 / 28)

# Define the neural network
class Net(nn.Module):
    def __init__(self):
        super(Net, self).__init__()
        self.conv1 = nn.Conv2d(1, 6, 5)
        self.pool = nn.MaxPool2d(2, 2)
        self.conv2 = nn.Conv2d(6, 16, 5)
        self.fc1 = nn.Linear(16 * 4 * 4, 120)
        self.fc2 = nn.Linear(120, 84)
        self.fc3 = nn.Linear(84, 10)

    def forward(self, x):
        x = self.pool(F.relu(self.conv1(x)))
        x = self.pool(F.relu(self.conv2(x)))
        x = x.view(-1, 16 * 4 * 4)
        x = F.relu(self.fc1(x))
        x = F.relu(self.fc2(x))
        x = self.fc3(x)
        return x

net = Net()



# Set loss function and optimizer
criterion = nn.CrossEntropyLoss()
optimizer = optim.SGD(net.parameters(), lr=0.001, momentum=0.9)

# Train the neural network
num_epochs = 20

for epoch in range(num_epochs):
    running_loss = 0.0
    for i, data in enumerate(trainloader, 0):
        inputs, labels = data


        optimizer.zero_grad()

        outputs = net(inputs)
        loss = criterion(outputs, labels)
        loss.backward()
        optimizer.step()

        running_loss += loss.item()

    print(f"Epoch {epoch + 1}, Loss: {running_loss / (i + 1)}")

print("Training completed.")

# Test the neural network
correct = 0
total = 0
images = []
labels = []
predictions = []
incorrect_images = []
incorrect_labels = []
incorrect_predictions = []

with torch.no_grad():
    for data in testloader:
        imgs, lbls = data
        outputs = net(imgs)
        _, predicted = torch.max(outputs.data, 1)
        total += lbls.size(0)
        correct += (predicted == lbls).sum().item()

        # Store the first 128 images, labels, and predictions for visualization
        if len(images) < 128:
            images.extend(imgs)
            labels.extend(lbls)
            predictions.extend(F.softmax(outputs, dim=1).tolist())

        # Store the incorrectly predicted images, labels, and predictions
        incorrect_indices = (predicted != lbls).nonzero(as_tuple=True)[0].tolist()
        if incorrect_indices:
            incorrect_images.extend(imgs[incorrect_indices])
            incorrect_labels.extend(lbls[incorrect_indices])
            incorrect_predictions.extend(F.softmax(outputs[incorrect_indices], dim=1).tolist())

print(f"Accuracy on 10,000 test images: {100 * correct / total}%")

# Show a 16x8 grid of test images with output layer neuron activations as histograms
show_images_with_histograms(images[:18], labels[:18], predictions[:18], nrows=3, ncols=6, upscale_factor=1)

# Show a 16x8 grid of incorrectly predicted test images with output layer neuron activations as histograms
show_images_with_histograms(incorrect_images[:18], incorrect_labels[:18], incorrect_predictions[:18], nrows=3, ncols=6, upscale_factor=1)
