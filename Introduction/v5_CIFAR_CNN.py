import torch
import torchvision
import torchvision.transforms as transforms
import torch.nn as nn
import torch.optim as optim
import torch.nn.functional as F
import matplotlib.pyplot as plt
import numpy as np
import os

plt.style.use('dark_background')

def main():
    def display_training_samples(trainloader, classes, nrows=3, ncols=10):
        images_by_class = {cls: [] for cls in classes}
        images_to_display = []

        for images, labels in trainloader:
            for img, label in zip(images, labels):
                class_name = classes[label.item()]
                if len(images_by_class[class_name]) < 3:
                    images_by_class[class_name].append((img, label.item()))
                    if sum(len(imgs) for imgs in images_by_class.values()) == 3 * len(classes):
                        break
            if sum(len(imgs) for imgs in images_by_class.values()) == 3 * len(classes):
                break

        for class_name, imgs in images_by_class.items():
            images_to_display.extend(imgs)

        fig, axes = plt.subplots(nrows=nrows, ncols=ncols, figsize=(ncols * 2, nrows * 2))

        for i, ax in enumerate(axes.flat):
            image, label = images_to_display[i]
            image = image.numpy()
            image = np.transpose(image, (1, 2, 0))
            image = (image / 2 + 0.5) * 255  # Un-normalize the image
            ax.imshow(image.astype(np.uint8))
            ax.set_title(f"Label: {classes[label]}")
            ax.axis('off')

        plt.subplots_adjust(hspace=0.5)
        plt.show()

    # Load and preprocess CIFAR-10 dataset
    transform = transforms.Compose(
    [transforms.Resize((32, 32)),
     transforms.ToTensor(),
     transforms.Normalize((0.5, 0.5, 0.5), (0.5, 0.5, 0.5))])

    trainset = torchvision.datasets.CIFAR10(root='./data', train=True,
                                        download=True, transform=transform)
    trainloader = torch.utils.data.DataLoader(trainset, batch_size=100,
                                          shuffle=True, num_workers=2)

    testset = torchvision.datasets.CIFAR10(root='./data', train=False,
                                       download=True, transform=transform)
    testloader = torch.utils.data.DataLoader(testset, batch_size=100,
                                         shuffle=False, num_workers=2)

    classes = ('plane', 'car', 'bird', 'cat', 'deer', 'dog', 'frog', 'horse', 'ship', 'truck')
    display_training_samples(trainloader, classes)


    # Define the CNN
    class Net(nn.Module):
        def __init__(self):
            super(Net, self).__init__()
            self.conv1 = nn.Conv2d(3, 6, 5)
            self.pool = nn.MaxPool2d(2, 2)
            self.conv2 = nn.Conv2d(6, 16, 5)
            self.fc1 = nn.Linear(16 * 5 * 5, 120)
            self.fc2 = nn.Linear(120, 84)
            self.fc3 = nn.Linear(84, 10)

        def forward(self, x):
            x = self.pool(F.relu(self.conv1(x)))
            x = self.pool(F.relu(self.conv2(x)))
            x = x.view(-1, 16 * 5 * 5)
            x = F.relu(self.fc1(x))
            x = F.relu(self.fc2(x))
            x = self.fc3(x)
            return x

    net = Net()

    # Set the loss function and optimizer
    criterion = nn.CrossEntropyLoss()
    optimizer = optim.SGD(net.parameters(), lr=0.001, momentum=0.9)

    # Train the network
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

    print("Finished training")

    # Functions for visualization
    def show_images_with_histograms_orig(images, labels, predictions, nrows=3, ncols=6, upscale_factor=1, hspace=0.5):
        fig, axes = plt.subplots(nrows=nrows * 2, ncols=ncols, figsize=(ncols * 3, nrows * 3))

        for i, ax in enumerate(axes.flat):
            row = i // ncols
            col = i % ncols
            if row % 2 == 0:
                img_idx = row // 2 * ncols + col
                image = images[img_idx].squeeze().numpy()
                image = np.transpose(image, (1, 2, 0))
                image = (image / 2 + 0.5) * 255  # Un-normalize the image
                ax.imshow(image.astype(np.uint8))
                ax.set_title(f"Label: {classes[labels[img_idx].item()]}")
            else:
                img_idx = (row - 1) // 2 * ncols + col
                pred = predictions[img_idx]
                ax.bar(range(10), pred, tick_label=classes)
                ax.set_ylim(0, 1)
                ax.set_title(f"Prediction: {classes[np.argmax(pred)]}")
            ax.axis('off')

        plt.subplots_adjust(hspace=hspace)
        plt.show()

    def show_images_with_histograms(images, labels, predictions, nrows=3, ncols=6, upscale_factor=1, hspace=0.5):
        fig, axes = plt.subplots(nrows=nrows * 2, ncols=ncols, figsize=(ncols * 3, nrows * 3))

        for i, ax in enumerate(axes.flat):
            row = i // ncols
            col = i % ncols
            if row % 2 == 0:
                img_idx = row // 2 * ncols + col
                image = images[img_idx].squeeze().numpy()
                image = np.transpose(image, (1, 2, 0))
                image = (image / 2 + 0.5) * 255  # Un-normalize the image
                ax.imshow(image.astype(np.uint8))
                ax.set_title(f"Label: {classes[labels[img_idx].item()]}")
            else:
                img_idx = (row - 1) // 2 * ncols + col
                pred = predictions[img_idx]
                ax.bar(range(10), pred, tick_label=classes)
                ax.set_ylim(0, 1)
                ax.set_yticks([0.0, 0.5, 1.0])
                ax.set_title(f"Prediction: {classes[np.argmax(pred)]}")

                # Set the x-axis tick labels
                ax.set_xticklabels(classes, rotation='vertical', fontsize=8)

            # ax.axis('off')

        plt.subplots_adjust(hspace=hspace)
        plt.show()

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

            if len(images) < 128:
                images.extend(imgs)
                labels.extend(lbls)
                predictions.extend(F.softmax(outputs, dim=1).tolist())

            incorrect_indices = (predicted != lbls).nonzero(as_tuple=True)[0].tolist()
            if incorrect_indices:
                incorrect_images.extend(imgs[incorrect_indices])
                incorrect_labels.extend(lbls[incorrect_indices])
                incorrect_predictions.extend(F.softmax(outputs[incorrect_indices], dim=1).tolist())

    print(f"Accuracy on 10,000 test images: {100 * correct / total}%")

    # # Show a 16x8 grid of test images with output layer neuron activations as histograms
    # show_images_with_histograms(images[:128], labels[:128], predictions[:128], nrows=16, ncols=8, hspace=1.5)
    #
    # # Show a 16x8 grid of incorrectly predicted test images with output layer neuron activations as histograms
    # show_images_with_histograms(incorrect_images[:128], incorrect_labels[:128], incorrect_predictions[:128], nrows=16,
    #                             ncols=8, hspace=1.5)

    # Show a 16x8 grid of test images with output layer neuron activations as histograms
    show_images_with_histograms(images[:12], labels[:12], predictions[:12], nrows=2, ncols=6, upscale_factor=1)

    # Show a 16x8 grid of incorrectly predicted test images with output layer neuron activations as histograms
    show_images_with_histograms(incorrect_images[:12], incorrect_labels[:12], incorrect_predictions[:12], nrows=2,
                                ncols=6, upscale_factor=1)


if __name__ == '__main__':
    main()
