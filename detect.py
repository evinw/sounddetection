import torch
import torch.nn as nn
import torch.optim as optim
import torchaudio

# Load the dataset of audio recordings
dataset = torchaudio.datasets.YESNO('path/to/dataset', download=True)

# Preprocess the data by converting audio files to MFCCs
transform = torchaudio.transforms.MFCC(sample_rate=8000, n_mfcc=40)

# Split the dataset into training and test sets
train_dataset, test_dataset = torch.utils.data.random_split(dataset, [80, 20])

# Define a neural network for classification
class Net(nn.Module):
    def __init__(self):
        super(Net, self).__init__()
        self.conv1 = nn.Conv2d(1, 6, (3,3))
        self.conv2 = nn.Conv2d(6, 16, (3,3))
        self.fc1 = nn.Linear(16 * 6 * 6, 120)
        self.fc2 = nn.Linear(120, 84)
        self.fc3 = nn.Linear(84, 2)

    def forward(self, x):
        x = F.max_pool2d(F.relu(self.conv1(x)), (2, 2))
        x = F.max_pool2d(F.relu(self.conv2(x)), (2, 2))
        x = x.view(-1, self.num_flat_features(x))
        x = F.relu(self.fc1(x))
        x = F.relu(self.fc2(x))
        x = self.fc3(x)
        return x

    def num_flat_features(self, x):
        size = x.size()[1:]
        num_features = 1
        for s in size:
            num_features *= s
        return num_features

# Initialize the model, loss function, and optimizer
model = Net()
criterion = nn.CrossEntropyLoss()
optimizer = optim.SGD(model.parameters(), lr=0.001, momentum=0.9)

# Train the model
for epoch in range(10):
    running_loss = 0.0
    for i, data in enumerate(train_dataset, 0):
        inputs, labels = data
        inputs = transform(inputs)
        inputs, labels = inputs.unsqueeze(1), labels.long()

        optimizer.zero_grad()

        outputs = model(inputs)
        loss = criterion(outputs, labels)
        loss.backward()
        optimizer.step()

        running_loss += loss.item()
    print('Epoch %d loss: %.3f' % (epoch + 1, running_loss / (i + 1)))

print('Finished Training')

# Test the model on the held-out test set
correct = 0
total = 0
with torch.no_grad():
    for data in test_dataset:
        inputs, labels = data
        inputs = transform(inputs)
        inputs, labels = inputs.unsqueeze(1), labels.long()
        outputs = model(inputs)
        _, predicted = torch.max(outputs.data, 1)
        total += labels.size(0)
        correct += (predicted == labels).sum().item()

print('Accuracy of the network on the test set: %d %%' % (100 * correct / total))
