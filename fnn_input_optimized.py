import torch
import torch.nn as nn
from torch.utils import data
import torchvision
import torchvision.transforms as transforms


# Device configuration
device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')

# Hyper-parameters 
input_size = 10
hidden_size = 50
num_classes = 2
num_epochs = 80
batch_size = 100
learning_rate = 0.001

train_data=[]
test_data=[]

for x in range(0,600):
    from random import randint
    images=[]
    labels=[]
    for i in range(0,100):
        temp=[]
        for j in range(0,10):
            temp.append(float(randint(0,9)))
        images.append(temp)
        labels.append(int(temp[9])%2)
    train_data.append((torch.tensor(images),torch.tensor(labels)))

for x in range(0,600):
    from random import randint
    images=[]
    labels=[]
    for i in range(0,100):
        temp=[]
        for j in range(0,10):
            temp.append(float(randint(0,9)))
        images.append(temp)
        labels.append(int(temp[9])%2)
    test_data.append((torch.tensor(images),torch.tensor(labels)))

# Fully connected neural network with one hidden layer
class NeuralNet(nn.Module):
    def __init__(self, input_size, hidden_size, num_classes):
        super(NeuralNet, self).__init__()
        self.fc1 = nn.Linear(input_size, hidden_size) 
        self.relu = nn.ReLU()
        self.fc2 = nn.Linear(hidden_size, num_classes)  
    
    def forward(self, x):
        out = self.fc1(x)
        out = self.relu(out)
        out = self.fc2(out)
        return out

model = NeuralNet(input_size, hidden_size, num_classes).to(device)

# Loss and optimizer
criterion = nn.CrossEntropyLoss()
optimizer = torch.optim.Adam(model.parameters(), lr=learning_rate)  

# Train the model
total_step = len(train_data)
for epoch in range(num_epochs):
    for i, (images, labels) in enumerate(train_data):  
        # Move tensors to the configured device
        images = images.reshape(-1,input_size).to(device)
        labels = labels.to(device)
        
        # Forward pass
        outputs = model(images)
        loss = criterion(outputs, labels)
        
        # Backward and optimize
        optimizer.zero_grad()
        loss.backward()
        optimizer.step()
        
        if (i+1) % 100 == 0:
            print ('Epoch [{}/{}], Step [{}/{}], Loss: {:.4f}' 
                   .format(epoch+1, num_epochs, i+1, total_step, loss.item()))

# Test the model
# In test phase, we don't need to compute gradients (for memory efficiency)
for x in model.parameters():
    print(x)
with torch.no_grad():
    correct = 0
    total = 0
    for images, labels in test_data:
        images = images.reshape(-1,input_size).to(device)
        labels = labels.to(device)
        outputs = model(images)
        _, predicted = torch.max(outputs.data, 1)
        total += labels.size(0)
        correct += (predicted == labels).sum().item()

    print('Accuracy of the network on the 6000 test images: {} %'.format(100 * correct / total))

