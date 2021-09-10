import torch
import torch.nn as nn
from torch.utils import data
import torchvision
import torchvision.transforms as transforms
from random import randint

# Device configuration
device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')

# Hyper-parameters 
maxSeqLen = 256
input_size= 1
hidden_size = 16
num_classes = 2
num_epochs = 6
num_layers=1
batch_size = 100
learning_rate = 0.001
numberOfBatches=1000

train_data=[]
test_data=[]


for x in range(0,numberOfBatches):
    images=[]
    labels=[]
    seqLength=randint(1,maxSeqLen)
    for i in range(0,batch_size):
        temp=[]
        for j in range(0,seqLength):
            temp.append([float(randint(0,1))])
        images.append(temp)
        labels.append(int(temp[0][0]))
    train_data.append((torch.tensor(images),torch.tensor(labels)))

for x in range(0,int(numberOfBatches/10)):
    images=[]
    labels=[]
    seqLength=randint(1,maxSeqLen)
    for i in range(0,batch_size):
        temp=[]
        for j in range(0,seqLength):
            temp.append([float(randint(0,1))])
        images.append(temp)
        labels.append(int(temp[0][0]))
    test_data.append((torch.tensor(images),torch.tensor(labels)))

# Recurrent neural network (many-to-one)
class RNN(nn.Module):
    def __init__(self, input_size, hidden_size, num_layers, num_classes):
        super(RNN, self).__init__()
        self.hidden_size = hidden_size
        self.num_layers = num_layers
        self.rnn = nn.RNN(input_size, hidden_size, num_layers, batch_first=True,nonlinearity='relu')
        self.fc = nn.Linear(hidden_size , num_classes)

    def forward(self, x):
        # Set initial hidden and cell states
        h0 = torch.zeros(self.num_layers, x.size(0), self.hidden_size).to(device)

        # Forward propagate LSTM
        out, _ = self.rnn(x, h0)

        # Decode the hidden state of the last time step
        out = self.fc(out[:,-1,:])
        return out

model = RNN(input_size=input_size, hidden_size=hidden_size,num_layers=num_layers ,num_classes=num_classes).to(device)

# Loss and optimizer
criterion = nn.CrossEntropyLoss()
optimizer = torch.optim.Adam(model.parameters(), lr=learning_rate)  

# Train the model
total_step = len(train_data)
for epoch in range(num_epochs):
    for i, (images, labels) in enumerate(train_data):  
        # Move tensors to the configured device
        images = images.to(device)
        labels = labels.to(device)
        
        # Forward pass
        outputs = model(images)
        # print(outputs.size())
        # print(labels.size())
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
counter=0
with torch.no_grad():
    correct = 0
    total = 0
    for images, labels in test_data:
        images = images.to(device)
        labels = labels.to(device)
        outputs = model(images)
        _, predicted = torch.max(outputs.data, 1)
        total += labels.size(0)
        # print(predicted)
        # print(outputs)
        correct += (predicted == labels).sum().item()
        counter+=1
        # if counter==10:
        #     break

    print('Accuracy of the network on the 6000 test images: {} %'.format(100 * correct / total))

