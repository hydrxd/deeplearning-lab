# EX1
# Approach 1

import torch
import torch.nn as nn
import matplotlib.pyplot as plt
import math

x = torch.linspace(-torch.pi,torch.pi,2000, dtype=torch.float32, device='cpu')
y = torch.sin(x)

x_train = x.reshape(-1,1)
y_train = y.reshape(-1,1)

hidden = 4
hidden_nodes = 5

class SimpleANN(nn.Module):
  def __init__(self):
    super().__init__()
    self.l1 = nn.Linear(1,hidden)
    self.l2 = nn.Linear(hidden,hidden_nodes)
    self.l3 = nn.Linear(hidden_nodes,hidden)
    self.l4 = nn.Linear(hidden,1)

  def forward(self, x):
    l1 = self.l1(x)
    h1 = torch.relu(l1)
    l2 = self.l2(h1)
    h2 = torch.tanh(l2)
    l3 = self.l3(h2)
    h3 = torch.sigmoid(l3)
    l4 = self.l4(h3)
    return l4

model = SimpleANN()
model

criterion = nn.MSELoss()
optimizer = torch.optim.SGD(model.parameters(), lr=0.3)

epochs = 10

for epoch in range(epochs):
  optimizer.zero_grad()
  op = model(x_train)
  loss = criterion(y_train,op)
  loss.backward()
  optimizer.step()
  print(f'Epoch: {epoch} Loss: {loss.item()}')

predicted = op.data.numpy()

plt.plot(x_train,predicted)
plt.plot(x_train,y_train)

# Approach 2

import torch
import torch.nn as nn
import matplotlib.pyplot as plt

torch.manual_seed(42)
x = torch.linspace(-torch.pi, torch.pi, 100).reshape(-1,1)
y = 3 * x**3 - 2 * x**2 + x + torch.randn(x.size()) * 0.2

class PolyReg(nn.Module):
  def __init__(self):
    super().__init__()
    self.poly = nn.Linear(degree,1,bias=False)

  def forward(self,x):
    x_poly = torch.cat([x**i for i in range(1, degree+1)], dim=1)
    return self.poly(x_poly)

degree = 3
model = PolyReg()
model

criterion = nn.MSELoss()
optimizer = torch.optim.SGD(model.parameters(), lr=0.001)

epochs = 2000
losses = []

for epoch in range(epochs):
  model.train()
  optimizer.zero_grad()
  op = model(x)
  loss = criterion(y,op)
  losses.append(loss.item())
  loss.backward()
  optimizer.step()
  if (epoch+1) % 200 == 0:
        print(f'Epoch {epoch+1}, Loss: {loss.item():.4f}')

plt.scatter(x,y, color='red', alpha=0.5)
plt.plot(x,op.data.numpy())

# EX2
# Approach 1

import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import DataLoader, TensorDataset
from sklearn.preprocessing import StandardScaler
from sklearn.model_selection import train_test_split
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import sklearn.metrics as metrics

data = pd.read_csv('/content/sample_data/EX2.csv')

data['Date'] = pd.to_datetime(data['Date'])
data['Date'] = (data['Date'] - data['Date'].min()).dt.days

scaler = StandardScaler()
data[['Open', 'High', 'Low', 'Close', 'Adj Close', 'Volume']] = scaler.fit_transform(data[['Open', 'High', 'Low', 'Close', 'Adj Close', 'Volume']])

data.head()

x = data.drop('Close',axis=1)
y = data['Close']

x_train, x_test, y_train, y_test = train_test_split(x,y,test_size=0.2, random_state=42)

x_train_tensor = torch.tensor(x_train.values, dtype=torch.float32)
x_test_tensor = torch.tensor(x_test.values, dtype=torch.float32)
y_train_tensor = torch.tensor(y_train.values, dtype=torch.float32)
y_test_tensor = torch.tensor(y_test.values, dtype=torch.float32)

train_dataset = TensorDataset(x_train_tensor,y_train_tensor)
test_dataset = TensorDataset(x_test_tensor,y_test_tensor)

train_loader = DataLoader(train_dataset,batch_size=64,shuffle=True)
test_loader = DataLoader(test_dataset,batch_size=64,shuffle=False)

class PolyNN(nn.Module):
    def __init__(self):
        super(PolyNN , self).__init__()
        self.fc1 = nn.Linear(x_train.shape[1], 200)
        self.bn1 = nn.BatchNorm1d(200)
        self.dropout1 = nn.Dropout(0.5)
        self.fc2 = nn.Linear(200, 128)
        self.bn2 = nn.BatchNorm1d(128)
        self.dropout2 = nn.Dropout(0.5)
        self.fc3 = nn.Linear(128, 64)
        self.bn3 = nn.BatchNorm1d(64)
        self.dropout3 = nn.Dropout(0.5)
        self.fc4 = nn.Linear(64, 1)

    def forward(self, x):
        x = torch.relu(self.bn1(self.fc1(x)))
        x = self.dropout1(x)
        x = torch.relu(self.bn2(self.fc2(x)))
        x = self.dropout2(x)
        x = torch.relu(self.bn3(self.fc3(x)))
        x = self.dropout3(x)
        x = self.fc4(x)
        return x

# Instantiate the model, define the loss function and optimizer
model = PolyNN()
criterion = nn.MSELoss()
optimizer = optim.Adam(model.parameters() , lr=0.001 , weight_decay=1e-5)

epochs = 100
train_losses = []
test_losses = []

for epoch in range(epochs):
  model.train()
  running_loss = 0.0
  for inputs,labels in train_loader:
    optimizer.zero_grad()
    ops = model(inputs).squeeze()
    loss = criterion(labels,ops)
    loss.backward()
    optimizer.step()
    running_loss += loss.item()

  train_losses.append(running_loss/len(train_loader))

  model.eval()
  test_loss = 0.0

  with torch.no_grad():
    for inputs,labels in test_loader:
      ops = model(inputs).squeeze()
      loss = criterion(labels,ops)
      test_loss += loss.item()

    test_losses.append(test_loss/len(test_loader))
    print(f'Epoch [{epoch+1}/{epochs}], Train Loss: {train_losses[-1]:.4f}, Test Loss: {test_losses[-1]:.4f}')

model.eval()
with torch.no_grad():
    y_test_pred = []
    y_test_true = []

    for batch_x, batch_y  in test_loader:
        outputs = model(batch_x)
        y_test_true.extend(batch_y)
        y_test_pred.extend(outputs)

    y_test_pred = np.array(y_test_pred)
    y_test_true = np.array(y_test_true)

    def discrete_values(values,threshold):
        return (values > threshold).astype(int)

    threshold = np.median(y)

    y_test_pred_dis = discrete_values(y_test_pred,threshold)
    y_test_true_dis = discrete_values(y_test_true,threshold)

    mse = metrics.mean_squared_error(y_test_true, y_test_pred)
    r2 = metrics.r2_score(y_test_true,y_test_pred)

    accuracy = metrics.accuracy_score(y_test_true_dis, y_test_pred_dis)
    precision = metrics.precision_score(y_test_true_dis, y_test_pred_dis)
    recall = metrics.recall_score(y_test_true_dis, y_test_pred_dis)
    f1 = metrics.f1_score(y_test_true_dis, y_test_pred_dis)

    print(mse, r2, accuracy, precision, recall, f1)


# Approach 2

import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim
import matplotlib.pyplot as plt
from torch.utils.data import Dataset, DataLoader, random_split
from sklearn.model_selection import ParameterGrid

x = torch.linspace(-torch.pi,torch.pi,2000)
y = torch.sin(x) + torch.randn(x.size())/10
y = y.reshape(-1,1)

class PolyDS(Dataset):
    def __init__(self,x,y):
        self.x = x
        self.y = y

    def __len__(self):
        return len(self.x)

    def __getitem__(self,idx):
        return self.x[idx], self.y[idx]

train_set, val_set, test_set = random_split(PolyDS(x,y), [1000, 500, 500])

class PolyReg(nn.Module):
    def __init__(self):
        super(PolyReg, self).__init__()
        self.fc1 = nn.Linear(1, 5)
        self.fc2 = nn.Linear(5, 9)
        self.fc3 = nn.Linear(9, 1)

    def forward(self, x):
        x = x.view(-1, 1)
        x = F.relu(self.fc1(x))
        x = F.relu(self.fc2(x))
        x = self.fc3(x)
        return x

param_grid = {
    'alpha': [0.1, 0.01],
    'batch_size': [32, 64],
    'epochs': [20, 50],
    'optim': [optim.Adam, optim.SGD]
}

val_loader = DataLoader(val_set, batch_size = 128, shuffle=True)
test_loader = DataLoader(test_set, batch_size = 128, shuffle=False)

best_val_loss = float('inf')
best_params = None
best_train_losses = []
best_val_losses = []

for params in ParameterGrid(param_grid):
    train_loader = DataLoader(train_set, batch_size=params['batch_size'], shuffle=True)
    model = PolyReg()
    optimizer = params['optim'](model.parameters(), lr=params['alpha'])
    criterion = nn.MSELoss()

    train_losses = []
    val_losses = []

    for epoch in range(params['epochs']):
        model.train()
        epoch_loss = 0.0
        for batch_idx, (inputs,targets) in enumerate(train_loader):
            optimizer.zero_grad()
            outputs = model(inputs)
            loss = criterion(outputs,targets)
            loss.backward()
            optimizer.step()
            epoch_loss += loss.item()

        train_loss = epoch_loss/len(train_loader)
        train_losses.append(train_loss)

        model.eval()
        val_loss = 0.0
        with torch.no_grad():
            for val_idx, (inputs,targets) in enumerate(val_loader):
                outputs = model(inputs)
                loss = criterion(outputs,targets)
                val_loss+=loss.item()

        val_loss/= len(val_loader)
        val_losses.append(val_loss)

        if epoch % 5 == 0 or epoch == params['epochs'] - 1:
            print(f"Epoch {epoch + 1}/{params['epochs']} | Train Loss: {train_loss:.4f} | Validation Loss: {val_loss:.4f}")

        if val_loss < best_val_loss:
            best_val_loss = val_loss
            best_params = params
            best_train_losses = train_losses
            best_val_losses = val_losses
            best_model = model

print("Best Parameters:", best_params)
print("Best Validation Loss:", best_val_loss)

avg_test_loss = 0.0
with torch.no_grad():
    for idx, (inputs, targets) in enumerate(test_loader):
        outputs = best_model(inputs)
        loss = criterion(outputs,targets)
        avg_test_loss += loss.item()

avg_test_loss/=len(test_loader)
print(avg_test_loss)

plt.plot(best_train_losses, label='Training Loss')
plt.plot(best_val_losses, label='Validation Loss')

def predict(model,x):
    model.eval()
    with torch.no_grad():
        return best_model(torch.tensor([[x]]))

inp = 3.0
predicted_y = predict(best_model, inp)
print(inp, predicted_y)

actual_y = torch.sin(torch.tensor(inp))
print(x, actual_y)

error = abs(predicted_y - actual_y)

# EX3
# Approach 1

import torch
import torchvision.models as models
import torchvision.transforms as transforms
from PIL import Image
import matplotlib.pyplot as plt
import requests
import json

model = models.vgg16(pretrained=True)
model.eval()

preprocess = transforms.Compose([
    transforms.Resize(256),
    transforms.CenterCrop(224),
    transforms.ToTensor(),
    transforms.Normalize(mean=[0.485,0.456,0.406], std=[0.229,0.224,0.225])
])

image_path = ("/content/umbrella.jpg")
input_image = Image.open(image_path).convert('RGB')
input_tensor = preprocess(input_image)
input_batch = input_tensor.unsqueeze(0)

activations = {}

def get_activations(name):
    def hook(model, input, output):
        activations[name] = output.detach()
    return hook

for idx, layer in enumerate(model.features):
    layer.register_forward_hook(get_activations(f"layer_{idx}"))

with torch.no_grad():
    output = model(input_batch)

def plot_activations(layer_activations,layer_num):
    num_feature_maps = min(4,layer_activations.shape[0])
    cols = 4
    rows = (num_feature_maps + cols -1) // cols

    fig,axes = plt.subplots(rows,cols, figsize=(15,rows*3))
    axes = axes.flatten()

    for i in range(num_feature_maps):
        axes[i].imshow(layer_activations[i].cpu(), cmap='ocean')
        axes[i].axis('off')

    plt.suptitle(f"Layer {layer_num} Feature Maps", fontsize=16)
    plt.tight_layout(rect=[0,0.03,1,0.95])
    plt.show()

max_layers = 12
layers_plotted = 0

for layer_name, act in activations.items():
    layer_num = int(layer_name.split('_')[1])
    act_squeezed = act.squeeze()

    if layers_plotted < max_layers:
        plot_activations(act_squeezed, layer_num)
        layers_plotted+=1
    else:
        break

LABELS_URL = "https://raw.githubusercontent.com/anishathalye/imagenet-simple-labels/master/imagenet-simple-labels.json"
response = requests.get(LABELS_URL)
labels = json.loads(response.text)
print(f"Available labels:\n {labels}")

probabilities = torch.nn.functional.softmax(output[0], dim=0)

top5_prob, top5_catid = torch.topk(probabilities, 5)
print("\nTop 5 predictions:")
for i in range(5):
    print(labels[top5_catid[i]], top5_prob[i].item())


# Approach 2

import torch
import torchvision.models as models
import torchvision.transforms as transforms
import matplotlib.pyplot
import requests
import json
from PIL import Image

LABELS_URL = "https://raw.githubusercontent.com/anishathalye/imagenet-simple-labels/master/imagenet-simple-labels.json"
response = requests.get(LABELS_URL)
labels = json.loads(response.text)
print(labels)

model = models.vgg16(pretrained = True)
model.eval()

preprocess = transforms.Compose([
    transforms.Resize(256),
    transforms.CenterCrop(224),
    transforms.ToTensor(),
    transforms.Normalize(mean=[0.485,0.456,0.406], std = [0.229, 0.224, 0.225])
])

image_path = "/content/umbrella.jpg"
inp_image = Image.open(image_path).convert('RGB')
input_tensor = preprocess(inp_image)
inp_batch = input_tensor.unsqueeze(0)

device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
model.to(device)
inp_batch = inp_batch.to(device)

with torch.no_grad():
    output = model(inp_batch)

probs = torch.softmax(output[0], dim=0)

top5_prob, top5_cat = torch.topk(probs,5)
print("top5:")
for i in range(5):
    print(labels[top5_cat[i]], top5_prob[i].item())

def get_activations(name):
    def hook(model,inp,out):
        activations[name] = out.detach()
    return hook

for idx, layer in enumerate(model.features):
    layer.register_forward_hook(get_activations(f"layer_{idx}"))

def plot_activations(layer_activations, layer_num):
    num_maps = 4
    fig,axes = plt.subplots(1,num_maps, figsize=(15,15))
    for i in range(num_maps):
        axes[i].imshow(layer_activations[i].cpu(), cmap='ocean')
        axes[i].axis('off')
        axes[i].set_title(f"Feature Map {i}")
    plt.suptitle(f"Layer {layer_num} Feature maps")
    plt.tight_layout(rect=[0,0.03,1,0.95])
    plt.show

for layer_name, act in activations.items():
    layer_num = int(layer_name.split('_')[1])
    act_unsq = act.squeeze()
    plot_activations(act_unsq, layer_num)

preprocess = transforms.Compose([
    transforms.Resize(256),
    transforms.CenterCrop(224),
    transforms.ToTensor(),
    transforms.Normalize(mean=[0.485,0.456,0.406], std=[0.229,0.224,0.225])
])

image_path = ("/content/umbrella.jpg")
input_image = Image.open(image_path).convert('RGB')
input_tensor = preprocess(input_image)
input_batch = input_tensor.unsqueeze(0)

activations = {}

def get_activations(name):
    def hook(model, input, output):
        activations[name] = output.detach()
    return hook

for idx, layer in enumerate(model.features):
    layer.register_forward_hook(get_activations(f"layer_{idx}"))

with torch.no_grad():
    output = model(input_batch)

def plot_activations(layer_activations,layer_num):
    num_feature_maps = min(4,layer_activations.shape[0])
    cols = 4
    rows = (num_feature_maps + cols -1) // cols

    fig,axes = plt.subplots(rows,cols, figsize=(15,rows*3))
    axes = axes.flatten()

    for i in range(num_feature_maps):
        axes[i].imshow(layer_activations[i].cpu(), cmap='ocean')
        axes[i].axis('off')

    plt.suptitle(f"Layer {layer_num} Feature Maps", fontsize=16)
    plt.tight_layout(rect=[0,0.03,1,0.95])
    plt.show()

max_layers = 12
layers_plotted = 0

for layer_name, act in activations.items():
    layer_num = int(layer_name.split('_')[1])
    act_squeezed = act.squeeze()

    if layers_plotted < max_layers:
        plot_activations(act_squeezed, layer_num)
        layers_plotted+=1
    else:
        break

LABELS_URL = "https://raw.githubusercontent.com/anishathalye/imagenet-simple-labels/master/imagenet-simple-labels.json"
response = requests.get(LABELS_URL)
labels = json.loads(response.text)
print(f"Available labels:\n {labels}")

probabilities = torch.nn.functional.softmax(output[0], dim=0)

top5_prob, top5_catid = torch.topk(probabilities, 5)
print("\nTop 5 predictions:")
for i in range(5):
    print(labels[top5_catid[i]], top5_prob[i].item())

# EX4
# Approach 1

import torch
import torch.nn as nn
import torch.optim as optim
import torch.nn.functional as F
import torchvision
import torchvision.transforms as transforms
from torch.utils.data import DataLoader
import matplotlib.pyplot as plt

transform = transforms.Compose([
    transforms.ToTensor(),
    transforms.Normalize((0.5,),(0.5,))
])

train_dataset = torchvision.datasets.MNIST(root= '/data', train=True, download = True, transform = transform)
test_dataset = torchvision.datasets.MNIST(root= '/data', train=False, download = True, transform = transform)

train_loader = DataLoader(train_dataset, batch_size=64, shuffle=True)
test_loader = DataLoader(test_dataset, batch_size=128, shuffle=False)

class CNN(nn.Module):
    def __init__(self):
        super(CNN,self).__init__()
        self.conv1 = nn.Conv2d(in_channels=1,out_channels=32,kernel_size=3, stride=1, padding=1)
        self.conv2 = nn.Conv2d(in_channels=32,out_channels=64,kernel_size=3, stride=1, padding=1)
        self.pool1 = nn.MaxPool2d(kernel_size=2,stride=2,padding=0)
        self.pool2 = nn.MaxPool2d(kernel_size=2,stride=2,padding=0)
        self.fc1 = nn.Linear(64*7*7,128)
        self.fc2 = nn.Linear(128,10)

    def forward(self,x):
        x = self.pool1(F.relu(self.conv1(x)))
        x = self.pool2(F.relu(self.conv2(x)))
        x = x.view(-1, 64*7*7)
        x = F.relu(self.fc1(x))
        x = self.fc2(x)
        return x

model = CNN()
criterion = nn.CrossEntropyLoss()
optimizer = optim.Adam(model.parameters(), lr=0.001)

device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
model.to(device)

epochs = 10
train_loss = []
train_acc = []

for epoch in range(epochs):
    model.train()
    running_loss = 0.0
    correct = 0
    total = 0

    for images, labels in train_loader:
        images, labels = images.to(device),labels.to(device)

        optimizer.zero_grad()

        outputs = model(images)
        loss = criterion(outputs,labels)
        loss.backward()
        optimizer.step()

        running_loss+=loss.item()
        _, predicted = torch.max(outputs.data,1)
        total+= labels.size(0)
        correct+=(predicted==labels).sum().item()

    epoch_loss = running_loss/len(train_loader)
    epoch_accuracy = 100*correct/total

    train_loss.append(epoch_loss)
    train_acc.append(epoch_accuracy)

    print(f"Epoch {epoch+1}/{epochs}, Loss: {epoch_loss:.4f}, Accuracy: {epoch_accuracy:.2f}%")

model.eval()
correct = 0
total = 0

with torch.no_grad():
    for images,labels in test_loader:
        images,labels = images.to(device), labels.to(device)
        outputs = model(images)
        _, predicted = torch.max(outputs.data, 1)
        total += labels.size(0)
        correct += (predicted == labels).sum().item()

test_acc = 100*correct / total
print(test_acc)

plt.plot(train_loss, label='Training Loss')
plt.plot(train_acc, label='Training Accuracy')

# Approach 2

import torch
import torch.nn as nn
import torch.optim as optim
from torchvision import datasets, transforms
from torch.utils.data import DataLoader
import matplotlib.pyplot as plt

class CNN(nn.Module):
    def __init__(self):
        super(CNN, self).__init__()
        self.conv1 = nn.Conv2d(1,32,kernel_size=3,padding=1)
        self.conv2 = nn.Conv2d(32,64,kernel_size=3,padding=1)
        self.pool = nn.MaxPool2d(2,2)
        self.fc1 = nn.Linear(64*14*14, 128)
        self.fc2 = nn.Linear(128,10)
    def forward(self,x):
        x = torch.relu(self.conv1(x))
        x = torch.relu(self.conv2(x))
        x = self.pool(x)
        x = x.view(-1,64*14*14)
        x = torch.relu(self.fc1(x))
        x = self.fc2(x)
        return x

transform = transforms.Compose([
    transforms.ToTensor(),
    transforms.Normalize((0.5,),(0.5,))
])

train_dataset = datasets.MNIST(root='/data', train=True, transform=transform, download=True)
test_dataset = datasets.MNIST(root='/data', train=False, transform=transform, download=True)
train_loader = DataLoader(train_dataset, batch_size=64, shuffle=True)
test_loader = DataLoader(test_dataset, batch_size=128, shuffle=False)

model = CNN()
criterion = nn.CrossEntropyLoss()
optimizer = optim.Adam(model.parameters(), lr=0.01)

epochs = 1
train_losses = []
test_losses = []

for epoch in range(epochs):
    model.train()
    running_loss = 0.0
    for images,labels in train_loader:
        optimizer.zero_grad()
        outputs = model(images)
        loss = criterion(outputs,labels)
        loss.backward()
        optimizer.step()
        running_loss+=loss.item()

    train_loss = running_loss/len(train_loader)
    train_losses.append(train_loss)
    print(f'Epoch [{epoch+1}/{epochs}], Training Loss: {train_loss:.4f}')

    model.eval()
    test_loss = 0.0
    correct = 0
    total = 0
    with torch.no_grad():
        for images,labels in test_loader:
            outputs = model(images)
            loss = criterion(outputs,labels)
            test_loss+=loss.item()
            _, predicted = torch.max(outputs.data,1)
            total+= labels.size(0)
            correct += (predicted == labels).sum().item()

    test_loss /= len(test_loader)
    test_losses.append(test_loss)
    accuracy = 100 * correct / total

    print(f'Epoch [{epoch+1}/{epochs}], Test Loss: {test_loss:.4f}, Accuracy: {accuracy:.2f}%')

plt.plot(range(1, epochs + 1), train_losses, label='Training Loss', marker='o')
plt.plot(range(1, epochs + 1), test_losses, label='Testing Loss', marker='o')

import numpy as np

def show_samples(images,labels, predictions, num_samples=5):
    images = images[:num_samples]
    labels = labels[:num_samples]
    predictions = predictions[:num_samples]

    images = images.numpy()

    fig, axes = plt.subplots(1,num_samples, figsize=(10,3))
    for i in range(num_samples):
        axes[i].imshow(np.squeeze(images[i]), cmap='gray')
        axes[i].set_title(f'Pred: {predictions[i]}, True: {labels[i]}')
        axes[i].axis('off')
    plt.show()

model.eval()
with torch.no_grad():
    data_iter = iter(test_loader)
    images,labels = next(data_iter)
    outputs = model(images)
    _,predictions = torch.max(outputs.data,1)

    show_samples(images,labels,predictions)

# EX5
# Approach 1

from datasets import load_dataset

dataset = load_dataset('imdb')
train_size = len(dataset['train'])
test_size = len(dataset['test'])
num_classes = len(set(dataset['train']['label']))

from transformers import AutoTokenizer
import torch
from torch.utils.data import DataLoader

tokenizer = AutoTokenizer.from_pretrained('distilbert-base-uncased')

def tokenize_and_pad(batch):
    return tokenizer(batch['text'], padding="max_length", truncation= True, max_length=500, return_tensors='pt')

tokenized_train = dataset['train'].map(tokenize_and_pad,batched=True)
tokenized_test = dataset['test'].map(tokenize_and_pad,batched=True)

from torch.utils.data import DataLoader
import torch

def collate_fn(batch):
    inputs = [item['input_ids'] for item in batch]
    labels = [item['label'] for item in batch]

    inputs = torch.tensor(inputs)
    labels = torch.tensor(labels, dtype = torch.float32)

    return inputs, labels

train_loader = DataLoader(tokenized_train, batch_size=64, shuffle=True, collate_fn = collate_fn)
test_loader = DataLoader(tokenized_test, batch_size= 128, shuffle=False, collate_fn = collate_fn)

import torch.nn as nn

class RNNModel(nn.Module):
    def __init__(self, vocab_size, embedding_dim, hidden_dim):
        super(RNNModel, self).__init__()
        self.embedding = nn.Embedding(vocab_size,embedding_dim)
        self.rnn = nn.LSTM(embedding_dim, hidden_dim, batch_first=True)
        self.fc = nn.Linear(hidden_dim,1)
    def forward(self,x):
        x = self.embedding(x)
        x, _ = self.rnn(x)
        x = self.fc(x[:,-1,:])
        return torch.sigmoid(x)

vocab_size = len(tokenizer)
embedding_dim = 32
hidden_dim = 32

model = RNNModel(vocab_size,embedding_dim, hidden_dim)

import torch.optim as optim

criterion = nn.BCELoss()
optimizer = optim.Adam(model.parameters())

device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
model.to(device)

import torch

def train(model,train_loader,criterion, optimizer, num_epochs=5):
    for epoch in range(num_epochs):
        running_loss = 0.0
        for i, (inputs, labels) in enumerate(train_loader):
            inputs,labels = inputs.to(device), labels.to(device)

            optimizer.zero_grad()
            outputs = model(inputs)
            loss = criterion(outputs,labels.unsqueeze(1))
            loss.backward()
            optimizer.step()

            running_loss += loss.item()

        print(f'Epoch [{epoch + 1}/{num_epochs}], Loss: {running_loss / len(train_loader)}')

train(model, train_loader, criterion, optimizer)

def predict_sentiment(model,reviews):
    model.eval()
    inputs = tokenizer(reviews, padding='max_length', truncation=True, max_length=500, return_tensors="pt")['input_ids']
    inputs = inputs.to(device)

    with torch.no_grad():
        outputs = model(inputs)
        predicted_sentiment = ["Positive" if pred > 0.5 else "Negative" for pred in outputs]

    return predicted_sentiment

custom_reviews = ["The movie was amazing!","The movie was terrible!"]
predicted_sentiments = predict_sentiment(model, custom_reviews)
print(predicted_sentiments)

# Approach 2

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import re
import torch
import torch.nn as nn
from torch.utils.data import Dataset, DataLoader
from torch.nn.utils.rnn import pad_sequence
from sklearn.model_selection import train_test_split
from collections import Counter
import sklearn.metrics as metrics

df = pd.read_csv('/content/imdb_ds.csv')

def clean_text(text):
    text = re.sub(r'[^a-zA-Z\s]','', text)
    return text.lower()

def tokenize(text):
    return text.split()

df['cleaned_text'] = df['review'].apply(clean_text)
df['tokens'] = df['cleaned_text'].apply(tokenize)

all_words = [word for tokens in df['tokens'] for word in tokens]
vocab = Counter(all_words)
word_to_idx = {word: idx+1 for idx, (word,_) in enumerate(vocab.most_common())}
df['indexed_tokens'] = df['tokens'].apply(lambda tokens: [word_to_idx[word] for word in tokens if word in word_to_idx])
df['sentiment'] = df['sentiment'].map({'negative':0, 'positive':1})
X = df['indexed_tokens']
y = df['sentiment']

X_train_val, X_test, y_train_val, y_test = train_test_split(X,y, test_size=0.2, random_state=42)
X_train, X_val, y_train, y_test = train_test_split(X_train_val, y_train_val, test_size=0.25, random_state=42)

class SentimentDataset(Dataset):
    def __init__(self,reviews,labels):
        super(SentimentDataset, self).__init__()
        self.reviews = reviews
        self.labels = labels

    def __len__(self):
        return len(self.reviews)

    def __getitem__(self,idx):
        review = self.reviews.iloc[idx]
        label = self.labels.iloc[idx]
        return torch.tensor(review,dtype=torch.long), torch.tensor(label,dtype=torch.long)

def collate_fn(batch):
    reviews,labels = zip(*batch)
    padded_reviews = pad_sequence([torch.tensor(review) for review in reviews], batch_first = True, padding_value = 0)
    labels = torch.tensor(labels, dtype=torch.long)
    return padded_reviews, labels

batch_size = 64
train_dataset = SentimentDataset(X_train, y_train)
val_dataset = SentimentDataset(X_val, y_val)
test_dataset = SentimentDataset(X_test, y_test)

train_loader = DataLoader(train_dataset, batch_size=64, shuffle=True, collate_fn=collate_fn)
val_loader = DataLoader(val_dataset, batch_size=64, shuffle=False, collate_fn=collate_fn)
test_loader = DataLoader(test_dataset, batch_size=64, shuffle=False, collate_fn=collate_fn)

class RNN(nn.Module):
    def __init__(self, vocab_size, embedding_dim, hidden_dim, output_dim):
        super(RNN,self).__init__()
        self.embedded = nn.Embedding(vocab_size + 1, embedding_dim)
        self.lstm = nn.LSTM(embedding_dim, hidden_dim, batch_first=True)
        self.fc = nn.Linear(hidden_dim, output_dim)

    def forward(self,x):
        x = self.embedding(x)
        x, (hn,cn) = self.lstm(x)
        x = self.fc(hn[-1])
        return x

def train_model_with_curve(model, train_loader, val_loader, criterion, optimizer, num_epochs, device):
    train_losses = []
    val_losses = []

    for epoch in range(num_epochs):
        model.train()
        epoch_train_loss = 0

        for reviews, labels in train_loader:
            reviews, labels = reviews.to(device), labels.to(device)
            optimizer.zero_grad()
            outputs = model(reviews)
            loss = criterion(outputs, labels)
            loss.backward()
            optimizer.step()
            epoch_train_loss += loss.item()

        avg_train_loss = epoch_train_loss / len(train_loader)
        train_losses.append(avg_train_loss)

        model.eval()
        epoch_val_loss = 0
        with torch.no_grad():
            for reviews, labels in val_loader:
                reviews, labels = reviews.to(device), labels.to(device)
                outputs = model(reviews)
                loss = criterion(outputs, labels)
                epoch_val_loss += loss.item()

        avg_val_loss = epoch_val_loss / len(val_loader)
        val_losses.append(avg_val_loss)

    plot_learning_curve(train_losses, val_losses)

def plot_learning_curve(train_losses, val_losses):
    plt.figure(figsize=(10, 5))
    plt.plot(train_losses, label='Training Loss')
    plt.plot(val_losses, label='Validation Loss')
    plt.title('Learning Curve')
    plt.xlabel('Epochs')
    plt.ylabel('Loss')
    plt.legend()
    plt.grid()
    plt.show()

def evaluate_model(model, test_loader, device):
    model.eval()
    correct = 0
    with torch.no_grad():
        for reviews, labels in test_loader:
            reviews, labels = reviews.to(device), labels.to(device)
            outputs = model(reviews)
            _, predicted = torch.max(outputs.data, 1)
            correct += (predicted == labels).sum().item()
    accuracy = correct / len(test_loader.dataset) * 100
    print(f'Test Accuracy: {accuracy:.2f}%')

def classify_unseen_data(model, unseen_data, word_to_idx, device):
    model.eval()
    unseen_data_cleaned = [clean_text(review) for review in unseen_data]
    unseen_data_tokenized = [tokenize(text) for text in unseen_data_cleaned]
    unseen_data_indexed = [[word_to_idx[word] for word in tokens if word in word_to_idx] for tokens in unseen_data_tokenized]

    # Pad the sequences
    unseen_data_tensor = pad_sequence([torch.tensor(review, dtype=torch.long) for review in unseen_data_indexed], batch_first=True, padding_value=0).to(device)

    with torch.no_grad():
        outputs = model(unseen_data_tensor)
        _, predictions = torch.max(outputs.data, 1)

    return predictions.cpu().numpy()

device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
vocab_size = len(word_to_idx)
embedding_dim = 100
hidden_dim = 128
output_dim = 2
num_epochs = 5

model = RNN(vocab_size, embedding_dim, hidden_dim, output_dim).to(device)
criterion = nn.CrossEntropyLoss()
optimizer = torch.optim.Adam(model.parameters())

train_model_with_curve(model, train_loader, val_loader, criterion, optimizer, num_epochs, device)
evaluate_model(model, test_loader, device)

unseen_reviews = [
    "The movie was fantastic! I loved every moment of it.",
    "It was a waste of time and I regret watching it."
]

predictions = classify_unseen_data(model, unseen_reviews, word_to_idx, device)
print("Predictions for unseen data:", ["Positive" if pred == 1 else "Negative" for pred in predictions])

# EX6
# Approach 1

import torch
from torch.utils.data import Dataset, DataLoader
from transformers import BertTokenizer, BertForSequenceClassification, Trainer, TrainingArguments
from sklearn.model_selection import train_test_split
import sklearn.metrics as metrics
import pandas as pd

df = pd.read_csv('/content/sentiment_analysis.csv')

class SentimentDataset(Dataset):
    def __init__(self, texts, labels, tokenizer, max_len):
        self.texts = texts
        self.labels = labels
        self.tokenizer = tokenizer
        self.max_len = max_len

    def __len__(self):
        return len(self.texts)

    def __getitem__(self,idx):
        text = str(self.texts[idx])
        label = self.labels[idx]

        encoding = self.tokenizer.encode_plus(
            text,
            add_special_tokens=True,
            max_length=self.max_len,
            return_token_type_ids=False,
            padding="max_length",
            truncation=True,
            return_attention_mask=True,
            return_tensors="pt"
        )

        return {
            'input_ids': encoding['input_ids'].flatten(),
            'attention_mask': encoding['attention_mask'].flatten(),
            'labels': torch.tensor(label,dtype=torch.long)
        }

tokenizer = BertTokenizer.from_pretrained('bert-base-uncased')

train_texts, val_texts, train_labels, val_labels = train_test_split(df['tweet'], df['label'], test_size=0.2, random_state=42)

train_dataset = SentimentDataset(train_texts.tolist(), train_labels.tolist(), tokenizer, max_len=128)
val_dataset = SentimentDataset(val_texts.tolist(), val_labels.tolist(), tokenizer, max_len=128)

model = BertForSequenceClassification.from_pretrained('bert-base-uncased', num_labels=2)
device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
model.to(device)

def compute_metrics(p):
    pred_labels = torch.argmax(torch.tensor(p.predictions), axis=1)
    true_labels = p.label_ids
    acc = metrics.accuracy_score(true_labels,pred_labels)
    precision,recall,f1,_ = metrics.precision_recall_fscore_support(true_labels,pred_labels, average='binary')
    return {'accuracy': acc, 'precision': precision, 'recall': recall, 'f1': f1}

training_args = TrainingArguments(
    output_dir='./results',
    num_train_epochs=3,
    per_device_train_batch_size=16,
    per_device_eval_batch_size=16,
    warmup_steps=500,
    weight_decay=0.01,
    evaluation_strategy='steps',
    eval_steps=500,
    logging_steps=500,
    load_best_model_at_end=True
)

trainer = Trainer(
    model=model,
    args = training_args,
    train_dataset=train_dataset,
    eval_dataset=val_dataset,
    compute_metrics=compute_metrics
)

trainer.train()

trainer.evaluate()

def predict_sentiment(text,model,tokenizer,max_len=128):
    encoding = tokenizer.encode_plus(
        text,
        add_special_tokens=True,
        max_length=max_len,
        return_token_type_ids=False,
        padding='max_length',
        truncation=True,
        return_attention_mask=True,
        return_tensors='pt'
    )

    input_ids = encoding['input_ids'].to(device)
    attention_mask = encoding['attention_mask'].to(device)

    with torch.no_grad():
        outputs = model(input_ids, attention_mask=attention_mask)
        logits = outputs.logits
        predicted_label = torch.argmax(logits, dim=1).item()

    return predicted_label

example_text = "Today was Great! #cool #fun"
predicted_label = predict_sentiment(example_text, model, tokenizer)

if predicted_label == 0:
    print("Positive")
else:
    print("Negative")

# Approach 2

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import re
import torch
import torch.nn as nn
from torch.utils.data import Dataset, DataLoader
from torch.nn.utils.rnn import pad_sequence
from sklearn.model_selection import train_test_split
from collections import Counter
from sklearn.metrics import classification_report, confusion_matrix

df = pd.read_csv('IMDB Dataset.csv')

def clean_text(text):
    text = re.sub(r'[^a-zA-Z\s]', '', text)
    return text.lower()

def tokenize(text):
    return text.split()

df['cleaned_text'] = df['review'].apply(clean_text)
df['tokens'] = df['cleaned_text'].apply(tokenize)

all_words = [word for tokens in df['tokens'] for word in tokens]
vocab = Counter(all_words)
word_to_idx = {word: idx + 1 for idx, (word, _) in enumerate(vocab.most_common())}
df['indexed_tokens'] = df['tokens'].apply(lambda tokens: [word_to_idx[word] for word in tokens if word in word_to_idx])
df['sentiment'] = df['sentiment'].map({'negative': 0, 'positive': 1})
X = df['indexed_tokens']
y = df['sentiment']

X_train_val, X_test, y_train_val, y_test = train_test_split(X, y, test_size=0.2, random_state=42)
X_train, X_val, y_train, y_val = train_test_split(X_train_val, y_train_val, test_size=0.25, random_state=42)

class TransformerModel(nn.Module):
    def __init__(self, vocab_size, embedding_dim, nhead, num_encoder_layers, hidden_dim, output_dim):
        super(TransformerModel, self).__init__()
        self.embedding = nn.Embedding(vocab_size + 1, embedding_dim)
        self.transformer_encoder = nn.TransformerEncoder(
            nn.TransformerEncoderLayer(embedding_dim, nhead, hidden_dim),
            num_layers=num_encoder_layers
        )
        self.fc = nn.Linear(embedding_dim, output_dim)

    def forward(self, x):
        x = self.embedding(x)
        x = x.permute(1, 0, 2)  # (batch_size, seq_len, embed_dim) -> (seq_len, batch_size, embed_dim)
        x = self.transformer_encoder(x)
        x = x.mean(dim=0)  # Average pooling across the sequence length
        return self.fc(x)

def train_model(model, train_loader, criterion, optimizer, num_epochs, device):
    for epoch in range(num_epochs):
        model.train()
        running_loss = 0.0
        for reviews, labels in train_loader:
            reviews, labels = reviews.to(device), labels.to(device)
            optimizer.zero_grad()
            outputs = model(reviews)
            loss = criterion(outputs, labels)
            loss.backward()
            optimizer.step()
            running_loss += loss.item()

        print(f'Epoch [{epoch + 1}/{num_epochs}], Loss: {running_loss / len(train_loader):.4f}')

def evaluate_model(model, test_loader, device):
    model.eval()
    correct = 0
    with torch.no_grad():
        for reviews, labels in test_loader:
            reviews, labels = reviews.to(device), labels.to(device)
            outputs = model(reviews)
            _, predicted = torch.max(outputs.data, 1)
            correct += (predicted == labels).sum().item()
    accuracy = correct / len(test_loader.dataset) * 100
    print(f'Test Accuracy: {accuracy:.2f}%')

def classify_unseen_data(model, unseen_data, word_to_idx, device):
    model.eval()
    unseen_data_cleaned = [clean_text(review) for review in unseen_data]  # Ensure you have this function
    unseen_data_tokenized = [tokenize(text) for text in unseen_data_cleaned]  # Ensure you have this function
    unseen_data_indexed = [[word_to_idx[word] for word in tokens if word in word_to_idx] for tokens in unseen_data_tokenized]

    unseen_data_tensor = pad_sequence([torch.tensor(review, dtype=torch.long) for review in unseen_data_indexed], batch_first=True, padding_value=0).to(device)

    with torch.no_grad():
        outputs = model(unseen_data_tensor)
        _, predictions = torch.max(outputs.data, 1)

    return predictions.cpu().numpy()

device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')

vocab_size = len(word_to_idx)
embedding_dim = 64
nhead = 8
num_encoder_layers = 2
hidden_dim = 128
output_dim = 2
num_epochs = 5

model = TransformerModel(vocab_size, embedding_dim, nhead, num_encoder_layers, hidden_dim, output_dim).to(device)
criterion = nn.CrossEntropyLoss()
optimizer = torch.optim.Adam(model.parameters())


train_model(model, train_loader, criterion, optimizer, num_epochs, device)
evaluate_model(model, test_loader, device)

unseen_data = [
    "This movie was fantastic! I loved it.",
    "I did not enjoy the film; it was boring."
]

predictions = classify_unseen_data(model, unseen_data, word_to_idx, device)
print(predictions)

# EX7
# Approach 1

import torch
import torch.nn as nn
import torch.optim as optim
import torchvision
from torchvision import transforms
import matplotlib.pyplot as plt

transform = transforms.Compose([transforms.ToTensor()])

train_dataset = torchvision.data.MNIST(root='./data',train=True, transform= transform, download=True)
test_dataset = torchvision.data.MNIST(root='./data', train=True, transform=tranforms, download=True)

epochs = 10
batch_size = 128
lr = 1e-03

train_loader = torch.utils.data.DataLoader(dataset=train_dataset, batch_size=batch_size, shuffle=True)
test_loader = torch.utils.data.DataLoader(dataset=test_dataset, batch_size=batch_size. shuffle=False)

def add_noise(imgs, noise_factor=0.5):
    noisy_imgs = imgs + noise_factor * torch.randn_like(imgs)
    noisy_imgs = torch.clip(noisy_imgs, 0., 1.)
    return noisy_imgs

class Autoencoder(nn.Module):
    def __init__(self):
        super(Autoencoder, self).__init__()
        self.encoder = nn.Sequential(
            nn.Linear(28*28,128)
            nn.ReLU(),
            nn.Linear(128,64),
            nn.ReLU(),
            nn.Linear(64,32)
        )
        self.decoder = nn.Sequential(
            nn.Linear(32,64),
            nn.ReLU(),
            nn.Linear(64,128),
            nn.ReLU()
            nn.Linear(128,28*28),
            nn.Sigmoid()
        )

    def forward(self,x):
        x = x.view(-1,28*28)
        encoded = self.encoder(x)
        decoded = self.decoder(encoded)
        decoded = decoded.view(-1,1,28,28)
        return decoded

device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
model = Autoencoder().to(device)
criterion = nn.MSELoss()
optimizer = optim.Adam(model.parameters(), lr=lr)

for epoch in range(epochs):
    for (images,_) in train_loader:
        optimizer.zero_grad()
        images = images.to(device)
        noisy_images = add_noise(images).to(device)

        outputs = model(noisy_images)
        loss = criterion(outputs,images)

        loss.backward()
        optimizer.step()

    print(f'Epoch [{epoch+1}/{num_epochs}], Loss: {loss.item():.4f}')

with torch.no_grad():
    for (images,_) in test_loader:
        images = images.to(device)
        noisy_images = add_noise(images).to(device)
        outputs = model(noisy_images)
        break

def show_images(images,title):
    images = images.cpu().numpy()
    fig, axes = plt.subplots(1,5, figsize=(10,2))
    for i, ax in enumerate(axes):
        ax.imshow(images[i][0], cmap='gray')
        ax.axis('off')
    fig.suptitle(title)
    plt.show()

show_images(images[:5])
show_images(noisy_images[:5])
show_images(outputs[:5])

# EX8
# Approach 1

import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import DataLoader
from torchvision import datasets, transforms
import torchvision

class Encoder(nn.Module):
    def __init__(self, input_dim,hidden_dim,latent_dim):
        super(Encoder, self).__init__()
        self.fc1 = nn.Linear(input_dim, hidden_dim)
        self.fc_mu = nn.Linear(hidden_dim, latent_dim)
        self.fc_logvar = nn.Linear(hidden_dim,latent_dim)

    def forward(self,x):
        h = torch.relu(self.fc1(x))
        mu = self.fc_mu(h)
        logvar = self.fc_logvar(h)
        return mu,logvar

class Decoder(nn.Module):
    def __init__(self, latent_dim, hidden_dim, output_dim):
        super(Decoder, self).__init__()
        self.fc1 = nn.Linear(latent_dim, hidden_dim)
        self.fc2 = nn.Linear(hidden_dim, output_dim)

    def forward(self,z):
        h = torch.relu(self.fc1(z))
        x_recon = torch.sigmoid(self.fc2(h))
        return x_recon

class VAE(nn.Module):
    def __init__(self, input_dim, hidden_dim, latent_dim):
        super(VAE, self).__init__()
        self.encoder = Encoder(input_dim,hidden_dim,latent_dim)
        self.decoder = Decoder(latent_dim,hidden_dim,input_dim)

    def reparameterize(self,mu,logvar):
        std = torch.exp(0.5 * logvar)
        eps = torch.randn_like(std)
        return mu + eps * std

    def forward(self,x):
        mu,logvar = self.encoder(x)
        z = self.reparameterize(mu,logvar)
        x_recon = self.decoder(z)
        return x_recon, mu, logvar

def loss_function(x_recon, x, mu, logvar):
    recon_loss = nn.functional.binary_cross_entropy(x_recon, x, reduction='sum')
    kl_loss = -0.5 * torch.sum(1+logvar - mu.pow(2) - logvar.exp())
    return recon_loss + kl_loss

def train_vae(model,dataloader,optimizer,device,epochs=1):
    model.train()
    for epoch in range(epochs):
        running_loss = 0.0
        for idx, (data,_) in enumerate(dataloader):
            data = data.view(-1,28*28).to(device)
            optimizer.zero_grad()
            recon_batch, mu, logvar = model(data)
            loss = loss_function(recon_batch,data,mu,logvar)
            loss.backward()
            optimizer.step()
            running_loss+=loss.item()

        print(f'Epoch [{epoch+1}/{epochs}], Loss: {running_loss / len(dataloader.dataset)}')

transform = transforms.Compose([transforms.ToTensor(), transforms.Lambda(lambda x: x.view(-1))])
train_dataset = datasets.MNIST(root='./data', train=True, transform = transform, download=True)
train_loader = DataLoader(train_dataset, batch_size=64, shuffle=True)

input_dim = 28*28
hidden_dim = 400
latent_dim = 20

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
vae = VAE(input_dim, hidden_dim, latent_dim).to(device)
optimizer = optim.Adam(vae.parameters())

train_vae(vae, train_loader, optimizer, device, epochs=10)

def generate_samples(model, num_samples, latent_dim, device):
    model.eval()
    with torch.no_grad():
        z = torch.randn(num_samples, latent_dim). to(device)
        samples = model.decoder(z).cpu()
        return samples.view(-1,1,28,28)

new_samples = generate_samples(vae,10,latent_dim,device)

import matplotlib.pyplot as plt
grid = torchvision.utils.make_grid(new_samples, nrow=5)
plt.imshow(grid.permute(1,2,0).detach().numpy())
plt.show()