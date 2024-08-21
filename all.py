## ANN

import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim
from torch.utils.data import Dataset, DataLoader, random_split
from sklearn.model_selection import ParameterGrid,GridSearchCV
import pandas as pd
import matplotlib.pyplot as plt
import numpy as np

x = torch.linspace(-torch.pi,torch.pi,2000)
y = torch.sin(x)
y += torch.randn(x.size())/5
y = y.reshape(-1,1)
y

class PolynomialDataset(Dataset):
    def __init__(self,x,y):
        self.x = x
        self.y = y
    
    def __len__(self):
        return len(self.x)
    
    def __getitem__(self,idx):
        return self.x[idx],self.y[idx]
    
dataset = PolynomialDataset(x,y)

t_set, v_set, test_set = random_split(dataset,[0.8,0.1,0.1])

val_loader = DataLoader(v_set, batch_size=64, shuffle=False)
test_loader = DataLoader(test_set, batch_size=64, shuffle=False)

class PolynomialRegression(nn.Module):
    def __init__(self):
        super().__init__()
        self.fc1 = nn.Linear(1,10)
        self.drop1 = nn.Dropout(0.5)
        self.fc2 = nn.Linear(10,5)
        self.fc3 = nn.Linear(5,3)
        self.drop2 = nn.Dropout(0.5)
        self.fc4 = nn.Linear(3,1)
        self.norm1 = nn.BatchNorm1d(5)
    
    def forward(self,x):
        x = x.view(-1,1)
        x = F.relu(self.fc1(x))
        x = self.drop1(x)
        x = F.sigmoid(self.fc2(x))
        x = self.norm1(x)
        x = self.drop2(x)
        x = F.tanh(self.fc3(x))
        x = self.fc4(x)
        return x
    
param_grid = {
    'epochs': [20,40],
    'lr': [0.001,0.01],
    'optimizer': [optim.Adam],
    'batch_size': [32,64]
}

losses = []
least_error = 9999999.9
best_params = {}
patience = 5 # number of epochs it doesnt improve continuously

for params in ParameterGrid(param_grid):
    unimproved_epochs = 0
    model = PolynomialRegression()
    criterion = nn.MSELoss()
    optimizer = params['optimizer'](model.parameters(), lr=params['lr'], weight_decay=1e-04)
    train_loader = DataLoader(t_set, batch_size = params['batch_size'], shuffle=True)

    for epoch in range(params['epochs']):
        for batch_idx,(inputs,targets) in enumerate(train_loader):
            optimizer.zero_grad()

            outputs = model(inputs)
            loss = criterion(outputs,targets)

            loss.backward()
            optimizer.step()

        #if epoch % 5 == 0:
            #print(f'Loss: {loss.item()}, epoch: {epoch}/{params['epochs']}')
        
        losses.append(loss.item())
        
        model.eval()
        validation_loss = 0.0
        with torch.no_grad():
            for ba_idx,(val_inputs,val_targets) in enumerate(val_loader):
                val_outputs = model(val_inputs)
                val_loss = criterion(val_targets,val_outputs)
                validation_loss+=val_loss
            
        avg = validation_loss/len(val_loader)
            
        if epoch % 10 == 0:
            print(params)
            print(f"Validation Loss: {avg}")

        if avg < least_error:
                torch.save(model,'ann_best_model.pt')
                least_error = avg
                best_params = params
        else:
            unimproved_epochs +=1
            if unimproved_epochs >= patience:
                print("Early stop for parameter set:")
                print(params)
                break

print(best_params)
least_error

model = torch.load('ann_best_model.pt')
test_loss = 0.0
model.eval()

t_inps = []
t_outs = []
t_targs = []


with torch.no_grad():
    for idx,(o_inputs,o_targets) in enumerate(test_loader):
        o_outputs = model(o_inputs)
        o_loss = criterion(o_outputs,o_targets)
        test_loss += o_loss.item()
        t_inps = o_inputs
        t_outs = o_outputs
        t_targs = o_targets
    
    avg = test_loss/len(test_loader)
    print(f"avg loss: {avg}")

x_np = x.numpy()
y_np = y.numpy()

plt.scatter(t_inps,t_targs)
plt.scatter(t_inps,t_outs)

model = torch.load('ann_best_model.pt')
test_loss = 0.0
model.eval()

t_inps = []
t_outs = []
t_targs = []


with torch.no_grad():
    for idx,(o_inputs,o_targets) in enumerate(train_loader):
        o_outputs = model(o_inputs)
        o_loss = criterion(o_outputs,o_targets)
        test_loss += o_loss.item()
        t_inps = o_inputs
        t_outs = o_outputs
        t_targs = o_targets
    
    avg = test_loss/len(test_loader)
    print(f"avg loss: {avg}")

## Classification

import pandas as pd
import numpy as np 
import matplotlib.pyplot as plt 

import torch
import torch.nn as nn
import torch.nn.functional as F 
import torch.optim as optim
from torch.utils.data import Dataset, DataLoader, random_split

from sklearn.model_selection import ParameterGrid
from sklearn.preprocessing import LabelEncoder
from sklearn.metrics import accuracy_score, classification_report

df = pd.read_csv('abalone.csv')

y = df['Type']
x = df.drop(['Type'],axis=1)

y.unique()

le = LabelEncoder()
le.fit(y)
y = le.transform(y)

x = torch.tensor(x.values, dtype=torch.float32)
y = torch.tensor(y, dtype=torch.long)

class MyDataset(Dataset):
    def __init__(self,x,y):
        self.x = x
        self.y = y
    
    def __len__(self):
        return len(self.x)

    def __getitem__(self,idx):
        return self.x[idx], self.y[idx]
    
dataset = MyDataset(x,y)

train_set, val_set, test_set = random_split(dataset,[0.8,0.1,0.1])

test_loader = DataLoader(test_set,batch_size=64,shuffle=False)
val_loader = DataLoader(val_set,batch_size=64,shuffle=False)

class ClassificationModel(nn.Module):
    def __init__(self):
        super().__init__()
        self.fc1 = nn.Linear(8, 20)
        self.norm1 = nn.BatchNorm1d(20)
        self.drop1 = nn.Dropout(0.5)
        self.fc2 = nn.Linear(20, 40)
        self.drop2 = nn.Dropout(0.5)
        self.o = nn.Linear(40, 3)  # 3 output classes
    
    def forward(self, x):
        x = F.relu(self.fc1(x))
        x = self.norm1(x)
        x = self.drop1(x)
        x = F.relu(self.fc2(x))
        x = self.drop2(x)
        x = self.o(x)  # Output raw logits, no activation here
        return x
    
param_grid = {
    'epochs': [10,20,30,40],
    'learning_rate': [0.001,0.01],
    'optimizer': [optim.Adam],
    'batch_size': [128,64,32]
}

best_loss = 999999999.9
best_params = {}
patience = 5

for params in ParameterGrid(param_grid):
    unimproved_epochs = 0
    model = ClassificationModel()
    criterion = nn.CrossEntropyLoss()
    optimizer = params['optimizer'](model.parameters(), lr = params['learning_rate'], weight_decay = 1e-04)
    train_loader = DataLoader(train_set, batch_size=params['batch_size'], shuffle=True)

    for epoch in range(params['epochs']):
        for batch_idx,(inputs,targets) in enumerate(train_loader):
            optimizer.zero_grad()

            outputs = model(inputs)
            loss = criterion(outputs,targets)

            loss.backward()
            optimizer.step()
        
        #if epoch % 5 == 0:
            #print(f"Error: {loss.item()}, epoch: {epoch}/{params['epochs']}")
        
        model.eval()
        validation_loss = 0.0
        total = 0
        correct = 0
        with torch.no_grad():
            for idx,(v_inputs,v_targets) in enumerate(val_loader):
                v_outputs = model(v_inputs)
                v_loss = criterion(v_outputs,v_targets)
                validation_loss+=v_loss.item()

                _, predicted = torch.max(v_outputs,1)
                total += v_targets.size(0)
                correct += (predicted == v_targets).sum().item()

            
            avg = validation_loss/len(val_loader)
            accuracy = correct/total
        print(100*accuracy)

        if avg < best_loss:
            best_loss = avg
            best_params = params
            unimproved_epochs = 0
            torch.save(model,'classify.pt')
        else:
            unimproved_epochs+=1
            if unimproved_epochs >= patience:
                print("Early stopping triggered")
                break

best_loss

best_params

model = torch.load('classify.pt')
test_loss = 0

model.eval()
all_targets = []
all_outputs = []
all_inputs = []

with torch.no_grad():
    for idx,(test_inputs,test_targets) in enumerate(test_loader):
        test_outputs = model(test_inputs)
        _, predictions = torch.max(test_outputs,1)

        all_targets.extend(test_targets.numpy())
        all_outputs.extend(predictions.numpy())
        all_inputs.extend(test_inputs.numpy())

all_targets = np.array(all_targets)
all_predictions = np.array(all_outputs)
print(accuracy_score(all_targets,all_predictions))
print(classification_report(all_targets,all_predictions))


## CNN

import torch
from torchvision import models,transforms
import matplotlib.pyplot as plt
from PIL import Image
import requests
import json
import numpy as np

print(models.list_models())

model = models.efficientnet_b4(pretrained=True) # Change Model Here
model.eval()

path = 'elephant.jpg'
inp_img = Image.open(path)

plt.imshow(inp_img)

preprocess = transforms.Compose([
    transforms.Resize(256),
    transforms.CenterCrop(224),
    transforms.ToTensor(),
    transforms.Normalize(mean=[0.485,0.456,0.406], std=[0.229,0.224,0.225])
])

if inp_img.mode != 'RGB':
    inp_img = inp_img.convert('RGB')

input_tensor = preprocess(inp_img)
input_batch = input_tensor.unsqueeze(0)
input_batch.shape

with torch.no_grad():
    output = model(input_batch)

probs = torch.nn.functional.softmax(output[0],dim=0)

output.shape

labels = 'https://raw.githubusercontent.com/anishathalye/imagenet-simple-labels/master/imagenet-simple-labels.json'
response = requests.get(labels)
label = json.loads(response.text)

top5_prob,top5_catid = torch.topk(probs,5)

for i in range(top5_prob.size(0)):
    print(label[top5_catid[i]],top5_prob[i].item())

