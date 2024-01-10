import sys
import os
sys.path.append("data/3DMNIST")
import torch
import torchvision
import numpy
import h5py
from torch import nn 
from matplotlib import pyplot as plt
from torch.utils.data import Dataset,DataLoader
from pathlib import Path
from Datasets import *
import random

SEED = 77
DEVICE = "cuda" if torch.cuda.is_available() else "cpu"
torch.manual_seed(SEED)

class TNetkd(nn.Module):
    def __init__(self,channel,k=3):
        super().__init__()
        self.k = k
        self.conv1  = nn.Conv1d(in_channels=channel,out_channels=128,kernel_size=1)
        self.conv2 = nn.Conv1d(in_channels=128,out_channels=256,kernel_size=1)
        self.conv3 = nn.Conv1d(in_channels=256,out_channels=1024,kernel_size=1)
        self.relu = nn.ReLU()
        self.norm1 = nn.BatchNorm1d(128)
        self.norm2 = nn.BatchNorm1d(256)
        self.norm3 = nn.BatchNorm1d(1024)
        self.fc1 = nn.Linear(1024,256)
        self.fc2 = nn.Linear(256,128)
        self.fc3 = nn.Linear(128,k*k)

        self.matrix = 0

    def forward(self,x:torch.Tensor):
        temp = x
        x = self.relu(self.norm1(self.conv1(x)))
        x = self.relu(self.norm2(self.conv2(x)))
        x = self.relu(self.norm3(self.conv3(x)))
        x = torch.max(x,dim=2)[0]
        x = self.fc1(x)
        x = self.fc2(x)
        x = self.fc3(x)
        iden = torch.from_numpy(numpy.array([numpy.eye(self.k)])).type(torch.float32).repeat(x.size()[0],1,1)
        iden = iden.to(DEVICE)
        x = x.view(-1,self.k,self.k)
        x = x + iden
        self.matrix = x[0]
        temp = temp.transpose(dim0 = 1,dim1 = 2)
        temp = temp.matmul(x)
        temp = temp.transpose(dim0 = 1,dim1 = 2)
        return temp
class ToyPointNet(nn.Module):
    def __init__(self,channel=3,out_feature=10):
        super().__init__()
        self.TNet3d = TNetkd(channel,3)
        self.mlp1 = nn.Sequential(
            nn.Conv1d(channel,64,1),
            nn.Conv1d(64,64,1)
        )
        self.TNet64d = TNetkd(64,64)
        self.mlp2 = nn.Sequential(
            nn.Conv1d(64,64,1),
            nn.Conv1d(64,128,1),
            nn.Conv1d(128,1024,1)
        )
        self.mlp3 = nn.Sequential(
            nn.Conv1d(1024,512,1),
            nn.Conv1d(512,256,1),
            nn.Conv1d(256,out_feature,1)
        )
        self.matrix_loss = 0
        
    def forward(self,x:torch.Tensor):
        x = self.TNet3d(x)
        x = self.mlp1(x)
        x = self.TNet64d(x)
        x = self.mlp2(x)
        x = torch.max(x,dim=2,keepdim=True)[0]
        x = self.mlp3(x)
        x = x.squeeze(dim=2)
        N1 = self.TNet3d.matrix.matmul(self.TNet3d.matrix.transpose(0,1))
        N2 = self.TNet64d.matrix.matmul(self.TNet64d.matrix.transpose(0,1))
        self.matrix_loss = torch.norm(N1 - torch.eye(N1.shape[0]).to(DEVICE), p='fro') + torch.norm(N2 - torch.eye(N2.shape[0]).to(DEVICE), p='fro')
        return x

def accuracy_fn(y_true:torch.Tensor,y_logits:torch.Tensor):
    eq =y_true.eq(y_logits.softmax(dim=1).argmax(dim=1))
    samecount  = eq.sum()
    return samecount/eq.size()[0]*100

def TrainModel(model:ToyPointNet,epochs,train_dataset=None,test_dataset=None,train_datasetname=None,test_datasetname=None):

    if not train_dataset and not train_datasetname:
        raise RuntimeError("failed to get train dataset in Trainning!")
    if not test_dataset and not test_datasetname:
        raise RuntimeError("failed to get train dataset in Trainning!")
    if not train_dataset:
        train_dataset = LoadDataset(train_datasetname)
    if not test_dataset:
        test_dataset = LoadDataset(test_datasetname)
    train_dataloader = DataLoader(train_dataset,32,shuffle=True)
    test_dataloader = DataLoader(test_dataset,32,shuffle=False)
    reg_weight = 0.1
    model.to(DEVICE)
    loss_fn = nn.CrossEntropyLoss()
    optimizer = torch.optim.Adam(model.parameters(),lr = 0.001,betas=(0.9,0.9))
    for epoch in range(epochs):
        print(f"Epoch:{epoch}:")
        model.train()   
        train_loss = 0
        train_acc = 0
        for X,y in train_dataloader:
            X = X.to(DEVICE)
            y = y.to(DEVICE)
            optimizer.zero_grad()
            y_logits = model(X)
            loss = loss_fn(y_logits,y) + reg_weight*model.matrix_loss
            acc = accuracy_fn(y,y_logits)
            train_loss += loss/len(train_dataloader)
            train_acc += acc/len(train_dataloader)
            loss.backward()
            optimizer.step()
        print(f"TrainLoss:{train_loss:.3f}  TrainAcc:{train_acc:.1f}%")
        model.eval()
        with torch.inference_mode():
            test_loss = 0
            test_acc = 0
            for X,y in test_dataloader:
                X = X.to(DEVICE)
                y = y.to(DEVICE)
                y_logits = model(X)
                test_loss += loss_fn(y_logits,y)/len(test_dataloader)
                test_acc += accuracy_fn(y,y_logits)/len(test_dataloader)
            print(f"TestLoss:{test_loss:.3f}  TestAcc:{test_acc:.1f}%")
    return model

def SaveModel(model:ToyPointNet,name = "ToyPointNet0.pth"):
    folder = Path("models")
    folder.mkdir(exist_ok=True)
    path = folder/name
    torch.save(model.state_dict(),path)

def LoadModel(name = "ToyPointNet0.pth"):
    folder = Path("models")
    path = folder/name
    model = ToyPointNet().to(DEVICE)
    model.load_state_dict(torch.load(path))
    return model
def Make_prediction(model:ToyPointNet,test_dataset=None,test_datasetname=None,num=3):
    if not test_dataset and not test_datasetname:
        raise RuntimeError("failed to get train dataset in Predicting!")
    if not test_dataset:
        test_dataset = LoadDataset(test_datasetname)
    fig = plt.figure(figsize=(10,10))
    samples = random.sample(list(test_dataset),num*num)
    for i,(X,y) in enumerate(samples):
        passin = X.unsqueeze(dim=0).to(DEVICE)
        y_pred = model(passin).softmax(dim=1).argmax(dim=1).item()
    
        points = X.numpy()
        ax = fig.add_subplot(num,num,i+1,projection="3d")
        if y_pred == y:
            ax.set_title(f"pred:{test_dataset.classes[y_pred]}|true:{test_dataset.classes[y]}",color ="g")
        else:
            ax.set_title(f"pred:{test_dataset.classes[y_pred]}|true:{test_dataset.classes[y]}",color ="r")
        ax.scatter(points[0,:],points[1,:],points[2,:],c=points[2,:],cmap="rainbow")
    fig.show()
    

        


    



        
