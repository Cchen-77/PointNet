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
import random

class PointCloudDataset_3DMNIST(Dataset):
    def __init__(self,path):
        super().__init__()
        self.classes = list(range(10))
        self.classes2idx = {idx:idx for idx in range(10)}
        with h5py.File(path) as hf:
            self.Xs = [numpy.array(random.sample(list(hf[key]["points"][:]),500)) for key in hf.keys()]
            self.ys = [hf[key].attrs["label"] for key in hf.keys()]
        for i in range(len(self.Xs)):
            self.Xs[i] = torch.tensor(self.Xs[i],dtype=torch.float32).transpose(0,1)
    def __len__(self):
        assert(len(self.Xs) == len(self.ys))
        return len(self.Xs)
    def __getitem__(self, index):
        return self.Xs[index],self.ys[index]
class PointCloudDataset_ModelNet_Sampled(Dataset):
    def __init__(self,path,shape_names_list):
        super().__init__()
        self.classes = []
        with open(shape_names_list,"r") as f:
            line = f.readline()
            while line:
                line = line.strip()
                self.classes.append(line)
                line = f.readline()
        self.classes2idx = {idx:idx for idx in range(10)}
        with h5py.File(path) as hf:
            self.Xs = list(hf["data"][:])
            self.ys = hf["label"][:]
        for i in range(len(self.Xs)):
            self.Xs[i] = torch.from_numpy(self.Xs[i]).transpose(0,1)
        self.ys = numpy.squeeze(self.ys,1)
        self.ys = list(self.ys)
        self.classes2idx = {theclass:idx for idx,theclass in enumerate(self.classes)}    
    def __len__(self):
        assert(len(self.Xs) == len(self.ys))
        return len(self.Xs)
    def __getitem__(self, index):
        return self.Xs[index],self.ys[index]
def find_classes(path:Path):
    classes = sorted([entry.name for entry in os.scandir(path) if entry.is_dir()])
    return classes,{theclass:idx for idx,theclass in enumerate(classes)}
class PointCloudDataset_ModelNet(Dataset):
    def __init__(self,path:Path,split = "train",points_sample_count = 100):
        super().__init__()
        self.classes,self.classes2idx = find_classes(path)
        self.Xs = []
        self.ys = []
        for off in list(path.glob(f"*/{split}/*.off")):
            X = []
            with open(off,"r") as f:
                #OFF MAGIC NUM
                f.readline()
                vertexcount = int(f.readline().split()[0])
                for _ in range(vertexcount):
                    vertex = [float(x) for x in f.readline().split()]
                    X.append(vertex)
                X = random.sample(X,points_sample_count)
                X = torch.tensor(numpy.array(X),dtype=torch.float32).transpose(0,1)
                self.Xs.append(X)
                y = self.classes2idx[off.parent.parent.stem]
                self.ys.append(y)

    def __len__(self):
        assert(len(self.Xs) == len(self.ys))
        return len(self.Xs)
    def __getitem__(self, index):
        return self.Xs[index],self.ys[index]

class BaseDataset(Dataset):
    def __init__(self,dataset):
        super().__init__()
        self.Xs = dataset.Xs[:]
        self.ys = dataset.ys[:]
        self.classes,self.classes2idx = dataset.classes,dataset.classes2idx
    def __len__(self):
        assert(len(self.Xs) == len(self.ys))
        return len(self.Xs)
    def __getitem__(self, index):
        return self.Xs[index],self.ys[index]
def SaveDataset(dataset,name):
    folder = Path("datasets")
    folder.mkdir(exist_ok=True)
    path = folder/name
    torch.save(dataset,path)
def LoadDataset(name):
    folder = Path("datasets")
    folder.mkdir(exist_ok=True)
    path = folder/name
    return BaseDataset(torch.load(path))