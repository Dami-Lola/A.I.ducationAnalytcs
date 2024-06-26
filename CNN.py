# -*- coding: utf-8 -*-
"""Copy of Proj_v02cp2.ipynb

Automatically generated by Colaboratory.

Original file is located at
    https://colab.research.google.com/drive/14BVop6w6TPcLxyE62bz_zsAbAJ89u894
"""

import torch
import numpy as np
import pandas as pd
from torch import optim, nn
import torch.nn.functional as F
from torch.utils.data import DataLoader
from torch.utils.data.dataset import Dataset
from torch.utils.data import random_split
import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.utils.data as td
import torchvision.transforms as transforms
import torchvision.datasets as datasets
import os
import matplotlib.pyplot as plt
from sklearn.metrics import classification_report, confusion_matrix, ConfusionMatrixDisplay

from google.colab import drive
drive_path = '/content/gdrive/'
drive.mount(drive_path)
folder_path = os.path.join(drive_path, "MyDrive","COMP6721 Project", "DATASETS", "emotional detection", "data set" ,)

# Assuming you have a custom dataset class named MyDataset
class MyDataset(Dataset):
    def __init__(self, data):
        self.data = data

    def __len__(self):
        return len(self.data)

    def __getitem__(self, index):
        return self.data[index]

# Create an instance of your dataset
normalize = transforms.Normalize(mean=[0.485],std=[0.225])
data_set_from_folder = datasets.ImageFolder(folder_path, transform=transforms.Compose([
    transforms.RandomHorizontalFlip(),
    transforms.RandomCrop(32, 4),
    transforms.Grayscale(num_output_channels=1),
    transforms.ToTensor(),
    normalize
    ]))
dataset = MyDataset(data_set_from_folder)

# Define the sizes for training, validation, and test sets
train_size = int(0.75 * len(dataset))
val_size = int(0.15 * len(dataset))
test_size = len(dataset) - train_size - val_size
batch_size = 32

# Use random_split to split the dataset
train_dataset, val_dataset, test_dataset = random_split(dataset, [train_size, val_size, test_size])

# Create DataLoader instances for each set if needed
train_loader = DataLoader(train_dataset, batch_size=batch_size, shuffle=True)
val_loader = DataLoader(val_dataset, batch_size=batch_size, shuffle=False)
test_loader = DataLoader(test_dataset, batch_size=batch_size, shuffle=False)

y_test = np.array([] , dtype='int64')
for _ , labels in iter(test_loader):
  y_test = np.append(y_test , labels.numpy())

class MainModel(nn.Module):
    def __init__(self,input_size, hidden_size, output_size):
        super().__init__()

        self.layer1=nn.Conv2d(1,32,3,padding=1,stride=1)
        self.B1 = nn.BatchNorm2d(32)
        self.layer2 = nn.Conv2d(32, 32, 3, padding=1, stride=1)
        self.B2 = nn.BatchNorm2d(32)
        self.Maxpool=nn.MaxPool2d(2)
        self.layer3 = nn.Conv2d(32, 64, 3, padding=1, stride=1)
        self.B3 = nn.BatchNorm2d(64)
        self.layer4 = nn.Conv2d(64, 64, 3, padding=1, stride=1)
        self.B4 = nn.BatchNorm2d(64)

        self.fc = nn.Linear(64*8*8, output_size)

    def forward(self, x):

        x = self.B1(F.leaky_relu(self.layer1(x)))
        x =  self.Maxpool(F.leaky_relu(self.layer2(x)))
        x=self.B2(x)
        x=self.B3(F.leaky_relu(self.layer3(x)))
        x = self.B4(self.Maxpool(F.leaky_relu(self.layer4(x))))

        return self.fc(x.view(x.size(0),-1))

if __name__ == '__main__':

    input_size = 1 * 32 * 32
    hidden_size = 32
    output_size = 4

epochs = 15
    model = MainModel(input_size, hidden_size, output_size)
    def accuracy(outputs, labels):
        _, preds = torch.max(outputs, dim=1)
        return torch.tensor(torch.sum(preds == labels).item() / len(preds))

    criterion = nn.CrossEntropyLoss()
    optimizer = optim.Adam(model.parameters(), lr=0.01)
    BestACC1=0
    BestACC2=0
    history1 = []
    history2=[]
    loss_epoc = []
    for epoch in range(epochs):
      model.train()
      running_loss = 0
      for instances, labels in train_loader:
          optimizer.zero_grad()

          output = model(instances)
          loss = criterion(output, labels)
          loss.backward()
          optimizer.step()

          running_loss += loss.item()

      print(running_loss / len(train_loader))
      loss_epoc.append(running_loss / len(train_loader))

      # VALIDATION
      model.eval()
      with torch.no_grad():
        allsamps1=0
        rightPred1=0

        for instances, labels in val_loader:
          output1 = model(instances)
          predictedClass=torch.max(output1,1)
          allsamps1+=output.size(0)
          rightPred1+=(torch.max(output1,1)[1]==labels).sum()

        ACC1=rightPred1/allsamps1
        print('Accuracy of validation is=',ACC1*100)
        if ACC1>BestACC1:
            BestACC1=ACC1
            torch.save(model, folder_path + '/trained_model.pth')
        history1.append(ACC1)

      #TEST
      model.eval()
      with torch.no_grad():
        allsamps2=0
        rightPred2=0
        for instances, labels in test_loader:
          output2 = model(instances)
          predictedClass=torch.max(output2,1)
          allsamps2+=output2.size(0)
          rightPred2+=(torch.max(output2,1)[1]==labels).sum()

        ACC2=rightPred2/allsamps2
        print('Accuracy of test is=',ACC2*100)
        if ACC2>BestACC2:
            BestACC2=ACC2
        history2.append(ACC2)

        print(history1)
        print(history2)

        print(loss_epoc)

        len(loss_epoc)

#Loss and ACCuracy for Test

        plt.plot(np.arange(len(loss_epoc)), loss_epoc,'r')
        plt.xlabel('Epoch')
        plt.ylabel('Loss')
        plt.show()

        plt.plot(np.arange(len(history2 )), history2 , 'b')
        plt.xlabel('Epoch')
        plt.ylabel('Accuracy')
        plt.show()

        from sklearn.metrics import confusion_matrix
        import seaborn as sn
        import pandas as pd

        y_pred = []
        y_true = []

        # iterate over test data
        for inputs, labels in test_loader:
                output = model(inputs)
                output = (torch.max(torch.exp(output), 1)[1]).data.cpu().numpy()
                y_pred.extend(output) # Save Prediction

                labels = labels.data.cpu().numpy()
                y_true.extend(labels) # Save Truth

        # constant for classes
        classes = ('neutral', 'angry', 'bored', 'focused')

        # Build confusion matrix

        cf_matrix = confusion_matrix(y_true, y_pred)
        cf_matrix
        df_cm = pd.DataFrame(cf_matrix / np.sum(cf_matrix, axis=1)[:, None], index = [i for i in classes],
                             columns = [i for i in classes])
        plt.figure(figsize = (12,7))
        sn.heatmap(df_cm, annot=True)
        plt.savefig('output.png')

        # Commented out IPython magic to ensure Python compatibility.
        # Magic function that renders the figure in a jupyter notebook
        # instead of displaying a figure object
        # %matplotlib inline

        classesIndexes, classesFrequency = np.unique(y_true, return_counts=True)
        pre_class_indexes , pre_class_fre = np.unique(y_pred, return_counts=True)

        # Setting default size of the plot
        plt.rcParams['figure.figsize'] = (10.0, 7.0)


        # Plotting histogram of 5 classes with their number of samples
        # Defining a figure object
        figure = plt.figure()


        # Plotting Bar chart
        plt.bar(classesIndexes, classesFrequency, align='center', alpha=0.1)
        plt.bar(pre_class_indexes, pre_class_fre, align='center', alpha=0.1)

        # Giving name to Y axis
        plt.ylabel('Class frequency', fontsize=16)


        # Giving names to every Bar along X axis
        plt.xticks(classesIndexes, ['neutral', 'angry', 'bored', 'focused'], fontsize=16)


        # Giving name to the plot
        plt.title('Histogram', fontsize=20)


        # Showing the plot
        plt.show()

        # import matplotlib.pyplot as plt
        # plt.hist(y_test)


        # Showing the main classification metrics
        print(classification_report(y_true, y_pred))


