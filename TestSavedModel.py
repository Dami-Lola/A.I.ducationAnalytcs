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

def testSavedModel():
      folder_path = ""
      BestTestACC = 0
      history = []
      model = torch.load(folder_path + '/trained_model.pth')
      model.eval()  # Set the model to evaluation mode for testing
      with torch.no_grad():
          correct_test = 0
          total_test = 0
          for instances, labels in test_loader:  # Use test_loader for final testing after training
              output = model(instances)
              _, predicted_test = torch.max(output, 1)
              total_test += labels.size(0)
              correct_test += (predicted_test == labels).sum().item()

          accuracy_test = correct_test / total_test
          print(f"Accuracy on Test Set: {accuracy_test * 100:.2f}%")

          if accuracy_test > BestTestACC:
              BestTestACC = accuracy_test
          history.append(accuracy_test)
          print(history)
if __name__ == '__main__':
      testSavedModel()