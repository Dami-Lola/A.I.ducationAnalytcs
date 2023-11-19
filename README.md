# A.I.ducationAnalytcs
Welcome to A.I.ducation Analytics, where the future of AI-driven academic feedback takes shape.

This work aims to demonstrate an end–to–end pipeline implementation of a Convolutional Neural Network (CNN–based) 
to classify five categories of Facial expression in a classroom setting, namely:
* Focused.
* Bored.
* Neutral.
* Angry.

## Files
cleaningcodeforcnnmodel.py: File handles the data cleaning

preprocessing.py: The data visualization file that handles these 3 tasks:
    # Task 1: Plot the class distribution
    # Task 2: Display 25 random images in a 5x5 grid
    # Task 3: Plot the pixel intensity histogram

CNN.py: The CNN network to run and train the models

TestSavedModel: To run a dataset on any of the saved models

TestSingleImage: To test a single images on any saved model

Dataset: Contains the cleaned dataset, 10 images of each class in training and testing folders

### Prerequisites
* Python 3.8.8

### Getting Started
These instructions get the files above running on the development environment. 
Have the dataset folder in the same directory as the .py files


## Installing
Install the following libraries in the development env
```
pip install seaborn
pip install matplotlib
pip install keras
pip install google.colab
```

## Running cleaningcodeforcnnmodel.py
1) The string concatenated with folder_path+"" is the folder name and path the clean data will be generated into
2) It has to be manually done for each class for training and testing
3) Run the file and the cleaned data will be generated to the folder created.
4) This was done manually to get the clean data for visualization. When running CNN, it will be generated automatically

## Running preprocessing.py
1) dataset_path has the path to the training and the testing folders. To view either one of them,comment the other one
2) Run the main in line 92 
3) The Class distribution, 25 random images and pixel density of the dataset will be presented.

## CNN.PY
1) Download the data set at link https://drive.google.com/drive/folders/1WjQ7nzhBlQzX9wYjfJN-YSLTq3sSrP4O?usp=sharing
2) The dataset have to be collected together. The program will clean and seprate the into the respective train, 
evaluation and testing loader.
3) Update the variable folder_path to the path where the dataset is stored
4) The main model currently has the main model settings. 
5) Run the program, the loss graph, accuracy graph, confusion matrix, including the performance table will be printed out

## TestSavedModel.py
1) Download any of the pretrain dataset https://drive.google.com/drive/folders/1WjQ7nzhBlQzX9wYjfJN-YSLTq3sSrP4O?usp=sharing
2) Update the variable folder_path to the path where the dataset is stored
3) Run the program to get the accuracy printed out

## TestSingleImage.py
1) Update the folder path to the path of the single image
2) Run the program to get the predicated class probability

## Versioning

We use Github for versioning. For the versions available, see the (https://github.com/Dami-Lola/A.I.ducationAnalytcs). 
The report was generate in LaTex

## Full Dataset
Link to the full dataset(cleaned) used for the project (https://drive.google.com/drive/folders/1hGzH4__sQvJTWzZPC2PliNfgltPBQ8Js?usp=sharing).

## Authors
* **Zahra pezeshki - Data Specialist**
* **Hema Reddy Muppidi - Training Specialist** 
* **Oluwadamilola Okafor - Evaluation Specialist**

See also the list of [contributors](https://github.com/your/project/contributors) who participated in this project.