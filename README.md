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
1) Download the data set at link https://drive.google.com/drive/folders/1Xez2jLl-JtV73wbujfPDdqpY_MvY-57x?usp=sharing
2) The dataset have to be collected together. The program will clean and seprate the into the respective train, 
evaluation and testing loader.
3) Update the variable folder_path to the path where the dataset is stored
4) Update the saved_model_path to where you'll have your saved model stored and also the updatedclass.csv
5) The main model currently has the main model settings. 
6) Run the program, every analysis from confusion matrix, to bias analysis and kflod validation will be done

## CNN.ipynb
1) Is the PyTorch format of CNN.PY
2) You can load up the file on google colab
3) Change the folder_path path to the folder that has the dataset 
4) Change the saved_model_path to the path where you'll store your saved model and also the updatedclass.csv
4) 
## TestSavedModel.py
1) Load the trained_model.pth from the saved model path
2) Run the program
3) The accuracy and the confusion matrix will be displayed

## TestSingleImage.py
1) Update the saved model path to the path of the single image
2) Run the program to get the predicated class probability
3) Predicted classes will be 0-Angry 1-Bored 2-Focused 3-Neutral

## 

## Versioning

We use Github for versioning. For the versions available, see the (https://github.com/Dami-Lola/A.I.ducationAnalytcs). 
The report was generate in LaTex

## Full Dataset
Link to the full dataset(cleaned) used for the project (https://drive.google.com/drive/folders/1hGzH4__sQvJTWzZPC2PliNfgltPBQ8Js?usp=sharing).

## New Dataset based on labeling by attributes
https://drive.google.com/drive/folders/1DYZ6y0x1f0hrlSdHrfPSu7gUjKMZ6jVE?usp=sharing

## Updated Dataset based on Bias Analysis
https://drive.google.com/drive/folders/1Xez2jLl-JtV73wbujfPDdqpY_MvY-57x?usp=sharing


## Link to saved model from part 2
https://drive.google.com/file/d/1jhoAOGFPj_MoC4nvLYUn7N7T6H1skikZ/view?usp=sharing
## Link to saved model from part 3
https://drive.google.com/file/d/1rJ0dnsvvsCTWCu2ufrTzYQiOOKCKAkDM/view?usp=sharing

## Authors
* **Zahra pezeshki - Data Specialist**
* **Hema Reddy Muppidi - Training Specialist** 
* **Oluwadamilola Okafor - Evaluation Specialist**

See also the list of [contributors](https://github.com/your/project/contributors) who participated in this project.