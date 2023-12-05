from sklearn.metrics import classification_report, confusion_matrix, ConfusionMatrixDisplay
import matplotlib.pyplot as plt
import seaborn as sn
import pandas as pd
import numpy as np
import torch

save_model_path = ''
# To evaluuate on test dataset and get confusion matrix

def testSavedModel():
    cnnModel = torch.load(save_model_path + '/best_model7.pth')
    cnnModel.eval()
    all_labels = []
    all_predictions = []
    with torch.no_grad():
        for images, labels, _ in test_loader:  # Unpacking to ignore attributes
            outputs = cnnModel(images)
            _, predicted = torch.max(outputs.data, 1)

            all_labels.extend(labels.cpu().numpy())
            all_predictions.extend(predicted.cpu().numpy())

    accuracy = np.mean(np.array(all_labels) == np.array(all_predictions))
    print('\nTest Accuracy of the model on the test images: {:.2f} %'.format(accuracy * 100))

    conf_mat = confusion_matrix(all_labels, all_predictions)
    print('\nConfusion Matrix\n')
    print(conf_mat, '\n')

    # Constant for classes - adjust as per your dataset
    classes = ('angry', 'bored', 'focused','neutral')  # Replace with actual class names
    df_cm = pd.DataFrame(conf_mat, index=classes, columns=classes)
    plt.figure(figsize=(12,7))
    sn.heatmap(df_cm, annot=True)
    plt.savefig('output.png')
    print(df_cm, '\n\n')
    print(classification_report(all_labels, all_predictions, target_names=classes))

if __name__ == '__main__':
    testSavedModel()