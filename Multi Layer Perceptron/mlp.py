#%% import dataset
import pandas as pd
import numpy as np
from model import multilayer

Data = pd.read_csv("iris.csv")
#%% 
X = Data.iloc[:, :-1].values
y = Data.iloc[:, -1].values
# 'setosa', 'versicolor', 'virginica'
class_mapping = {label: index for index, label in enumerate(np.unique(y))}
y = np.array([class_mapping[cls] for cls in y])
#%% split dataset
from sklearn.model_selection import train_test_split
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)
datasets = np.array(Data)
#%% 
input_size = X.shape[1]  #jumlah fitur
hidden_size = 8  # hidden_size yang diinginkan
output_size = len(np.unique(y))  #jumlah class
mlp = multilayer(input_size, hidden_size, output_size)
mlp.train(X_train, y_train.reshape(-1, 1)) 
#%% prediksi 
predictions = mlp.predict(X_test)

def threshold_predictions(predictions):
    return np.argmax(predictions, axis=1)
# print(predictions)
# mengubah prediction ke predlabel
predicted_labels = threshold_predictions(predictions)
#%% akurasi 
from sklearn.metrics import accuracy_score
akurasi = accuracy_score(y_test, predicted_labels) * 100
print(f"Accuracy: {akurasi:.2f}%")

#%% plotting
import matplotlib.pyplot as plt

plt.scatter(X_train[:, 0], X_train[:, 1], marker="o", edgecolors="black", c=y_train, s=30, alpha=0.5, label='Training')
plt.scatter(X_test[:, 0], X_test[:, 1], marker="x", c=y_test, s=100, label='Testing')
plt.xlabel('Sepal length')
plt.ylabel('sepal width')
plt.legend()

plt.show()