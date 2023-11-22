#%% import dataset iris
import pandas as pd
Data = pd.read_csv('iris.csv')

#%% Testing
import numpy as np
datasets = np.array(Data)
Testing = pd.concat([Data[0:10],Data[50:60]])
Xtest = np.array(Testing[['sepal length (cm)', 'sepal width (cm)']])
Ytest = np.concatenate((datasets[0:10,4],datasets[50:60,4]))
#%% Training
Training = pd.concat([Data[10:50],Data[60:100]])
Xtrain = np.array(Training[['sepal length (cm)', 'sepal width (cm)']])
Ytrain = np.concatenate((datasets[10:50,4],datasets[60:100,4]))
#%% 
from Perceptron import perceptron

p = perceptron(learning_rate=0.02, iterasi=1100)
p.train(Xtrain,Ytrain)
prediction = p.prediksi(Xtest)

def accuracy(y_true, y_pred):
    accuracy = np.sum(y_true == y_pred) / len(y_true)*100
    return accuracy

akurasi = accuracy(Ytest, prediction)
print(akurasi)
#%% plotting 
import matplotlib.pyplot as plt
plt.scatter(Xtrain[:, 0], Xtrain[:, 1], marker="o", edgecolors="black", c=Ytrain, s=30, alpha=0.5)
plt.scatter(Xtest[:, 0], Xtest[:, 1], marker="x", c=Ytest, s=100)
plt.xlabel('Sepal Length (cm)')
plt.ylabel('Sepal Width (cm)')

x0_1 = np.amin(Xtrain[:, 0])
x0_2 = np.amax(Xtrain[:, 0])

x1_1 = (-p.weight[0]*x0_1 - p.ias)/p.weight[1]
x1_2 = (-p.weight[0]*x0_2 - p.bias)/p.weight[1]
plt.plot([x0_1, x0_2], [x1_1, x1_2], "r")
plt.show()

#%%
