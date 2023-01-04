import numpy as np
import pandas as pd
from sklearn import metrics
from sklearn import datasets
from sklearn.model_selection import train_test_split
from sklearn import neural_network

#Veri okuması yapılır.
input=pd.read_csv(r"C:\Users\mehme\Desktop\Yapay Sinir Ağları MLP\girdi.csv")
print(input)
print("------------------------------------------------")
#Veriler array formatına dönüştürülür.
input_array=(input.values)
print(input_array)
#Veriler 300 sample olacak şekilde ve her sample 5 eleman içerecek şekilde boyutlandırılır.
X=input_array.reshape(300,5)
print(X)
print(X.shape)
print("------------------------------------------------")

#Veri okuması yapılır.
output=pd.read_csv(r"C:\Users\mehme\Desktop\Yapay Sinir Ağları MLP\cikti.csv")
print(output)
print("------------------------------------------------")
#Veriler array formatına dönüştürülür.
output_array=(output.values)
print(output_array)
#Veriler 300 sample olacak şekilde boyutlandırılır.
Y=output_array.reshape(300,1)
print(Y)
print(Y.shape)
print("------------------------------------------------")

x_train,x_test,y_train,y_test=train_test_split(X,Y,test_size=0.25)
print("X Eğitim:",x_train)
print(x_train.shape)
print("------------------------------------------------")
print("X Test:",x_test)
print(x_test.shape)
print("------------------------------------------------")
print("Y Eğitim:",y_train)
print(y_train.shape)
print("------------------------------------------------")
print("Y Test:",y_test)
print(y_test.shape)

clf=neural_network.MLPClassifier(hidden_layer_sizes=50,max_iter=40,momentum=0.5,solver="adam")
clf.fit(x_train,y_train)

y_pred=clf.predict(x_test)

#Doğruluk Oranı
print("Tahmin Edilen Y:",y_pred)
print("Test Verisi   Y:",y_test)
acs=metrics.accuracy_score(y_test,y_pred)
print("Doğruluk Oranı:",acs)

