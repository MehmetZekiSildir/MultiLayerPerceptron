#Gerekli Modüllerin,Bu Modüllere Bağlı Metod ve Fonksiyonların Eklenmesi

from sklearn import metrics
from sklearn import datasets
from sklearn.model_selection import train_test_split
from sklearn import neural_network

#Veri Setlerini Random Bir Şekilde Oluşturma

#x:Girdi y:Çıktı verileri make_classification fonksiyonu ile 100 adet random olarak üretilmiştir.
x,y=datasets.make_classification(n_features=5,n_classes=2,n_samples=100)
#Üretilen girdi ve çıktı kendi içinde de eğitim veirleri ve test verileri olarak ayırıyoruz.
x_train,x_test,y_train,y_test=train_test_split(x,y)
print("X Eğitim Verileri:",end="\n")
print(x_train)
print(x_train.shape)#Shape metodu bize Verinin boyutunu verir.

print("---------------------------------------------------------------")
print("X Test Verileri:",end="\n")
print(x_test)
print(x_test.shape)
print("---------------------------------------------------------------")
print("Y Eğitim Verileri:",end="\n")
print(y_train)
print(y_train.shape)
print("---------------------------------------------------------------")
print("Y Test Verileri:",end="\n")
print(y_test)
print(y_test.shape)
print("---------------------------------------------------------------")


#Model Eğitimi
#Model Eğitimi için bir iterasyon değeri girilir.Bu metod  bir değişkene atanır.
clf=neural_network.MLPClassifier(max_iter=30,momentum=0.8)
#Fit fonksiyonu eğitim setlerini parametre olarak alarak ağı eğitmeye çalışır.Koşul olarak girilen max_iter alınır.
clf.fit(x_train,y_train)

#Tahmin
#Predict metoduna x_test parametresi verilerek y_pred üretliyor.
y_pred=clf.predict(x_test)

#Doğruluk Oranı
print("Tahmin Edilen Y:",y_pred)
print("Test Verisi   Y:",y_test)
acs=metrics.accuracy_score(y_test,y_pred)
print("Doğruluk Oranı:",acs)

