'''
1. Supervised Learning ve Sınıflandırma
Denetimli öğrenmede modelimizi özellikler (features) ve bu özelliklere karşılık gelen etiketler (labels/targets) 
ile eğitiriz.

Eğer tahmin etmek istediğimiz etiket sürekli bir değerse buna Regresyon (Regression), belirli kategorilerden 
(örneğin: Spam / Spam Değil) oluşuyorsa buna **Sınıflandırma (Classification)** denir.

2. Veri Setini Bölme (Train/Test Split)
Bir modelin başarısını ölçmek için veriyi ikiye ayırırız:

Eğitim Seti (Training Set): Modelin kalıpları ve kuralları öğrendiği kısım (%70-%80).

Test Seti (Test Set): Modelin daha önce hiç görmediği veriler üzerinde ne kadar 
iyi tahmin yaptığını ölçtüğümüz kısım (%20-%30).

3. Scikit-Learn Standart İş Akışı (API Tasarımı)
Scikit-Learn kütüphanesindeki neredeyse tüm algoritmalar şu üç temel adımı takip eder:

Model Tanımlama: model = SınıflandırıcıAdı()

Modeli Eğitme (.fit): model.fit(X_train, y_train) -> Model özellikleri ve etiketleri öğrenir.

Tahmin Yapma (.predict): tahminler = model.predict(X_test) -> Model yeni veriler için etiket üretir.'''

# Gerekli kütüphaneleri içe aktarıyoruz
import numpy as np
from sklearn.datasets import load_iris
from sklearn.model_selection import train_test_split
from sklearn.neighbors import KNeighborsClassifier
from sklearn.metrics import accuracy_score

# GÖREV 1: Veriyi Yükleme
iris = load_iris()
X = iris.data  # Özellikler (Features)
y = iris.target  # Etiketler (Labels)

# GÖREV 2: Veriyi Train ve Test olarak bölme
# İpucu: train_test_split fonksiyonunu kullanın ve test_size ile random_state parametrelerini ayarlayın.
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

# GÖREV 3: Modeli Tanımlama
# İpucu: KNeighborsClassifier sınıfından n_neighbors=3 olan bir nesne oluşturun.
model = KNeighborsClassifier(n_neighbors=3)

# GÖREV 4: Modeli Eğitme
# İpucu: .fit() metodunu uygun verilerle çağırın.
model.fit(X_train, y_train)

# GÖREV 5: Tahmin Yapma ve Başarı Ölçme
# İpucu: .predict() metodu ile X_test üzerinden tahmin alın ve accuracy_score ile y_test ile karşılaştırın.
y_pred = model.predict(X_test)
basari_orani = accuracy_score(y_test, y_pred)

# Sonucu yazdırma
print(f"Modelin Doğruluk Skoru: {basari_orani}") ##Modelin Doğruluk Skoru: 1.0

#kullanılan dataset iris olduğu için doğruluk skoru 1.0 çıktı ama hayır overfit değil, 
#çünkü iris datasetinde sınıflar birbirinden oldukça ayrık ve kolayca sınıflandırılabilir.
#kanıt:

# Görev A: Modelin Eğitim Verileri Üzerindeki Tahmini
y_train_pred = model.predict(X_train)
train_basari = accuracy_score(y_train, y_train_pred)

# Görev B: Modelin Test Verileri Üzerindeki Tahmini (Zaten Yapmıştın)
y_test_pred = model.predict(X_test)
test_basari = accuracy_score(y_test, y_test_pred)

# Sonuçları Kıyaslama
print(f"Eğitim Seti (Train) Doğruluk Skoru : {train_basari:.4f}")
print(f"Test Seti (Test) Doğruluk Skoru     : {test_basari:.4f}")