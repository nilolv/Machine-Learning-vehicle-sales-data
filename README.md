
# Araç Satış Verisi Analizi

Bu depo, araç satış verilerini çeşitli makine öğrenimi algoritmaları kullanarak kapsamlı bir şekilde analiz etmektedir. Bu projede kullanılan veri seti [Kaggle](https://www.kaggle.com/datasets/syedanwarafridi/vehicle-sales-data) kaynağından alınmıştır.Burdan ulaşabilirsiniz



 ## Kaggle 
Projemi burdan görüntüleyebilirsiniz https://www.kaggle.com/code/nihalzerenkuruoglu/ml-firstproject1 




## İçindekiler
- [Giriş](#giriş)
- [Kurulum](#kurulum)
- [Veri Açıklaması](#veri-açıklaması)
- [Veri Ön İşleme](#veri-ön-işleme)
- [Keşifsel Veri Analizi](#keşifsel-veri-analizi)
- [Modelleme](#modelleme)
  - [Doğrusal Regresyon](#doğrusal-regresyon)
  - [Karar Ağaçları](#karar-ağaçları)
  - [K-Means Kümeleme](#k-means-kümeleme)
- [Sonuçlar](#sonuçlar)
- [Sonuç](#sonuç)

## Giriş
Bu proje, araç satış verilerini analiz ederek araçların satış fiyatlarını denetimli öğrenme algoritmaları kullanarak tahmin etmeyi ve denetimsiz öğrenme algoritmaları kullanarak desenleri belirlemeyi amaçlamaktadır.

## Kurulum
Bu projeyi çalıştırmak için Python 3 ve aşağıdaki kütüphanelerin kurulu olması gerekmektedir:
```python
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.model_selection import train_test_split, GridSearchCV
from sklearn.linear_model import LinearRegression
from sklearn.metrics import mean_squared_error, mean_absolute_error
from sklearn.preprocessing import LabelEncoder
from sklearn.tree import DecisionTreeRegressor
from sklearn.cluster import KMeans
```

## Veri Açıklaması
Veri seti, araçların marka, model, donanım, gövde, şanzıman, eyalet, renk, iç mekan ve satıcı gibi çeşitli özelliklerini içermektedir. Hedef değişken ise araçların satış fiyatıdır.

## Veri Ön İşleme
1. **Veriyi Yükleme**:
    ```python
    df = pd.read_csv('/kaggle/input/vehicle-sales-data/car_prices.csv')
    print(df.head())
    ```

2. **Veri Temizleme**:
    - Eksik değerleri kontrol etme:
        ```python
        print(df.info())
        print(df.describe())
        print(df.isnull().sum())
        ```
    - Eksik değerleri görselleştirme:
        ```python
        plt.figure(figsize=(12, 6))
        sns.heatmap(df.isnull(), cbar=False, cmap='viridis')
        plt.title('Eksik Veriler')
        plt.show()
        ```
    - Eksik değerleri kaldırma:
        ```python
        df.dropna(inplace=True)
        ```

3. **Kategorik Değişkenleri Kodlama**:
    ```python
    le = LabelEncoder()
    kategorik_sütunlar = ['make', 'model', 'trim', 'body', 'transmission', 'state', 'color', 'interior', 'seller']
    for sütun in kategorik_sütunlar:
        df[sütun] = le.fit_transform(df[sütun])
    ```

4. **Veriyi Bölme**:
    ```python
    X = df.drop('sellingprice', axis=1)
    y = df['sellingprice']
    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)
    ```

## Keşifsel Veri Analizi
Özelliklerin ve hedef değişkenin dağılımını görselleştirme.

## Modelleme

### Doğrusal Regresyon
1. **Modeli Eğitme**:
    ```python
    lr_model = LinearRegression()
    lr_model.fit(X_train, y_train)
    ```
2. **Tahmin Yapma**:
    ```python
    y_pred_lr = lr_model.predict(X_test)
    ```
3. **Modeli Değerlendirme**:
    ```python
    print("Lineer Regresyon - Ortalama Karesel Hata:", mean_squared_error(y_test, y_pred_lr))
    print("Lineer Regresyon - Ortalama Mutlak Hata:", mean_absolute_error(y_test, y_pred_lr))
    ```

### Karar Ağaçları
1. **Modeli Eğitme**:
    ```python
    dt_model = DecisionTreeRegressor()
    dt_model.fit(X_train, y_train)
    ```
2. **Tahmin Yapma**:
    ```python
    y_pred_dt = dt_model.predict(X_test)
    ```
3. **Modeli Değerlendirme**:
    ```python
    print("Karar Ağaçları - Ortalama Karesel Hata:", mean_squared_error(y_test, y_pred_dt))
    print("Karar Ağaçları - Ortalama Mutlak Hata:", mean_absolute_error(y_test, y_pred_dt))
    ```

### K-Means Kümeleme
1. **Modeli Eğitme**:
    ```python
    kmeans = KMeans(n_clusters=3, n_init=10)
    X_kmeans = X.select_dtypes(include=[np.number])
    kmeans.fit(X_kmeans)
    df['cluster'] = kmeans.labels_
    ```
2. **Kümeleri Görselleştirme**:
    ```python
    plt.figure(figsize=(12, 6))
    sns.scatterplot(data=df, x='odometer', y='sellingprice', hue='cluster', palette='viridis')
    plt.title('K-Means Kümeleme Sonuçları')
    plt.show()
    ```

## Sonuçlar
- **Doğrusal Regresyon**:
    - Ortalama Karesel Hata: `mean_squared_error(y_test, y_pred_lr)`
    - Ortalama Mutlak Hata: `mean_absolute_error(y_test, y_pred_lr)`

- **Karar Ağaçları**:
    - Ortalama Karesel Hata: `mean_squared_error(y_test, y_pred_dt)`
    - Ortalama Mutlak Hata: `mean_absolute_error(y_test, y_pred_dt)`

- **K-Means Kümeleme**:
    - Küme boyutları: `df['cluster'].value_counts()`

## Sonuç
Bu proje, araç satış fiyatlarını tahmin etmek ve verilerdeki desenleri belirlemek için çeşitli makine öğrenimi algoritmalarının uygulanmasını göstermektedir. Sonuçlar, doğrusal regresyon ve karar ağaçlarının fiyat tahmininde etkili olduğunu, K-Means kümelemenin ise verilerdeki farklı grupları belirlemede yardımcı olduğunu göstermektedir.
