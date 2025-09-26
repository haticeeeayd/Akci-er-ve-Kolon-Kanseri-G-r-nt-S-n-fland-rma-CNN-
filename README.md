# Akciğer ve Kolon Kanseri Görüntü Sınıflandırma (CNN)

Bu proje, histopatolojik mikroskop görüntülerinden akciğer ve kolon kanseri sınıflarını Konvolüsyonel Sinir Ağı (CNN) ile otomatik olarak sınıflandırır. Model; Keras/TensorFlow ile yazıldı, veri artırma (data augmentation), EarlyStopping ve ReduceLROnPlateau gibi düzenlileştirme/kontrol mekanizmaları kullanıldı.

Son deneyimizde test doğruluğu ~%95 elde edildi.

# Akciğer ve Kolon Kanseri Histopatolojik Görüntü Sınıflandırması (CNN)

Bu proje, LC25000 veri seti kullanılarak akciğer ve kolon kanseri hücre tiplerini sınıflandırmak için bir Convolutional Neural Network (CNN) modeli geliştirmeyi amaçlamaktadır. Çalışma Kaggle üzerinde gerçekleştirilmiş ve TensorFlow/Keras kullanılarak uygulanmıştır.

---

##  Veri Seti
Projede kullanılan veri seti: **Lung and Colon Cancer Histopathological Images (LC25000)**  
Bu veri setinde 5 sınıf bulunmaktadır:
- **colon_aca**: Kolon Adenokarsinom
- **colon_n**: Normal Kolon Doku
- **lung_aca**: Akciğer Adenokarsinom
- **lung_n**: Normal Akciğer Doku
- **lung_scc**: Akciğer Yassı Hücreli Karsinom  

Veri seti train/validation/test olarak ayrılmış ve `ImageDataGenerator` ile veri arttırma (data augmentation) yapılmıştır.
https://www.kaggle.com/datasets/andrewmvd/lung-and-colon-cancer-histopathological-images/data

---



##  Metrikler
Projenizde gerçekleştirdiğimiz çalışma sonucunda aşağıdaki metrikler elde edilmiştir.  
Bu metrikler, modelin yalnızca kod seviyesinde değil, anlam olarak da doğru bir şekilde çalıştığını göstermektedir.  


- Eğitim ve doğrulama süresince **Accuracy** ve **Loss** grafikleri incelenmiştir.
- Modelin overfitting yapmadığı, validation accuracy değerinin yüksek olduğu gözlenmiştir.
- Confusion Matrix ve Classification Report yardımıyla sınıf bazında Precision, Recall ve F1-Score değerlendirilmiştir.
- Özellikle **lung_aca** ve **lung_scc** sınıflarında az miktarda karışıklık gözlense de genel başarı yüksektir (%97 accuracy).

---

##  Sonuçlar

###  Confusion Matrix & Classification Report
Modelin test seti üzerindeki performansı:

| Sınıf      | Precision | Recall | F1-Score |
|-----------|-----------|--------|----------|
| colon_aca | 1.00      | 0.99   | 0.99 |
| colon_n   | 1.00      | 1.00   | 1.00 |
| lung_aca  | 0.90      | 0.96   | 0.92 |
| lung_n    | 1.00      | 1.00   | 1.00 |
| lung_scc  | 0.95      | 0.90   | 0.92 |

**Genel doğruluk (Accuracy): %97**

---

## Grad-CAM Görselleştirme
Modelin hangi bölgelerden etkilendiğini görmek için **Grad-CAM** uygulanmıştır.  
Bu sayede modelin karar verirken dikkate aldığı kritik alanlar görselleştirilmiş ve modelin açıklanabilirliği artırılmıştır.

---

##  Gelecek Çalışmalar
- Veri setine daha fazla görüntü eklenerek genelleme başarısı artırılabilir.
- Transfer learning (ör. ResNet, EfficientNet) denenebilir.
- Model, gerçek zamanlı bir web arayüzü ile entegre edilerek kullanılabilir hale getirilebilir.

---

## 🔗 Kaggle Çalışması
Projeye ait Kaggle notebook bağlantısı buraya eklenebilir:  
📎 https://www.kaggle.com/code/haticeaydoan/akbank2

