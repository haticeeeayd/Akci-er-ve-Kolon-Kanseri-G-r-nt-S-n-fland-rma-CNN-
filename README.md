# Akciğer ve Kolon Kanseri Görüntü Sınıflandırma (CNN)

Bu proje, histopatolojik mikroskop görüntülerinden akciğer ve kolon kanseri sınıflarını Konvolüsyonel Sinir Ağı (CNN) ile otomatik olarak sınıflandırır. Model; Keras/TensorFlow ile yazıldı, veri artırma (data augmentation), EarlyStopping ve ReduceLROnPlateau gibi düzenlileştirme/kontrol mekanizmaları kullanıldı.

Overifitting sorunu çözülemedi

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


## Model Değerlendirme Metrikleri

Modelin eğitim sürecinde yalnızca kayıp (loss) değerine bakmak yeterli değil; modelin gerçekten doğru tahminler yapıp yapmadığını anlamak için farklı değerlendirme metriklerine de bakıyorum. Bu metrikler sayesinde modelin performansını daha somut ve yorumlanabilir hale getiriyorum.

### Kullanılan Temel Metrikler

**Doğruluk (Accuracy):**  
En temel ve en çok kullanılan metriktir. Doğru sınıflandırılan örneklerin toplam tahminlere oranıdır.  
Modelin genel başarısını hızlıca görmek için kullanıyorum.

**Hassasiyet (Precision):**  
Modelin pozitif olarak sınıflandırdığı örneklerden ne kadarının gerçekten pozitif olduğunu ölçer.  
Yanlış pozitiflerin maliyetli olduğu senaryolarda önemlidir. (Bu projede yanlış sınıfa atanan akciğer görsellerini azaltmak için önemli.)

**Duyarlılık (Recall):**  
Gerçekte pozitif olan örneklerin ne kadarının doğru bir şekilde pozitif sınıflandırıldığını ölçer.  
Yanlış negatiflerin maliyetli olduğu durumlarda önemlidir. (Örneğin bir kanser görüntüsünü "normal" sanmak istemeyiz.)

**F1-Skor:**  
Hassasiyet ve Duyarlılığın harmonik ortalamasıdır. Precision ve Recall arasında bir denge sunar ve dengesiz veri setlerinde güvenilir bir metrik sağlar.

Bu metrikleri eğitim sonunda `classification_report` ve `confusion_matrix` ile görselleştirip her sınıf için ayrı ayrı Precision, Recall, F1-Score değerlerini inceliyorum. Böylece modelin hangi sınıflarda iyi, hangi sınıflarda daha zayıf performans gösterdiğini net bir şekilde görebiliyorum.




---

## Grad-CAM Görselleştirme
Modelin hangi bölgelerden etkilendiğini görmek için **Grad-CAM** uygulanmıştır.  
Bu sayede modelin karar verirken dikkate aldığı kritik alanlar görselleştirilmiş ve modelin açıklanabilirliği artırılmıştır.

---

##  Gelecek Çalışmalar
-Overifitting sorununun çözümü
- Veri setine daha fazla görüntü eklenerek genelleme başarısı artırılabilir.
- Transfer learning (ör. ResNet, EfficientNet) denenebilir.
- Model, gerçek zamanlı bir web arayüzü ile entegre edilerek kullanılabilir hale getirilebilir.

---

## 🔗 Kaggle Çalışması
Projeye ait Kaggle notebook bağlantısı buraya eklenebilir:  
📎 https://www.kaggle.com/code/haticeaydoan/akbank2

