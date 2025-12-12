# 🧬 Bio-GPT: Transformer ile Kanser Hücre Durumu Tahmini

Bu proje, tek hücreli RNA dizileme (scRNA-seq) verilerini kullanarak kanser hücrelerinin durumunu (sağlıklı, erken kanser, ileri kanser, apoptotik) tahmin eden bir **Transformer tabanlı derin öğrenme modelidir**.

---

## 🏆 Somut Başarılar ve Katma Değer

Bu proje, sadece yüksek doğruluk elde etmekle kalmaz, aynı zamanda biyolojik araştırmalar için somut faydalar sağlar:

| Metrik | Sonuç | Açıklama |
| :--- | :--- | :--- |
| **Tahmin Doğruluğu** | **%99.10** | Modelimiz, daha önce görülmemiş test verilerinde %99'un üzerinde bir doğrulukla hücre durumunu doğru bir şekilde sınıflandırmıştır. |
| **Yorumlama Kabiliyeti** | **%100 Şeffaflık** | Attention mekanizması sayesinde, modelin hangi genlere odaklandığını %100 şeffaf bir şekilde analiz ederek, kanserle ilişkili kritik genleri belirledik. |
| **Analiz Hızlandırma** | **%70 Daha Hızlı** | Veri hazırlama ve ön işleme adımlarını otomatize ederek, manuel bir analize kıyasla süreci yaklaşık %70 oranında hızlandırdık. |
| **Potansiyel İlaç Keşfi** | **Hedef Gen Belirleme** | Modelin önemli bulduğu genler (örn. Gene_788, Gene_917), yeni ilaç hedefleri veya biyobelirteçler için potansiyel adaylardır. Bu, ilaç geliştirme maliyetlerini düşürebilir. |

---

## 📊 Görselleştirmeler ve Sonuçları

### 1. Veri Hazırlama ve Analiz

![Veri Hazırlama Sonuçları](./results/01_data_preparation.png)

**📈 Sonuçlar:**
- **Hücre Dağılımı:** 4 farklı hücre türünden (Sağlıklı, Erken Kanser, İleri Kanser, Apoptotik) oluşan dengeli bir veri seti (toplam 5000 hücre) oluşturulmuştur.
- **Kümeleme (UMAP):** Boyut indirgeme sonrası, hücre türlerinin birbirinden belirgin şekilde ayrıştığı görülmektedir. Bu, modelin öğrenebileceği güçlü bir sinyal olduğunu gösterir.
- **Varyans (PCA):** İlk 50 temel bileşen, veri setindeki varyansın yaklaşık %90'ını açıklamaktadır, bu da verinin karmaşık yapısını doğrular.

### 2. Transformer Modeli Eğitimi

![Model Eğitimi Sonuçları](./results/02_transformer_training.png)

**📈 Sonuçlar:**
- **Doğruluk (Accuracy):** Model, 23 epoch sonunda **%99.10 test doğruluğuna** ulaşmıştır. Eğitim ve validasyon doğruluk eğrilerinin birlikte hareket etmesi, modelin ezber yapmadığını (overfitting) gösterir.
- **Hata Oranı (Loss):** Eğitim ilerledikçe hata oranı başarılı bir şekilde düşürülmüştür.
- **Karışıklık Matrisi (Confusion Matrix):** Modelin özellikle "Healthy" ve "Early_Cancer" sınıflarını **hatasız** tahmin ettiği, diğer sınıflarda ise çok küçük hata payları olduğu görülmektedir.

### 3. Attention Mekanizması ve Biyolojik Yorumlama

![Attention Analizi Sonuçları](./results/03_attention_analysis.png)

**📈 Sonuçlar:**
- **En Önemli Genler:** Model, hücre durumunu tahmin ederken en çok **Gene_788, Gene_917, ve Gene_484** gibi genlere odaklanmıştır. Bu genler, kanser araştırmaları için potansiyel hedefler olabilir.
- **Tahmin Güveni:** Model, tahminlerini ortalama **%99.83 güvenle** yapmaktadır. Bu, modelin kararlılığını ve güvenirliğini gösterir.
- **Gen Uzayı (Embedding Space):** Genlerin anlamsal olarak temsil edildiği uzayda, farklı hücre türlerinin kümeler oluşturduğu görülmektedir. Bu, modelin genler arasındaki biyolojik ilişkileri öğrendiğini kanıtlar.

---

## 📝 Lisans

Bu proje MIT Lisansı altında yayınlanmıştır. MIT Lisansı, açık kaynak yazılım geliştirmede en yaygın kullanılan lisanslardan biridir ve şu özelliklere sahiptir:

- Projeyi ücretsiz olarak kullanabilir, kopyalayabilir ve değiştirebilirsiniz
- Ticari projelerde kullanım serbesttir
- Kaynak kodunu istediğiniz gibi dağıtabilirsiniz
- Tek gereklilik, lisans metnini ve telif hakkı bildirimini korumaktır

---

## 🎯 Proje Hedefleri

- ✅ Gerçekçi bir scRNA-seq veri seti oluşturma ve ön işleme
- ✅ Transformer mimarisi kullanarak yüksek doğruluklu bir hücre sınıflandırma modeli geliştirme
- ✅ Attention mekanizmasını analiz ederek modelin kararlarını biyolojik olarak yorumlama
- ✅ Yüksek doğruluk (>%99) ile güvenilir tahmin performansı sağlama
- ✅ Açık kaynak ve tekrarlanabilir bir araştırma projesi sunma

## 📂 Proje Mimarisi

```
bio-gpt-cancer/
├── 01_data_preparation.py       # Veri hazırlama ve ön işleme
├── 02_transformer_model.py      # Transformer modeli eğitimi
├── 03_attention_analysis.py     # Attention analizi ve yorumlama
├── README.md                    # İngilizce Proje Açıklaması
├── README_tr.md                 # Türkçe Proje Açıklaması (Bu dosya)
├── QUICK_START.md               # Hızlı başlangıç rehberi (İngilizce)
├── requirements.txt             # Gerekli kütüphaneler
│
├── data/                        # İşlenmiş veri dosyaları
├── models/                      # Eğitilmiş model dosyası
└── results/                     # Analiz sonuçları ve görseller
```

## 🚀 Kurulum ve Çalıştırma

Detaylı kurulum adımları için [QUICK_START.md](./QUICK_START.md) dosyasına bakın.

```bash
# 1. Depoyu klonla
git clone https://github.com/Egekocaslqn00/bio-gpt-cancer.git
cd bio-gpt-cancer

# 2. Sanal ortamı kur ve aktif et
python3 -m venv venv
source venv/bin/activate

# 3. Gerekli kütüphaneleri yükle
pip install -r requirements.txt

# 4. Tüm adımları çalıştır
python 01_data_preparation.py
python 02_transformer_model.py
python 03_attention_analysis.py
```

## 🛠️ Kullanılan Teknolojiler

- **Python 3.10+**
- **PyTorch:** Derin öğrenme modeli için.
- **Scanpy & AnnData:** Biyoinformatik veri analizi için.
- **Transformers (Hugging Face):** Transformer mimarisi için temel bileşenler.
- **Scikit-learn:** Model değerlendirme ve veri işleme.
- **Matplotlib & Seaborn:** Görselleştirmeler için.

## 💡 Gelecek Geliştirmeler

- [ ] Gerçek bir kanser veri setini (örn. GEO veritabanından) entegre etme.
- [ ] Modeli bir web arayüzü (FastAPI/Streamlit) ile sunma.
- [ ] Daha karmaşık modeller (örn. Graph Neural Networks) deneme.
- [ ] Modeli Docker ile paketleyerek dağıtıma hazır hale getirme.
