# Bio-GPT: Hızlı Başlangıç Rehberi

## 🚀 5 Dakikalık Kurulum

### Adım 1: Depoyu Klonla
```bash
git clone https://github.com/yourusername/bio-gpt-cancer.git
cd bio-gpt-cancer
```

### Adım 2: Virtual Environment Oluştur
```bash
python3 -m venv venv
source venv/bin/activate  # Linux/Mac
# veya
venv\Scripts\activate  # Windows
```

### Adım 3: Bağımlılıkları Yükle
```bash
pip install -r requirements.txt
```

### Adım 4: Projeyi Çalıştır
```bash
# Tüm adımları sırayla çalıştır
python 01_data_preparation.py
python 02_transformer_model.py
python 03_attention_analysis.py
```

---

## 📊 Beklenen Sonuçlar

### Veri Hazırlama (01_data_preparation.py)
- ✅ 5000 hücre, 2000 gen veri seti oluşturulur
- ✅ 1000 yüksek varyans gen seçilir
- ✅ UMAP görselleştirmesi oluşturulur
- **Süre**: ~2-3 dakika

### Model Eğitimi (02_transformer_model.py)
- ✅ 1.1M parametreli Transformer modeli eğitilir
- ✅ **Test Accuracy: 99.10%**
- ✅ Eğitim grafikleri oluşturulur
- **Süre**: ~5-10 dakika

### Attention Analizi (03_attention_analysis.py)
- ✅ Gen önem sıralaması belirlenir
- ✅ Model tahmin güvenliği analiz edilir
- ✅ Embedding uzayı görselleştirilir
- **Süre**: ~2-3 dakika

---

## 📁 Çıktı Dosyaları

```
results/
├── 01_data_preparation.png          # Veri analizi görselleştirmesi
├── 02_transformer_training.png      # Model eğitimi grafikleri
├── 03_attention_analysis.png        # Attention analizi
├── gene_importance.csv              # Top 50 önemli gen
├── embedding_statistics.csv         # Embedding istatistikleri
├── prediction_confidence.csv        # Tahmin güvenliği
└── model_info.csv                   # Model parametreleri

data/
├── adata_full.h5ad                  # Tam veri seti (77 MB)
├── adata_hvg.h5ad                   # HVG veri seti (42 MB)
└── metadata.csv                     # Veri seti metadatası

models/
└── best_model.pth                   # Eğitilmiş model (4.3 MB)
```

---

## 💻 Sistem Gereksinimleri

| Gereksinim | Minimum | Önerilen |
|-----------|---------|----------|
| **RAM** | 8 GB | 16 GB |
| **Disk** | 500 MB | 2 GB |
| **CPU** | 2 cores | 4+ cores |
| **GPU** | Opsiyonel | NVIDIA (CUDA) |
| **Python** | 3.8+ | 3.10+ |

---

## 🔧 Sorun Giderme

### Problem: "ModuleNotFoundError: No module named 'torch'"
**Çözüm:**
```bash
pip install --upgrade torch transformers
```

### Problem: "CUDA out of memory"
**Çözüm:**
- GPU kullanmadan CPU'da çalıştır (otomatik)
- Batch size'ı azalt (02_transformer_model.py'da `batch_size = 16`)

### Problem: "Permission denied" (Linux/Mac)
**Çözüm:**
```bash
chmod +x *.py
```

---

## 📚 Dosya Açıklamaları

### 01_data_preparation.py
- Sentetik scRNA-seq veri seti oluşturur
- Kalite kontrol ve normalizasyon yapar
- PCA ve UMAP boyut indirgeme uygular
- **Çıktı**: `data/adata_hvg.h5ad`

### 02_transformer_model.py
- Transformer mimarisi tanımlar
- Modeli eğitir ve değerlendirir
- Early stopping ile overfitting'i önler
- **Çıktı**: `models/best_model.pth`

### 03_attention_analysis.py
- Attention mekanizmasını analiz eder
- Gen önem sıralaması belirler
- Tahmin güvenliğini hesaplar
- **Çıktı**: `results/gene_importance.csv`

---

## 🎯 Sonraki Adımlar

### 1. Gerçek Veri Kullan
```python
# GEO'dan veri indir
import GEOparse
gse = GEOparse.get_GEO(geo='GSE161529')
```

### 2. Model Parametrelerini Ayarla
```python
# 02_transformer_model.py'da değiştir
num_layers = 6  # 4'ten 6'ya
embedding_dim = 256  # 128'den 256'ya
```

### 3. Yeni Hücre Türleri Ekle
```python
# 01_data_preparation.py'da değiştir
n_cell_types = 6  # 4'ten 6'ya
```

### 4. Web Arayüzü Oluştur
```bash
pip install fastapi uvicorn
# API geliştir ve deploy et
```

---

## 📖 Kaynaklar

- **PyTorch Tutorials**: https://pytorch.org/tutorials/
- **Scanpy Documentation**: https://scanpy.readthedocs.io/
- **Transformer Paper**: https://arxiv.org/abs/1706.03762
- **scRNA-seq Best Practices**: https://www.nature.com/articles/s41576-023-00586-x

---

## 🤝 Katkı Yapın

Geliştirmeleri ve hata düzeltmelerini pull request olarak gönderin!

---

## 📝 Lisans

MIT License - Detaylar için LICENSE dosyasına bakın.

---

**Başarılar! 🚀**

Sorular veya sorunlar için GitHub Issues açın.
