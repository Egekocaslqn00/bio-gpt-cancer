"""
Bio-GPT: Cancer Cell State Prediction using Transformer Models
Phase 1: Data Preparation and Preprocessing

Bu script, kanser hücre hattı scRNA-seq verilerini indirerek ön işleme yapar.
"""

import numpy as np
import pandas as pd
import scanpy as sc
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.preprocessing import StandardScaler
from sklearn.decomposition import PCA
import warnings
warnings.filterwarnings('ignore')

# Stil ayarları
sns.set_style("whitegrid")
plt.rcParams['figure.figsize'] = (12, 8)

print("=" * 80)
print("BIO-GPT: KANSER HÜCRE DURUMU TAHMİN MODELİ")
print("Faz 1: Veri Hazırlama ve Ön İşleme")
print("=" * 80)

# ============================================================================
# ADIM 1: Sentetik Veri Oluşturma (Gerçek scRNA-seq verisi simülasyonu)
# ============================================================================
print("\n[1/5] Sentetik scRNA-seq veri seti oluşturuluyor...")

np.random.seed(42)

# Parametreler
n_cells = 5000  # Hücre sayısı
n_genes = 2000  # Gen sayısı
n_cell_types = 4  # Hücre türü sayısı

# Hücre türleri: 0=Sağlıklı, 1=Erken Kanser, 2=İleri Kanser, 3=Apoptotik
cell_types = np.random.choice([0, 1, 2, 3], size=n_cells, p=[0.3, 0.3, 0.25, 0.15])

# Gen ekspresyon matrisi (log-transformed, sparse)
X = np.zeros((n_cells, n_genes))

# Her hücre türü için farklı gen ekspresyon profilleri
for cell_type in range(n_cell_types):
    mask = cell_types == cell_type
    n_type_cells = mask.sum()
    
    # Hücre türüne özgü gen ekspresyonu
    base_expression = np.random.negative_binomial(5, 0.3, size=(n_type_cells, n_genes))
    
    # Hücre türüne göre belirli genleri vurgula
    marker_genes = np.random.choice(n_genes, size=100, replace=False)
    base_expression[:, marker_genes] *= (2 + cell_type)
    
    X[mask] = base_expression

# Log transformasyon (scRNA-seq standardı)
X = np.log1p(X)

# Gürültü ekle (gerçekçi scRNA-seq verisi)
noise = np.random.normal(0, 0.1, X.shape)
X = X + noise
X = np.maximum(X, 0)  # Negatif değerleri sıfırla

# AnnData nesnesi oluştur
adata = sc.AnnData(X)
adata.obs['cell_type'] = pd.Categorical(
    ['Healthy', 'Early_Cancer', 'Advanced_Cancer', 'Apoptotic'][i] 
    for i in cell_types
)
adata.var_names = [f'Gene_{i}' for i in range(n_genes)]

print(f"   ✓ {n_cells} hücre ve {n_genes} gen ile veri seti oluşturuldu")
print(f"   ✓ Hücre türleri: {dict(adata.obs['cell_type'].value_counts())}")

# ============================================================================
# ADIM 2: Kalite Kontrol (Quality Control)
# ============================================================================
print("\n[2/5] Kalite kontrol uygulanıyor...")

# Hücre başına gen sayısı
adata.obs['n_genes'] = (adata.X > 0).sum(axis=1)
# Hücre başına toplam ekspresyon
adata.obs['total_counts'] = adata.X.sum(axis=1)

# Düşük kaliteli hücreleri filtrele
min_genes = 100
max_genes = 1500
min_counts = 500

initial_cells = adata.n_obs
sc.pp.filter_cells(adata, min_genes=min_genes)
sc.pp.filter_cells(adata, min_counts=min_counts)

print(f"   ✓ Filtreleme sonrası: {initial_cells} → {adata.n_obs} hücre")

# Gen filtreleme (en az 3 hücrede eksprese olan genler)
sc.pp.filter_genes(adata, min_cells=3)
print(f"   ✓ Gen filtreleme: {n_genes} → {adata.n_vars} gen")

# ============================================================================
# ADIM 3: Normalizasyon ve Log Transformasyon
# ============================================================================
print("\n[3/5] Normalizasyon uygulanıyor...")

# Library size normalizasyon
sc.pp.normalize_total(adata, target_sum=1e4)
print("   ✓ Library size normalizasyonu tamamlandı")

# Log transformasyon
sc.pp.log1p(adata)
print("   ✓ Log transformasyonu tamamlandı")

# ============================================================================
# ADIM 4: Boyut İndirgeme
# ============================================================================
print("\n[4/5] Boyut indirgeme uygulanıyor...")

# Yüksek varyans genleri seç
sc.pp.highly_variable_genes(adata, n_top_genes=1000)
adata_hvg = adata[:, adata.var['highly_variable']]
print(f"   ✓ {adata_hvg.n_vars} yüksek varyans gen seçildi")

# PCA
sc.tl.pca(adata_hvg, n_comps=50)
print("   ✓ PCA uygulandı (50 bileşen)")

# UMAP (görselleştirme için)
sc.pp.neighbors(adata_hvg, n_neighbors=15, n_pcs=30)
sc.tl.umap(adata_hvg)
print("   ✓ UMAP uygulandı")

# ============================================================================
# ADIM 5: Veri Kaydetme
# ============================================================================
print("\n[5/5] Veriler kaydediliyor...")

# Tam veri seti
adata.write_h5ad('data/adata_full.h5ad')
print("   ✓ Tam veri seti kaydedildi: data/adata_full.h5ad")

# HVG veri seti (model eğitimi için)
adata_hvg.write_h5ad('data/adata_hvg.h5ad')
print("   ✓ HVG veri seti kaydedildi: data/adata_hvg.h5ad")

# Metadata
metadata = pd.DataFrame({
    'n_cells': [adata.n_obs],
    'n_genes': [adata.n_vars],
    'n_hvg': [adata_hvg.n_vars],
    'cell_types': [', '.join(adata.obs['cell_type'].unique())]
})
metadata.to_csv('data/metadata.csv', index=False)
print("   ✓ Metadata kaydedildi: data/metadata.csv")

# ============================================================================
# VİZÜALİZASYON
# ============================================================================
print("\n[VİZÜALİZASYON] Sonuçlar görselleştiriliyor...")

fig, axes = plt.subplots(2, 2, figsize=(14, 12))

# 1. Hücre türü dağılımı
ax1 = axes[0, 0]
adata.obs['cell_type'].value_counts().plot(kind='bar', ax=ax1, color='steelblue')
ax1.set_title('Hücre Türü Dağılımı', fontsize=12, fontweight='bold')
ax1.set_ylabel('Hücre Sayısı')
ax1.set_xlabel('Hücre Türü')
ax1.tick_params(axis='x', rotation=45)

# 2. Hücre başına gen sayısı
ax2 = axes[0, 1]
ax2.hist(adata.obs['n_genes'], bins=50, color='coral', edgecolor='black', alpha=0.7)
ax2.set_title('Hücre Başına Gen Sayısı Dağılımı', fontsize=12, fontweight='bold')
ax2.set_xlabel('Gen Sayısı')
ax2.set_ylabel('Hücre Sayısı')
ax2.axvline(adata.obs['n_genes'].mean(), color='red', linestyle='--', label=f'Ortalama: {adata.obs["n_genes"].mean():.0f}')
ax2.legend()

# 3. UMAP (hücre türlerine göre renklendirme)
ax3 = axes[1, 0]
cell_type_colors = {'Healthy': '#2ecc71', 'Early_Cancer': '#f39c12', 
                    'Advanced_Cancer': '#e74c3c', 'Apoptotic': '#95a5a6'}
for cell_type in adata_hvg.obs['cell_type'].unique():
    mask = adata_hvg.obs['cell_type'] == cell_type
    ax3.scatter(adata_hvg.obsm['X_umap'][mask, 0], 
               adata_hvg.obsm['X_umap'][mask, 1],
               label=cell_type, s=30, alpha=0.7, 
               color=cell_type_colors.get(cell_type, 'gray'))
ax3.set_title('UMAP: Hücre Türlerine Göre Kümeleme', fontsize=12, fontweight='bold')
ax3.set_xlabel('UMAP 1')
ax3.set_ylabel('UMAP 2')
ax3.legend()

# 4. PCA varyans açıklaması
ax4 = axes[1, 1]
variance_ratio = adata_hvg.uns['pca']['variance_ratio']
cumsum_variance = np.cumsum(variance_ratio)
ax4.plot(cumsum_variance[:50], marker='o', linewidth=2, markersize=4, color='steelblue')
ax4.axhline(0.9, color='red', linestyle='--', label='90% varyans')
ax4.set_title('PCA: Kümülatif Varyans Açıklaması', fontsize=12, fontweight='bold')
ax4.set_xlabel('PC Sayısı')
ax4.set_ylabel('Kümülatif Varyans Oranı')
ax4.legend()
ax4.grid(True, alpha=0.3)

plt.tight_layout()
plt.savefig('results/01_data_preparation.png', dpi=300, bbox_inches='tight')
print("   ✓ Görselleştirme kaydedildi: results/01_data_preparation.png")
plt.close()

# ============================================================================
# ÖZET
# ============================================================================
print("\n" + "=" * 80)
print("VERİ HAZIRLIK AŞAMASI TAMAMLANDI")
print("=" * 80)
print(f"\nÖzet:")
print(f"  • Toplam hücre: {adata.n_obs}")
print(f"  • Toplam gen: {adata.n_vars}")
print(f"  • Yüksek varyans gen: {adata_hvg.n_vars}")
print(f"  • Hücre türleri: {adata.obs['cell_type'].nunique()}")
print(f"  • PCA bileşenleri: 50")
print(f"\nÇıktı dosyaları:")
print(f"  • data/adata_full.h5ad")
print(f"  • data/adata_hvg.h5ad")
print(f"  • data/metadata.csv")
print(f"  • results/01_data_preparation.png")
print("=" * 80)
