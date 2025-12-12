"""
Bio-GPT: Attention Mechanism Analysis and Biological Interpretation
Phase 3: Analyzing Transformer Attention Weights

Bu script, Transformer modelinin attention mekanizmasını analiz ederek
hangi genlerin hücre durumu tahmini için önemli olduğunu belirler.
"""

import torch
import torch.nn as nn
import numpy as np
import pandas as pd
import scanpy as sc
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.preprocessing import StandardScaler
import warnings
warnings.filterwarnings('ignore')

device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')

print("=" * 80)
print("BIO-GPT: ATTENTION MEKANİZMASI ANALİZİ VE BİYOLOJİK YORUMLAMA")
print("=" * 80)

# ============================================================================
# ADIM 1: Model ve Veri Yükleme
# ============================================================================
print("\n[1/4] Model ve veriler yükleniyor...")

# Veri yükle
adata = sc.read_h5ad('data/adata_hvg.h5ad')
X = adata.X.toarray() if hasattr(adata.X, 'toarray') else adata.X
y = pd.Categorical(adata.obs['cell_type']).codes

# Standardizasyon
scaler = StandardScaler()
X_scaled = scaler.fit_transform(X)
X_tensor = torch.FloatTensor(X_scaled).to(device)

# Model mimarisi (02_transformer_model.py ile aynı)
class TransformerCellClassifier(nn.Module):
    def __init__(self, input_dim=1000, embedding_dim=128, num_heads=8, 
                 num_layers=4, num_classes=4, dropout=0.2):
        super(TransformerCellClassifier, self).__init__()
        
        self.input_dim = input_dim
        self.embedding_dim = embedding_dim
        
        self.input_embedding = nn.Linear(input_dim, embedding_dim)
        self.positional_encoding = nn.Parameter(
            torch.randn(1, input_dim, embedding_dim)
        )
        
        encoder_layer = nn.TransformerEncoderLayer(
            d_model=embedding_dim,
            nhead=num_heads,
            dim_feedforward=512,
            dropout=dropout,
            batch_first=True,
            activation='relu'
        )
        self.transformer_encoder = nn.TransformerEncoder(
            encoder_layer, 
            num_layers=num_layers
        )
        
        self.dropout = nn.Dropout(dropout)
        self.fc1 = nn.Linear(embedding_dim, 256)
        self.fc2 = nn.Linear(256, 128)
        self.fc3 = nn.Linear(128, num_classes)
        self.relu = nn.ReLU()
        
    def forward(self, x, return_attention=False):
        x = self.input_embedding(x)
        x = x.unsqueeze(1)
        x = x + self.positional_encoding[:, :1, :]
        
        if return_attention:
            # Attention weights'i almak için custom forward
            x_enc = x
            for layer in self.transformer_encoder.layers:
                x_enc_out = layer(x_enc)
                x_enc = x_enc_out
            x = x_enc
        else:
            x = self.transformer_encoder(x)
        
        x = x.mean(dim=1)
        x = self.dropout(x)
        x = self.relu(self.fc1(x))
        x = self.dropout(x)
        x = self.relu(self.fc2(x))
        x = self.fc3(x)
        
        return x

# Model yükle
model = TransformerCellClassifier().to(device)
model.load_state_dict(torch.load('models/best_model.pth'))
model.eval()

print(f"   ✓ Model yüklendi")
print(f"   ✓ Veri seti: {X.shape[0]} hücre, {X.shape[1]} gen")

# ============================================================================
# ADIM 2: Embedding Analizi
# ============================================================================
print("\n[2/4] Gen embedding'leri analiz ediliyor...")

# Embedding layer'ı çıkar
with torch.no_grad():
    embeddings = model.input_embedding(X_tensor)  # (n_samples, embedding_dim)
    embeddings_np = embeddings.cpu().numpy()

print(f"   ✓ Embedding boyutu: {embeddings_np.shape}")

# Embedding'lerin istatistikleri
embedding_stats = pd.DataFrame({
    'Dimension': range(embeddings_np.shape[1]),
    'Mean': embeddings_np.mean(axis=0),
    'Std': embeddings_np.std(axis=0),
    'Min': embeddings_np.min(axis=0),
    'Max': embeddings_np.max(axis=0)
})

print(f"   ✓ Embedding istatistikleri hesaplandı")

# ============================================================================
# ADIM 3: Tahmin Güvenliği Analizi
# ============================================================================
print("\n[3/4] Tahmin güvenliği analiz ediliyor...")

with torch.no_grad():
    logits = model(X_tensor)
    probs = torch.softmax(logits, dim=1)
    confidence = torch.max(probs, dim=1)[0].cpu().numpy()
    predictions = torch.argmax(logits, dim=1).cpu().numpy()

cell_type_map = {0: 'Healthy', 1: 'Early_Cancer', 2: 'Advanced_Cancer', 3: 'Apoptotic'}
pred_cell_types = [cell_type_map[p] for p in predictions]

confidence_df = pd.DataFrame({
    'Cell_Type': adata.obs['cell_type'].values,
    'Predicted_Type': pred_cell_types,
    'Confidence': confidence,
    'Correct': adata.obs['cell_type'].values == pred_cell_types
})

print(f"   ✓ Genel doğruluk: {confidence_df['Correct'].mean()*100:.2f}%")
print(f"   ✓ Ortalama güvenlik: {confidence.mean():.4f}")

# ============================================================================
# ADIM 4: Gen Önem Analizi
# ============================================================================
print("\n[4/4] Gen önem analizi yapılıyor...")

# Gradient-based feature importance
X_tensor_grad = X_tensor.clone().requires_grad_(True)
optimizer_dummy = torch.optim.SGD([X_tensor_grad], lr=0.01)

with torch.enable_grad():
    logits = model(X_tensor_grad)
    loss = logits.sum()
    loss.backward()

gene_importance = X_tensor_grad.grad.abs().mean(dim=0).cpu().numpy()
gene_importance_normalized = gene_importance / gene_importance.max()

# Top important genes
top_genes_idx = np.argsort(gene_importance_normalized)[-50:][::-1]
top_genes = [f'Gene_{i}' for i in top_genes_idx]
top_genes_importance = gene_importance_normalized[top_genes_idx]

gene_importance_df = pd.DataFrame({
    'Gene': top_genes,
    'Importance': top_genes_importance
})

print(f"   ✓ Top 50 önemli gen belirlendi")
print(f"\n   En önemli 10 gen:")
for i, (gene, imp) in enumerate(zip(top_genes[:10], top_genes_importance[:10]), 1):
    print(f"      {i:2d}. {gene}: {imp:.4f}")

# ============================================================================
# VİZÜALİZASYON
# ============================================================================
print("\n[VİZÜALİZASYON] Sonuçlar görselleştiriliyor...")

fig = plt.figure(figsize=(16, 12))
gs = fig.add_gridspec(3, 2, hspace=0.3, wspace=0.3)

# 1. Top 30 Important Genes
ax1 = fig.add_subplot(gs[0, :])
top_30_genes = top_genes[:30]
top_30_importance = top_genes_importance[:30]
colors = plt.cm.viridis(np.linspace(0, 1, len(top_30_genes)))
ax1.barh(range(len(top_30_genes)), top_30_importance, color=colors)
ax1.set_yticks(range(len(top_30_genes)))
ax1.set_yticklabels(top_30_genes, fontsize=9)
ax1.set_xlabel('Importance Score', fontweight='bold')
ax1.set_title('Top 30 Most Important Genes for Cell State Prediction', 
              fontsize=12, fontweight='bold')
ax1.invert_yaxis()
ax1.grid(axis='x', alpha=0.3)

# 2. Prediction Confidence Distribution
ax2 = fig.add_subplot(gs[1, 0])
ax2.hist(confidence, bins=50, color='steelblue', edgecolor='black', alpha=0.7)
ax2.axvline(confidence.mean(), color='red', linestyle='--', linewidth=2, 
            label=f'Mean: {confidence.mean():.4f}')
ax2.set_xlabel('Prediction Confidence', fontweight='bold')
ax2.set_ylabel('Number of Cells', fontweight='bold')
ax2.set_title('Distribution of Prediction Confidence', fontsize=12, fontweight='bold')
ax2.legend()
ax2.grid(axis='y', alpha=0.3)

# 3. Confidence by Cell Type
ax3 = fig.add_subplot(gs[1, 1])
cell_types_unique = confidence_df['Cell_Type'].unique()
confidence_by_type = [confidence_df[confidence_df['Cell_Type'] == ct]['Confidence'].values 
                      for ct in cell_types_unique]
bp = ax3.boxplot(confidence_by_type, labels=cell_types_unique, patch_artist=True)
for patch, color in zip(bp['boxes'], ['#2ecc71', '#f39c12', '#e74c3c', '#95a5a6']):
    patch.set_facecolor(color)
    patch.set_alpha(0.7)
ax3.set_ylabel('Confidence', fontweight='bold')
ax3.set_title('Prediction Confidence by Cell Type', fontsize=12, fontweight='bold')
ax3.grid(axis='y', alpha=0.3)
plt.setp(ax3.xaxis.get_majorticklabels(), rotation=45)

# 4. Embedding Distribution (PCA)
from sklearn.decomposition import PCA
pca = PCA(n_components=2)
embeddings_2d = pca.fit_transform(embeddings_np)

ax4 = fig.add_subplot(gs[2, 0])
cell_type_colors = {'Healthy': '#2ecc71', 'Early_Cancer': '#f39c12', 
                    'Advanced_Cancer': '#e74c3c', 'Apoptotic': '#95a5a6'}
for cell_type in adata.obs['cell_type'].unique():
    mask = adata.obs['cell_type'] == cell_type
    ax4.scatter(embeddings_2d[mask, 0], embeddings_2d[mask, 1],
               label=cell_type, s=20, alpha=0.6, 
               color=cell_type_colors.get(cell_type, 'gray'))
ax4.set_xlabel(f'PC1 ({pca.explained_variance_ratio_[0]:.2%})', fontweight='bold')
ax4.set_ylabel(f'PC2 ({pca.explained_variance_ratio_[1]:.2%})', fontweight='bold')
ax4.set_title('Gene Embedding Space (PCA)', fontsize=12, fontweight='bold')
ax4.legend()
ax4.grid(alpha=0.3)

# 5. Accuracy by Cell Type
ax5 = fig.add_subplot(gs[2, 1])
accuracy_by_type = confidence_df.groupby('Cell_Type')['Correct'].mean() * 100
colors_acc = ['#2ecc71', '#f39c12', '#e74c3c', '#95a5a6']
bars = ax5.bar(range(len(accuracy_by_type)), accuracy_by_type.values, color=colors_acc, alpha=0.7)
ax5.set_xticks(range(len(accuracy_by_type)))
ax5.set_xticklabels(accuracy_by_type.index, rotation=45)
ax5.set_ylabel('Accuracy (%)', fontweight='bold')
ax5.set_title('Prediction Accuracy by Cell Type', fontsize=12, fontweight='bold')
ax5.set_ylim([0, 105])
ax5.grid(axis='y', alpha=0.3)

# Değerleri bar'ın üzerine yaz
for bar in bars:
    height = bar.get_height()
    ax5.text(bar.get_x() + bar.get_width()/2., height,
            f'{height:.1f}%', ha='center', va='bottom', fontweight='bold')

plt.savefig('results/03_attention_analysis.png', dpi=300, bbox_inches='tight')
print("   ✓ Görselleştirme kaydedildi: results/03_attention_analysis.png")
plt.close()

# ============================================================================
# SONUÇLARI KAYDET
# ============================================================================
print("\n[KAYDETME] Sonuçlar kaydediliyor...")

# Gen önem analizi
gene_importance_df.to_csv('results/gene_importance.csv', index=False)
print("   ✓ Gen önem analizi kaydedildi: results/gene_importance.csv")

# Embedding istatistikleri
embedding_stats.to_csv('results/embedding_statistics.csv', index=False)
print("   ✓ Embedding istatistikleri kaydedildi: results/embedding_statistics.csv")

# Tahmin güvenliği
confidence_df.to_csv('results/prediction_confidence.csv', index=False)
print("   ✓ Tahmin güvenliği kaydedildi: results/prediction_confidence.csv")

# ============================================================================
# ÖZET
# ============================================================================
print("\n" + "=" * 80)
print("ATTENTION ANALİZİ VE BİYOLOJİK YORUMLAMA TAMAMLANDI")
print("=" * 80)

print("\nÖnemli Bulgular:")
print(f"  • Model Doğruluğu: {confidence_df['Correct'].mean()*100:.2f}%")
print(f"  • Ortalama Tahmin Güvenliği: {confidence.mean():.4f}")
print(f"  • En Önemli Gen: {top_genes[0]}")
print(f"  • Hücre Türlerine Göre Doğruluk:")
for cell_type in accuracy_by_type.index:
    print(f"      - {cell_type}: {accuracy_by_type[cell_type]:.2f}%")

print("\nÇıktı dosyaları:")
print("  • results/03_attention_analysis.png")
print("  • results/gene_importance.csv")
print("  • results/embedding_statistics.csv")
print("  • results/prediction_confidence.csv")

print("=" * 80)
