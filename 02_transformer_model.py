"""
Bio-GPT: Cancer Cell State Prediction using Transformer Models
Phase 2: Transformer Model Architecture and Training

Bu script, scRNA-seq verileri için Transformer tabanlı bir model eğitir.
"""

import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import DataLoader, TensorDataset
import numpy as np
import pandas as pd
import scanpy as sc
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler
import warnings
warnings.filterwarnings('ignore')

# Cihaz ayarı
device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
print(f"Cihaz: {device}")

print("=" * 80)
print("BIO-GPT: TRANSFORMER MODELİ OLUŞTURMA VE EĞİTİM")
print("=" * 80)

# ============================================================================
# ADIM 1: Veri Yükleme
# ============================================================================
print("\n[1/6] Veriler yükleniyor...")

adata = sc.read_h5ad('data/adata_hvg.h5ad')
X = adata.X.toarray() if hasattr(adata.X, 'toarray') else adata.X
y = pd.Categorical(adata.obs['cell_type']).codes

print(f"   ✓ Veri yüklendi: {X.shape[0]} hücre, {X.shape[1]} gen")
print(f"   ✓ Hücre türleri: {np.unique(y)}")

# ============================================================================
# ADIM 2: Veri Standardizasyonu ve Bölümleme
# ============================================================================
print("\n[2/6] Veri standardizasyonu ve bölümleme...")

scaler = StandardScaler()
X_scaled = scaler.fit_transform(X)

# Train-test bölümü
X_train, X_test, y_train, y_test = train_test_split(
    X_scaled, y, test_size=0.2, random_state=42, stratify=y
)

# Validation bölümü
X_train, X_val, y_train, y_val = train_test_split(
    X_train, y_train, test_size=0.2, random_state=42, stratify=y_train
)

print(f"   ✓ Eğitim seti: {X_train.shape[0]} örnek")
print(f"   ✓ Validasyon seti: {X_val.shape[0]} örnek")
print(f"   ✓ Test seti: {X_test.shape[0]} örnek")

# PyTorch tensörleri
X_train_t = torch.FloatTensor(X_train).to(device)
y_train_t = torch.LongTensor(y_train).to(device)
X_val_t = torch.FloatTensor(X_val).to(device)
y_val_t = torch.LongTensor(y_val).to(device)
X_test_t = torch.FloatTensor(X_test).to(device)
y_test_t = torch.LongTensor(y_test).to(device)

# DataLoader
batch_size = 32
train_dataset = TensorDataset(X_train_t, y_train_t)
train_loader = DataLoader(train_dataset, batch_size=batch_size, shuffle=True)

print(f"   ✓ Batch boyutu: {batch_size}")

# ============================================================================
# ADIM 3: Transformer Mimarisi
# ============================================================================
print("\n[3/6] Transformer mimarisi tanımlanıyor...")

class TransformerCellClassifier(nn.Module):
    """
    Hücre durumu tahmini için Transformer tabanlı sınıflandırıcı.
    
    Mimarı:
    - Input: Gen ekspresyon profili (1000 gen)
    - Embedding: Genleri 128 boyutlu vektörlere dönüştür
    - Transformer: 4 katmanlı, 8 başlı attention
    - Output: 4 hücre türü sınıfı
    """
    
    def __init__(self, input_dim=1000, embedding_dim=128, num_heads=8, 
                 num_layers=4, num_classes=4, dropout=0.2):
        super(TransformerCellClassifier, self).__init__()
        
        self.input_dim = input_dim
        self.embedding_dim = embedding_dim
        
        # Input embedding
        self.input_embedding = nn.Linear(input_dim, embedding_dim)
        
        # Positional encoding (genler için)
        self.positional_encoding = nn.Parameter(
            torch.randn(1, input_dim, embedding_dim)
        )
        
        # Transformer encoder
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
        
        # Classification head
        self.dropout = nn.Dropout(dropout)
        self.fc1 = nn.Linear(embedding_dim, 256)
        self.fc2 = nn.Linear(256, 128)
        self.fc3 = nn.Linear(128, num_classes)
        
        self.relu = nn.ReLU()
        
    def forward(self, x):
        # x shape: (batch_size, input_dim)
        
        # Embedding
        x = self.input_embedding(x)  # (batch_size, embedding_dim)
        x = x.unsqueeze(1)  # (batch_size, 1, embedding_dim)
        
        # Positional encoding ekle
        x = x + self.positional_encoding[:, :1, :]
        
        # Transformer encoder
        x = self.transformer_encoder(x)  # (batch_size, 1, embedding_dim)
        
        # Global average pooling
        x = x.mean(dim=1)  # (batch_size, embedding_dim)
        
        # Classification head
        x = self.dropout(x)
        x = self.relu(self.fc1(x))
        x = self.dropout(x)
        x = self.relu(self.fc2(x))
        x = self.fc3(x)
        
        return x

# Model oluştur
model = TransformerCellClassifier(
    input_dim=X_train.shape[1],
    embedding_dim=128,
    num_heads=8,
    num_layers=4,
    num_classes=len(np.unique(y)),
    dropout=0.2
).to(device)

print(f"   ✓ Model oluşturuldu")
print(f"   ✓ Toplam parametreler: {sum(p.numel() for p in model.parameters()):,}")

# ============================================================================
# ADIM 4: Eğitim Ayarları
# ============================================================================
print("\n[4/6] Eğitim ayarları yapılandırılıyor...")

criterion = nn.CrossEntropyLoss()
optimizer = optim.Adam(model.parameters(), lr=0.001, weight_decay=1e-5)
scheduler = optim.lr_scheduler.ReduceLROnPlateau(
    optimizer, mode='min', factor=0.5, patience=10
)

num_epochs = 100
best_val_loss = float('inf')
patience = 20
patience_counter = 0
history = {'train_loss': [], 'val_loss': [], 'train_acc': [], 'val_acc': []}

print(f"   ✓ Optimizer: Adam (lr=0.001)")
print(f"   ✓ Scheduler: ReduceLROnPlateau")
print(f"   ✓ Epoch sayısı: {num_epochs}")

# ============================================================================
# ADIM 5: Eğitim Döngüsü
# ============================================================================
print("\n[5/6] Model eğitiliyor...")

for epoch in range(num_epochs):
    # Training
    model.train()
    train_loss = 0.0
    train_correct = 0
    train_total = 0
    
    for X_batch, y_batch in train_loader:
        optimizer.zero_grad()
        
        outputs = model(X_batch)
        loss = criterion(outputs, y_batch)
        
        loss.backward()
        optimizer.step()
        
        train_loss += loss.item()
        _, predicted = torch.max(outputs.data, 1)
        train_total += y_batch.size(0)
        train_correct += (predicted == y_batch).sum().item()
    
    train_loss /= len(train_loader)
    train_acc = 100 * train_correct / train_total
    
    # Validation
    model.eval()
    with torch.no_grad():
        val_outputs = model(X_val_t)
        val_loss = criterion(val_outputs, y_val_t).item()
        _, val_predicted = torch.max(val_outputs.data, 1)
        val_acc = 100 * (val_predicted == y_val_t).sum().item() / y_val_t.size(0)
    
    history['train_loss'].append(train_loss)
    history['val_loss'].append(val_loss)
    history['train_acc'].append(train_acc)
    history['val_acc'].append(val_acc)
    
    # Early stopping
    if val_loss < best_val_loss:
        best_val_loss = val_loss
        patience_counter = 0
        torch.save(model.state_dict(), 'models/best_model.pth')
    else:
        patience_counter += 1
    
    scheduler.step(val_loss)
    
    if (epoch + 1) % 10 == 0:
        print(f"   Epoch {epoch+1:3d}/{num_epochs} | "
              f"Train Loss: {train_loss:.4f} | Val Loss: {val_loss:.4f} | "
              f"Train Acc: {train_acc:.2f}% | Val Acc: {val_acc:.2f}%")
    
    if patience_counter >= patience:
        print(f"   Early stopping at epoch {epoch+1}")
        break

print(f"   ✓ Eğitim tamamlandı")

# ============================================================================
# ADIM 6: Test Seti Değerlendirmesi
# ============================================================================
print("\n[6/6] Test seti değerlendiriliyor...")

model.load_state_dict(torch.load('models/best_model.pth'))
model.eval()

with torch.no_grad():
    test_outputs = model(X_test_t)
    test_loss = criterion(test_outputs, y_test_t).item()
    _, test_predicted = torch.max(test_outputs.data, 1)
    test_acc = 100 * (test_predicted == y_test_t).sum().item() / y_test_t.size(0)

print(f"   ✓ Test Loss: {test_loss:.4f}")
print(f"   ✓ Test Accuracy: {test_acc:.2f}%")

# ============================================================================
# VİZÜALİZASYON
# ============================================================================
print("\n[VİZÜALİZASYON] Sonuçlar görselleştiriliyor...")

fig, axes = plt.subplots(2, 2, figsize=(14, 10))

# 1. Training ve Validation Loss
ax1 = axes[0, 0]
ax1.plot(history['train_loss'], label='Training Loss', linewidth=2, marker='o', markersize=3)
ax1.plot(history['val_loss'], label='Validation Loss', linewidth=2, marker='s', markersize=3)
ax1.set_title('Model Loss', fontsize=12, fontweight='bold')
ax1.set_xlabel('Epoch')
ax1.set_ylabel('Loss')
ax1.legend()
ax1.grid(True, alpha=0.3)

# 2. Training ve Validation Accuracy
ax2 = axes[0, 1]
ax2.plot(history['train_acc'], label='Training Accuracy', linewidth=2, marker='o', markersize=3)
ax2.plot(history['val_acc'], label='Validation Accuracy', linewidth=2, marker='s', markersize=3)
ax2.set_title('Model Accuracy', fontsize=12, fontweight='bold')
ax2.set_xlabel('Epoch')
ax2.set_ylabel('Accuracy (%)')
ax2.legend()
ax2.grid(True, alpha=0.3)

# 3. Confusion Matrix
from sklearn.metrics import confusion_matrix, classification_report
y_test_np = y_test_t.cpu().numpy()
y_pred_np = test_predicted.cpu().numpy()
cm = confusion_matrix(y_test_np, y_pred_np)

cell_types = ['Healthy', 'Early_Cancer', 'Advanced_Cancer', 'Apoptotic']
ax3 = axes[1, 0]
sns.heatmap(cm, annot=True, fmt='d', cmap='Blues', ax=ax3, 
            xticklabels=cell_types, yticklabels=cell_types, cbar=False)
ax3.set_title('Confusion Matrix (Test Set)', fontsize=12, fontweight='bold')
ax3.set_ylabel('True Label')
ax3.set_xlabel('Predicted Label')

# 4. Classification Report
ax4 = axes[1, 1]
ax4.axis('off')
report = classification_report(y_test_np, y_pred_np, target_names=cell_types)
ax4.text(0.1, 0.5, report, fontfamily='monospace', fontsize=10, verticalalignment='center')
ax4.set_title('Classification Report', fontsize=12, fontweight='bold')

plt.tight_layout()
plt.savefig('results/02_transformer_training.png', dpi=300, bbox_inches='tight')
print("   ✓ Görselleştirme kaydedildi: results/02_transformer_training.png")
plt.close()

# ============================================================================
# ÖZET VE KAYDETME
# ============================================================================
print("\n" + "=" * 80)
print("TRANSFORMER MODELİ EĞİTİMİ TAMAMLANDI")
print("=" * 80)

# Model bilgilerini kaydet
model_info = {
    'Architecture': 'Transformer Encoder',
    'Input Dimension': X_train.shape[1],
    'Embedding Dimension': 128,
    'Number of Heads': 8,
    'Number of Layers': 4,
    'Total Parameters': sum(p.numel() for p in model.parameters()),
    'Training Samples': X_train.shape[0],
    'Validation Samples': X_val.shape[0],
    'Test Samples': X_test.shape[0],
    'Best Validation Loss': best_val_loss,
    'Final Test Accuracy': f'{test_acc:.2f}%',
    'Final Test Loss': f'{test_loss:.4f}',
    'Device': str(device)
}

model_info_df = pd.DataFrame(list(model_info.items()), columns=['Parameter', 'Value'])
model_info_df.to_csv('results/model_info.csv', index=False)

print("\nModel Özeti:")
for key, value in model_info.items():
    print(f"  • {key}: {value}")

print("\nÇıktı dosyaları:")
print("  • models/best_model.pth")
print("  • results/02_transformer_training.png")
print("  • results/model_info.csv")

print("=" * 80)
