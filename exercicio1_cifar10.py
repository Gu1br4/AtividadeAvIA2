import matplotlib
matplotlib.use('Agg')

import copy
import os
import random
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import DataLoader, Dataset
import torchvision
import torchvision.transforms as transforms
from PIL import Image
from sklearn.model_selection import train_test_split
from sklearn.metrics import ConfusionMatrixDisplay, classification_report, confusion_matrix

# -- Configuração --------------------------------------------------------------

SEED = 2025
random.seed(SEED)
np.random.seed(SEED)
torch.manual_seed(SEED)
torch.cuda.manual_seed_all(SEED)
torch.backends.cudnn.deterministic = True
torch.backends.cudnn.benchmark = False

BATCH_SIZE = 128
EPOCHS     = 50
DEVICE     = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
CLASS_NAMES = [
    'airplane', 'automobile', 'bird', 'cat', 'deer',
    'dog', 'frog', 'horse', 'ship', 'truck',
]

os.makedirs('outputs', exist_ok=True)

print(f'PyTorch: {torch.__version__}')
print(f'Device:  {DEVICE}')


# -- 1) Dataset ----------------------------------------------------------------

print('\n-- 1) Leitura e Organização do Dataset --')

raw_train = torchvision.datasets.CIFAR10(root='./data', train=True,  download=True)
raw_test  = torchvision.datasets.CIFAR10(root='./data', train=False, download=True)

X_all = np.concatenate([raw_train.data, raw_test.data], axis=0)
y_all = np.array(raw_train.targets + raw_test.targets)

dfgen = {'features': X_all, 'labels': y_all}

print(f'Features: {dfgen["features"].shape}  dtype={dfgen["features"].dtype}')
print(f'Labels:   {dfgen["labels"].shape}')
print(f'Range pixels: [{dfgen["features"].min()}, {dfgen["features"].max()}]')

unique, counts = np.unique(dfgen['labels'], return_counts=True)
contagem = pd.DataFrame({
    'Classe':     [CLASS_NAMES[i] for i in unique],
    'Código':     unique,
    'Quantidade': counts,
})
print('\nContagem por classe:')
print(contagem.to_string(index=False))

np.random.seed(SEED)
idx_random = np.random.choice(len(dfgen['features']), size=10, replace=False)

fig, axes = plt.subplots(2, 5, figsize=(12, 5))
for i, ax in enumerate(axes.flat):
    idx = idx_random[i]
    ax.imshow(dfgen['features'][idx])
    ax.set_title(f"{CLASS_NAMES[dfgen['labels'][idx]]} ({dfgen['labels'][idx]})")
    ax.axis('off')
plt.suptitle('10 Amostras Aleatórias do CIFAR-10', fontsize=14)
plt.tight_layout()
plt.savefig('outputs/01_amostras_aleatorias.png', dpi=150, bbox_inches='tight')
plt.close()
print('Gráfico salvo: outputs/01_amostras_aleatorias.png')


# -- 2) Pré-processamento e Data Augmentation ----------------------------------

print('\n-- 2) Pré-processamento e Data Augmentation --')

# Estatísticas do CIFAR-10 (por canal)
CIFAR10_MEAN = (0.4914, 0.4822, 0.4465)
CIFAR10_STD  = (0.2470, 0.2435, 0.2616)

train_transform = transforms.Compose([
    transforms.RandomCrop(32, padding=4),
    transforms.RandomHorizontalFlip(),
    transforms.ToTensor(),
    transforms.Normalize(CIFAR10_MEAN, CIFAR10_STD),
])
eval_transform = transforms.Compose([
    transforms.ToTensor(),
    transforms.Normalize(CIFAR10_MEAN, CIFAR10_STD),
])

print(f'Média por canal (R, G, B): {CIFAR10_MEAN}')
print(f'Desvio por canal (R, G, B): {CIFAR10_STD}')

# Visualizar augmentation
sample_img = Image.fromarray(dfgen['features'][0])
fig, axes = plt.subplots(1, 6, figsize=(14, 3))
axes[0].imshow(sample_img)
axes[0].set_title('Original')
axes[0].axis('off')

aug_only = transforms.Compose([
    transforms.RandomCrop(32, padding=4),
    transforms.RandomHorizontalFlip(),
])
for i in range(1, 6):
    axes[i].imshow(aug_only(sample_img))
    axes[i].set_title(f'Aug {i}')
    axes[i].axis('off')
plt.suptitle('Exemplos de Data Augmentation', fontsize=14)
plt.tight_layout()
plt.savefig('outputs/02_data_augmentation.png', dpi=150, bbox_inches='tight')
plt.close()
print('Gráfico salvo: outputs/02_data_augmentation.png')

# -- 3) Divisão 70 / 15 / 15 --------------------------------------------------

print('\n-- 3) Divisão dos Dados (70% treino / 15% val / 15% teste) --')

X_train_idx, X_temp_idx, y_train, y_temp = train_test_split(
    np.arange(len(y_all)), y_all,
    test_size=0.30, random_state=SEED, stratify=y_all,
)
X_val_idx, X_test_idx, y_val, y_test = train_test_split(
    X_temp_idx, y_temp,
    test_size=0.50, random_state=SEED, stratify=y_temp,
)

total = len(y_all)
print(f'Treino:    {len(X_train_idx):>6} ({len(X_train_idx)/total*100:.1f}%)')
print(f'Validação: {len(X_val_idx):>6} ({len(X_val_idx)/total*100:.1f}%)')
print(f'Teste:     {len(X_test_idx):>6} ({len(X_test_idx)/total*100:.1f}%)')

def contagem_por_classe(y, nome):
    u, c = np.unique(y, return_counts=True)
    return pd.Series(c, index=[CLASS_NAMES[i] for i in u], name=nome)

tabela_split = pd.DataFrame({
    'Treino (70%)':    contagem_por_classe(y_train, 'Treino'),
    'Validação (15%)': contagem_por_classe(y_val,   'Validação'),
    'Teste (15%)':     contagem_por_classe(y_test,  'Teste'),
})
tabela_split['Total'] = tabela_split.sum(axis=1)
tabela_split.loc['TOTAL'] = tabela_split.sum()
print('\nAmostras por classe em cada partição:')
print(tabela_split.astype(int).to_string())

class CIFAR10Split(Dataset):
    def __init__(self, images, labels, transform=None):
        self.images    = images
        self.labels    = labels
        self.transform = transform

    def __len__(self):
        return len(self.labels)

    def __getitem__(self, idx):
        img = Image.fromarray(self.images[idx])
        if self.transform:
            img = self.transform(img)
        return img, int(self.labels[idx])

g = torch.Generator().manual_seed(SEED)

train_ds = CIFAR10Split(dfgen['features'][X_train_idx], y_train, train_transform)
val_ds   = CIFAR10Split(dfgen['features'][X_val_idx],   y_val,   eval_transform)
test_ds  = CIFAR10Split(dfgen['features'][X_test_idx],  y_test,  eval_transform)

train_loader = DataLoader(train_ds, batch_size=BATCH_SIZE, shuffle=True,  generator=g, num_workers=0)
val_loader   = DataLoader(val_ds,   batch_size=BATCH_SIZE, shuffle=False, num_workers=0)
test_loader  = DataLoader(test_ds,  batch_size=BATCH_SIZE, shuffle=False, num_workers=0)

# -- 4) Arquitetura CNN — clfgen ----------------------------------------------─

print('\n-- 4) Arquitetura da CNN (clfgen) --')

class ClfGen(nn.Module):
    def __init__(self, num_classes=10):
        super().__init__()
        self.block1 = nn.Sequential(
            nn.Conv2d(3,  32, 3, padding=1), nn.BatchNorm2d(32),  nn.ReLU(),
            nn.Conv2d(32, 32, 3, padding=1), nn.BatchNorm2d(32),  nn.ReLU(),
            nn.MaxPool2d(2, 2),
        )
        self.block2 = nn.Sequential(
            nn.Conv2d(32, 64, 3, padding=1), nn.BatchNorm2d(64), nn.ReLU(),
            nn.Conv2d(64, 64, 3, padding=1), nn.BatchNorm2d(64), nn.ReLU(),
            nn.MaxPool2d(2, 2),
        )
        self.head = nn.Sequential(
            nn.Dropout(0.3),
            nn.Flatten(),
            nn.Linear(64 * 8 * 8, 256), nn.ReLU(),
            nn.Dropout(0.5),
            nn.Linear(256, num_classes),
        )

    def forward(self, x):
        return self.head(self.block2(self.block1(x)))

clfgen = ClfGen().to(DEVICE)
total_params = sum(p.numel() for p in clfgen.parameters())
print(f'Parâmetros totais: {total_params:,}')
print(clfgen)

# -- 5) Treinamento ------------------------------------------------------------

print('\n-- 5) Treinamento com Validação --')

criterion = nn.CrossEntropyLoss()
optimizer = optim.Adam(clfgen.parameters(), lr=1e-3)
scheduler = optim.lr_scheduler.ReduceLROnPlateau(
    optimizer, mode='max', factor=0.2, patience=5, min_lr=1e-5
)

history = {'train_loss': [], 'train_acc': [], 'val_loss': [], 'val_acc': []}
best_val_acc  = 0.0
best_weights  = None
no_improve    = 0
PATIENCE      = 10

def run_epoch(loader, train=True):
    clfgen.train(train)
    total_loss, correct, total = 0.0, 0, 0
    ctx = torch.enable_grad() if train else torch.no_grad()
    with ctx:
        for X, y in loader:
            X, y = X.to(DEVICE), y.to(DEVICE)
            if train:
                optimizer.zero_grad()
            logits = clfgen(X)
            loss   = criterion(logits, y)
            if train:
                loss.backward()
                optimizer.step()
            total_loss += loss.item() * len(y)
            correct    += (logits.argmax(1) == y).sum().item()
            total      += len(y)
    return total_loss / total, correct / total


for epoch in range(1, EPOCHS + 1):
    train_loss, train_acc = run_epoch(train_loader, train=True)
    val_loss,   val_acc   = run_epoch(val_loader,   train=False)

    history['train_loss'].append(train_loss)
    history['train_acc'].append(train_acc)
    history['val_loss'].append(val_loss)
    history['val_acc'].append(val_acc)

    scheduler.step(val_acc)

    if val_acc > best_val_acc:
        best_val_acc = val_acc
        best_weights = copy.deepcopy(clfgen.state_dict())
        torch.save(best_weights, 'outputs/melhor_clfgen.pt')
        tag = ' ✓ salvo'
        no_improve = 0
    else:
        no_improve += 1
        tag = ''

    print(f'Época {epoch:3d}/{EPOCHS} | '
          f'loss {train_loss:.4f} acc {train_acc:.4f} | '
          f'val_loss {val_loss:.4f} val_acc {val_acc:.4f}{tag}')

    if no_improve >= PATIENCE:
        print(f'\nEarly stopping na época {epoch} (sem melhora por {PATIENCE} épocas).')
        break

best_epoch = history['val_acc'].index(best_val_acc) + 1
print(f'\nMelhor val_acc: {best_val_acc:.4f} (época {best_epoch})')

fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(14, 5))
ax1.plot(history['train_acc'], label='Treino',    color='steelblue')
ax1.plot(history['val_acc'],   label='Validação', color='orange')
ax1.set_title('Acurácia por Época')
ax1.set_xlabel('Época')
ax1.set_ylabel('Acurácia')
ax1.legend()
ax1.grid(True, alpha=0.3)

ax2.plot(history['train_loss'], label='Treino',    color='steelblue')
ax2.plot(history['val_loss'],   label='Validação', color='orange')
ax2.set_title('Perda (Loss) por Época')
ax2.set_xlabel('Época')
ax2.set_ylabel('Loss')
ax2.legend()
ax2.grid(True, alpha=0.3)

plt.suptitle('Curvas de Treino vs. Validação', fontsize=14)
plt.tight_layout()
plt.savefig('outputs/03_curvas_treino_validacao.png', dpi=150, bbox_inches='tight')
plt.close()
print('Gráfico salvo: outputs/03_curvas_treino_validacao.png')


# -- 6) Avaliação no Conjunto de Teste ----------------------------------------

print('\n-- 6) Avaliação no Conjunto de Teste --')

clfgen.load_state_dict(torch.load('outputs/melhor_clfgen.pt', map_location=DEVICE))
clfgen.eval()

test_loss, test_acc = run_epoch(test_loader, train=False)
print(f'\nAcurácia: {test_acc:.4f} ({test_acc*100:.2f}%)')
print(f'Loss:     {test_loss:.4f}')

all_preds, all_labels = [], []
with torch.no_grad():
    for X, y in test_loader:
        all_preds.extend(clfgen(X.to(DEVICE)).argmax(1).cpu().tolist())
        all_labels.extend(y.tolist())

y_pred  = np.array(all_preds)
y_true  = np.array(all_labels)
cm      = confusion_matrix(y_true, y_pred)

fig, ax = plt.subplots(figsize=(10, 8))
ConfusionMatrixDisplay(confusion_matrix=cm, display_labels=CLASS_NAMES).plot(
    ax=ax, cmap='Blues', values_format='d', xticks_rotation=45
)
ax.set_title('Matriz de Confusão — Conjunto de Teste', fontsize=14)
plt.tight_layout()
plt.savefig('outputs/04_matriz_confusao.png', dpi=150, bbox_inches='tight')
plt.close()
print('Gráfico salvo: outputs/04_matriz_confusao.png')

print('\n=== Relatório de Classificação ===')
print(classification_report(y_true, y_pred, target_names=CLASS_NAMES, digits=4))

cm_no_diag = cm.copy()
np.fill_diagonal(cm_no_diag, 0)
confusoes = [
    (CLASS_NAMES[i], CLASS_NAMES[j], cm_no_diag[i, j])
    for i in range(10) for j in range(10)
    if i != j and cm_no_diag[i, j] > 0
]
confusoes.sort(key=lambda x: x[2], reverse=True)

print('\nTop 10 pares de confusão (real -> predito):')
for real, pred, qtd in confusoes[:10]:
    print(f'  {real:>12} -> {pred:<12}: {qtd} erros')

print('\nAcurácia por classe:')
for i in range(10):
    acc = cm[i, i] / cm[i].sum()
    print(f'  {CLASS_NAMES[i]:>12}: {acc:.4f} ({acc*100:.2f}%)')
