# =============================================================================
# Pipeline de Classificação CIFAR-10 com TensorFlow/Keras
# Avaliação Prática 2 - IA e Aprendizado de Máquina
# =============================================================================

import matplotlib
matplotlib.use('Agg')
import numpy as np
import tensorflow as tf
from tensorflow import keras
from tensorflow.keras import layers, models, callbacks
import matplotlib.pyplot as plt
from sklearn.model_selection import train_test_split
from sklearn.metrics import confusion_matrix, classification_report, ConfusionMatrixDisplay
import pandas as pd
import random
import os

# =====================================================================
# 0) Configuração de Semente
# =====================================================================
SEED = 2025
random.seed(SEED)
np.random.seed(SEED)
tf.random.set_seed(SEED)
os.environ['TF_DETERMINISTIC_OPS'] = '1'

print(f'TensorFlow version: {tf.__version__}')
print(f'GPU disponível: {tf.config.list_physical_devices("GPU")}')

CLASS_NAMES = ['airplane', 'automobile', 'bird', 'cat', 'deer',
               'dog', 'frog', 'horse', 'ship', 'truck']

# =====================================================================
# 1) Leitura e Organização do Dataset (df_gen)
# =====================================================================
print('\n' + '='*60)
print('1) Leitura e Organização do Dataset')
print('='*60)

(x_train_full, y_train_full), (x_test_orig, y_test_orig) = keras.datasets.cifar10.load_data()

X_all = np.concatenate([x_train_full, x_test_orig], axis=0)
y_all = np.concatenate([y_train_full, y_test_orig], axis=0).flatten()

df_gen = {'features': X_all, 'labels': y_all}

print(f'Forma dos dados (features): {df_gen["features"].shape}')
print(f'Forma dos dados (labels):   {df_gen["labels"].shape}')
print(f'Tipo dos pixels: {df_gen["features"].dtype}')
print(f'Range dos pixels: [{df_gen["features"].min()}, {df_gen["features"].max()}]')

# Contagem por classe
unique, counts = np.unique(df_gen['labels'], return_counts=True)
contagem = pd.DataFrame({'Classe': [CLASS_NAMES[i] for i in unique],
                         'Código': unique,
                         'Quantidade': counts})
print('\nContagem por classe:')
print(contagem.to_string(index=False))

# 10 amostras aleatórias
np.random.seed(SEED)
idx_random = np.random.choice(len(df_gen['features']), size=10, replace=False)

fig, axes = plt.subplots(2, 5, figsize=(12, 5))
for i, ax in enumerate(axes.flat):
    idx = idx_random[i]
    ax.imshow(df_gen['features'][idx])
    ax.set_title(f"{CLASS_NAMES[df_gen['labels'][idx]]} ({df_gen['labels'][idx]})")
    ax.axis('off')
plt.suptitle('10 Amostras Aleatórias do CIFAR-10', fontsize=14)
plt.tight_layout()
plt.savefig('01_amostras_aleatorias.png', dpi=150, bbox_inches='tight')
plt.close()
print('Gráfico salvo: 01_amostras_aleatorias.png')

# =====================================================================
# 2) Pré-processamento e Data Augmentation
# =====================================================================
print('\n' + '='*60)
print('2) Pré-processamento e Data Augmentation')
print('='*60)

X_all_norm = df_gen['features'].astype('float32') / 255.0
y_all = df_gen['labels']

mean_per_channel = X_all_norm.mean(axis=(0, 1, 2))
std_per_channel = X_all_norm.std(axis=(0, 1, 2))
print(f'Média por canal (R, G, B): {mean_per_channel}')
print(f'Desvio por canal (R, G, B): {std_per_channel}')

# Função de data augmentation via tf.image (aplicada apenas no treino)
def augment(image, label):
    image = tf.image.pad_to_bounding_box(image, 4, 4, 40, 40)  # padding=4
    image = tf.image.random_crop(image, size=[32, 32, 3])       # RandomCrop(32)
    image = tf.image.random_flip_left_right(image)               # RandomHorizontalFlip
    return image, label

# Visualizar augmentation
sample_img = X_all_norm[0]
fig, axes = plt.subplots(1, 6, figsize=(14, 3))
axes[0].imshow(sample_img)
axes[0].set_title('Original')
axes[0].axis('off')
for i in range(1, 6):
    aug_img, _ = augment(tf.constant(sample_img), 0)
    axes[i].imshow(aug_img.numpy())
    axes[i].set_title(f'Aug {i}')
    axes[i].axis('off')
plt.suptitle('Exemplos de Data Augmentation', fontsize=14)
plt.tight_layout()
plt.savefig('02_data_augmentation.png', dpi=150, bbox_inches='tight')
plt.close()
print('Gráfico salvo: 02_data_augmentation.png')

# =====================================================================
# 3) Divisão dos Dados em Treino / Validação / Teste (70/15/15)
# =====================================================================
print('\n' + '='*60)
print('3) Divisão dos Dados (70% treino / 15% val / 15% teste)')
print('='*60)

X_train, X_temp, y_train, y_temp = train_test_split(
    X_all_norm, y_all,
    test_size=0.30,
    random_state=SEED,
    stratify=y_all
)

X_val, X_test, y_val, y_test = train_test_split(
    X_temp, y_temp,
    test_size=0.50,
    random_state=SEED,
    stratify=y_temp
)

print(f'Treino:    {X_train.shape[0]} amostras ({X_train.shape[0]/len(X_all_norm)*100:.1f}%)')
print(f'Validação: {X_val.shape[0]} amostras ({X_val.shape[0]/len(X_all_norm)*100:.1f}%)')
print(f'Teste:     {X_test.shape[0]} amostras ({X_test.shape[0]/len(X_all_norm)*100:.1f}%)')

# Tabela de amostras por classe em cada partição
def contagem_por_classe(y, nome):
    unique, counts = np.unique(y, return_counts=True)
    return pd.Series(counts, index=[CLASS_NAMES[i] for i in unique], name=nome)

tabela_split = pd.DataFrame({
    'Treino (70%)': contagem_por_classe(y_train, 'Treino'),
    'Validação (15%)': contagem_por_classe(y_val, 'Validação'),
    'Teste (15%)': contagem_por_classe(y_test, 'Teste')
})
tabela_split['Total'] = tabela_split.sum(axis=1)
tabela_split.loc['TOTAL'] = tabela_split.sum()
tabela_split = tabela_split.astype(int)
print('\nAmostras por classe em cada partição:')
print(tabela_split.to_string())

# Criar tf.data.Datasets
BATCH_SIZE = 128

train_ds = tf.data.Dataset.from_tensor_slices((X_train, y_train))
train_ds = train_ds.shuffle(len(X_train), seed=SEED)
train_ds = train_ds.map(augment, num_parallel_calls=tf.data.AUTOTUNE)
train_ds = train_ds.batch(BATCH_SIZE).prefetch(tf.data.AUTOTUNE)

val_ds = tf.data.Dataset.from_tensor_slices((X_val, y_val))
val_ds = val_ds.batch(BATCH_SIZE).prefetch(tf.data.AUTOTUNE)

test_ds = tf.data.Dataset.from_tensor_slices((X_test, y_test))
test_ds = test_ds.batch(BATCH_SIZE).prefetch(tf.data.AUTOTUNE)

# =====================================================================
# 4) Arquitetura da CNN (clf_gen)
# =====================================================================
print('\n' + '='*60)
print('4) Arquitetura da CNN (clf_gen)')
print('='*60)

def build_cnn(input_shape=(32, 32, 3), num_classes=10):
    inputs = keras.Input(shape=input_shape)
    x = inputs

    # Layer 1: Conv(3->32) + BN + ReLU; Conv(32->32) + BN + ReLU; MaxPool(2x2)
    x = layers.Conv2D(32, (3, 3), padding='same')(x)
    x = layers.BatchNormalization()(x)
    x = layers.ReLU()(x)
    x = layers.Conv2D(32, (3, 3), padding='same')(x)
    x = layers.BatchNormalization()(x)
    x = layers.ReLU()(x)
    x = layers.MaxPooling2D((2, 2))(x)

    # Layer 2: Conv(32->64) + BN + ReLU; Conv(64->64) + BN + ReLU; MaxPool(2x2)
    x = layers.Conv2D(64, (3, 3), padding='same')(x)
    x = layers.BatchNormalization()(x)
    x = layers.ReLU()(x)
    x = layers.Conv2D(64, (3, 3), padding='same')(x)
    x = layers.BatchNormalization()(x)
    x = layers.ReLU()(x)
    x = layers.MaxPooling2D((2, 2))(x)

    # Head: Dropout(0.3) -> Flatten -> Dense(256) + ReLU -> Dropout(0.5) -> Dense(10) + Softmax
    x = layers.Dropout(0.3)(x)
    x = layers.Flatten()(x)
    x = layers.Dense(256)(x)
    x = layers.ReLU()(x)
    x = layers.Dropout(0.5)(x)
    outputs = layers.Dense(num_classes, activation='softmax')(x)

    model = models.Model(inputs, outputs, name='clf_gen')
    return model

clf_gen = build_cnn()
clf_gen.summary()

# =====================================================================
# 5) Treinamento com Validação para Melhoria de Desempenho
# =====================================================================
print('\n' + '='*60)
print('5) Treinamento com Validação')
print('='*60)

optimizer = keras.optimizers.Adam(learning_rate=1e-3)

clf_gen.compile(
    optimizer=optimizer,
    loss='sparse_categorical_crossentropy',
    metrics=['accuracy']
)

reduce_lr = callbacks.ReduceLROnPlateau(
    monitor='val_accuracy',
    factor=0.2,
    patience=5,
    min_lr=1e-5,
    verbose=1
)

early_stop = callbacks.EarlyStopping(
    monitor='val_accuracy',
    patience=10,
    restore_best_weights=True,
    verbose=1
)

checkpoint = callbacks.ModelCheckpoint(
    'melhor_clf_gen.keras',
    monitor='val_accuracy',
    save_best_only=True,
    verbose=1
)

EPOCHS = 50

history = clf_gen.fit(
    train_ds,
    epochs=EPOCHS,
    validation_data=val_ds,
    callbacks=[reduce_lr, early_stop, checkpoint],
    verbose=2
)

# Curvas treino vs. validação
fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(14, 5))

ax1.plot(history.history['accuracy'], label='Treino')
ax1.plot(history.history['val_accuracy'], label='Validação')
ax1.set_title('Acurácia por Época')
ax1.set_xlabel('Época')
ax1.set_ylabel('Acurácia')
ax1.legend()
ax1.grid(True)

ax2.plot(history.history['loss'], label='Treino')
ax2.plot(history.history['val_loss'], label='Validação')
ax2.set_title('Perda (Loss) por Época')
ax2.set_xlabel('Época')
ax2.set_ylabel('Loss')
ax2.legend()
ax2.grid(True)

plt.suptitle('Curvas de Treino vs. Validação', fontsize=14)
plt.tight_layout()
plt.savefig('03_curvas_treino_validacao.png', dpi=150, bbox_inches='tight')
plt.close()
print('Gráfico salvo: 03_curvas_treino_validacao.png')

print(f'\nMelhor acurácia de validação: {max(history.history["val_accuracy"]):.4f} '
      f'(época {np.argmax(history.history["val_accuracy"]) + 1})')

# =====================================================================
# 6) Avaliação no Conjunto de Teste
# =====================================================================
print('\n' + '='*60)
print('6) Avaliação no Conjunto de Teste')
print('='*60)

clf_gen = keras.models.load_model('melhor_clf_gen.keras')

test_loss, test_accuracy = clf_gen.evaluate(test_ds, verbose=0)
print(f'\n=== Resultado Final no Conjunto de Teste ===')
print(f'Acurácia: {test_accuracy:.4f} ({test_accuracy*100:.2f}%)')
print(f'Loss:     {test_loss:.4f}')

y_pred_probs = clf_gen.predict(test_ds)
y_pred = np.argmax(y_pred_probs, axis=1)

# Matriz de confusão
cm = confusion_matrix(y_test, y_pred)

fig, ax = plt.subplots(figsize=(10, 8))
disp = ConfusionMatrixDisplay(confusion_matrix=cm, display_labels=CLASS_NAMES)
disp.plot(ax=ax, cmap='Blues', values_format='d', xticks_rotation=45)
ax.set_title('Matriz de Confusão - Conjunto de Teste', fontsize=14)
plt.tight_layout()
plt.savefig('04_matriz_confusao.png', dpi=150, bbox_inches='tight')
plt.close()
print('Gráfico salvo: 04_matriz_confusao.png')

# Relatório de classificação
print('\n=== Relatório de Classificação por Classe ===')
print(classification_report(y_test, y_pred, target_names=CLASS_NAMES, digits=4))

# Análise de erros
print('\n=== Análise de Erros ===')
cm_no_diag = cm.copy()
np.fill_diagonal(cm_no_diag, 0)

confusoes = []
for i in range(10):
    for j in range(10):
        if i != j and cm_no_diag[i, j] > 0:
            confusoes.append((CLASS_NAMES[i], CLASS_NAMES[j], cm_no_diag[i, j]))

confusoes.sort(key=lambda x: x[2], reverse=True)

print(f'\nTop 10 pares de confusão (real -> predito):')
for real, pred, qtd in confusoes[:10]:
    print(f'  {real:>12} -> {pred:<12}: {qtd} erros')

print(f'\nAcurácia por classe:')
for i in range(10):
    acc_classe = cm[i, i] / cm[i].sum()
    print(f'  {CLASS_NAMES[i]:>12}: {acc_classe:.4f} ({acc_classe*100:.2f}%)')
