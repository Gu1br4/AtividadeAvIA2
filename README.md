# Avaliação Prática 2 — CIFAR-10 Classification Pipeline

**Disciplina:** Inteligência Artificial e Aprendizado de Máquina  
**Instituição:** PUC-Campinas — Engenharia de Software  
**Framework:** PyTorch 2.12 (CPU)  
**Semente:** `random_state = 2025`

---

## Estrutura do Projeto

```
AtividadeAvIA2/
├── exercicio1_cifar10.py      # Script principal do pipeline
├── cifar10_pipeline.ipynb     # Notebook com pipeline completo
├── requirements.txt           # Dependências do projeto
├── outputs/                   # Gerado ao rodar o script
│   ├── 01_amostras_aleatorias.png
│   ├── 02_data_augmentation.png
│   ├── 03_curvas_treino_validacao.png
│   ├── 04_matriz_confusao.png
│   └── melhor_clfgen.pt       # Melhor modelo salvo
└── data/                      # Baixado automaticamente na 1ª execução
    └── cifar-10-batches-py/
```

---

## Pré-requisitos

- Python 3.12+ (testado em 3.14)
- [uv](https://docs.astral.sh/uv/) (gerenciador de pacotes recomendado) **ou** pip

---

## Instalação das Dependências

### Opção 1 — uv (recomendado)

```powershell
# Criar ambiente virtual
uv venv .venv

# Ativar ambiente (PowerShell)
.venv\Scripts\Activate.ps1

# Instalar dependências
uv pip install pandas matplotlib scikit-learn pillow
uv pip install torch torchvision --index-url https://download.pytorch.org/whl/cpu
```

### Opção 2 — pip comum

```powershell
python -m venv .venv
.venv\Scripts\Activate.ps1
pip install -r requirements.txt --index-url https://download.pytorch.org/whl/cpu
```

---

## Como Rodar

### Script Python (terminal)

```powershell
# Ativar ambiente virtual primeiro
.venv\Scripts\Activate.ps1

# Rodar pipeline completo
python -X utf8 exercicio1_cifar10.py
```

> **Nota:** Na primeira execução o CIFAR-10 (~170 MB) é baixado automaticamente para `data/`.

### Notebook (VS Code ou Jupyter)

1. Selecione o kernel `.venv` no VS Code:  
   `Ctrl+Shift+P` → **Python: Select Interpreter** → `.venv\Scripts\python.exe`
2. Abra `cifar10_pipeline.ipynb`
3. **Run All**

### Google Colab (GPU gratuita — recomendado)

1. Acesse [colab.research.google.com](https://colab.research.google.com)
2. Faça upload de `cifar10_pipeline.ipynb`
3. `Runtime` → `Change runtime type` → **T4 GPU**
4. Execute todas as células

No Colab, adicione na primeira célula:
```python
!pip install torch torchvision
```

---

## Tempo de Execução Estimado

| Hardware | Tempo por época | Total (50 épocas) |
|----------|----------------|-------------------|
| CPU      | ~5–10 min      | ~4–8 horas        |
| GPU (T4) | ~20–30 seg     | ~15–25 min        |

---

## Pipeline

| Etapa | Descrição |
|-------|-----------|
| 1 | Carrega CIFAR-10, cria `dfgen` com 60.000 imagens |
| 2 | Normaliza + data augmentation (RandomCrop, RandomHorizontalFlip) |
| 3 | Divisão estratificada 70/15/15 com semente 2025 |
| 4 | CNN `clfgen`: 2 blocos conv + BN + ReLU + MaxPool, head com Dropout |
| 5 | Treino com Adam, ReduceLROnPlateau, EarlyStopping (patience=10) |
| 6 | Avaliação: acurácia, matriz de confusão, relatório por classe |

---

## Outputs Gerados

| Arquivo | Conteúdo |
|---------|----------|
| `outputs/01_amostras_aleatorias.png` | 10 imagens aleatórias com classe |
| `outputs/02_data_augmentation.png`  | Exemplos de augmentation |
| `outputs/03_curvas_treino_validacao.png` | Curvas de loss e acurácia |
| `outputs/04_matriz_confusao.png`    | Matriz de confusão no teste |
| `outputs/melhor_clfgen.pt`          | Pesos do melhor modelo |
