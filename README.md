# Pipeline de Classificação de Texto

Aplicação web para classificação de comentários em português usando um pipeline configurável de NLP — pré-processamento, vetorização e classificação.

---

## Pré-requisitos

- Python 3.10+
- [Homebrew](https://brew.sh) (macOS)

---

## Instalação

### 1. Clone o repositório e entre na pasta

```bash
cd pipeline_classificadores-
```

### 2. Crie e ative o ambiente virtual

```bash
python3 -m venv .venv
source .venv/bin/activate
```

### 3. Instale as dependências Python

```bash
pip install flask pandas scikit-learn spacy nltk gensim lightgbm openpyxl numpy
```

### 4. Instale o modelo de português do spaCy


```bash
python -m spacy download pt_core_news_sm
```

### 5. Instale os recursos do NLTK

```bash
python -c "import nltk; nltk.download('stopwords'); nltk.download('rslp')"
```

### 6. Instale a dependência OpenMP (necessária para o LightGBM no macOS)

```bash
brew install libomp
```

---

## Rodando a aplicação web

```bash
cd src
python app.py
```

Acesse no browser:

```
http://127.0.0.1:5000
```

> Use `127.0.0.1`, não `localhost` — alguns setups de macOS têm problema de resolução de DNS com `localhost`.

> Não abra o arquivo HTML diretamente pelo VS Code (Live Server). O app precisa ser servido pelo Flask.

---

## Como usar

1. **Arquivo** — selecione ou arraste uma planilha `.xlsx`. O teste pode ser feito com a planilha `NPS_COMENTÁRIOS.xlsx` disponibilizada pela Zamp (já presente na pasta `data/`), ou com qualquer outra planilha que possua as colunas `Comentários` (texto a classificar) e `Classificação` (rótulos para treino).
2. **Etapas do pipeline** — ative ou desative clicando nas pills: Pré-processamento, Vetorização, Classificação.
3. **Configure cada etapa** — os parâmetros aparecem abaixo de cada etapa ativa.
4. **Executar pipeline** — clique no botão. Ao terminar, as métricas (Acurácia, F1, Precisão, Recall) são exibidas e um botão de download do resultado aparece.

---

## Rodando via linha de comando (sem interface web)

```bash
cd src
python pipeline.py
```

A entrada e saída são configuradas em [src/config.py](src/config.py):

```python
config_pipeline = {
    'input_path':  '../data/NPS_COMENTÁRIOS.xlsx',
    'output_path': '../data/NPS_RESULTADO.xlsx',
    'steps': ['pre_processing', 'vectorization', 'classification'],
}
```

---

## Opções de configuração

### Pré-processamento

| Parâmetro | Opções | Descrição |
|---|---|---|
| `normalization` | `lemmatization` \| `stemming` \| `None` | Forma de redução morfológica |
| `lowercase` | `True` / `False` | Converte para minúsculas |
| `normalize_unicode` | `True` / `False` | Remove acentos |
| `remove_noise` | `True` / `False` | Remove pontuação e números |
| `remove_stopwords` | `True` / `False` | Remove stopwords em português |

### Vetorização

| Estratégia | Descrição |
|---|---|
| `tfidf_svd` | TF-IDF + redução dimensional com SVD (padrão) |
| `tfidf` | TF-IDF puro |
| `bow` | Bag of Words |
| `word2vec` | Embeddings Word2Vec |

### Classificadores

| Modelo | Identificador |
|---|---|
| Regressão Logística | `logistic_regression` |
| Linear SVC | `linear_svc` |
| Random Forest | `random_forest` |
| Naive Bayes Multinomial | `multinomial_nb` |
| Naive Bayes Bernoulli | `bernoulli_nb` |
| LightGBM | `lightgbm` |

---

## Estrutura do projeto

```
pipeline_classificadores-/
├── data/
│   ├── NPS_COMENTÁRIOS.xlsx   # planilha de entrada
│   └── NPS_RESULTADO.xlsx     # resultado gerado
└── src/
    ├── app.py                 # servidor Flask (interface web)
    ├── pipeline.py            # execução via linha de comando
    ├── config.py              # configurações padrão
    ├── pre_processor.py       # etapa de pré-processamento
    ├── vectorizer.py          # etapa de vetorização
    ├── classifier.py          # etapa de classificação
    └── templates/
        └── index.html         # interface web
```

---

## Configurando diretamente no código

Toda a configuração feita pela interface web também pode ser definida diretamente em [src/config.py](src/config.py), sem precisar rodar o servidor Flask. É a forma mais direta de usar o pipeline.

Edite os dicionários no arquivo e rode `python pipeline.py`:

```python
# Pré-processamento
config_pre_processing = {
    'lowercase': True,
    'normalize_unicode': True,
    'normalization': 'lemmatization',  # 'lemmatization' | 'stemming' | None
    'remove_noise': True,
    'remove_stopwords': True,
}

# Vetorização
config_vectorizer = {
    'strategy': 'tfidf_svd',  # 'bow' | 'tfidf' | 'tfidf_svd' | 'word2vec'
    'svd_n_components': 100,
    'tfidf_max_features': 100,
    'tfidf_ngram_range': (1, 2),
    'tfidf_sublinear_tf': True,
    'tfidf_norm': 'l2',
}

# Classificador
config_classifier = {
    'model': 'logistic_regression',  # veja tabela de modelos acima
    'params': {
        'C': 1.0,
        'max_iter': 1000,
        'solver': 'lbfgs',
    }
}

# Pipeline: defina quais etapas executar e em qual ordem
config_pipeline = {
    'input_path':  '../data/NPS_COMENTÁRIOS.xlsx',
    'output_path': '../data/NPS_RESULTADO.xlsx',
    'steps': ['pre_processing', 'vectorization', 'classification'],
}
```

Para rodar:

```bash
cd src
python pipeline.py
```

O resultado é salvo automaticamente no caminho definido em `output_path`, e as métricas são impressas no terminal.
