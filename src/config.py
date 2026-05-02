# Configuração pré processamento
config_pre_processing = {
    'lowercase': True,
    'normalize_unicode': True,
    'normalization': 'lemmatization',  # escolha entre: 'lemmatization' | 'stemming' | None
    'remove_noise': False,
    'remove_stopwords': False
}

# Configuração de vetorização 
config_vectorizer = {
    'strategy': 'tfidf_svd',  # 'bow' | 'tfidf' | 'tfidf_svd' | 'word2vec'
    
    # Parâmetros BoW
    'bow_binary': False,
    'bow_max_features': 100,
    
    # Parâmetros TF-IDF
    'tfidf_sublinear_tf': True,
    'tfidf_ngram_range': (1, 2),
    'tfidf_max_features': 100,
    'tfidf_norm': 'l2',
    
    # Parâmetros TF-IDF + SVD
    'svd_n_components': 100,  # 50, 100 ou 300
    
    # Parâmetros Word2Vec
    'w2v_vector_size': 100
}

# Configuração do pipeline
# steps: ordem de execução das etapas — altere a lista para controlar o fluxo
# Etapas disponíveis: 'pre_processing' | 'vectorization' | 'classification'
config_pipeline = {
    'input_path':  '../data/NPS_COMENTÁRIOS.xlsx',
    'output_path': '../data/NPS_RESULTADO.xlsx',
    'steps': ['pre_processing', 'vectorization', 'classification'],
}

# Configuração do classificador
# model: 'multinomial_nb' | 'bernoulli_nb' | 'logistic_regression' | 'linear_svc' | 'random_forest' | 'lightgbm'
# params: hiperparâmetros explícitos passados diretamente ao construtor do modelo
config_classifier = {
    'model': 'logistic_regression',
    'params': {
        'C': 1.0,
        'max_iter': 1000,
        'solver': 'lbfgs',
    }
}

