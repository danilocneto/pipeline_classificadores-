# Configuração pré processamento
config_pre_processing = {
    'lowercase': True,
    'normalize_unicode': True,
    'normalization': 'lemmatization',  # escolha entre: 'lemmatization' | 'stemming' | None
    'remove_noise': True,
    'remove_stopwords': True
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
