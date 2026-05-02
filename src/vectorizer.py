import numpy as np
from sklearn.feature_extraction.text import CountVectorizer, TfidfVectorizer
from sklearn.decomposition import TruncatedSVD
from gensim.models import Word2Vec
class TextVectorizer:
    
    def __init__(self, config: dict):
        self.config = config
        self.strategy = config['strategy']
    
    def _round_vectors(self, vectors):
        # Arredonda todos os vetores para 2 casas decimais
        return [[round(float(val), 2) for val in vec] for vec in vectors]
    
    def fit_transform(self, texts: list) -> list:
        
        if self.strategy == 'bow':
            vectorizer = CountVectorizer(
                binary=self.config.get('bow_binary', False),
                max_features=self.config.get('bow_max_features', 100)
            )
            X = vectorizer.fit_transform(texts)
            vectors = [X[i].toarray()[0] for i in range(len(texts))]
            return self._round_vectors(vectors)
        
        elif self.strategy == 'tfidf':
            vectorizer = TfidfVectorizer(
                sublinear_tf=self.config.get('tfidf_sublinear_tf', False),
                ngram_range=self.config.get('tfidf_ngram_range', (1, 1)),
                max_features=self.config.get('tfidf_max_features', 100),
                norm=self.config.get('tfidf_norm', 'l2')
            )
            X = vectorizer.fit_transform(texts)
            vectors = [X[i].toarray()[0] for i in range(len(texts))]
            return self._round_vectors(vectors)
        
        elif self.strategy == 'tfidf_svd':
            vectorizer = TfidfVectorizer(
                sublinear_tf=self.config.get('tfidf_sublinear_tf', False),
                ngram_range=self.config.get('tfidf_ngram_range', (1, 1)),
                max_features=self.config.get('tfidf_max_features', 1000),
                norm=self.config.get('tfidf_norm', 'l2')
            )
            X_tfidf = vectorizer.fit_transform(texts)
            svd = TruncatedSVD(n_components=self.config.get('svd_n_components', 100))
            X_svd = svd.fit_transform(X_tfidf)
            return self._round_vectors(X_svd)
        
        elif self.strategy == 'word2vec':
            tokens = [t.split() for t in texts]
            model = Word2Vec(
                tokens,
                vector_size=self.config.get('w2v_vector_size', 100),
                window=5,
                min_count=1,
                workers=4
            )
            
            vecs = []
            for t in tokens:
                palavras_validas = [w for w in t if w in model.wv]
                if palavras_validas:
                    vec = np.mean([model.wv[w] for w in palavras_validas], axis=0)
                else:
                    vec = np.zeros(self.config.get('w2v_vector_size', 100))
                vecs.append(vec)
            return self._round_vectors(vecs)
        
        return []

