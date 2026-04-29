import pandas as pd
import config as config

import spacy

import unicodedata
from nltk.stem import RSLPStemmer
from nltk.corpus import stopwords

class BasicTextPreprocessor:
    
    def __init__(self, config: dict):
    # Inicializa o processador com as configurações fornecidas
        self.config = config
        
        self.nlp = None
        if config.get('normalization') == 'lemmatization':
            self.nlp = spacy.load("pt_core_news_sm")
        
        self.stemmer = None
        if config.get('normalization') == 'stemming':
            self.stemmer = RSLPStemmer()
        
        self.stop_words = set(stopwords.words('portuguese'))

        
    
    
    def tokenize(self, text: str) -> list:
    #Divide texto em tokens
        if self.nlp:
            doc = self.nlp(text)
            return [token.text for token in doc]
        else:
            return text.split()
    
    
    def lowercase_and_normalize_unicode(self, tokens: list) -> list:
    # Converte minúsculas e remove acentos
        normalized_tokens = []
        
        for token in tokens:
            if self.config.get('lowercase'):
                token = token.lower()
            
            if self.config.get('normalize_unicode'):
                token = unicodedata.normalize('NFKD', token)
                token = ''.join([c for c in token if not unicodedata.combining(c)])
            
            normalized_tokens.append(token)
        
        return normalized_tokens
    
    
    def lemmatize_or_stem(self, tokens: list) -> list:
    # Reduz palavras à forma base (lemma ou stem)
        if self.config.get('normalization') == 'lemmatization':
            text = ' '.join(tokens)
            doc = self.nlp(text)
            return [token.lemma_ for token in doc]
        
        elif self.config.get('normalization') == 'stemming':
            return [self.stemmer.stem(token) for token in tokens]
        
        else:
            return tokens
    
    
    def remove_noise_and_stopwords(self, tokens: list) -> list:
  # Remove pontuação, números e stopwords
        clean_tokens = []
        
        for token in tokens:
            if self.config.get('remove_noise'):
                if not token.isalnum() or token.isdigit() or len(token) < 2:
                    continue
            
            if self.config.get('remove_stopwords'):
                if token.lower() in self.stop_words:
                    continue
            
            clean_tokens.append(token)
        
        return clean_tokens
    
    
    def pre_process(self, text: str) -> str:
    # Pipeline completo: aplica todas as 4 etapas
        tokens = self.tokenize(text)
        tokens = self.lowercase_and_normalize_unicode(tokens)
        tokens = self.lemmatize_or_stem(tokens)
        tokens = self.remove_noise_and_stopwords(tokens)
        return ' '.join(tokens)
    
df = pd.read_excel('../data/NPS_COMENTÁRIOS.xlsx')

pre_processor = BasicTextPreprocessor(config.config_pre_processing)
df['texto_processado'] = df['Comentários'].apply(
    lambda x: pre_processor.pre_process(x) if pd.notna(x) else ""
)

df.to_excel('../data/NPS_COMENTA_RIOS_PROCESSADO.xlsx', index=False)

