# Este arquivo é o ponto de entrada da aplicação Flask. Ele define as rotas, processa os dados enviados pelo usuário, executa as etapas de pré-processamento, vetorização e classificação, e retorna os resultados em formato Excel.

import io
import os
import sys
import numpy as np
import pandas as pd
from flask import Flask, render_template, request, send_file, jsonify

sys.path.insert(0, os.path.dirname(__file__))
from pre_processor import BasicTextPreprocessor
from vectorizer import TextVectorizer
from classifier import Classifier

app = Flask(__name__)
app.config['MAX_CONTENT_LENGTH'] = 32 * 1024 * 1024  # 32 MB


def _bool(v):
    return v == 'true'


def build_configs(form):
    pre = {
        'lowercase':         _bool(form.get('lowercase', 'true')),
        'normalize_unicode': _bool(form.get('normalize_unicode', 'true')),
        'normalization':     form.get('normalization', 'lemmatization') or None,
        'remove_noise':      _bool(form.get('remove_noise', 'true')),
        'remove_stopwords':  _bool(form.get('remove_stopwords', 'true')),
    }
    if pre['normalization'] == 'none':
        pre['normalization'] = None

    vec = {
        'strategy':           form.get('strategy', 'tfidf_svd'),
        'bow_binary':         _bool(form.get('bow_binary', 'false')),
        'bow_max_features':   int(form.get('bow_max_features', 100)),
        'tfidf_sublinear_tf': _bool(form.get('tfidf_sublinear_tf', 'true')),
        'tfidf_ngram_range':  (1, int(form.get('tfidf_ngram_max', 2))),
        'tfidf_max_features': int(form.get('tfidf_max_features', 100)),
        'tfidf_norm':         form.get('tfidf_norm', 'l2'),
        'svd_n_components':   int(form.get('svd_n_components', 100)),
        'w2v_vector_size':    int(form.get('w2v_vector_size', 100)),
    }

    model = form.get('model', 'logistic_regression')
    clf_params = {}
    if model == 'logistic_regression':
        clf_params = {
            'C':        float(form.get('lr_C', 1.0)),
            'max_iter': int(form.get('lr_max_iter', 1000)),
            'solver':   form.get('lr_solver', 'lbfgs'),
        }
    clf = {'model': model, 'params': clf_params}

    steps_order = ['pre_processing', 'vectorization', 'classification']
    steps = [s for s in steps_order if form.get(s) == 'on']

    return pre, vec, clf, steps


@app.route('/')
def index():
    return render_template('index.html')


@app.route('/run', methods=['POST'])
def run():
    file = request.files.get('file')
    if not file or file.filename == '':
        return jsonify(error='Nenhum arquivo enviado.'), 400

    try:
        df = pd.read_excel(file)
    except Exception as e:
        return jsonify(error=f'Erro ao ler arquivo: {e}'), 400

    if 'Comentários' not in df.columns:
        cols = ', '.join(df.columns.tolist())
        return jsonify(error=f'Coluna "Comentários" não encontrada. Colunas disponíveis: {cols}'), 400

    pre_cfg, vec_cfg, clf_cfg, steps = build_configs(request.form)

    try:
        if 'pre_processing' in steps:
            preprocessor = BasicTextPreprocessor(pre_cfg)
            df['texto_processado'] = df['Comentários'].apply(
                lambda x: preprocessor.pre_process(str(x)) if pd.notna(x) else ''
            )

        if 'vectorization' in steps:
            col = 'texto_processado' if 'texto_processado' in df.columns else 'Comentários'
            vectorizer = TextVectorizer(vec_cfg)
            df['vetor'] = vectorizer.fit_transform(df[col].fillna('').tolist())

        metrics = None
        if 'classification' in steps:
            classifier = Classifier(clf_cfg)
            X = df['vetor'].tolist()
            y = df['Classificação'].tolist()
            classifier.fit(X, y)
            df['predicao'] = classifier.predict(X)
            from sklearn.metrics import accuracy_score, f1_score, precision_score, recall_score
            y_pred = df['predicao'].tolist()
            metrics = {
                'accuracy':  round(accuracy_score(y, y_pred), 4),
                'f1':        round(f1_score(y, y_pred, average='weighted', zero_division=0), 4),
                'precision': round(precision_score(y, y_pred, average='weighted', zero_division=0), 4),
                'recall':    round(recall_score(y, y_pred, average='weighted', zero_division=0), 4),
            }

    except Exception as e:
        return jsonify(error=str(e)), 500

    out_cols = [c for c in df.columns if c != 'vetor']
    buf = io.BytesIO()
    df[out_cols].to_excel(buf, index=False)
    buf.seek(0)

    import base64
    b64 = base64.b64encode(buf.read()).decode()
    return jsonify(metrics=metrics, file_b64=b64, filename='resultado.xlsx')


if __name__ == '__main__':
    app.run(debug=True, port=5000)
