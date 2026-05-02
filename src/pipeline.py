import pandas as pd
from pre_processor import BasicTextPreprocessor
from vectorizer import TextVectorizer
from classifier import Classifier
import config as cfg


_VALID_STEPS = {'pre_processing', 'vectorization', 'classification'}


class Pipeline:

    def __init__(self, pipeline_config: dict):
        self.input_path = pipeline_config['input_path']
        self.output_path = pipeline_config['output_path']
        self.steps = pipeline_config['steps']

        unknown = set(self.steps) - _VALID_STEPS
        if unknown:
            raise ValueError(f"Etapas desconhecidas: {unknown}. Opções: {_VALID_STEPS}")

        self._handlers = {
            'pre_processing':  self._run_pre_processing,
            'vectorization':   self._run_vectorization,
            'classification':  self._run_classification,
        }

    def run(self) -> pd.DataFrame:
        df = pd.read_excel(self.input_path)

        for step in self.steps:
            print(f"[pipeline] executando: {step}")
            df = self._handlers[step](df)

        df.to_excel(self.output_path, index=False)
        print(f"[pipeline] resultado salvo em: {self.output_path}")
        return df

    def _run_pre_processing(self, df: pd.DataFrame) -> pd.DataFrame:
        preprocessor = BasicTextPreprocessor(cfg.config_pre_processing)
        df['texto_processado'] = df['Comentários'].apply(
            lambda x: preprocessor.pre_process(x) if pd.notna(x) else ''
        )
        return df

    def _run_vectorization(self, df: pd.DataFrame) -> pd.DataFrame:
        vectorizer = TextVectorizer(cfg.config_vectorizer)
        df['vetor'] = vectorizer.fit_transform(df['texto_processado'].tolist())
        return df

    def _run_classification(self, df: pd.DataFrame) -> pd.DataFrame:
        classifier = Classifier(cfg.config_classifier)
        X = df['vetor'].tolist()
        y = df['Classificação'].tolist()
        classifier.fit(X, y)
        df['predicao'] = classifier.predict(X)
        classifier.evaluate(X, y)
        return df


# Execução
pipeline = Pipeline(cfg.config_pipeline)
pipeline.run()
