import os
from typing import Tuple, Union

from beir.retrieval import models
from sentence_transformers import SentenceTransformer
from sentence_transformers.models import Transformer

from custom_pooling import CustomPooling


class CustomBERT(models.SentenceBERT):
    def __init__(self, model_path: Union[str, Tuple] = None, **kwargs):
        super().__init__(model_path=None, **kwargs)
        word_embedding_model = Transformer.load(model_path)
        pooling_model = CustomPooling.load(os.path.join(model_path, '1_CustomPooling'))

        self.q_model = SentenceTransformer(modules=[word_embedding_model, pooling_model])
        self.doc_model = self.q_model
