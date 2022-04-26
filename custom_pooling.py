import json
import os
from typing import Dict

import torch
from sentence_transformers.models import Pooling
from torch import Tensor


class CustomPooling(Pooling):
    def __init__(self, word_embedding_dimension, pooling_mode='mean',
                 pooling_mode_max_tokens: bool = False,
                 pooling_mode_mean_tokens: bool = True):
        assert pooling_mode in {'mean', 'max'}
        super().__init__(word_embedding_dimension, pooling_mode=pooling_mode)

        self.config_keys = ['word_embedding_dimension', 'pooling_mode_mean_tokens', 'pooling_mode_max_tokens']
        self.pooling_output_dimension = word_embedding_dimension

    def forward(self, features: Dict[str, Tensor]):
        token_embeddings = features['token_embeddings']
        pooling_mask = features['pooling_mask']

        ## Pooling strategy
        if self.pooling_mode_max_tokens:
            input_mask_expanded = pooling_mask.unsqueeze(-1).expand(token_embeddings.size()).float()
            token_embeddings[input_mask_expanded == 0] = -1e9  # Set padding tokens to large negative value
            max_over_time = torch.max(token_embeddings, 1)[0]
            output_vector = max_over_time
        else:
            input_mask_expanded = pooling_mask.unsqueeze(-1).expand(token_embeddings.size()).float()
            sum_embeddings = torch.sum(token_embeddings * input_mask_expanded, 1)
            sum_mask = input_mask_expanded.sum(1)
            sum_mask = torch.clamp(sum_mask, min=1e-9)
            output_vector = sum_embeddings / sum_mask

        features.update({'sentence_embedding': output_vector})
        return features

    @staticmethod
    def load(input_path):
        with open(os.path.join(input_path, 'config.json')) as fIn:
            config = json.load(fIn)

        return CustomPooling(**config)
