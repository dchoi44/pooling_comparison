from typing import List

import nltk
from torch import Tensor
from transformers import BatchEncoding
from transformers.models.bert.tokenization_bert_fast import BertTokenizerFast


class CustomTokenizer(BertTokenizerFast):
    def __init__(self, *args, **kwargs):
        nltk.download('stopwords')
        super().__init__(*args, **kwargs)
        self.stopwords: List[List[int]] = super().__call__(nltk.corpus.stopwords.words('english'),
                                                           add_special_tokens=False)['input_ids']

    def __call__(self, text, **kwargs):
        kwargs['text'] = text
        batch_encoding: BatchEncoding = super().__call__(**kwargs)
        batch_encoding['pooling_mask'] = self._build_pooling_mask(batch_encoding)
        return batch_encoding

    def _build_pooling_mask(self, batch_encoding: BatchEncoding) -> Tensor:
        pooling_mask = batch_encoding['attention_mask'].detach().clone()
        for sw in self.stopwords:
            sw_indices = (batch_encoding['input_ids'] == sw[0]).nonzero()
            for index in sw_indices:
                try:
                    if all([batch_encoding['input_ids'][index[0], index[1]+i] == sw[i] for i in range(1, len(sw))]):
                        pooling_mask[index[0], index[1]:index[1]+len(sw)] = 0
                except IndexError:
                    continue

        return pooling_mask
