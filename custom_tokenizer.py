from typing import List

import nltk
from torch import Tensor
from transformers import BatchEncoding
from transformers.models.bert.tokenization_bert_fast import BertTokenizerFast


class CustomTokenizer(BertTokenizerFast):
    def __init__(self, *args, **kwargs):
        self.sw_mode = kwargs.get('sw_mode')
        super().__init__(*args, **kwargs)
        if self.sw_mode == 'nltk':
            nltk.download('stopwords')
            self.stopwords = nltk.corpus.stopwords.words('english')
        elif self.sw_mode == 'lucene':
            self.stopwords = ["a", "an", "and", "are", "as", "at", "be", "but", "by",
                              "for", "if", "in", "into", "is", "it",
                              "no", "not", "of", "on", "or", "such",
                              "that", "the", "their", "then", "there", "these",
                              "they", "this", "to", "was", "will", "with"]
        else:
            raise ValueError(f'Unexpected sw_mode: {self.sw_mode}')
        print(f'CustomTokenizer initialized, using {self.sw_mode} stopwords')
        self.stopwords: List[List[int]] = super().__call__(self.stopwords,
                                                           add_special_tokens=False)['input_ids']
        self.register_for_auto_class()

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
                    if all([batch_encoding['input_ids'][index[0], index[1]+i] == sw[i] for i in range(1, len(sw))]) \
                            and (batch_encoding.word_ids(index[0])[index[1]+len(sw)-1] != batch_encoding.word_ids(index[0])[index[1]+len(sw)]):
                        pooling_mask[index[0], index[1]:index[1]+len(sw)] = 0
                except IndexError:
                    continue

        return pooling_mask
