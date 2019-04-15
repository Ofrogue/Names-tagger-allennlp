# manual classes

from typing import Iterator, List, Dict

import torch
import torch.optim as optim
import numpy as np

from allennlp.data import Instance
from allennlp.data.fields import TextField, SequenceLabelField, LabelField

from allennlp.data.dataset_readers import DatasetReader

from allennlp.common.file_utils import cached_path

from allennlp.data.token_indexers import TokenIndexer, SingleIdTokenIndexer, TokenCharactersIndexer
from allennlp.data.tokenizers import Token

from allennlp.data.vocabulary import Vocabulary

from allennlp.models import Model

from allennlp.modules.text_field_embedders import TextFieldEmbedder, BasicTextFieldEmbedder
from allennlp.modules.token_embedders import Embedding
from allennlp.modules.seq2seq_encoders import Seq2SeqEncoder, PytorchSeq2SeqWrapper
from allennlp.nn.util import get_text_field_mask, sequence_cross_entropy_with_logits

from allennlp.training.metrics import CategoricalAccuracy

from allennlp.data.iterators import BucketIterator

from allennlp.training.trainer import Trainer

from allennlp.predictors import SentenceTaggerPredictor

import unicodedata

import string

# torch.manual_seed(1)


all_letters = string.ascii_letters + " .,;'"
# Turn a Unicode string to plain ASCII, thanks to https://stackoverflow.com/a/518232/2809427
def unicodeToAscii(s):
    return ''.join(
        c for c in unicodedata.normalize('NFD', s)
        if unicodedata.category(c) != 'Mn'
        and c in all_letters
    )


@DatasetReader.register('names-tagger-reader')
class NamesDatasetReader(DatasetReader):
    """
    DatasetReader for names  
    """

    def __init__(self, token_indexers: Dict[str, TokenIndexer] = None) -> None:
        super().__init__(lazy=False)
        self.token_indexers = token_indexers or {"tokens": SingleIdTokenIndexer()}

    def text_to_instance(self, tokens: List[Token], tags: List[str] = None) -> Instance:
        name_field = TextField(tokens, self.token_indexers)
        fields = {"name": name_field}

        if tags:
            label_field = SequenceLabelField(labels=tags, sequence_field=name_field)
            fields["label"] = label_field

        return Instance(fields)


    def _read(self, file_path: str) -> Iterator[Instance]:
        batch_split = 1
        lines = open(file_path).read().split('\n')
        batches = [[unicodeToAscii(line).strip().split() for line in lines[i:i+batch_split]] for i in range(int(len(lines)/batch_split))]
        for batch in batches:
            names, tags = zip(*(pair for pair in batch))
            yield self.text_to_instance([Token(name) for name in names], tags)
       



