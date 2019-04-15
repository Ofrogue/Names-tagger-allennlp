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

torch.manual_seed(1)


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
        batch_split = 2
        lines = open(file_path).read().split('\n')
        batches = [[unicodeToAscii(line).strip().split() for line in lines[i:i+batch_split]] for i in range(int(len(lines)/batch_split))]
        for batch in batches:
            names, tags = zip(*(pair for pair in batch))
            yield self.text_to_instance([Token(name) for name in names], tags)
       


@Model.register('names-tagger')                
class NamesTagger(Model):

    def __init__(self,

                 word_embeddings: TextFieldEmbedder,

                 encoder: Seq2SeqEncoder,

                 vocab: Vocabulary) -> None:

        super().__init__(vocab)
        self.word_embeddings = word_embeddings
        self.encoder = encoder

        self.hidden2tag = torch.nn.Linear(in_features=encoder.get_output_dim(),
                                          out_features=vocab.get_vocab_size('labels'))

        self.accuracy = CategoricalAccuracy()

    def forward(self,
                name: Dict[str, torch.Tensor],
                label: torch.Tensor = None) -> Dict[str, torch.Tensor]:

        mask = get_text_field_mask(name)

        embeddings = self.word_embeddings(name)

        encoder_out = self.encoder(embeddings, mask)

        tag_logits = self.hidden2tag(encoder_out)
        output = {"tag_logits": tag_logits}

        if label is not None:
            # print('>>>>...............>>>>........>>>>', label.size(), label)
            self.accuracy(tag_logits, label)
            output["loss"] = sequence_cross_entropy_with_logits(tag_logits, label, mask)

        return output

    def get_metrics(self, reset: bool = False) -> Dict[str, float]:
        return {"accuracy": self.accuracy.get_metric(reset)}


# In practice you'd probably do this from the command line:
#   $ allennlp train names_tagger.jsonnet -s serialization_dir --include-package data_reader
# allennlp train names_tagger.jsonnet -s serialization_dir --include-package namestagger


#reader = NamesDatasetReader()

#train_dataset = reader.read('data/train.txt')
#validation_dataset = reader.read('data/val.txt')



#vocab = Vocabulary.from_instances(train_dataset + validation_dataset)




'''

reader = NamesDatasetReader()

train_dataset = reader.read('data/train.txt')
validation_dataset = reader.read('data/val.txt')

vocab = Vocabulary.from_instances(train_dataset + validation_dataset)

EMBEDDING_DIM = 6
HIDDEN_DIM = 6

token_embedding = Embedding(num_embeddings=vocab.get_vocab_size('tokens'),
                            embedding_dim=EMBEDDING_DIM)
word_embeddings = BasicTextFieldEmbedder({"tokens": token_embedding})

lstm = PytorchSeq2SeqWrapper(torch.nn.LSTM(EMBEDDING_DIM, HIDDEN_DIM, batch_first=True))

model = LstmTagger(word_embeddings, lstm, vocab)

if torch.cuda.is_available():
    cuda_device = 0

    model = model.cuda(cuda_device)
else:

    cuda_device = -1

optimizer = optim.SGD(model.parameters(), lr=0.1)

iterator = BucketIterator(batch_size=2, sorting_keys=[("sentence", "num_tokens")])

iterator.index_with(vocab)

trainer = Trainer(model=model,
                  optimizer=optimizer,
                  iterator=iterator,
                  train_dataset=train_dataset,
                  validation_dataset=validation_dataset,
                  patience=10,
                  num_epochs=1000,
                  cuda_device=cuda_device)

trainer.train()


'''