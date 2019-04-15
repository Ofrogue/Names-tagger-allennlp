from overrides import overrides

from allennlp.common.util import JsonDict
from allennlp.data import Instance
from allennlp.predictors.predictor import Predictor
from allennlp.data.tokenizers import Token


@Predictor.register('names-classifier')
class NamesTaggerPredictor(Predictor):
    """"Predictor wrapper for the NamesTagger"""
    
    @overrides
    def _json_to_instance(self, json_dict: JsonDict) -> Instance:
        name = json_dict['name']
        
        return self._dataset_reader.text_to_instance(tokens=[Token(name)])