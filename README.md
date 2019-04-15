# Names-tagger-allennlp

This is an implementation of [tutorial](https://pytorch.org/tutorials/intermediate/char_rnn_classification_tutorial.html) done with
the help of [allennlp](https://github.com/allenai/allennlp) framework. The model predicts the origination of a surname based on a 
character-level LSTM representation. The baseline was taken from allennlp's [tutorial](https://github.com/allenai/allennlp/tree/master/tutorials/tagger).
Character repressentations are gained with the help of allennlp methods. A surname is passed into the model and treated as a sentence
with one word. There is a redundant LSTM layer into the model because it builds a representation of only one word. So that layer works as an 
extra fully-connected one. This issue was decided to be saved due to lack of documentation about the framework.
# Data
[Data](https://download.pytorch.org/tutorial/data.zip) is the same as into the [tutorial](https://pytorch.org/tutorials/intermediate/char_rnn_classification_tutorial.html): txt
files with surnames. To prepare data for training call ```prepare_data.py``` and ```split_data.py```
# Configuration
Config can be seen into ```names_tagger.jsonnet```. The dimension of character-level LSTM is ```6```, optimizer is ```adam```.
# Training 
To train a model use 
```
allennlp train -s serialization_dir names_tagger.jsonnet --include-package namestagger
```
# Prediction
Example of a prediction
```
allennlp predict ./serialization_dir/model.tar.gz ./data/test.txt --include-package namestagger --predictor names-classifier
```
Lines in a file ```test.txt``` are represented as JSONs:
```{"name": "SAMPLE_NAME"}```
