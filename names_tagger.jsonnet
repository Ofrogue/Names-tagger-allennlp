// jsonnet allows local variables like this
local word_embedding_dim = 0;
local char_embedding_dim = 6;
local embedding_dim = word_embedding_dim + char_embedding_dim;
local hidden_dim = 6;
local num_epochs = 100;
local patience = 10;
local batch_size = 2;
local learning_rate = 0.1;

{
    "train_data_path": './data/train.txt',
    "validation_data_path": './data/val.txt',
    "dataset_reader": {
        "type": "names-tagger-reader",
        "token_indexers": {
            "token_characters": { "type": "characters" }
        }
    },
    "model": {
        "type": "names-tagger",
        "word_embeddings": {
            
            "token_embedders": {
                "token_characters": {
                    "type": "character_encoding",
                    "embedding": {
                        "embedding_dim": char_embedding_dim,
                    },
                    "encoder": {
                        "type": "lstm",
                        "input_size": char_embedding_dim,
                        "hidden_size": char_embedding_dim
                    }
                }
            }
        },
        "encoder": {
            "type": "lstm",
            "input_size": embedding_dim,
            "hidden_size": hidden_dim
        }
    },
    "iterator": {
        "type": "basic",
        "batch_size": batch_size,
    },
    "trainer": {
        "num_epochs": num_epochs,
        "optimizer": {

            "type": "adam"
        },
        "patience": patience
    }
}


