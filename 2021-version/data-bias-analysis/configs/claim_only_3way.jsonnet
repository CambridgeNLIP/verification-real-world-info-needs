local transformer_model = "bert-base-uncased";
local transformer_dim = 768;
local cls_is_last_token = false;

{
  "dataset_reader": {
    "type": "qwic_claim_only",
    "token_indexers": {
      "tokens": {
        "type": "pretrained_transformer",
        "model_name": transformer_model,
        "max_length": 256
      }
    },
    "tokenizer": {
      "type": "pretrained_transformer",
      "model_name": transformer_model
    }
  },
  "train_data_path": "ncmace95/train.jsonl",
  "validation_data_path": "ncmace95/dev.jsonl",
  "model": {
    "type": "simple_perclass",
    "text_field_embedder": {
      "token_embedders": {
        "tokens": {
          "type": "pretrained_transformer",
          "model_name": transformer_model,
          "max_length": 256
        }
      }
    },
    "seq2vec_encoder": {
       "type": "cls_pooler",
       "embedding_dim": transformer_dim,
       "cls_is_last_token": cls_is_last_token
    },
    "feedforward": {
      "input_dim": transformer_dim,
      "num_layers": 1,
      "hidden_dims": transformer_dim,
      "activations": "tanh"
    },
    "dropout": 0.1
  },
  "data_loader": {
    "batch_sampler": {
      "type": "bucket",
      "batch_size" : 16
    }
  },
  "trainer": {
    "num_epochs": 3,
    "cuda_device" : 0,
    "validation_metric": "+accuracy",
    "learning_rate_scheduler": {
      "type": "slanted_triangular",
      "cut_frac": 0.06
    },
    "optimizer": {
      "type": "huggingface_adamw",
      "lr": 2e-5,
      "weight_decay": 0.1,
    }
  }
}