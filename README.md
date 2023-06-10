# Create doc-bin for training dataset
```
python3 preprocess.py dataset/training out/dataset/training.spacy
```

# Create doc-bin for dev dataset
```
python3 preprocess.py dataset/dev out/dataset/dev.spacy
```

# To create base config
Follow https://spacy.io/usage/training#quickstart

# To create config file from base config
```
python3 -m spacy init fill-config base_config.cfg config.cfg
```