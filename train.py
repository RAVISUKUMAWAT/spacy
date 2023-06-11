import locale
locale.setlocale(locale.LC_ALL, 'en_US.UTF-8')
import spacy
from spacy.cli.train import train
import os

print(os. getcwd())

# gpu = spacy.prefer_gpu()
# print(gpu)

# can override config info with overrides
# the tutorial config file doesn't have the paths for train/dev corpora
# going to just run this for a few epochs, see how it works
train("./config.cfg",
      output_path='out/medicine_model',
      overrides={"paths.train": "./out/dataset/training.spacy", 
                 "paths.dev": "./out/dataset/dev.spacy",
                 "training.max_epochs": 1})
