For this project the following dependencies are to be installed.

    pip install -q transformers pytorch_lightning nervaluate
    
    Linux:
    wget -q https://raw.githubusercontent.com/dumitrescustefan/ronec/master/data/train.json
    wget -q https://raw.githubusercontent.com/dumitrescustefan/ronec/master/data/valid.json
    wget -q https://raw.githubusercontent.com/dumitrescustefan/ronec/master/data/test.json

    Windows:
    iwr -outf train.json -Uri https://raw.githubusercontent.com/dumitrescustefan/ronec/master/data/train.json -UseBasicParsing
    iwr -outf valid.json -Uri https://raw.githubusercontent.com/dumitrescustefan/ronec/master/data/valid.json -UseBasicParsing
    iwr -outf test.json -Uri https://raw.githubusercontent.com/dumitrescustefan/ronec/master/data/test.json -UseBasicParsing


In `src/train.py` is the training and evaluation for the model.

TODO add code for saving and loading the model plus writing code for evaluation.