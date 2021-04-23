# Purpose
We want to gain useful generalizable knowledge about the inner workings of non-finetuned transformer language models. We achieve this by running a number of experiments to probe the hidden layer activations using different datasets, model sizes, pooling methods and classifiers, and visualize this interactively with a StreamLit app. This information will help us in future embedding experiments and eegi, among other things.

# Usage
1. Prepare data by the running get_data.py script: ```python get_data.py <data_save_path>```
2. Create classifiers
Classifiers come in two flavours: sklearn and pytorch, these are taken care off by the base classes SingleStepOpt and MultiStepOpt resp. (See Modules section).

Sklearn-type classifiers need following the attributes:
- name: identifier name for logging
- classifier: sklearn classifier. Needs to have a fit and transform method. If transform is not available the forward methods needs to be overwritten.
- discrete_targets: boolean to indicate whether the classifier requires binary (0 or 1) targets.

For example
``` py
class LDA(SingleStepOpt):
    def __init__(self, input_size):
        super().__init__()
        self.name = 'LDA'
        self.discrete_targets = True
        self.classifier = LinearDiscriminantAnalysis(n_components=1)
```
Pytorch-type classifiers need
- name attribute: identifier name for logging
- forward method

For example
``` py
class SingleLayer(MultiStepOpt):
    def __init__(self, input_size):
        super().__init__()
        self.name = 'SingleLayer'
        self.linear = nn.Linear(input_size, 1)
    
    def forward(self, X):
        return self.linear(X)
```

4. Call run_exp. For example
``` py
import glob
import torch
import torch.nn as nn
from pipeline import NLPDataset, LMModel, PoolToken, SingleStepOpt, MultiStepOpt, RunExperiments, evaluator
from classifiers import LDA, SingleLayer

device = 'cuda' if torch.cuda.is_available() else 'cpu'
n_folds = 10
dataset_paths = glob.glob('data/*')
poolers = [PoolToken(layer = 1), PoolToken(quantile = 0.4), PoolToken(layer = 'all', layer_method = 'mean')]
classifiers = [LDA, SingleLayer]
model_names = ['gpt2', 'gpt2-large']

exps = RunExperiments(n_folds, dataset_paths, poolers, classifiers, model_names, evaluator, device)
exps.run_all()
```

5. Visualise: run visualisation_experiments.py: ```streamlit run streamlit_app.py```

# Modules
The program consists of the following parts
## NLPDataset
This loads the text into a class. Can be given either a path to a data-json file or dict.

\_\_init\_\_ args:
- json_input: Path to dataset json file or loaded json file itself

Methods:
- load_embeds_into_memory: Given a model and pooler, processes all embeddings and stores them
- \_\_len\_\_
- \_\_getitem\_\_

## LMModel
(Hugging Face) Language Model Model Wrapper

init args:
- model_name: Hugging face model name or dictionary that contains a model and tokenizer

Methods:
- \_\_call\_\_: Takes a list of strings and returns Hugging face style output dictionary

## PoolToken
Selects (combinations of) tokens from hugging face output dictionary

\_\_init\_\_ args:
- layer: Which layer the tokens are to be selected from
    - layer index
    - list of layer indices
    - -1: second to last layer
    - 'all': all layers except the first and last one
- quantile: Which quantile of tokens is selected (0. = first token, 1. = last)
    - quantile
    - -1: last token
- layer_method: If multiple layers are selected, layer_method can be used to reduce them
    - 'mean', 'max', 'min': pool over layer dimension
    - 'extend': concatenates embeddings in embed dimension

Methods:
- \_\_call\_\_: Takes hugging face output dictionary and returns (batch_size, hidden_size) sized tensor of embeddings

## Classifiers
### SingleStepOpt
Base class for sklearn style classifiers.
Distinguished between regression (continuous targets) and classification (binary targets)

\_\_init\_\_ args: None

Methods:
- fit: fits classifier and sets optimal thresholds if the task is classification
- set_thresholds: Uses a validation set to find optimal thresholds for both accuracy and F1
- predict: Returns predictions for targets, using the thresholds if necessary
- forward: Maps inputs to targets

### MultiStepOpt
Base class for pytorch style classifiers, uses gradient descent.

\_\_init\_\_ args:
- batch_size
- max_epoch

Methods:
- init_optimizer: Initializes optimizer, scheduler and loss function
- training_step: Training step used in fit method
- test_step: Test step used in fit method for validation and in predict to get predictions
- fit: Training loop, stores losses
- predict: Given testset, returns its corresponding predictions from the classifier
    
## RunExperiments
Runs all possible combinations of experiments given datasets, poolers, classifiers and language models, and logs them

\_\_init\_\_ args:
- n_folds: Number of folds for cross validation (Not real folds cause sample will typically overlap)
- dataset_paths
- poolers
- classifiers
- model_names
- evaluator
- device
- logs_folder: Folder in which the experiments logs are to be stored, will create folder if it doesn't already exist

Methods:
- log_data: Logs experiment results to logs_folder
- run_all: single_batch_limit is not yet implemented

# TODO
- make concat work properly
- log val_size, test_size
- add classification using attention
- BERT type models
- log number of params per model
- larger datasets: generate embeddings on-the-fly
