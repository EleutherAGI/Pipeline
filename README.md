# Purpose
We want to gain useful generalizable knowledge about the inner workings of non-finetuned transformer language models. We achieve this by running a number of experiments to probe the hidden layer activations using different datasets, model sizes, pooling methods and classifiers, and visualize this interactively with a StreamLit app. This information will help us in future embedding experiments and eegi, among other things.

# How to use
1. Prepare data by the running get_data.py script. -data_save_path
2. Create classifiers
Classifiers come in two flavours: sklearn and pytorch.


BaseClass Sklearn style classifiers (SingleStepOpt) get fitted by a single step and depending on the output, require a set of thresholds to be found. Pytorch style classifiers (MultiStepOpt) use gradient descent and are optimizer over multiple training epochs.
In both cases they need an input_size argument and a str name attribute for logging.
Sklearn classifiers need a classifier and a bool whether or not the targets are discrete (binary). For example
``` py
class LDA(SingleStepOpt):
    def __init__(self, input_size):
        super().__init__()
        self.name = 'LDA'
        self.discrete_targets = True
        self.classifier = LinearDiscriminantAnalysis(n_components=1)
```
Pytorch classifiers need a forward method. For example
``` py
class SingleLayer(MultiStepOpt):
    def __init__(self, input_size):
        super().__init__()
        self.name = 'SingleLayer'
        self.linear = nn.Linear(input_size, 1)
    
    def forward(self, X):
        return self.linear(X)
```

forward

4. Call run_exp
5. Visualise

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
(Hugging Face) Language Model Wrapper

init args:
- model_name: Hugging face model name or dictionary that contains a model and tokenizer

Methods:
- \_\_call\_\_: Takes a list of strings and returns Hugging face style output dictionary

## PoolToken
Selects (combinations of) tokens from hugging face output dictionary

\_\_init\_\_ args:
layer: Which layer the tokens are to be selected from
-- layer index
-- list of layer indices
-- -1: second to last layer
-- 'all': all layers except the first and last one
quantile: Which quantile of tokens is selected (0. = first token, 1. = last)
-- quantile
-- -1: last token
layer_method: If multiple layers are selected, layer_method can be used to reduce them
-- 'mean', 'max', 'min': pool over layer dimension
-- 'extend': concatenates embeddings in embed dimension

Methods:
- \_\_call\_\_: Takes hugging face output dictionary and returns (batch_size, hidden_size) sized tensor of embeddings
    
## evaluator
returns MSE for regression and MSE, accuracy and F1 for binary

## Classifiers
### SingleStepOpt
This corresponds to sklearn style optimizers

### MultiStepOpt
Pytorch style neural nets

# TODO
- make concat work properly
