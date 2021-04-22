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

# How it works
The program consists of the following parts
## NLPDataset
This loads the text into a class. Can be given either a path to a data-json file or dict.

load_embeds_into_memory prepares and pools the whole dataset

## LMModel
Language model model, takes huggingface name or a dict {'model':model, 'tokenizer':tokenizer}

its forward takes a list of texts and outputs a dictionary

## PoolToken
pools tokens from the output dictionary

it supports selection of layers and quantiles

layer: single number to select one, list to select multiple, 'all' to select all, -1 to select last
    layer excludes first and last layer by default, can be overwritten by giving single number
    quantile: -1 or 1.0 to select last, all to select all, list to select multiple (might cause double instances)
methods: mean, max, min, extend, concat
    extend adds samples along the embed axis, concat adds them to a new axis
    can only concat one
    
## evaluator
returns MSE for regression and MSE, accuracy and F1 for binary

## Classifiers
### SingleStepOpt
This corresponds to sklearn style optimizers

### MultiStepOpt
Pytorch style neural nets
