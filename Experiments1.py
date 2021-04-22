import glob
import itertools
import numpy as np
import torch
import torch.nn as nn
from pipeline import NLPDataset, LMModel, PoolToken, SingleStepOpt, MultiStepOpt, RunExperiments, evaluator

class LDA(SingleStepOpt):
    def __init__(self, input_size):
        super().__init__()
        self.name = 'LDA'
        self.discrete_targets = True
        self.classifier = LinearDiscriminantAnalysis(n_components=1)

class SingleLayer(MultiStepOpt):
    def __init__(self, input_size):
        super().__init__()
        self.name = 'SingleLayer'
        self.linear = nn.Linear(input_size, 1)
    
    def forward(self, X):
        return self.linear(X)

device = 'cuda' if torch.cuda.is_available() else 'cpu'
n_folds = 10
dataset_paths = glob.glob('data/*')
poolers = [PoolToken(layer = l, quantile = q) for l, q in itertools.product(range(12), np.arange(0, 1.2, 0.2))]
classifiers = [LDA, SingleLayer]
model_names = ['gpt2']

exps = RunExperiments(n_folds, dataset_paths, poolers, classifiers, model_names, evaluator, device)
exps.run_all()