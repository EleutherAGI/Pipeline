import os
import time
import json
import numpy as np
import itertools
from functools import partial
import torch
import torch.nn as nn
from torch.optim.lr_scheduler import ReduceLROnPlateau
from torch.optim import Adam
from tqdm import tqdm

from sklearn.discriminant_analysis import LinearDiscriminantAnalysis, QuadraticDiscriminantAnalysis
from sklearn.neighbors import KNeighborsClassifier, NearestCentroid
from sklearn.svm import SVC
from sklearn.gaussian_process import GaussianProcessClassifier
from sklearn.gaussian_process.kernels import RBF
from sklearn.naive_bayes import GaussianNB
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import accuracy_score, f1_score

from torch.utils.data import Dataset, DataLoader, random_split

from transformers import AutoModel, AutoTokenizer


class NLPDataset(Dataset):
    """
    Dataset class
    
    init args
    ----------
    json_input : str or dict
        Path to dataset json file or loaded json file itself
    
    load_embeds_into_memory
    -------
    Given a model and pooler, processes all embeddings and stores them
    """
    def __init__(self, json_input):
        if isinstance(json_input, str):
            with open(json_input) as json_file:
                self.json = json.load(json_file)
        else:
            self.json = json_input
        self.sentences = list(self.json['data']['sentences'].values())
        self.y = list(self.json['data']['labels'].values())
        self.meta_dict = {k: self.json[k] for k in set(list(self.json.keys())) - set(['data'])}
        
        self.data_in_memory = False
        self.batch_size = 256
        del self.json
    
    def load_embeds_into_memory(self, model, pooler):
        data_iter = DataLoader(self.sentences, batch_size = self.batch_size, shuffle = False,
                               num_workers=0, drop_last = False, pin_memory=False)
        
        hidden_mult, extra_dim = pooler.hidden_mult, pooler.extra_dim
        if extra_dim == 0:
            X_tot = torch.zeros(len(self.sentences), hidden_mult*model.model.config.hidden_size)
        else:
            X_tot = torch.zeros(len(self.sentences), hidden_mult*model.model.config.hidden_size, extra_dim)
        for i, sentence in enumerate(data_iter):
            idx = i*self.batch_size
            output = model(sentence)
            embed = pooler(output)
            X_tot[idx:idx+len(sentence)] = embed
            del embed
        if extra_dim != 0:
            X_tot = X_tot.transpose(1, 2).reshape(len(self.sentences)*extra_dim, -1)
            self.y = list(np.tile(np.array(self.y), (extra_dim, 1)).T.reshape(-1))
        self.X = X_tot

    def __len__(self):
        return len(self.y)

    def __getitem__(self, idx):
        return self.X[idx], self.y[idx]

class LMModel():
    """
    (Hugging Face) Language Model Wrapper
    
    init args
    ----------
    model_name : str or dict
        Hugging face model name or dictionary that contains a model and tokenizer
    
    __call__
    -------
    Takes a list of strings and returns Hugging face style output dictionary
    """
    def __init__(self, model_name, device):
        self.device = device
        if isinstance(model_name, str):
            self.tokenizer = AutoTokenizer.from_pretrained(model_name)
            self.tokenizer.pad_token = self.tokenizer.eos_token
            self.model = AutoModel.from_pretrained(model_name, return_dict=True)
        elif isinstance(model_name, dict):
            self.tokenizer = model_name['tokenizer']
            self.model = model_name['model']
            
        self.model = self.model.eval().to(self.device)

    def __call__(self, text):
        inputs = self.tokenizer(text, padding = True,  return_tensors="pt")
        with torch.no_grad():
            outputs = self.model(input_ids = inputs.input_ids.to(self.device),
                                 attention_mask=inputs.attention_mask.to(self.device),
                                 output_attentions = False, output_hidden_states = True,
                                 return_dict = True)
        outputs['attention_mask'] = inputs.attention_mask
        return outputs


class PoolToken():
    """
    Selects (combinations of) tokens from hugging face output dictionary
    
    init args
    ----------
    layer: int, list, str
        Which layer the tokens are to be selected from
        - layer index
        - list of layer indices
        - -1: second to last layer
        - 'all': all layers except the first and last one
    quantile: float, int
        Which quantile of tokens is selected (0. = first token, 1. = last)
        - quantile
        - -1: last token
    layer_method: str
        If multiple layers are selected, layer_method can be used to reduce them
        - 'mean', 'max', 'min': pool over layer dimension
        - 'extend': concatenates embeddings in embed dimension
    
    __call__
    -------
    Takes hugging face output dictionary and returns (batch_size, hidden_size) sized tensor of embeddings
    """
    def __init__(self, layer=-1, quantile=-1, layer_method = None, quantile_method = None):
        #assert not (isinstance(layer, list) and isinstance(quantile, list)), ''
        self.layer = layer
        self.quantile = quantile
        self.layer_method = layer_method
        self.quantile_method = quantile_method

        #assert not (self.layer_method == 'extend' and self.quantile_method == 'extend'), 'Cannot ha'
        assert not (self.layer_method == 'concat' and self.quantile_method == 'concat'), 'Cannot ha'

        # extend hidden size
        self.hidden_mult = 1
        if self.layer_method == 'extend' and isinstance(self.layer, list):
            self.hidden_mult *= len(self.layer)
        if self.quantile_method == 'extend' and isinstance(self.quantile, list):
            self.hidden_mult *= len(self.quantile)
        
        # add extra dim
        self.extra_dim = 0
        if self.layer_method == 'concat' and isinstance(self.layer, list):
            self.extra_dim = len(self.layer)
        if self.quantile_method == 'concat' and isinstance(self.quantile, list):
            self.extra_dim = len(self.quantile)
    
    def get_layer_idx(self, layer, n_layers):
        if isinstance(layer, int):
            if layer == -1:
                layer = [n_layers-1]
            else:
                layer = [layer]
        elif layer == 'all':
            layer = list(range(1, n_layers-1))
        return torch.LongTensor(layer)

    def get_quantile_idx(self, quantile, attention_mask):
        if quantile == -1: quantile = 1.0
        batch_size, seq_len = attention_mask.shape
        
        idx_quantile = torch.round(quantile*(attention_mask.sum(1)-1))
        return idx_quantile.long()
    
    def __call__(self, output_dict):
        n_layers = len(output_dict['hidden_states'])
        batch_size, seq_len, hidden_size = output_dict['hidden_states'][0].shape

        #if self.extra_dim:
        #    X = torch.zeros(batch_size, self.hidden_mult*hidden_size, self.extra_dim)
        #else:
        #    X = torch.zeros(batch_size, self.hidden_mult*hidden_size)

        X = torch.zeros(n_layers, batch_size, seq_len, hidden_size)
        for l in range(n_layers):
            X[l] = output_dict['hidden_states'][l]
        
        # select layers
        idx_layer = self.get_layer_idx(self.layer, n_layers)
        X = X[idx_layer]

        if self.layer_method == 'mean':
            assert len(idx_layer) > 1
            X = X.mean(0, keepdim = True)
        elif self.layer_method == 'max':
            assert len(idx_layer) > 1
            X = X.max(0, keepdim = True)[0]
        elif self.layer_method == 'min':
            assert len(idx_layer) > 1
            X = X.min(0, keepdim = True)[0]

        # select quantiles
        attention_mask = output_dict['attention_mask']
        idx_quantile = self.get_quantile_idx(self.quantile, attention_mask)
        X = X[:, torch.arange(batch_size), idx_quantile]

        del output_dict

        X = X.transpose(0, 1)
        if self.layer_method == 'extend':
            X = X.reshape(batch_size, -1)
        return X.squeeze() #X.reshape(-1, X.shape[-1])[torch.LongTensor(idx_quantile)]


class SingleStepOpt():
    """
    Base class for sklearn style classifiers
    Distinguished between regression (continuous targets) and classification (binary targets)
    
    init args
    ----------
    None
    
    fit
    -------
    fits classifier and sets optimal thresholds if the task is classification
        
    set_thresholds
    -------
    Uses a validation set to find optimal thresholds for both accuracy and F1
    
    predict
    -------
    Returns predictions for targets, using the thresholds if necessary
    
    forward
    -------
    Maps inputs to targets
    """
    def __init__(self, n_steps = 20):
        self.acc_threshold = None
        self.single_step = True

        # threshold args
        self.n_steps = n_steps
    
    def fit(self, trainset, valset, pooler, single_batch, classification):
        self.classification = classification
        self.single_batch = single_batch

        # if the dataset is too large, we could take an ensemble of classifiers
        if single_batch:
            X, y = np.array(trainset.X), np.array(trainset.y) #trainset.dataset.X, trainset.dataset.y
            if self.discrete_targets:
                y = np.round(y)
            self.classifier.fit(X, y)
        else:
            pass
            self.ensemble = 0
        
        # thresholds only required for classification
        if classification:
            self.set_thresholds(valset)
        
    def set_thresholds(self, valset):
        y_target = np.array(valset.y)# valset.dataset.y
        if self.single_batch:
            y_pred = self.forward(np.array(valset.X))
        else:
            pass

        acc_thresholds = []
        F1_thresholds = []

        y_min, y_max = y_pred.min(), y_pred.max()
        #if self.binary:
        eps = (y_max-y_min)/self.n_steps
        for theta in np.arange(y_min, y_max, eps):
            y_rounded = np.where(y_pred > theta, 1, 0)
            acc_thresholds.append(accuracy_score(y_target, y_rounded))
            F1_thresholds.append(f1_score(y_target, y_rounded))
        
        self.acc_threshold = np.argmax(acc_thresholds)*eps + y_min
        self.F1_threshold = np.argmax(F1_thresholds)*eps + y_min
    
    def predict(self, testset):
        y_target = np.array(testset.y) #testset.dataset.y
        y_pred = self.forward(np.array(testset.X)) #testset.dataset.X)
        if self.classification:
            y_acc = np.where(y_pred > self.acc_threshold, 1, 0)
            y_F1 = np.where(y_pred > self.F1_threshold, 1, 0)
            return y_pred, y_acc, y_F1, y_target
        else:
            return y_pred, y_target
    
    def forward(self, X):
        return self.classifier.transform(X)

class MultiStepOpt(nn.Module):
    """
    Base class for pytorch style classifiers
    
    init args
    ----------
    batch_size
    max_epoch
    
    init_optimizer
    -------
    Initializes optimizer, scheduler and loss function
    
    training_step
    -------
    Training step used in fit method
    
    test_step
    -------
    Test step used in fit method for validation and in predict to get predictions
        
    fit
    -------
    Training loop, stores losses
    
    predict
    -------
    Given testset, returns its corresponding predictions from the classifier
    """
    def __init__(self, batch_size = 128, max_epoch = 50):
        super().__init__()
        self.single_step = False
        self.train_loss, self.val_loss, self.test_loss = [], [], []
        
        self.batch_size = batch_size
        self.max_epoch = max_epoch
    
    def init_optimizer(self, classification):
        self = self.to(self.device)
        self.classification = classification
        if self.classification:
            self.loss_fn = nn.BCEWithLogitsLoss()
        else:
            self.loss_fn = nn.MSELoss()
        self.optimizer = Adam(self.parameters(), lr=1e-3)
        self.scheduler = ReduceLROnPlateau(self.optimizer)
    
    def training_step(self, X, y):
        X, y = X.to(self.device), y.to(self.device, dtype = torch.float32)
        y_pred = self(X).squeeze()
        loss = self.loss_fn(y_pred, y)

        self.train_loss.append(loss.item())

        self.optimizer.zero_grad()
        loss.backward()
        self.optimizer.step()

    def test_step(self, X, y, val):
        X = X.to(self.device)
        y = y.to(torch.float32)
        with torch.no_grad():
            y_pred = self(X).squeeze()
        y_pred = y_pred.detach().cpu()
        
        loss = self.loss_fn(y_pred, y)
        if val:
            self.val_loss.append(loss.item())
            self.scheduler.step(loss)
        else:
            self.test_loss.append(loss.item())
            return y_pred, y
    
    def get_dataloader(self, dataset):
        return DataLoader(dataset, batch_size = self.batch_size,
                          shuffle = True, num_workers = 0, drop_last = False, pin_memory = False)
    
    def fit(self, trainset, valset, pooler, single_batch, classification):
        self.init_optimizer(classification)

        for t in range(self.max_epoch):
            # train loop
            train_data_iter = iter(self.get_dataloader(trainset))
            for X, y in train_data_iter:
                self.training_step(X, y)

            # val loop
            self = self.eval()
            val_data_iter = iter(self.get_dataloader(valset))
            for X, y in val_data_iter:
                self.test_step(X, y, val = True)
            self = self.train()

    def predict(self, testset):
        test_data_iter = iter(self.get_dataloader(testset))
        y_pred, y_target = [], []
        self = self.eval()
        for X, y in test_data_iter:
            y_pred_i, y_target_i = self.test_step(X, y, val = False)
            if self.classification:
                y_pred_i = nn.Sigmoid()(y_pred_i)
            y_pred.append(y_pred_i)
            y_target.append(y_target_i)
        self = self.train()

        y_pred, y_target = torch.cat(y_pred).numpy(), torch.cat(y_target).numpy()

        if self.classification:
            y_acc = np.where(y_pred > 0.5, 1, 0)
            y_F1 = np.where(y_pred > 0.5, 1, 0)
            return y_pred, y_acc, y_F1, y_target
        else:
            return y_pred, y_target

    def forward(self, X, y):
        raise RuntimeError('Need to implement forward method')


def evaluator(ys, classification):
    """
    Returns prediction scores given a tuple of predictions and targets
    """
    if classification:
        y_pred, y_acc, y_F1, y_target = ys
        MSE = np.mean((y_pred - y_target)**2)
        acc = accuracy_score(y_acc, y_target)
        F1 = f1_score(y_F1, y_target)
        return [MSE.item(), acc.item(), F1.item()]
    else:
        y_pred, y_target = ys
        MSE = np.mean((y_pred - y_target)**2)
        return [MSE.item()]

class SimpleDataset(Dataset):
    def __init__(self, X, y):
        self.X = X
        self.y = y

    def __len__(self):
        return len(self.y)

    def __getitem__(self, idx):
        return self.X[idx], self.y[idx]

def real_random_split(dataset, splits):
    train_len, val_len, test_len = splits
    idxs = torch.randperm(len(dataset))
    new_X = dataset.X[idxs]
    new_y = torch.Tensor(dataset.y)[idxs]
    ds1 = SimpleDataset(new_X[:train_len], list(new_y[:train_len]))
    ds2 = SimpleDataset(new_X[train_len:train_len+val_len], list(new_y[train_len:train_len+val_len]))
    ds3 = SimpleDataset(new_X[-test_len:], list(new_y[-test_len:]))
    return ds1, ds2, ds3


'''
def collate_fn(data, model):
    if 
    text, y = zip(*data)
    X = model(list(text))
    return X, torch.Tensor(y)
'''

def tot_len(*args):
    tot_len = 1
    for arg in args:
        tot_len *= len(arg)
    return tot_len

class RunExperiments():
    """
    Runs a batch of experiments and logs them
    
    init args
    ----------
    n_folds: int
        Number of folds for cross validation (Not real folds cause sample will typically overlap)
    dataset_paths: list
    poolers: list
    classifiers: list
    model_names: list
    evaluator: function
    device: str
    logs_folder: str
        Folder in which the experiments logs are to be stored, will create folder if it doesn't already exist

    log_data
    -------
    Logs experiment results to logs_folder
    
    run_all(self, , val_size = 0.1, test_size = 0.2)
    -------
    single_batch_limit is not yet implemented
    """
    def __init__(self, n_folds, dataset_paths, poolers, classifiers, model_names, evaluator, device, logs_folder = 'experiment_logs'):
        self.dataset_paths = dataset_paths
        self.poolers = poolers
        self.classifiers = classifiers
        self.model_names = model_names
        self.evaluator = evaluator
        self.n_folds = n_folds
        self.device = device
        self.logs_folder = logs_folder
    
    def log_data(self, all_metrics, meta_dict, pooler, classifier, model_name):
        try:
            os.mkdir(self.logs_folder)
        except:
            pass
        
        try:
            numbers = os.listdir(self.logs_folder)
            numbers = [int(number.split('_')[-1].split('.')[0]) for number in numbers]
            new_num = max(numbers)+1
        except:
            new_num = 0
        
        exp_dict = {}
        exp_dict['metadata'] = meta_dict
        layer, quantile = pooler.layer, pooler.quantile
        layer_method, quantile_method = pooler.layer_method, pooler.quantile_method
        exp_dict['expdata'] = {'model':model_name, 'classifier':classifier.name, 'layer':layer,
                               'quantile':quantile, 'layer_method':layer_method,
                               'quantile_method':quantile_method, 'attention':False}
        exp_dict['data'] = {'metrics':all_metrics}
        try:
            exp_dict['data']['losses'] = [classifier.train_loss.tolist(), classifier.val_loss, classifier.test_loss]
        except:
            exp_dict['data']['losses'] = [[0], [0], [0]]
        
        with open(os.path.join(self.logs_folder, f'experiment_{new_num}.json'), 'w') as fp:
            json.dump(exp_dict, fp)
    
    def run_all(self, single_batch_limit=True, val_size = 0.1, test_size = 0.2):

        len_1 = tot_len(self.dataset_paths, self.poolers, self.model_names)
        len_2 = tot_len(self.classifiers)

        for k, (dataset_path, pooler, model_name) in enumerate(itertools.product(self.dataset_paths, self.poolers, self.model_names)):
            # load model and data
            model = LMModel(model_name, self.device)
            dataset = NLPDataset(dataset_path)

            single_batch = True #(model.config.n_layers * data.meta_dict['sentences'] < single_batch_limit)

            if single_batch:
                # load all embeddings into memory
                #data.batch_size
                dataset.load_embeds_into_memory(model, pooler)
                del model.model, model.tokenizer

            for n, classifier_uninit in enumerate(self.classifiers):

                all_metrics = []
                t = time.time()
                for fold in range(self.n_folds):
                    # split data
                    test_len = int(test_size*len(dataset))
                    val_len = int(val_size*len(dataset))
                    train_len = len(dataset) - test_len - val_len
                    # random_split doesn't actually split the data itself, it keeps the data the same and selects a subset from the indices
                    trainset, valset, testset = real_random_split(dataset, [train_len, val_len, test_len])

                    input_size = trainset[0][0].shape[-1]
                    classifier = classifier_uninit(input_size)
                    classifier.device = self.device

                    # fit data
                    classification = (dataset.meta_dict['task'] == 'Classification')
                    
                    # assert dataset.meta_dict['task'] is either classification or regression
                    classifier.fit(trainset, valset, pooler, single_batch, classification)

                    # make predictions
                    ys = classifier.predict(testset)

                    # evaluate predictions
                    metrics = self.evaluator(ys, classification)

                    # add to store
                    all_metrics += [metrics]
                #print(all_metrics)
                self.log_data(all_metrics, dataset.meta_dict, pooler, classifier, model_name)
                print(f'time: {round(time.time()-t, 1)}s', f'progress: {k*len_2 + n + 1}/{len_1*len_2}')
                t = time.time()
                del classifier