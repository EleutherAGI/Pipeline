from argparse import ArgumentParser
import os
import pandas as pd
import json
from datasets import load_dataset
import numpy as np
import urllib.request
import tarfile

'''
parser = ArgumentParser()

parser.add_argument('data_save_path', type = str, help='Creates folder for data storage')

args = parser.parse_args()

try:
    os.mkdir(args.data_save_path)
except:
    raise RuntimeError('Folder already exists')
    # ask to go ahead anyways
'''



def amazon_polarity(data_save_path, force_classification = False):
    # # Amazon polarity
    # https://huggingface.co/datasets/amazon_polarity

    dataset = load_dataset('amazon_polarity')
    X1 = pd.DataFrame(dataset['train'])
    X2 = pd.DataFrame(dataset['test'])
    X = pd.concat([X1, X2])
    X = X[['content', 'label']]
    X = X.rename(columns = {'content':'sentences', 'label':'labels'})
    len_fn = lambda x: len(x.split(' '))
    lens = [len_fn(x) for x in X['sentences']]
    mean_len = round(sum(lens)/len(lens), 1)
    N = len(lens)
    top_level_dict = {}
    top_level_dict['data'] = X.to_dict('list')
    top_level_dict['name'] = 'Amazon polarity'
    top_level_dict['link'] = 'https://huggingface.co/datasets/amazon_polarity'
    top_level_dict['topic'] = 'Sentiment'
    top_level_dict['task'] = 'Classification'
    top_level_dict['size'] = N
    top_level_dict['average length'] = mean_len
    top_level_dict['multiple'] = False
    top_level_dict['type'] = 'Ordinal'
    top_level_dict['n_categories'] = 2
    top_level_dict['labels'] = {'0':'Negative', '1':'Positive'}
    top_level_dict['language'] = 'En'
    with open(os.path.join(data_save_path, 'amazonpolarity.json'), 'w') as outfile:
        json.dump(top_level_dict, outfile)

def app_reviews(data_save_path, force_classification = False):
    # # App reviews
    # https://huggingface.co/datasets/app_reviews

    dataset = load_dataset('app_reviews')


    X = pd.DataFrame(dataset['train'])


    X = X[['review', 'star']]
    X = X.rename(columns = {'review':'sentences', 'star':'labels'})


    X['labels'] = (X['labels']-1)/4
    if force_classification: X['labels'] = X['labels'].round()

    len_fn = lambda x: len(x.split(' '))
    lens = [len_fn(x) for x in X['sentences']]
    mean_len = round(sum(lens)/len(lens), 1)
    N = len(lens)


    top_level_dict = {}
    top_level_dict['data'] = X.to_dict('list')
    top_level_dict['name'] = 'App review'
    top_level_dict['link'] = 'https://huggingface.co/datasets/app_reviews'
    top_level_dict['topic'] = 'Sentiment'
    top_level_dict['task'] = 'Regression' if not force_classification else 'Classification'
    top_level_dict['size'] = N
    top_level_dict['average length'] = mean_len
    top_level_dict['multiple'] = False
    top_level_dict['type'] = 'Ordinal'
    top_level_dict['n_categories'] = 5
    top_level_dict['labels'] = {'0':'Negative', '1':'Positive'}
    top_level_dict['language'] = 'En'

    with open(os.path.join(data_save_path, 'app_review.json'), 'w') as outfile:
        json.dump(top_level_dict, outfile)


def colbert(data_save_path, force_classification = False):
    # # ColBERT
    source_path = 'https://github.com/Moradnejad/ColBERT-Using-BERT-Sentence-Embedding-for-Humor-Detection/raw/master/Data'
    save_path = os.path.join(data_save_path, 'temp')
    try:
        os.mkdir(save_path)
    except:
        pass

    urllib.request.urlretrieve(source_path+'/train.csv', os.path.join(save_path, 'train.csv'))
    urllib.request.urlretrieve(source_path+'/dev.csv', os.path.join(save_path, 'dev.csv'))

    X1 = pd.DataFrame(pd.read_csv(os.path.join(save_path, 'train.csv'), sep=','))
    X3 = pd.DataFrame(pd.read_csv(os.path.join(save_path, 'dev.csv'), sep=','))

    X = pd.concat([X1, X3])

    X = X[['text', 'humor']]
    X = X.rename(columns = {'text':'sentences', 'humor':'labels'})
    
    X['labels'] = [1 if humor else 0 for humor in X['labels']]

    len_fn = lambda x: len(x.split(' '))
    lens = [len_fn(x) for x in X['sentences']]
    mean_len = round(sum(lens)/len(lens), 1)
    N = len(lens)


    top_level_dict = {}
    top_level_dict['data'] = X.to_dict('list')
    top_level_dict['name'] = 'ColBERT'
    top_level_dict['link'] = 'https://github.com/Moradnejad/ColBERT-Using-BERT-Sentence-Embedding-for-Humor-Detection'
    top_level_dict['topic'] = 'Humour'
    top_level_dict['task'] = 'Classification'
    top_level_dict['size'] = N
    top_level_dict['average length'] = mean_len
    top_level_dict['multiple'] = False
    top_level_dict['type'] = 'Ordinal'
    top_level_dict['n_categories'] = 2
    top_level_dict['labels'] = {'0':'Not funny', '1':'Funny'}
    top_level_dict['language'] = 'En'


    with open(os.path.join(data_save_path, 'colbert.json'), 'w') as outfile:
        json.dump(top_level_dict, outfile)


def dutch_social(data_save_path, force_classification = False):
    # # Dutch social
    # https://huggingface.co/datasets/dutch_social

    dataset = load_dataset('dutch_social')


    X1 = pd.DataFrame(dataset['train'])
    X2 = pd.DataFrame(dataset['validation'])
    X3 = pd.DataFrame(dataset['test'])

    X = pd.concat([X1, X2, X3])


    X = X[['text_translation', 'sentiment_pattern']]
    X = X.rename(columns = {'text_translation':'sentences', 'sentiment_pattern':'labels'})

    X['labels'] = (X['labels']+1)/2
    if force_classification: X['labels'] = X['labels'].round()

    len_fn = lambda x: len(x.split(' '))
    lens = [len_fn(x) for x in X['sentences']]
    mean_len = round(sum(lens)/len(lens), 1)
    N = len(lens)

    top_level_dict = {}
    top_level_dict['data'] = X.to_dict('list')
    top_level_dict['name'] = 'Dutch social'
    top_level_dict['link'] = 'https://huggingface.co/datasets/dutch_social'
    top_level_dict['topic'] = 'Sentiment'
    top_level_dict['task'] = 'Regression' if not force_classification else 'Classification'
    top_level_dict['size'] = N
    top_level_dict['average length'] = mean_len
    top_level_dict['multiple'] = False
    top_level_dict['type'] = 'Ordinal'
    top_level_dict['n_categories'] = 'inf'
    top_level_dict['labels'] = {'0':'Negative', '1':'Positive'}
    top_level_dict['language'] = 'En'

    with open(os.path.join(data_save_path, 'dutch_social.json'), 'w') as outfile:
        json.dump(top_level_dict, outfile)


def hate_speech(data_save_path, force_classification = False):
    # # Hate speech18
    # https://huggingface.co/datasets/hate_speech18

    dataset = load_dataset('hate_speech18')

    X = pd.DataFrame(dataset['train'])

    X = X[(X['label'] == 1) | (X['label'] == 0)]

    X = X[['text', 'label']]
    X = X.rename(columns = {'text':'sentences', 'label':'labels'})


    len_fn = lambda x: len(x.split(' '))
    lens = [len_fn(x) for x in X['sentences']]
    mean_len = round(sum(lens)/len(lens), 1)
    N = len(lens)

    top_level_dict = {}
    top_level_dict['data'] = X.to_dict('list')
    top_level_dict['name'] = 'Stormfront hate speech'
    top_level_dict['link'] = 'https://huggingface.co/datasets/hate_speech18'
    top_level_dict['topic'] = 'Hate speech'
    top_level_dict['task'] = 'Classification'
    top_level_dict['size'] = N
    top_level_dict['average length'] = mean_len
    top_level_dict['multiple'] = False
    top_level_dict['type'] = 'Ordinal'
    top_level_dict['n_categories'] = 2
    top_level_dict['labels'] = {'0':'Not hate speech', '1':'Hate speech'}
    top_level_dict['language'] = 'En'

    with open(os.path.join(data_save_path, 'hate_speech18.json'), 'w') as outfile:
        json.dump(top_level_dict, outfile)

def liar(data_save_path, force_classification = False):
    # # Liar
    # https://huggingface.co/datasets/amazon_polarity

    dataset = load_dataset('liar')


    X1 = pd.DataFrame(dataset['train'])
    X2 = pd.DataFrame(dataset['validation'])
    X3 = pd.DataFrame(dataset['test'])

    X = pd.concat([X1, X2, X3])

    X['label'] = (5-X['label'])/5
    
    X = X[['statement', 'label']]
    X = X.rename(columns = {'statement':'sentences', 'label':'labels'})
    if force_classification: X['labels'] = X['labels'].round()

    len_fn = lambda x: len(x.split(' '))
    lens = [len_fn(x) for x in X['sentences']]
    mean_len = round(sum(lens)/len(lens), 1)
    N = len(lens)

    top_level_dict = {}
    top_level_dict['data'] = X.to_dict('list')
    top_level_dict['name'] = 'Pants on fire'
    top_level_dict['link'] = 'https://huggingface.co/datasets/liar'
    top_level_dict['topic'] = 'Truth'
    top_level_dict['task'] = 'Regression' if not force_classification else 'Classification'
    top_level_dict['size'] = N
    top_level_dict['average length'] = mean_len
    top_level_dict['multiple'] = False
    top_level_dict['type'] = 'Ordinal'
    top_level_dict['n_categories'] = 6
    top_level_dict['labels'] = {'0':'False', '1':'True'}
    top_level_dict['language'] = 'En'

    with open(os.path.join(data_save_path, 'liar.json'), 'w') as outfile:
        json.dump(top_level_dict, outfile)

def microedit(data_save_path, force_classification = False):
    # # humicroedit

    dataset = load_dataset('humicroedit', 'subtask-1')


    X1 = pd.DataFrame(dataset['train'])
    X2 = pd.DataFrame(dataset['test'])
    X3 = pd.DataFrame(dataset['validation'])
    X4 = pd.DataFrame(dataset['funlines'])

    X = pd.concat([X1, X2, X3, X4])

    def replace_fn(x, word):
        x = x.split('<')
        part1 = x[0]
        part2 = x[1].split('>')[1]
        return part1 + word + part2

    replaced = [replace_fn(orig, word) for orig, word in zip(X['original'], X['edit'])]


    X['replaced'] = replaced


    X = X[['replaced', 'meanGrade']]
    X = X.rename(columns = {'replaced':'sentences', 'meanGrade':'labels'})

    X['labels'] = X['labels']/3
    if force_classification: X['labels'] = X['labels'].round()

    len_fn = lambda x: len(x.split(' '))
    lens = [len_fn(x) for x in X['sentences']]
    mean_len = round(sum(lens)/len(lens), 1)
    N = len(lens)

    top_level_dict = {}
    top_level_dict['data'] = X.to_dict('list')
    top_level_dict['name'] = 'Humicro edit'
    top_level_dict['link'] = 'https://huggingface.co/datasets/humicroedit'
    top_level_dict['topic'] = 'Humour'
    top_level_dict['task'] = 'Regression' if not force_classification else 'Classification'
    top_level_dict['size'] = N
    top_level_dict['average length'] = mean_len
    top_level_dict['multiple'] = False
    top_level_dict['type'] = 'Ordinal'
    top_level_dict['n_categories'] = 'inf'
    top_level_dict['labels'] = {'0':'Not funny', '1':'Funny'}
    top_level_dict['language'] = 'En'

    with open(os.path.join(data_save_path, 'microedit.json'), 'w') as outfile:
        json.dump(top_level_dict, outfile)

def sst(data_save_path, force_classification = False):
    # # Stanford Sentiment Treebank
    # https://huggingface.co/datasets/sst

    dataset = load_dataset('sst')

    X1 = pd.DataFrame(dataset['train'])
    X2 = pd.DataFrame(dataset['validation'])
    X3 = pd.DataFrame(dataset['test'])

    X = pd.concat([X1, X2, X3])


    X = X[['sentence', 'label']]
    X = X.rename(columns = {'sentence':'sentences', 'label':'labels'})
    if force_classification: X['labels'] = X['labels'].round()

    len_fn = lambda x: len(x.split(' '))
    lens = [len_fn(x) for x in X['sentences']]
    mean_len = round(sum(lens)/len(lens), 1)
    N = len(lens)

    top_level_dict = {}
    top_level_dict['data'] = X.to_dict('list')
    top_level_dict['name'] = 'Stanford Sentiment Treebank'
    top_level_dict['link'] = 'https://huggingface.co/datasets/sst'
    top_level_dict['topic'] = 'Sentiment'
    top_level_dict['task'] = 'Regression' if not force_classification else 'Classification'
    top_level_dict['size'] = N
    top_level_dict['average length'] = mean_len
    top_level_dict['multiple'] = False
    top_level_dict['type'] = 'Ordinal'
    top_level_dict['n_categories'] = 'inf'
    top_level_dict['labels'] = {'0':'Negative', '1':'Positive'}
    top_level_dict['language'] = 'En'

    with open(os.path.join(data_save_path, 'sst.json'), 'w') as outfile:
        json.dump(top_level_dict, outfile)


def tweets_hate_speech(data_save_path, force_classification = False):
    # # Tweets hate speech detection
    # https://huggingface.co/datasets/tweets_hate_speech_detection

    dataset = load_dataset('tweets_hate_speech_detection')

    X = pd.DataFrame(dataset['train'])


    X = X[['tweet', 'label']]
    X = X.rename(columns = {'tweet':'sentences', 'label':'labels'})
    

    len_fn = lambda x: len(x.split(' '))
    lens = [len_fn(x) for x in X['sentences']]
    mean_len = round(sum(lens)/len(lens), 1)
    N = len(lens)

    top_level_dict = {}
    top_level_dict['data'] = X.to_dict('list')
    top_level_dict['name'] = 'Tweets hate speech'
    top_level_dict['link'] = 'https://huggingface.co/datasets/tweets_hate_speech_detection'
    top_level_dict['topic'] = 'Hate speech'
    top_level_dict['task'] = 'Classification'
    top_level_dict['size'] = N
    top_level_dict['average length'] = mean_len
    top_level_dict['multiple'] = False
    top_level_dict['type'] = 'Ordinal'
    top_level_dict['n_categories'] = 2
    top_level_dict['labels'] = {'0':'Not hate speech', '1':'Hate speech'}
    top_level_dict['language'] = 'En'

    with open(os.path.join(data_save_path, 'tweets_hate_speech_detection.json'), 'w') as outfile:
        json.dump(top_level_dict, outfile)

def commonsense(data_save_path, force_classification = False):
    url = "https://people.eecs.berkeley.edu/~hendrycks/ethics.tar"
    tar_path = os.path.join(data_save_path, 'temp')
    try:
        os.mkdir(tar_path)
    except:
        pass
    
    urllib.request.urlretrieve(url, os.path.join(tar_path, 'ethics.tar'))
    my_tar = tarfile.open(os.path.join(tar_path, 'ethics.tar'))
    my_tar.extractall(tar_path)
    my_tar.close()

    file_names = [os.path.join(tar_path, 'ethics/commonsense/cm_test.csv'),
                  os.path.join(tar_path, 'ethics/commonsense/cm_train.csv')]

    X_total = pd.DataFrame()
    for file_name in file_names:
        X = pd.read_csv(file_name)
        X = X[X['is_short'] == True]
        X_total = pd.concat((X_total, X), axis = 0)

    X = X_total

    X = X[['input', 'label']]
    X = X.rename(columns = {'input':'sentences', 'label':'labels'})

    len_fn = lambda x: len(x.split(' '))
    lens = [len_fn(x) for x in X['sentences']]
    mean_len = round(sum(lens)/len(lens), 1)
    N = len(lens)
    
    top_level_dict = {}
    top_level_dict['data'] = X.to_dict('list')
    top_level_dict['name'] = 'ETHICS Commonsense'
    top_level_dict['link'] = 'https://people.eecs.berkeley.edu/~hendrycks/ethics.tar'
    top_level_dict['topic'] = 'Normativity'
    top_level_dict['task'] = 'Classification'
    top_level_dict['size'] = N
    top_level_dict['average length'] = mean_len
    top_level_dict['multiple'] = False
    top_level_dict['type'] = 'Ordinal'
    top_level_dict['n_categories'] = 2
    top_level_dict['labels'] = {'0':'Normative', '1':'Non-normative'}
    top_level_dict['language'] = 'En'

    #os.rmdir(tar_path)
    with open(os.path.join(data_save_path, 'ethicscommonsense.json'), 'w') as outfile:
        json.dump(top_level_dict, outfile)
