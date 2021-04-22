from argparse import ArgumentParser
import os
import pandas as pd
import json
from datasets import load_dataset

parser = ArgumentParser()

parser.add_argument('data_save_path', type = str, default='train_clean_speech', help='Creates folder for data storage')

args = parser.parse_args()

try:
    os.mkdir(args.data_save_path)
except:
    raise RuntimeError('Folder already exists')
    # ask to go ahead anyways
# # Amazon polarity
# https://huggingface.co/datasets/amazon_polarity

'''

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
top_level_dict['data'] = X.to_dict()
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


with open(os.path.join(args.data_save_path, 'amazonpolarity.json'), 'w') as outfile:
    json.dump(top_level_dict, outfile)
'''

# # App reviews
# https://huggingface.co/datasets/app_reviews

dataset = load_dataset('app_reviews')


X = pd.DataFrame(dataset['train'])


X = X[['review', 'star']]
X = X.rename(columns = {'review':'sentences', 'star':'labels'})


X['labels'] = (X['labels']-1)/4


len_fn = lambda x: len(x.split(' '))
lens = [len_fn(x) for x in X['sentences']]
mean_len = round(sum(lens)/len(lens), 1)
N = len(lens)


top_level_dict = {}
top_level_dict['data'] = X.to_dict()
top_level_dict['name'] = 'App review'
top_level_dict['link'] = 'https://huggingface.co/datasets/app_reviews'
top_level_dict['topic'] = 'Sentiment'
top_level_dict['task'] = 'Regression'
top_level_dict['size'] = N
top_level_dict['average length'] = mean_len
top_level_dict['multiple'] = False
top_level_dict['type'] = 'Ordinal'
top_level_dict['n_categories'] = 5
top_level_dict['labels'] = {'0':'Negative', '1':'Positive'}
top_level_dict['language'] = 'En'

with open(os.path.join(args.data_save_path, 'app_review.json'), 'w') as outfile:
    json.dump(top_level_dict, outfile)


# # ColBERT

X1 = pd.DataFrame(pd.read_csv('D:/GIS/train.txt', sep=','))
X3 = pd.DataFrame(pd.read_csv('D:/GIS/dev.txt', sep=','))

X = pd.concat([X1, X3])


X = X[['text', 'humor']]
X = X.rename(columns = {'text':'sentences', 'humor':'labels'})


X['labels'] = [1 if humor else 0 for humor in X['labels']]


len_fn = lambda x: len(x.split(' '))
lens = [len_fn(x) for x in X['sentences']]
mean_len = round(sum(lens)/len(lens), 1)
N = len(lens)


top_level_dict = {}
top_level_dict['data'] = X.to_dict()
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


with open(os.path.join(args.data_save_path, 'colbert.json'), 'w') as outfile:
    json.dump(top_level_dict, outfile)


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


len_fn = lambda x: len(x.split(' '))
lens = [len_fn(x) for x in X['sentences']]
mean_len = round(sum(lens)/len(lens), 1)
N = len(lens)

top_level_dict = {}
top_level_dict['data'] = X.to_dict()
top_level_dict['name'] = 'Dutch social'
top_level_dict['link'] = 'https://huggingface.co/datasets/dutch_social'
top_level_dict['topic'] = 'Sentiment'
top_level_dict['task'] = 'Regression'
top_level_dict['size'] = N
top_level_dict['average length'] = mean_len
top_level_dict['multiple'] = False
top_level_dict['type'] = 'Ordinal'
top_level_dict['n_categories'] = 'inf'
top_level_dict['labels'] = {'0':'Negative', '1':'Positive'}
top_level_dict['language'] = 'En'

with open(os.path.join(args.data_save_path, 'dutch_social.json'), 'w') as outfile:
    json.dump(top_level_dict, outfile)


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
top_level_dict['data'] = X.to_dict()
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

with open(os.path.join(args.data_save_path, 'hate_speech18.json'), 'w') as outfile:
    json.dump(top_level_dict, outfile)


# # Liar
# https://huggingface.co/datasets/amazon_polarity

dataset = load_dataset('liar')


X1 = pd.DataFrame(dataset['train'])
X2 = pd.DataFrame(dataset['validation'])
X3 = pd.DataFrame(dataset['test'])

X = pd.concat([X1, X2, X3])

X['label'] = (5-X['label'])/5
X['label'].unique()

X = X[['statement', 'label']]
X = X.rename(columns = {'statement':'sentences', 'label':'labels'})


len_fn = lambda x: len(x.split(' '))
lens = [len_fn(x) for x in X['sentences']]
mean_len = round(sum(lens)/len(lens), 1)
N = len(lens)

top_level_dict = {}
top_level_dict['data'] = X.to_dict()
top_level_dict['name'] = 'Pants on fire'
top_level_dict['link'] = 'https://huggingface.co/datasets/liar'
top_level_dict['topic'] = 'Truth'
top_level_dict['task'] = 'Regression'
top_level_dict['size'] = N
top_level_dict['average length'] = mean_len
top_level_dict['multiple'] = False
top_level_dict['type'] = 'Ordinal'
top_level_dict['n_categories'] = 6
top_level_dict['labels'] = {'0':'False', '1':'True'}
top_level_dict['language'] = 'En'

with open(os.path.join(args.data_save_path, 'liar.json'), 'w') as outfile:
    json.dump(top_level_dict, outfile)


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


len_fn = lambda x: len(x.split(' '))
lens = [len_fn(x) for x in X['sentences']]
mean_len = round(sum(lens)/len(lens), 1)
N = len(lens)

top_level_dict = {}
top_level_dict['data'] = X.to_dict()
top_level_dict['name'] = 'Humicro edit'
top_level_dict['link'] = 'https://huggingface.co/datasets/humicroedit'
top_level_dict['topic'] = 'Humour'
top_level_dict['task'] = 'Regression'
top_level_dict['size'] = N
top_level_dict['average length'] = mean_len
top_level_dict['multiple'] = False
top_level_dict['type'] = 'Ordinal'
top_level_dict['n_categories'] = 'inf'
top_level_dict['labels'] = {'0':'Not funny', '1':'Funny'}
top_level_dict['language'] = 'En'

with open(os.path.join(args.data_save_path, 'microedit.json'), 'w') as outfile:
    json.dump(top_level_dict, outfile)


# # Stanford Sentiment Treebank
# https://huggingface.co/datasets/sst

dataset = load_dataset('sst')

X1 = pd.DataFrame(dataset['train'])
X2 = pd.DataFrame(dataset['validation'])
X3 = pd.DataFrame(dataset['test'])

X = pd.concat([X1, X2, X3])


X = X[['sentence', 'label']]
X = X.rename(columns = {'sentence':'sentences', 'label':'labels'})


len_fn = lambda x: len(x.split(' '))
lens = [len_fn(x) for x in X['sentences']]
mean_len = round(sum(lens)/len(lens), 1)
N = len(lens)


top_level_dict = {}
top_level_dict['data'] = X.to_dict()
top_level_dict['name'] = 'Stanford Sentiment Treebank'
top_level_dict['link'] = 'https://huggingface.co/datasets/sst'
top_level_dict['topic'] = 'Sentiment'
top_level_dict['task'] = 'Regression'
top_level_dict['size'] = N
top_level_dict['average length'] = mean_len
top_level_dict['multiple'] = False
top_level_dict['type'] = 'Ordinal'
top_level_dict['n_categories'] = 'inf'
top_level_dict['labels'] = {'0':'Negative', '1':'Positive'}
top_level_dict['language'] = 'En'

with open(os.path.join(args.data_save_path, 'sst.json'), 'w') as outfile:
    json.dump(top_level_dict, outfile)


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
top_level_dict['data'] = X.to_dict()
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

with open(os.path.join(args.data_save_path, 'tweets_hate_speech_detection.json'), 'w') as outfile:
    json.dump(top_level_dict, outfile)