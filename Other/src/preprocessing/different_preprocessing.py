import re
import nltk
import numpy as np
import pandas as pd
from sklearn.model_selection import train_test_split
from nltk.tokenize import word_tokenize
from nltk.stem import WordNetLemmatizer
import spacy
from PyDictionary import PyDictionary
from nltk.corpus import stopwords
import copy


class Different_Preprocessing:

    def __init__(self, num_words, seq_len, path_train, path_test):
        self.data_train = path_train
        self.data_test = path_test
        self.num_words = num_words
        self.seq_len = seq_len
        self.vocabulary = dict()
        self.ind_word = dict()
        self.x_after_split = None

        self.x_train = None
        self.x_test = None
        self.y_train = None
        self.y_test = None

        nlp = spacy.load('en_core_web_md')
        self.dictionary = PyDictionary()
        # nltk.download('wordnet')

        self.STOPWORDS = set(stopwords.words('english'))
        self.SYMBOLS = re.compile('[^0-9a-z #+_]')
        self.stemmer = WordNetLemmatizer()

    def do_things(self):
        self.load_data()
        self.clean_text()
        self.text_tokenization()
        self.build_vocabulary()
        self.word_to_idx()
        self.padding_sentences()

    def load_data(self):
        # Reads the raw csv file and split into
        # sentences (x) and target (y)

        df = pd.read_csv(self.data_train)
        # df.drop(['id','keyword','location'], axis=1, inplace=True)
        # print(df)
        self.x_train = df['text'].values
        self.y_train = df['target'].values
        for index, ele in enumerate(self.y_train):
            self.y_train[index] = int(ele)

        df = pd.read_csv(self.data_test)
        # df.drop(['id','keyword','location'], axis=1, inplace=True)

        self.x_test = df['text'].values
        self.y_test = df['target'].values
        for index, ele in enumerate(self.y_test):
            self.y_test[index] = int(ele)

    def clean_text(self):
        self.x_train = [x.lower() for x in self.x_train]
        # Removes Special Characters
        self.x_train = [re.sub(r'\W', ' ', x) for x in self.x_train]
        # Removes single characters
        self.x_train = [re.sub(r'\s+[a-zA-Z]\s+', ' ', x) for x in self.x_train]
        # Removes single characters from the start
        self.x_train = [re.sub(r'[^A-Za-z]+', ' ', x) for x in self.x_train]
        self.x_train = [re.sub(r'\^[a-zA-Z]\s+', ' ', x) for x in self.x_train]
        # replace multiple spaces with single space
        self.x_train = [re.sub(r'\s+', ' ', x, flags=re.I) for x in self.x_train]
        self.x_train = [''.join([i for i in x if not i.isdigit()]) for x in self.x_train]

        self.x_test = [x.lower() for x in self.x_test]
        # Removes Special Characters
        self.x_test = [re.sub(r'\W', ' ', x) for x in self.x_test]
        # Removes single characters
        self.x_test = [re.sub(r'\s+[a-zA-Z]\s+', ' ', x) for x in self.x_test]
        # Removes single characters from the start
        self.x_test = [re.sub(r'[^A-Za-z]+', ' ', x) for x in self.x_test]
        self.x_test = [re.sub(r'\^[a-zA-Z]\s+', ' ', x) for x in self.x_test]
        # replace multiple spaces with single space
        self.x_test = [re.sub(r'\s+', ' ', x, flags=re.I) for x in self.x_test]
        self.x_test = [''.join([i for i in x if not i.isdigit()]) for x in self.x_test]

    # self.x_raw = [([self.stemmer.lemmatize(word) for word in x.split()]) for x in self.x_raw]

    def text_tokenization(self):
        # Tokenizes each sentence by implementing the nltk tool
        self.x_train = [word_tokenize(x) for x in self.x_train]
        self.x_test = [word_tokenize(x) for x in self.x_test]

        self.x_after_split = copy.deepcopy(self.x_train)

    def build_vocabulary(self):
        # Builds the vocabulary and keeps the "x" most frequent words
        # self.vocabulary = dict()
        fdist = nltk.FreqDist()
        for sentence in self.x_train:
            for word in sentence:
                fdist[word] += 1
        common_words = fdist.most_common(self.num_words)
        # common_words = fdist.most_common(len(fdist))

        for idx, word in enumerate(common_words):
            self.vocabulary[word[0]] = (idx + 1)
            self.ind_word[idx] = word[0]

    def word_to_idx(self):
        # By using the dictionary (vocabulary), it is transformed
        # each token into its index based representation

        temp = list()
        for sentence in self.x_train:
            temp_sentence = list()
            for word in sentence:
                if word in self.vocabulary.keys():
                    temp_sentence.append(self.vocabulary[word])
            temp.append(temp_sentence)
        self.x_train = temp

        temp = list()
        for sentence in self.x_test:
            temp_sentence = list()
            for word in sentence:
                if word in self.vocabulary.keys():
                    temp_sentence.append(self.vocabulary[word])
                # else:
                #     temp_sentence.append(-1)
            temp.append(temp_sentence)
        self.x_test = temp

    def padding_sentences(self):
        # Each sentence which does not fulfill the required len
        # it's padded with the index 0

        pad_idx = 0
        temp = list()

        for sentence in self.x_train:
            while len(sentence) < self.seq_len:
                sentence.insert(len(sentence), pad_idx)
            temp.append(sentence)
        self.x_train = np.array(temp)

        temp = list()
        for sentence in self.x_test:
            while len(sentence) < self.seq_len:
                sentence.insert(len(sentence), pad_idx)
            temp.append(sentence)
        self.x_test = np.array(temp)

    def inx_to_sent(self, indicies):
        sent = list()
        for ele in indicies:
            sent.append(self.ind_word[ele])
        return sent


