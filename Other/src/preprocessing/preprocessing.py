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


class Preprocessing:

    def __init__(self, num_words, seq_len, path):
        self.data = path
        self.num_words = num_words
        self.seq_len = seq_len
        self.vocabulary = dict()
        self.x_tokenized = None
        self.x_padded = None
        self.x_raw = None
        self.y = None
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

    def do_things(self, test_size):
        self.load_data()
        self.clean_text()
        self.text_tokenization()
        self.build_vocabulary()
        self.word_to_idx()
        self.padding_sentences()
        self.split_data()
        self.create_x_file(test_size)

    def create_x_file(self, test_size):
        self.x_after_split = list()
        self.x_after_split, temp, temp1, temp2 = train_test_split(self.x_raw, self.y, test_size=test_size,
                                                                  random_state=42)

    def load_data(self):
        # Reads the raw csv file and split into
        # sentences (x) and target (y)

        df = pd.read_csv(self.data)
        # df.drop(['id','keyword','location'], axis=1, inplace=True)

        self.x_raw = df['text'].values
        self.y = df['target'].values
        for index, ele in enumerate(self.y):
            self.y[index] = int(ele)

    def clean_text(self):
        self.x_raw = [x.lower() for x in self.x_raw]
        # Removes Special Characters
        self.x_raw = [re.sub(r'\W', ' ', x) for x in self.x_raw]
        # Removes single characters
        self.x_raw = [re.sub(r'\s+[a-zA-Z]\s+', ' ', x) for x in self.x_raw]
        # Removes single characters from the start
        self.x_raw = [re.sub(r'[^A-Za-z]+', ' ', x) for x in self.x_raw]
        self.x_raw = [re.sub(r'\^[a-zA-Z]\s+', ' ', x) for x in self.x_raw]
        # replace multiple spaces with single space
        self.x_raw = [re.sub(r'\s+', ' ', x, flags=re.I) for x in self.x_raw]
        self.x_raw = [''.join([i for i in x if not i.isdigit()]) for x in self.x_raw]

    # self.x_raw = [([self.stemmer.lemmatize(word) for word in x.split()]) for x in self.x_raw]

    def text_tokenization(self):
        # Tokenizes each sentence by implementing the nltk tool
        self.x_raw = [word_tokenize(x) for x in self.x_raw]

    def build_vocabulary(self):
        # Builds the vocabulary and keeps the "x" most frequent words
        # self.vocabulary = dict()
        fdist = nltk.FreqDist()
        for sentence in self.x_raw:
            for word in sentence:
                fdist[word] += 1
        common_words = fdist.most_common(self.num_words)
        # common_words = fdist.most_common(len(fdist))

        for idx, word in enumerate(common_words):
            self.vocabulary[word[0]] = (idx + 1)

    def word_to_idx(self):
        # By using the dictionary (vocabulary), it is transformed
        # each token into its index based representation

        self.x_tokenized = list()
        for sentence in self.x_raw:
            temp_sentence = list()
            for word in sentence:
                if word in self.vocabulary.keys():
                    temp_sentence.append(self.vocabulary[word])
            self.x_tokenized.append(temp_sentence)

    def padding_sentences(self):
        # Each sentence which does not fulfill the required len
        # it's padded with the index 0

        pad_idx = 0
        self.x_padded = list()

        for sentence in self.x_tokenized:
            while len(sentence) < self.seq_len:
                sentence.insert(len(sentence), pad_idx)
            self.x_padded.append(sentence)
        self.x_padded = np.array(self.x_padded)

    def split_data(self, test_size=0.25):
        self.x_train, self.x_test, self.y_train, self.y_test = train_test_split(self.x_padded, self.y,
                                                                                test_size=test_size, random_state=42)
        for ele in self.x_test:
            print(ele)
