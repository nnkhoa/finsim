import random
import logging

from optparse import OptionParser

import gensim
import numpy
from nltk.tokenize import word_tokenize
from gensim.models import Word2Vec
logging.basicConfig(level="INFO")


def get_args():
    parser = OptionParser()

    parser.add_option("--embedding-path", dest="embedding_path",
                        help="path to Word Embedding")
    parser.add_option("--binary", dest="binary", default=False,
                        help="Whether pretrained embedding is in binary format or not")
    parser.add_option("--terms-path", dest="terms_path", 
                        help="path to list of terms given by finsm")
    # parser.add_option("--exclude-topics", dest="exclude_topics", default=None,
    #                     help="File containing list of topics to be excluded from querying")
    # parser.add_option("--include-topics", dest="include_topics", default=None,
    #                     help="File containing list of topics to be included in querying")
    # parser.add_option("--tickers", dest="tickers", default=None,
    #                     help="File containing list of tickers to be included in querying")
    # parser.add_option("--keywords", dest="keywords", default=None,
    #                     help="File containing list of keywords to be included in querying")
    parser.add_option("--output", dest="output", default="../data/outputs/temp.json",
                        help="Path to output the results")
    (options, args) = parser.parse_args()

    return options


def loadWord2Vec(embedding_path, binary):
	return gensim.models.KeyedVectors.load_word2vec_format(embedding_path, binary=binary)


def get_X_Y(embeddings, term_list):
    X, Y = [], []
    for term in term_list:
        phrase_vector = numpy.zeros((embeddings.vector_size,))
        for word in word_tokenize(term["term"]):
            try:
                phrase_vector += embeddings.wv[word.lower()]
            except KeyError:
                logging.debug(f"Word {word.lower()} not found.")
                pass
        if type(X) == list:
            X = numpy.array(phrase_vector)
        else:
            X = numpy.vstack([X, phrase_vector])
        Y.append(term["label"])
    return X, Y


def train_test_split(term_list, seed=0):
    random.Random(seed).shuffle(term_list)
    
    train_list = term_list[:-int(0.8*len(term_list))]
    rest_list = term_list[-int(0.8*len(term_list)):]

    valid_list = rest_list[:-int(0.5*len(rest_list))]
    test_list = rest_list[-int(0.5*len(rest_list)):]
    
    return train_list, valid_list, test_list


def train(X, Y, model):
    model.fit(X, Y)
    return model
