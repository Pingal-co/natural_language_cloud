
'''
Intent classification: Classify what pingal hears: (greet, agree, disagree, connect, ...)

We are using pre-trained word embeddings without "fine-tuning"
Spacy uses the 300-dimensional GloVe common crawl vectors by default

Intents are built with a bag of words or n-grams vector model & doesn't take word order into account.
'''


from __future__ import unicode_literals
import json
import os
import logging
from collections import defaultdict

import sense2vec
import spacy
import numpy as np
from scipy.spatial.distance import cosine

from preprocess import clean_text
from data_utils import flatten
from sense import Sense
from parse import Parse

from nanoservice import Requester
from nanoservice.encoder import JSONEncoder
from config import registry

from keras.layers import Dense, Input, Flatten, Embedding
from keras.layers import Conv1D, MaxPooling1D, Dropout, LSTM, Bidirectional, TimeDistributed
from keras.models import Sequential, model_from_json
from keras.regularizers import l2

# use theano as the backend
os.environ['KERAS_BACKEND']='theano'

class Intent(object):
    """
      Finds the intent : Dialog or Brain. 

    """
    def __init__(self, 
                 training_data="./data/bot_intent.json", 
                 vec_size = 128,
                 max_sentence_len=15, 
                 classifier="cnn",
                 model_file="./data/model/intent",
                 retrain=False):

        self.client = Requester(registry['NLP_SERVICE_ADDRESS'], encoder=JSONEncoder())
        self.vec_size = vec_size
        self.max_sentence_len = max_sentence_len
        self.classifier = classifier
        #self.nlp = nlp
        #self.sense = Sense(nlp, word_vectors)
        #self.parse = Parse(nlp)
        #self.pos = ['NOUN', 'VERB', 'ADJ', 'ORG', 'PERSON', 'FAC', 'PRODUCT', 'LOC', 'GPE']
        # load intent data
        # open |> json.loads |> flatten |> intent2vec
        if os.path.isfile(model_file+'.h5') and not retrain:
            self.load_model(model_file) 
        else:
            # model has not been trained yet
            print "retraining ..."
            self.intents = flatten(json.load(open(training_data)))
            self.trained = False
            # retrain the model at load time
            if retrain:
                print "data loaded. Starting to train ..."
                if self.classifier == "cnn": self.train_cnn()
                if self.classifier == "two-layer-cnn": self.train_twolayer_cnn()
                if self.classifier == "cnn_lstm": self.train_cnn_lstm()
                if self.classifier == "cosine": self.train_cosine()
                if self.classifier == "bidirectional_lstm": self.train_bidirectional_lstm()
                
                if self.trained:
                    self.save_model(model_file)
                    self.load_model(model_file)                
                
        if self.classifier == "cosine":
            self.predict = self.cosine_classify
        else:
            self.predict = self.cnn_classify

        #self.intent2vec(self.intents)
        # Set threshold ; min_similarity=0.7,
        # self.threshold = min_similarity
        logging.info("Intent query")

    def get_vectors_cloud(self,phrase):
        '''calls the NLP cloud service to get the vector
        '''
        vector, _err = self.client.call("vector", phrase)
        return vector

    def get_pos_cloud(self, sentence):
        '''calls the NLP cloud service to get the vector
        '''
        phrases, err = self.client.call("parse", sentence, 'pos')
        return phrases

    def sentence2vec(self, sent):
        """
            Average pooling of word vectors
        """
        vec = np.zeros(self.vec_size)
        pos = self.get_pos_cloud(sent)
        for phrase in pos.split():
            # filter out the words that are not in self.pos
            v = self.get_vectors_cloud(phrase)
            if len(v) == self.vec_size:
                vec+= v
        
        norm = np.linalg.norm(vec)
        if norm!=0:
            vec /= norm
        return vec

    def sentence2matrix(self, words, pos=True):
        """
         Bag of vector models 
        """
        if not pos:
            words = self.get_pos_cloud(words)
        words = words.split() 
        matrix = np.zeros((self.max_sentence_len, self.vec_size))
        for i in range(min(self.max_sentence_len, len(words))):
            v = self.get_vectors_cloud(words[i])
            if len(v) < self.vec_size: 
                v = np.zeros(self.vec_size)
            matrix[i] = v
        return matrix

    def train_cosine(self, intents=None):
        '''
            turn intent classes into vectors and take the mean vector to represent the class
        '''
        if not intents:
            intents = self.intents
        self.intent_vectors = defaultdict(lambda : np.zeros(self.vec_size))
        for intent in intents:
            # convert sentence to vectors for training
            for sent in intents[intent]:
                self.intent_vectors[intent] += self.sentence2vec(sent)
            
            # compute the mean vector
            self.intent_vectors[intent] /= np.linalg.norm(self.intent_vectors[intent])
        
        self.trained = True


    def cosine_classify(self, input):
        '''
            compute the similarity between input and intent_class_vectors
            choose the highest scoring class

            O(N): We can do an O(N) approach
            - By learning a summarized vector for a intent class using bidirectional RNNs | LSTMs (deep_learning.py)
            - By using a mean vector from doc2vec encodings of each sentence in intent class. (query2vec)

        '''
        input_vector = self.sentence2vec(input)


        scores = {}
        for intent in self.intent_vectors:
            try:
                scores[intent] = 1 - cosine(input_vector, self.intent_vectors[intent])
            except ValueError:
                scores[intent] = np.nan

        scores = sorted(scores.items(), key=lambda (k,v): v, reverse=True)
        return scores

        #best_score = 0
        #probable_intent = 'dialog'
        # choose the best class
        #for topic in self.intents:
        #    intent = self.intents[topic]
            # score = max([self.sense.similarity(input_vector, intent_vector) for intent_vector in intent['intent_vector']]) if intent['intent_vector'] else 0
        #    if score > best_score and score > self.threshold:
        #        best_score = score
        #        probable_intent = topic

        #return probable_intent, best_score

    def prepare_train_data(self):
        class_labels = self.intents.keys()
        labels_index = dict(zip(class_labels, range(len(class_labels))))

        sentences = []
        classes = []
        # multi-class classification: 1-hot vector representation of all classes
        for label in class_labels:
            for sentence in self.intents[label]:
                class_bucket = [0]*len(class_labels)
                class_bucket[labels_index[label]]=1
                sentences.append(self.get_pos_cloud(sentence))
                classes.append(class_bucket)

        # store vectors
        train_vectors = np.zeros(shape=(len(sentences), self.max_sentence_len, self.vec_size))
        # bag_of_vectors representaton for each sentence
        for i in range(len(sentences)):
            train_vectors[i] = self.sentence2matrix(sentences[i])
        
        classes = np.array(classes, dtype=np.int)

        return class_labels, train_vectors, classes

    def train_cnn(self, 
            n_gram=2, 
            num_filters=1200, 
            max_sentence_len=15,
            vec_size=128,
            drop_out=0.0,
            activation='softmax',
            w_l2reg=0.0,
            b_l2reg=0.0,
            optimizer='adam'
        ):
        """
        Returns the convolutional neural network (CNN/ConvNet) for word-embedded vectors.
        Reference: Yoon Kim, "Convolutional Neural Networks for Sentence Classification,"
        
        Model = Input |> Embeddings |> Conv1D |> Dropout |> MaxPooling |> Flatten |> Dense |> Optimzer

        # Arguments
        num_filters: number of filters, the dimensionality of the output space (Default: 1200)
        n_gram: n-gram, or the length of the 1D convolution window of CNN/ConvNet (Default: 2)
        max_sentence_len: maximum number of words in a sentence (Default: 15)
        vec_size: length of the embedded vectors in the model (Default: 128)
        drop_out: dropout rate for CNN/ConvNet (Default: 0.0)
        activation: activation function. Options: softplus, softsign, relu, tanh, sigmoid, hard_sigmoid, linear. (Default: 'softmax')
        w_l2reg: L2 regularization coefficient (Default: 0.0)
        b_l2reg: L2 regularization coefficient for bias (Default: 0.0)
        optimizer: optimizer for gradient descent. Options: sgd, rmsprop, adagrad, adadelta, adam, adamax, nadam. (Default: adam)
        
        # Output
        keras sequantial model for CNN/ConvNet for Word-Embeddings
        
        # Type
        num_filters: int
        n_gram: int
        max_sentence_len: int
        vec_size: int
        dropout: float
        activation: str
        w_l2reg: float
        b_l2reg: float
        optimizer: str

        """
        self.max_sentence_len = max_sentence_len
        # convert data to training input vectors
        self.class_labels, train_vectors, classes = self.prepare_train_data()

        # build deep neural net
        model = Sequential()
        model.add(Conv1D(filters=num_filters,
                     kernel_size=n_gram,
                     padding='valid',
                     activation='relu',
                     input_shape=(max_sentence_len, vec_size)
                    ))
        if drop_out > 0.0:
            model.add(Dropout(drop_out))

        model.add(MaxPooling1D(pool_size=self.max_sentence_len - n_gram + 1))
        model.add(Flatten())
        model.add(Dense(len(self.class_labels),
                    activation=activation,
                    kernel_regularizer=l2(w_l2reg),
                    bias_regularizer=l2(b_l2reg))
              )

        model.compile(loss='categorical_crossentropy', optimizer=optimizer)


        #model.add(Dense(len(self.class_labels), activation='softmax'))
        #model.compile(loss='categorical_crossentropy', optimizer='rmsprop')

        # train the model
        model.fit(train_vectors, classes)
        self.model = model
        # flag
        self.trained = True

    def train_twolayer_cnn(self,  
                num_filters_1=1200, 
                num_filters_2=600,
                n_gram=2,
                window_size_2=10,
                max_sentence_len=15,
                vec_size=128,
                drop_out_1=0.0,
                drop_out_2=0.0,
                activation='softmax',
                w_l2reg=0.0,
                b_l2reg=0.0,
                optimizer='adam'
            ):
        """
        Returns the two-layer convolutional neural network (CNN/ConvNet) for word-embedded vectors.
        - two layers of CNN, maxpooling, dense
        Model = Input |> Embeddings |> Conv1D |> Dropout |> Conv1D |> Dropout |> MaxPooling |> Flatten |> Dense |> Optimzer

        """
        self.max_sentence_len = max_sentence_len
        # convert data to training input vectors
        self.class_labels, train_vectors, classes = self.prepare_train_data()

        # build deep neural net
        model = Sequential()
        # first_layer
        model.add(Conv1D(filters=num_filters_1,
                     kernel_size=n_gram,
                     padding='valid',
                     activation='relu',
                     input_shape=(max_sentence_len, vec_size)))
        if drop_out_1 > 0.0:
            model.add(Dropout(drop_out_1))

        model.add(Conv1D(filters=num_filters_2,
                        kernel_size=window_size_2,
                        padding='valid',
                        activation='relu'))
        if drop_out_2 > 0.0:
            model.add(Dropout(drop_out_2))

        model.add(MaxPooling1D(pool_size=self.max_sentence_len - n_gram - window_size_2 + 1))
        model.add(Flatten())
        model.add(Dense(len(self.class_labels),
                    activation=activation,
                    kernel_regularizer=l2(w_l2reg),
                    bias_regularizer=l2(b_l2reg))
              )
        model.compile(loss='categorical_crossentropy', optimizer=optimizer)

        # train the model
        model.fit(train_vectors, classes)
        self.model = model
        # flag
        self.trained = True

    def train_cnn_lstm(self, 
            n_gram=2, 
            num_filters=1200, 
            max_sentence_len=15,
            vec_size=128,
            drop_out=0.0,
            lstm_outdim=1200,
            lstm_dropout=0.2,
            activation='softmax',
            w_l2reg=0.0,
            b_l2reg=0.0,
            optimizer='adam'
        ):
        """
        Returns the CNN-LSTM (CNN/ConvNet) for word-embedded vectors.
        Reference: Chunting Zhou, Chonglin Sun, Zhiyuan Liu, Francis Lau, "A C-LSTM Neural Network for Text Classification"
        :param lstm_outdim: output dimension for the LSTM networks (Default: 1200)
        :param lstm_dropout: dropout rate for LSTM (Default: 0.2)
        
        Model = Input |> Embeddings |> Conv1D |> Dropout |> MaxPooling |> LSTM |> Dropout  |> Dense |> Optimzer
        """
        self.max_sentence_len = max_sentence_len
        # convert data to training input vectors
        self.class_labels, train_vectors, classes = self.prepare_train_data()

        # build deep neural net
        model = Sequential()
        model.add(Conv1D(filters=num_filters,
                     kernel_size=n_gram,
                     padding='valid',
                     activation='relu',
                     input_shape=(max_sentence_len, vec_size)))
        if drop_out > 0.0:
            model.add(Dropout(drop_out))

        #model.add(MaxPooling1D(pool_length=self.max_sentence_len - n_gram + 1))
        model.add(LSTM(lstm_outdim))
        if lstm_dropout > 0.0:
            model.add(Dropout(lstm_dropout))
        model.add(Dense(len(self.class_labels),
                    activation=activation,
                    kernel_regularizer=l2(w_l2reg),
                    bias_regularizer=l2(b_l2reg))
              )
        model.compile(loss='categorical_crossentropy', optimizer=optimizer)


        #model.add(Dense(len(self.class_labels), activation='softmax'))
        #model.compile(loss='categorical_crossentropy', optimizer='rmsprop')

        # train the model
        model.fit(train_vectors, classes)
        self.model = model
        # flag
        self.trained = True

    def train_bidirectional_lstm(self, 
            max_sentence_len=15,
            vec_size=128,
            lstm_outdim=1200,
            lstm_dropout=0.2,
            activation='softmax',
            w_l2reg=0.0,
            b_l2reg=0.0,
            optimizer='rmsprop'
        ):
        """
        Returns the Bidirectional-LSTM/RNN for word-embedded vectors.
        :param lstm_outdim: output dimension for the LSTM networks (Default: 1200)
        :param lstm_dropout: dropout rate for LSTM (Default: 0.2)
        
        Model = Input |> Embeddings |> LSTM |> Bidirectional |> Dropout  |> Dense |> Optimzer
        """
        self.max_sentence_len = max_sentence_len
        # convert data to training input vectors
        self.class_labels, train_vectors, classes = self.prepare_train_data()
        max_words = 20000 # top most_common words
        # build deep neural net
        model = Sequential()
        #model.add(Input(shape=(max_sentence_len, vec_size), dtype='int32'))
        #model.add(Embedding(max_words,
        #                    vec_size,
        #                    input_length=max_sentence_len,
        #                    weights=[train_vectors],                    
        #                    trainable=True))
        model.add(Bidirectional(LSTM(lstm_outdim), input_shape=(max_sentence_len, vec_size)))
        if lstm_dropout > 0.0:
            model.add(Dropout(lstm_dropout))
        model.add(Dense(len(self.class_labels),
                    activation=activation,
                    kernel_regularizer=l2(w_l2reg),
                    bias_regularizer=l2(b_l2reg))
              )
        model.compile(loss='categorical_crossentropy', 
                      optimizer=optimizer,
                      metrics=['acc'])


        #model.add(Dense(len(self.class_labels), activation='softmax'))
        #model.compile(loss='categorical_crossentropy', optimizer='rmsprop')

        # train the model
        model.fit(train_vectors, classes)
        self.model = model
        # flag
        self.trained = True

    def save_model(self, filename):
        """
        saves the model into a JSON file, and an HDF5 file (.h5).
        """
        if not self.trained:
            print 'Model not trained.'
        
        # save the model
        model_json = self.model.to_json()
        open(filename + '.json', 'wb').write(model_json)
        self.model.save_weights(filename + '.h5')
        # save the labels
        label_file = open(filename+'_labels.txt', 'w')
        label_file.write('\n'.join(self.class_labels))
        label_file.close()
    
    def load_model(self, filename):
        model = model_from_json(open(filename +'.json', 'rb').read())
        model.load_weights(filename+'.h5')
        self.model = model

        label_file = open(filename+'_labels.txt', 'r')
        self.class_labels = label_file.readlines()
        self.class_labels = map(lambda s: s.strip(), self.class_labels)
        label_file.close()
        self.trained = True
        

    def cnn_classify(self, input):
        # get vector
        input_matrix = np.array([self.sentence2matrix(input)])
        
        # classification using cnn
        predictions = self.model.predict(input_matrix)
        # output
        scores = {}
        for index, class_label in zip(range(len(self.class_labels)), self.class_labels):
            scores[class_label] = predictions[0][index]

        scores = sorted(scores.items(), key=lambda (k,v): v, reverse=True)
        return scores

    