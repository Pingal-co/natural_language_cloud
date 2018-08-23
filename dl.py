'''
Retraining things 
E.g. : 
- Sentence & User Vector Encoding: Sooner or later we have to build user vectors based on his/her preferences. To understand what I mean: refer https://explosion.ai/blog/deep-learning-formula-nlp
- Intent Classifier: what type of query | quesstion is the user asking: Calculation (One plus two), Unit Converstion (20F to Celsius), Date, time, weather Questions, or any wit.ai style
- Emotions Classifier
- Sentiments Classifier
- Entity Recognition
  

spaCy: is used to load the GloVe vectors, perform the feature extraction
Keras: is used to build and train the network
'''
import numpy
import spacy

from keras.models import Sequential, model_from_json
from keras.layers import LSTM, Dense, Embedding, Dropout, Bidirectional
from keras.layers import TimeDistributed
from keras.optimizers import Adam


def train(train_texts, train_labels, dev_texts, dev_labels,
        lstm_shape, lstm_settings, lstm_optimizer, batch_size=100, nb_epoch=5,
        by_sentence=True):
    print("Loading spaCy")
    nlp = spacy.load('en', entity=False)
    embeddings = get_embeddings(nlp.vocab)
    model = compile_lstm(embeddings, lstm_shape, lstm_settings)
    print("Parsing texts...")
    train_docs = list(nlp.pipe(train_texts, batch_size=5000, n_threads=3))
    dev_docs = list(nlp.pipe(dev_texts, batch_size=5000, n_threads=3))
    if by_sentence:
        train_docs, train_labels = get_labelled_sentences(train_docs, train_labels)
        dev_docs, dev_labels = get_labelled_sentences(dev_docs, dev_labels)
        
    train_X = get_features(train_docs, lstm_shape['max_length'])
    dev_X = get_features(dev_docs, lstm_shape['max_length'])
    model.fit(train_X, train_labels, validation_data=(dev_X, dev_labels),
              nb_epoch=nb_epoch, batch_size=batch_size)
    return model

def compile_lstm(embeddings, shape, settings):
    model = Sequential()
    model.add(
        Embedding(
            embeddings.shape[0],
            embeddings.shape[1],
            input_length=shape['max_length'],
            trainable=False,
            weights=[embeddings],
            mask_zero=True
        )
    )
    model.add(TimeDistributed(Dense(shape['nr_hidden'], bias=False)))
    model.add(Bidirectional(LSTM(shape['nr_hidden'], dropout_U=settings['dropout'],
                                 dropout_W=settings['dropout'])))
    model.add(Dense(shape['nr_class'], activation='sigmoid'))
    model.compile(optimizer=Adam(lr=settings['lr']), loss='binary_crossentropy',
		  metrics=['accuracy'])
    return model

def get_embeddings(vocab):
    max_rank = max(lex.rank+1 for lex in vocab if lex.has_vector)
    vectors = numpy.ndarray((max_rank+1, vocab.vectors_length), dtype='float32')
    for lex in vocab:
        if lex.has_vector:
            vectors[lex.rank + 1] = lex.vector
    return vectors

def get_labelled_sentences(docs, doc_labels):
    labels = []
    sentences = []
    for doc, y in zip(docs, doc_labels):
        for sent in doc.sents:
            sentences.append(sent)
            labels.append(y)
    return sentences, numpy.asarray(labels, dtype='int32')

def get_features(docs, max_length):
    docs = list(docs)
    Xs = numpy.zeros((len(docs), max_length), dtype='int32')
    for i, doc in enumerate(docs):
        j = 0
        for token in doc:
            if token.has_vector and not token.is_punct and not token.is_space:
                Xs[i, j] = token.rank + 1
                j += 1
                if j >= max_length:
                    break
    return Xs
