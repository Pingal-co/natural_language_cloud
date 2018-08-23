"""
Data file can be
  1. A .bin file that contains pre-trained word vectors (word2vec binary format as in https://code.google.com/p/word2vec/ )
  2. A .gz file with the GloVe format (item then a list of floats in plain text)
  3. A plain text file with the same format as Glove

  fname= 'data/vectors/google_vectors/GoogleNews-vectors-negative300.bin'
"""
import os
import gzip
import struct
#import sense2vec
from nanoservice import Requester
from nanoservice.encoder import JSONEncoder
from config import registry

"""
class WordVectors:
    '''Expose WordVectors object.'''
    def __init__(self, path=None):
        self.model = sense2vec.load()
        self.pos = ['NOUN', 'VERB', 'ADJ', 'ORG', 'PERSON', 'FAC','PRODUCT', 'LOC', 'GPE']
    
    def get_vectors(self, phrase):
        # get the vector from service
        key = self.sanitize_phrase(phrase)
        if not phrase or not key:
            # return {'key': '', 'text': phrase, 'results': [], 'count': 0}
            return []
        freq, vec = self.model[key]
        return phrase, vec

    def sanitize_phrase(self, phrase):
        phrase = phrase.replace(' ', '_')
        if "|" in phrase:
            text, pos = phrase.rsplit('|', 1)
            key = text + '|' + pos.upper()
            return key if key in self.model else None
        
        # autofill best part_of_speech info in the phrase
        return self.find_best_phrase(phrase) 

    def find_best_phrase(self, phrase):
        '''
            finds the highest scoring phrase and add its POS
        '''
        freqs = []
        candidates = [phrase, phrase.upper(), phrase.title()] if phrase.islower() else [phrase]
        for candidate in candidates:
            for pos in self.pos:
                key = candidate + '|' + pos
                if key in self.model:
                    freq, _ = self.model[key]
                    freqs.append((freq, key))
        return max(freqs)[1] if freqs else None
"""
client = Requester(registry['NLP_SERVICE_ADDRESS'], encoder=JSONEncoder())
# model = WordVectors()

def load_vectors(fname):
    '''
        load files with different vector formats
    '''
    if fname.endswith('.gz'):
        # use gzip to read file
        opened = gzip.open(fname)
        fname = fname[:-3]

    else:
        # simply open the file
        opened = open(fname)

    if fname.endswith('.bin'):
        # make a generator: yield word and vector
        # first line of word2vec format contains number_of_words and size_of_vectors
        _words, size = (int(x) for x in opened.readline().strip().split())
        # struct: uses 'f' for float format to unpack
        fmt = 'f' * size

        while True:
            pos = opened.tell()
            buf = opened.read(1024)
            if buf == '' or buf == '\n':
                return
            i = buf.index(' ')
            word = buf[:i]
            opened.seek(pos + i + 1)

            vec = struct.unpack(fmt, opened.read(4 * size))

            yield word, vec

    elif fname.endswith('.txt'):
        for line in opened:
            items = line.strip().split()
            yield items[0], [float(x) for x in items[1:]]


def get_vectors_file(fname, num=float('inf')):
    '''use the saved file in one of the standard vector formats to get the vector
    '''
    i = 0
    # returns a word & vector
    for line in load_vectors(fname):
        yield line
        i += 1
        if i >= num:
            break

def get_vectors_cloud(phrase):
    '''calls the NLP cloud service to get the vector
    '''
    vector, _err = client.call("vector", phrase)
    return phrase, vector

"""
def get_vectors(phrase):
    '''use the local vectors object to get the vector
    '''
    return model.get_vectors(phrase)
"""


