"""
web-scale approximate nearest neighbor (ann) search over word vectors
* Using annoy, Falconn using LSH is another promising library for ann
* lmdb is used as in-memory database like in Caffe-based deep learning apps

We can use this to build and search any vector-based retreival service if model vectors are saved in a file.
* Static: Reddit, Wikipedia, Glove, Google vectors
* Sentence or Document Summarization: RNN, LSTM, CNN transformed vectors

"""
import numpy
import os
import shutil

import annoy
import lmdb
# nanoservices
from nanoservice import Responder
from nanoservice.encoder import JSONEncoder

from load import get_vectors_file
from config import registry


class VectorStore(object):
    '''
        Expose Vector cloud service.
    '''
    def __init__(self, fname='data/vectors/google_vectors/GoogleNews-vectors-negative300.bin'):
        self.fname = fname
        self.lmdb_path = fname + '.lmdb'
        self.anndb = fname + '.annoy'

        # cache store word - id map 
        self.env = lmdb.open(self.lmdb_path, map_size=int(1e9))

        # annoy parameters
        self.search_k = 100000
        self.metric = 'euclidean'
        self.number_of_trees = 40
        self.size = 300

        # load annoy
        _word, vec = get_vectors_file(self.fname).next()
        self.size = len(vec)

        self.ann = annoy.AnnoyIndex(self.size, self.metric)
        self.load_index()

    def load_index(self):
        if os.path.exists(self.anndb):
            self.ann.load(self.anndb)
 
    # build the annoy index
    def build_index(self):
        i = 0
        print "Building Index ...."
        ann = annoy.AnnoyIndex(self.size, self.metric)
        with self.env.begin(write=True) as txn:
            for word, vec in get_vectors_file(self.fname):
                # add the vector to annoy index
                ann.add_item(i, vec)
                # use the same id to point to word
                id = 'i%d' % i
                # make it a string
                word = 'w' + word
                # index by id
                txn.put(id, word)
                # index by word
                txn.put(word, id)
                i += 1
                # print the progress
                if i % 1000 == 0:
                    print i, "..."
            
        # build the forest of trees. More trees give higher precision when querying
        ann.build(self.number_of_trees)
        # save the index to disk
        ann.save(self.anndb)
        # load the new index
        self.ann.load(self.anndb)
        return "Built ann index of size: {}, and loaded it in memory".format(i)

    def clean_index(self):
        if os.path.exists(self.lmdb_path):
            shutil.rmtree(self.lmdb_path)
        if os.path.exists(self.anndb):
            os.remove(self.anndb)
        return "Cleaned lmdb database and ann index"

    def get_word_vector(self, word):
        # get index by word
        word = 'w' + word
        if isinstance(word, unicode):
            word = word.encode('utf-8')

        with self.env.begin() as txn:
            id = int(txn.get(word)[1:])
            return self.ann.get_item_vector(id)

    def get_expression_vector(self, expression):
        """
         Compute analogy: Find similar groups like the group in location X
         # queen == king -man +woman
         # stanford == MIT +paloalto -boston
        """
            
        with self.env.begin() as txn:
            ids = []
            for word in expression.strip().split():
                # positive or negative
                sign=1
                if word[0] in ['+', '-']:
                    word, sign = word[1:], int(word[0] + '1')
                id = int(txn.get('w' + word)[1:])
                ids.append((sign, id))

            vecs = [(sign, self.ann.get_item_vector(id)) for sign, id in ids]
            # looks odd
            vecs = [ [sign * value for value in vec] for sign, vec in vecs]
            # compute the sum vector
            vec = numpy.sum(vecs, axis=0)
        return vec, ids

    def analogy(self, expression, n=50):
        with self.env.begin() as txn:
            vec, ids = self.get_expression_vector(expression)
            results=[]
            for id, distance in zip(*self.ann.get_nns_by_vector(vec, n, self.search_k, True)):
                # exclude the ids
                if id not in ids:
                    word = txn.get('i%d' % id)[1:]
                    #print '%50s\t%f' % (word, distance)
                    results.append((word, distance))

        return results

    def nearest_neighbors(self, word, n=50):
        
        with self.env.begin() as txn:
            vec = self.get_word_vector(word)
            results=[]
            for id in self.ann.get_nns_by_vector(vec, n, self.search_k):
                nn_word = txn.get('i%d' % id)[1:]
                results.append(nn_word)

        return {'nn': results}


class ANNService(object):
    ''' Create an index cloud service for Deep Learning based vectors index'''
    def __init__(self, store, address='ipc:///tmp/ann-service.sock', encoder=JSONEncoder):
        self.store = store
        self.address = address
        self.encoder = encoder
        self.service = Responder(self.address, encoder=self.encoder())
        self.service.register('build_index', self.store.build_index)
        self.service.register('clean_index', self.store.clean_index)
        self.service.register('word_vector', self.store.get_word_vector)
        self.service.register('nn', self.store.nearest_neighbors)

  
if __name__ == '__main__':
    print "Loading Index in Memory"
    fname= 'data/vectors/google_vectors/GoogleNews-vectors-negative300.bin'
    vs = VectorStore(fname)
    print "Registering ...."
    s = ANNService(vs, address=registry['ANN_SERVICE_ADDRESS'])
    print "Starting  Index service at {}".format(registry['ANN_SERVICE_ADDRESS'])
    s.service.start()




