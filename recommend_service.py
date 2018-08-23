"""
Recommendation based on nearest neighbor
web-scale approximate nearest neighbor search over word vectors
* Using annoy
* lmdb is used as in-memory database 

We use this to build and search any in-memory vector-based service.
* dynamic (continous indexing and search) 
* User-Interest vectors

"""

import numpy
import os
import shutil

import annoy
import lmdb
# nanoservices
from nanoservice import Responder
from nanoservice.encoder import JSONEncoder

from load import get_vectors_cloud
from config import registry
import db

class UserVectorStore(object):
    '''
        Expose User Vector cloud service.
    '''
    def __init__(self, path):
        # data in-memory
        self.lmdb_path = path + '.lmdb'
        self.anndb = path + '.annoy'

        # cache user: id - vector map
        self.env = lmdb.open(self.lmdb_path, map_size=int(1e9))
        self.index_size = 0

        # annoy parameters
        self.search_k = 100000
        self.metric = 'euclidean'
        self.number_of_trees = 40
        _phrase, vec = get_vectors_cloud(u"king")
        self.size = len(vec)
        #print vec, self.size

        self.ann = annoy.AnnoyIndex(self.size, self.metric)
        self.load_index()

    def load_index(self):
        if os.path.exists(self.anndb):
            self.ann.load(self.anndb)

    # build the annoy index
    def add_index(self, user_rooms):
        # user_rooms = [{user_id: '', room_id: '', topic: ''}]
        #print user_rooms
        self.ann = annoy.AnnoyIndex(self.size, self.metric)
        with self.env.begin(write=True) as txn:
            for user_room in user_rooms:
                phrase = user_room.get("topic")
                if phrase:
                    id = 'i%d' % self.index_size
                    # make it a string
                    room_word = 'w' + str(user_room.get("user_id")) + ':' + str(user_room.get("room_id")) + ':' + str(phrase)
                    if isinstance(room_word, unicode):
                        room_word = room_word.encode('utf-8')
                    # avoid duplicate user-rooms vector
                    #print txn.get(room_word)  
                    #if not txn.get(room_word):
                    #print user_room 
                    # get the vector
                    if isinstance(phrase, str):
                        phrase = phrase.decode('utf-8')                
                    _phrase, vec = get_vectors_cloud(phrase)
                    #print vec
                    # add the vector to annoy index
                    self.ann.add_item(self.index_size, vec)
                    # use the same id to point to word
                    # index by id
                    txn.put(id, room_word)
                    # index by user_room
                    txn.put(room_word, id)
                    self.index_size += 1

        return "Added user rooms to index. New size: {}".format(self.index_size)
    
    def build_index(self):      
        # build the forest of trees. More trees give higher precision when querying
        self.ann.build(self.number_of_trees)
        # save the index to disk
        self.ann.save(self.anndb)
        #self.ann.unload()
        # load the new index
        self.ann.load(self.anndb)
        return "Built ann index of size: {}, and loaded it in memory".format(self.index_size)

    def build_index_from_db(self): 
        user_rooms = db.get("select id, user_id, topic from rooms;")     
        # map transform into # user_rooms = [{user_id: '', room_id: '', topic: ''}]
        user_rooms = map(lambda ur: {'room_id': ur[0], 'user_id': ur[1], 'topic': ur[2]}, user_rooms)
        #print user_rooms
        self.add_index(user_rooms)
        # build the forest of trees. More trees give higher precision when querying
        self.ann.build(self.number_of_trees)
        # save the index to disk
        self.ann.save(self.anndb)
        #self.ann.unload()
        # load the new index
        self.ann.load(self.anndb)
        response = "Built ann index of size: {}, and loaded it in memory".format(self.index_size)
        print response
        return response

    def clean_index(self):
        if os.path.exists(self.lmdb_path):
            shutil.rmtree(self.lmdb_path)
        if os.path.exists(self.anndb):
            os.remove(self.anndb)
        return "Cleaned lmdb database and user index"

    def get_word_vector(self, room_word):
        # get index by word
        room_word = 'w' + room_word
        if isinstance(room_word, unicode):
            room_word = room_word.encode('utf-8')

        with self.env.begin() as txn:
            # default if room_word is not in index(should never touch this condition)
            if txn.get(room_word):            
                id = int(txn.get(room_word)[1:])
                return self.ann.get_item_vector(id)
            else:
                return []

    def nearest_neighbors(self, room_word, n=10):
        if self.index_size <=1 :
            return []

        
        with self.env.begin() as txn:
            results=set([])
            users=set([])
            # recommend should exclude the user himself
            user = room_word.split(":")[0]
            users.add(user)

            vec = self.get_word_vector(room_word) 
                     
            if vec:
                ids = self.ann.get_nns_by_vector(vec, n, self.search_k)
                #print ids
                for id in ids:
                    nn_word = txn.get('i%d' % id)[1:]
                    if room_word != nn_word:
                        #print nn_word
                        user = nn_word.split(":")[0]
                        if user not in users:
                            # only unique users
                            results.add(nn_word)
                            users.add(user)

        print "nn for {}".format(room_word)
        print list(results)
        print "======="
        return list(results)

class RecommendService(object):
    ''' Create an cache service for User-Room Vectors for Introduction radars'''
    def __init__(self, index, address='ipc:///tmp/recommend-service.sock', encoder=JSONEncoder):
        self.index = index
        self.address = address
        self.encoder = encoder
        self.service = Responder(self.address, encoder=self.encoder())
        self.service.register('add_index', self.index.add_index)
        self.service.register('build_index', self.index.build_index)
        self.service.register('build_index_db', self.index.build_index_from_db)
        self.service.register('clean_index', self.index.clean_index)
        self.service.register('word_vector', self.index.get_word_vector)
        self.service.register('nn', self.index.nearest_neighbors)

  
if __name__ == '__main__':
    print "Loading  Index in Memory"
    path= 'data/vectors/recommend'
    index = UserVectorStore(path)
    print "Registering ...."
    s = RecommendService(index, address=registry['RECOMMEND_SERVICE_ADDRESS'])
    print "Starting Recommend service at {}".format(registry['RECOMMEND_SERVICE_ADDRESS'])
    s.service.start()




