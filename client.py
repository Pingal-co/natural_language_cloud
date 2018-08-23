from nanoservice import Requester
from nanoservice.encoder import JSONEncoder
from config import registry


nlp_client = Requester(registry['NLP_SERVICE_ADDRESS'], encoder=JSONEncoder())
ann_client = Requester(registry['ANN_SERVICE_ADDRESS'], encoder=JSONEncoder())
recommend_client = Requester(registry['RECOMMEND_SERVICE_ADDRESS'], encoder=JSONEncoder())
dialog_client = Requester(registry['DIALOG_SERVICE_ADDRESS'], encoder=JSONEncoder())
intent_client = Requester(registry['INTENT_SERVICE_ADDRESS'], encoder=JSONEncoder())

# test topic similarity  service
def test_similarity_service(msg):
    res, err = nlp_client.call("similar", msg)
    print("--- Similarity service response for: {}".format(msg))
    print("{}".format(res))

# test topic similarity  service
def test_nlp_service(msg):
    res, err = nlp_client.call("all", msg)
    print("--- NLP all service response for: {}".format(msg))
    print("{}".format(res))

# test topic similarity  service
def test_parse_service(sentence):
    res, err = nlp_client.call("parse", sentence)
    print("--- Parse service responses for: {}".format(sentence))
    print("{}".format(res))

# test word vector  service
def test_vector_service(phrase):
    res, err = nlp_client.call("vector", phrase)
    print("--- Vector service responses for: {}".format(phrase))
    print("{}".format(res))

# test dialog service
def test_dailog_service(user, msg):
    res, err = dialog_client.call("reply", user, msg)
    print("--- Dialog service response for: {} by {} ".format(msg, user))
    print("{}".format(res))

# test static vector service
def test_build_static_index_service():
    print("--- Starting ANN Build service. Takes few minutes to finish. Check on server side -- ")
    res, err = ann_client.call("build_index")
    print("--- ANN Build service response for: -- ")
    print("{}".format(res))

def test_ann_wordvec_service(msg):
    res, err = ann_client.call("word_vector", msg)
    print("--- ANN Word Vector service response for: -- {} ".format(msg))
    print("{}".format(res))

def test_ann_service(msg):
    res, err = ann_client.call("nn", msg)
    print("--- ANN Nearest Neighbor service response for: -- {} ".format(msg))
    print("{}".format(res))

def test_clean_static_index_service():
    res, err = ann_client.call("clean_index")
    print("--- ANN Clean service response for: -- ")
    print("{}".format(res))

# test dynamic vector service
def test_add_recommend_service(user_rooms):
    # user_rooms = [{user_id: '', room_id: '', topic: ''}]
    print("--- Starting recommend -- ")
    res, err = recommend_client.call("add_index", user_rooms)
    print("--- recommend add service response for: -- ")
    print("{}".format(res))

def test_build_recommend_service():
    print("--- Starting recommend Build service. Takes few minutes to finish. Check on server side -- ")
    res, err = recommend_client.call("build_index")
    print("--- recommend build service response for: -- ")
    print("{}".format(res))

def test_build_recommend_service_db():
    print("--- Starting recommend Build from db service. Takes few minutes to finish. Check on server side -- ")
    res, err = recommend_client.call("build_index_db")
    print("--- recommend build service response for: -- ")
    print("{}".format(res))
    print("=================")

def test_recommend_wordvec_service(room_word):
    # room_word = user_room.user_id + ':' + user_room.room_id
    res, err = recommend_client.call("word_vector", room_word)
    print("--- recommend  Word Vector service response for: -- {} ".format(room_word))
    print("{}".format(res))

def test_recommend_service(room_word):
    # room_word = user_room.user_id + ':' + user_room.room_id
    res, err = recommend_client.call("nn", room_word)
    print("--- recommend  Nearest Neighbor service response for: -- {} ".format(room_word))
    print("{}".format(res))
    print "test complete"

def test_clean_recommend_service():
    res, err = recommend_client.call("clean_index")
    print("--- recommend Clean service response for: -- ")
    print("{}".format(res))

# test intent  service
def test_intent_service(msg):
    res, err = intent_client.call("predict", msg)
    print("--- Intent service response for: {}".format(msg))
    print("{}".format(res))
    res, err = intent_client.call("echo", msg)
    print("{}".format(res))
    print("--- Intent echo for: {}".format(msg))
    return ''


if __name__ == '__main__':
    test_nlp = True
    test_intent = False
    test_recommend = False
    test_db = False
    test_dialog = False
    # nlp service test
    if test_nlp:
        print "testing nlp service"
        test_similarity_service("tom brady")
        test_vector_service("tom brady")
        test_parse_service("I like Tom Brady")
        test_nlp_service("I plan to visit MIT tomorrow")
        test_nlp_service("I like MIT")
    
    if test_intent:
        print "testing intent service"
        test_intent_service("I like cricket")
    
    if test_recommend:
        # index service test
        # test_build_static_index_service()
        # test_ann_service("king")
        # test_clean_static_index_service()

        # user cache service test
        print "testing recommend service"
        test_clean_recommend_service()
        if test_db:
            test_build_recommend_service_db()
            # 2:9:Dhoni ; 3:23:Occulus
            user_room = {'topic': 'Occulus', 'user_id': 3, 'room_id': 23}
            room_word = str(user_room.get("user_id")) + ':' + str(user_room.get("room_id")) + ':' + str(user_room.get("topic"))
        else:
            user_rooms = [
                        {'user_id': 1, 'room_id': 1, 'topic': 'king'},
                        {'user_id': 2, 'room_id': 2, 'topic': 'king'}
                        ]
            test_add_recommend_service(user_rooms)
            test_build_recommend_service()
            
            user_rooms = [
                        {'user_id': 3, 'room_id': 3, 'topic': 'king'},
                        {'user_id': 2, 'room_id': 4, 'topic': 'prince'},
                        {'user_id': 4, 'room_id': 5, 'topic': 'queen'},
                        {'user_id': 3, 'room_id': 6, 'topic': 'man'}
                    ]
            test_add_recommend_service(user_rooms)
            test_build_recommend_service()
            room_word = str(1) + ':' + str(1) + ":" + "king"

        
        test_recommend_service(room_word)
    
    if test_dialog:
        # dialog service test
        print "testing dialog service"
        test_dailog_service("user1", "hi")
        test_dailog_service("user1", "Who are you")
        test_dailog_service("user1", "What is your favorite color")
        test_dailog_service("user1", "I like MIT")
    #   test_dailog_service("user1", "What is 3 plus 5")
        # must exit from topic: otherwise above queries are asked in context of the topic
        test_dailog_service("user2", "pingal help")
        test_dailog_service("user2","north")
    #    test_dailog_service("user2","What is 3 plus 5")
    #    test_dailog_service("user1","What is 3 plus 5")
        test_dailog_service("user2", "exit")
