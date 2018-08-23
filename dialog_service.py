#!/usr/bin/python

from rivescript import RiveScript
# nanoservices
from nanoservice import Responder
from nanoservice.encoder import JSONEncoder
from config import registry
import json

class DialogEngine(object):
    '''
        Expose Dialog cloud service.
    '''
    def __init__(self, path="./data/dialogs/"):
        # data in-memory
        self.engine = RiveScript()
        self.path = path
        self.engine.load_directory(path)
        self.engine.sort_replies()

    def reply(self, user, msg):
        ''' 
            user (str): A unique user ID for the person requesting a reply.
                This could be e.g. a screen name or nickname. It's used internally
                to store user variables (including topic and history), so if your
                bot has multiple users each one should have a unique ID.
            msg (str): The user's message. This is allowed to contain
                punctuation and such, but any extraneous data such as HTML tags
                should be removed in advance.
        '''
        return json.loads(json.dumps({"data": self.engine.reply(user, msg)}))


class DialogService(object):
    ''' Create an dialog cloud service for Chatbots'''
    def __init__(self, engine, address='ipc:///tmp/dialog-service.sock', encoder=JSONEncoder):
        self.engine = engine
        self.address = address
        self.encoder = encoder
        self.service = Responder(self.address, encoder=self.encoder())
        self.service.register('reply', self.engine.reply)


  
if __name__ == '__main__':
    print "Loading  dialogs"
    path= './data/dialogs/'
    dialog = DialogEngine(path)
    print "Registering ...."
    print "Starting Dialog service at {}".format(registry['DIALOG_SERVICE_ADDRESS'])
    s = DialogService(dialog, address=registry['DIALOG_SERVICE_ADDRESS'])
    s.service.start()
