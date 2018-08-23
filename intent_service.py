#!/usr/bin/python

# nanoservices
from nanoservice import Responder
from nanoservice.encoder import JSONEncoder
from config import registry
import json
import os
# intent class
from deep_learning_intent import Intent


class IntentService(object):
    ''' Expose Intent cloud service.'''
    def __init__(self,
                 retrain=False,
                 address='ipc:///tmp/intent-service.sock',
                 encoder=JSONEncoder):
        self.engine = Intent(retrain=retrain)
        self.address = address
        self.encoder = encoder
        self.service = Responder(self.address, encoder=self.encoder())
        self.service.register('predict', self.predict)
        self.service.register('echo', self.echo)

    def predict(self, msg):
        response = self.engine.predict(msg)
        print response[0][0]
        return str(response[0][0])
    
    def echo(self, msg):
        return 'hello {}'.format(msg)

 
if __name__ == '__main__':
    print "Registering ...."
    s = IntentService(retrain=False, address=registry['INTENT_SERVICE_ADDRESS'])
    print "Starting INTENT service at {}".format(registry['INTENT_SERVICE_ADDRESS'])
    s.service.start()
