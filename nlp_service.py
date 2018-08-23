#!/usr/bin/env python
from __future__ import unicode_literals
import os
# nanoservices
from nanoservice import Responder
#from nanoservice import Service
from nanoservice.encoder import JSONEncoder

import json
import sense2vec
import spacy

from sense import Sense
from parse import Parse

import logging
from config import registry

class NLPService(object):
    '''Expose NLP cloud service.'''
    def __init__(self):
        self.nlp = spacy.load('en')
        self.word_vectors = sense2vec.load()
        self.handlers = {
            'similar': Sense(self.nlp, self.word_vectors),
            'parse': Parse(self.nlp),
            'vector': Sense(self.nlp, self.word_vectors),
            #'intent': Intent(self.nlp, self.word_vectors)
            # 'converse':
            # 'person':
            # 'address':
            # 'date':
            # 'email':
        }

    def get(self, query='', handle='similar', **kwargs):
        self.handler = self.handlers[handle]
        #print "handle: {}".format(handle)
        #print kwargs
        #return json.dumps(self.handler(query, kwargs))
        return self.handler(query, **kwargs)
    
    def intent(self, msg):
        return self.get(query=msg, handle='intent')
    
    def similar(self, msg):
        return self.get(query=msg, handle='similar')
    
    def parse(self, msg, parser='sentence'):
        return self.get(query=msg, handle='parse', parser=parser)  

    def vector(self, msg, service='vector'):
        #print "using vector service"
        return self.get(query=msg, handle='vector', service=service)            

    def validate_person(self, msg):
        return self.get(query=msg, handle='person')
    
    def validate_address(self, msg):
        return self.get(query=msg, handle='address')

    def validate_email(self, msg):
        return self.get(query=msg, handle='email')

    def validate_date(self, msg):
        return self.get(query=msg, handle='date')

    def converse(self, msg):
        return self.get(query=msg, handle='converse')

    def echo(self, msg):
        return msg

    def get_all(self, msg):
        #print "using all"
        #print msg
        parser = self.handlers['parse']
        connector = self.handlers['similar']
        parsed = parser(msg, parser='sentence')
        
        # message is just keywords and not a sentence
        if len(msg.split()) <= 2:
            keyphrases = [msg]           
        else:
            keyphrases = parsed["keyphrases"]

        index_terms = []
        #print keyphrases
        for phrase in keyphrases:
            terms = connector(phrase)
            if terms:
                index_terms.append({ phrase: terms})
        
        if not index_terms:
            for noun in parsed["nouns"]:
                terms = connector(noun)
                if terms:
                    index_terms.append({ noun: terms})

        parsed["index_terms"] = index_terms
        print msg
        return parsed

class ConfigService(object):
    '''Expose NLP cloud service.'''
    def __init__(self, nlp, address='ipc:///tmp/nlp-service.sock', encoder=JSONEncoder):
        self.nlp = nlp
        self.address = address
        self.encoder = encoder
        #self.service = Responder(self.address, encoder=self.encoder)
        self.service = Responder(self.address, encoder=self.encoder())
        # Introduction Planner: build the cognitive map
        self.service.register('similar', self.nlp.similar)
        # understand the action and predict the bot to perform the action  
        self.service.register('intent', self.nlp.intent)
         # predict the next action your bot should perform
        self.service.register('converse', self.nlp.converse)
        # (Understand) : parse a message into structured data :-> 
        # extract event, datetime, location, person, numbers, ...
        self.service.register('all', self.nlp.get_all)
        # the full parse 
        self.service.register('parse', self.nlp.parse)
        # word vector
        self.service.register('vector', self.nlp.vector)
        # validate parser
        self.service.register('validate_email', self.nlp.validate_email)
        self.service.register('validate_person', self.nlp.validate_person)
        self.service.register('validate_date', self.nlp.validate_date)
        self.service.register('validate_address', self.nlp.validate_address)

        self.service.register('echo', self.nlp.echo)

if __name__ == '__main__':
    print "Loading  NLP in Memory"
    nlp = NLPService()
    print "Registering ...."
    s = ConfigService(nlp, address=registry['NLP_SERVICE_ADDRESS'])
    #s = ConfigService(nlp)
    print "Starting  NLP service at {}".format(registry['NLP_SERVICE_ADDRESS'])
    s.service.start()
