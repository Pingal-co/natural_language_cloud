'''
 Game: let people test pingal's mind
 E.g.:  Let them try their favourite band, food, slang words, technical things, person, ...
 
'''

# loading reddit vectors (based on reddit comments, 2015). Needs 1GB of memory
#model = sense2vec.load()

from __future__ import unicode_literals
import logging



class Sense(object):
    """
        Sense2Vec = Word + Part-of-Speech + Named Entities -> Learn Embedding Vectors
        So, you learn multiple embeddings for each word using these NLP annotations. 
        Not surprisingly, it yields better results than standard Word2Vec.
        
        Papers: https://arxiv.org/abs/1511.06388 ; http://wwwusers.di.uniroma1.it/~navigli/pubs/ACL_2015_Iacobaccietal.pdf
        Implementation: Based on https://demos.explosion.ai/sense2vec/
        Sense2Vec implementaion is based on Gensim and uses spacy for NLP annotations over reddit comments

    """

    def __init__(self, nlp, word2vec):
        self.nlp = nlp
        self.model = word2vec
        self.lemmatizer = self.nlp.vocab.morphology.lemmatizer
        self.pos = ['NOUN', 'VERB', 'ADJ', 'ORG', 'PERSON', 'FAC','PRODUCT', 'LOC', 'GPE']
        logging.info("Sense query")

    def __call__(self, phrase, n=50, service="similar"):
        #print "service: {}".format(service)

        if service == 'vector':
            return self.get_vector(phrase)
        logging.info("Sanitizing phrase")
        # word2vec is dependent on the key in the model
        key = self.sanitize_phrase(phrase)
        if not phrase or not key:
            # return {'key': '', 'text': phrase, 'results': [], 'count': 0}
            return []
        freq, _ = self.model[key]
        text = key.rsplit('|', 1)[0].replace('_', ' ')
        results = self.get_results(key)
        # return {'key': key, 'text': text, 'results': results, 'count': freq}
        return results    
    
    def get_vector(self, phrase):
        # get the vector from service
        # phrase = 'Tom_Brady|PERSON'
        key = self.sanitize_phrase(phrase)
        if not phrase or not key:
            return []
        freq, vec = self.model[key]
        return vec.tolist()

    def sanitize_phrase(self, phrase):
        phrase = phrase.replace(' ', '_')
        if "|" in phrase:
            text, pos = phrase.rsplit('|', 1)
            key = text + '|' + pos.upper()
            return key if key in self.model else None
        
        # autofill best part_of_speech info in the phrase
        return self.find_best_phrase(phrase) 

    def find_best_phrase(self, phrase):
        """
            finds the highest scoring phrase and add its POS
        """
        freqs = []
        candidates = [phrase, phrase.upper(), phrase.title()] if phrase.islower() else [phrase]
        for candidate in candidates:
            for pos in self.pos:
                key = candidate + '|' + pos
                if key in self.model:
                    freq, _ = self.model[key]
                    freqs.append((freq, key))
        return max(freqs)[1] if freqs else None

    def get_results(self, phrase, n=50):
        # remove the original
        text = phrase.rsplit('|', 1)[0].replace('_', ' ')
        results = []
        seen = set([text])
        # use basic lemmatization for keys
        seen.add(self.base_key(phrase))
        for words, score in self.get_similar(phrase, n * 2):
            freq, _ = self.model[words]
            base = self.base_key(words)
            if base not in seen:
                #print base, words
                """
                results.append({
                    'score': score,
                    'key': words,
                    'text': words.split('|')[0].replace('_', ' '),
                    'count': freq
                })
                """
                results.append(words.split('|')[0].replace('_', ' '))
                seen.add(base)
            if len(results) >= n:
                break
        return results


    def base_key(self, phrase):
        # dictionary keys
        if '|' not in phrase:
            return phrase.lower()
        text, pos = phrase.rsplit('|', 1)
        head = text.split('_')[-1]
        return min(self.lemmatizer(head, pos))

    def query_vector(self, phrase):
        # phrase = 'Tom_Brady|PERSON'
        key = self.sanitize_phrase(phrase)
        if key not in self.model:
            return []
        query_vector = self.model[key][1]
        return query_vector

    def get_similar(self, phrase, n=20):
        '''
            given a phrase, output most similar terms. 
            
            
        '''
        # phrase = u"natural_language_processing|NOUN"
        # most_similar_terms(['Donald_Trump|PERSON'])
        key = self.sanitize_phrase(phrase)
        if key not in self.model:
            return []
        freq, query_vector = self.model[phrase]
        words, scores = self.model.most_similar(query_vector, n)
        return zip(words,scores)

        # we must also filter the results based on Entity label
        
        
    def get_similar_by_tag(self, word, tag="PERSON", n=20):
        # most_similar_tag("find"; tag="VERB")
        # phrase = "natural_language_processing"; tag="NOUN"
        #  phrase = "Donald_Trump" ; tag="PERSON"
        phrase = word + '|' + tag
        if phrase not in self.model:
            return []
        freq, query_vector = self.model[phrase]
        words, scores = self.model.most_similar(query_vector, n)
        return zip(words, scores)


    def similarity(self, phrase1, phrase2):
        '''
            Test the similarity among two phrases
            phrase1 = u'multiplayer_game|NOUN'
            phrase2 = u'game|NOUN'
        '''
        if phrase1 not in self.model:
            return []
        if phrase2 not in self.model:
            return []
        return self.model.similarity(phrase1, phrase2)