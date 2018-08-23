'''
 Parse the incoming text and add NLP annotations
 Helper functions to identify the words in the text that look like:
   - Person, product [objects, vehicles, foods, ...], organization [companies, agencies, institutions, ...]
   - Date, time
   - Location : {GPE: [countries, cities, states], LOC: [mountain ranges, water bodies] 
   - Media: books, songs, etc
   - Language
   - Nationalities, religious & political groups
   - Event [Hurricanes, battles, sports events, etc]
 
 e.g. 
 - get me a flight from New York to Boston leaving 20 December and arriving on January 5th
 - introduce me to people talking about Donald Trump Immigration policies
 - connect me to cricket enthusiast
 - show me people shopping for study table
 - find greek people around me
 - find people working on react native
 - chat on superbowl game
 - Favorite band, technical things, slang words

 Return: {
    intent: "",
    person: [],
    product: "",
    organization: "",
    location: [],
    address: {},
    datetime: "",
    media: [],
    email: "",
    event_title: "",
    action_verbs: []
    keywords: [],
    similar_terms: [],
    index_terms: [],
    nationalities:"",
    religious_political_groups: ""
 }

preprocess |> parse_sentence |> map(extractors)
'''
from __future__ import unicode_literals

import spacy
import re
import logging

from preprocess import clean_text
from date_parse import datetime_parsing
from extract import extract_entity, extract_address, extract_info


#from spacy.en import English
#nlp=English()
#nlp = spacy.load('en')

class Parse(object):
    """
      Parses the word, sentence or a document in English language
      - extracts token and sentences
      - identifies parts of speech
      - creates dependency parse trees for each sentence 
      - identifies entities and labels by types such as person, organization, location, events, products and media
      - better date and time parser
    """
    def __init__(self, nlp):
        self.nlp = nlp
        logging.info("Parse query")
    
    def __call__(self, document, parser='sentence'):
        if parser == 'pos':
            return  self.words_pos(document)
        if parser == 'document':
            return self.parse_doc(document)
        return self.parse_sentence(document) 

        
    def join_entity(self, doc):
        for ent in doc.ents:
            ent.merge(ent.root.tag_, ent.text, ent.label_)
    
    def join_noun_phrases(self, doc):
        for np in doc.noun_chunks:
            while len(np) > 1 and np[0].dep_ not in ('advmod', 'amod', 'compound'):
                np = np[1:]
            np.merge(np.root.tag_, np.text, np.root.ent_type_)


    def words_pos(self, input):
        """
            add pos or entity info to each word in the input
        """
        doc = self.nlp(clean_text(input))
        self.join_entity(doc)
        strings = []
        for sent in doc.sents:
            if sent.text.strip():
                strings.append(' '.join(self.represent_word(w) for w in sent if not w.is_space))
        if strings:
            return '\n'.join(strings) + '\n'
        else:
            return ''

    def represent_word(self, word):
        """
            word_pos = word_token + entity_or_pos
        """
        text = word.text.title()
        text = re.sub(r'\s', '_', text)
        tag = word.ent_type_ if word.ent_type_ else word.pos_
        if not tag:
            tag = '?'
        return text + '|' + tag

    def find_keyphrases(self, line):
        if not line.text:
            line = self.nlp(clean_text(line))

        keyphrases = set([word.text for word in line.ents])

        # add nouns that have word_vectors
        nouns = [w.text for w in line if w.tag_ in ['NN', 'NNP']]
        # add noun chunks that have word vectors
        candidates = [word.text for word in line.noun_chunks]

        # add noun_chunks if noun in noun chunks else add noun
        if nouns and candidates:
            for noun in nouns:
                add_noun = True
                for candidate in candidates:
                    if noun in candidate:
                        add_noun = False
                        keyphrases.add(candidate)
                if add_noun:
                    keyphrases.add(noun)

        return keyphrases

    def parse_sentence(self, input):
        '''
            parse each sentence and return the spacy's nlp properties
            Pipeline: sentence |> nlp.tokenizer |> nlp.tagger |> nlp.parser |> nlp.entity
        '''
        line = self.nlp(clean_text(input))
        
        keyphrases = self.find_keyphrases(line)
        output = {
            'text': line.text,
            "entities": [(entity.text, entity.label_) for entity in line.ents],
            # Given we will be dealing with small text, I am assuming we do not need to compute word frequency to find keywords
            "keyphrases": list(keyphrases),
            'words': [(w.text, w.tag_, w.pos_, w.ent_type_, w.dep_) for w in line],
            'persons': [(entity.text, extract_entity(entity.text)["data"]) for entity in line.ents if entity.label_ == 'PERSON'],
            'products': [entity.text for entity in line.ents if entity.label_ == 'PRODUCT'],
            'organizations': [(entity.text, extract_entity(entity.text)["data"]) for entity in line.ents if entity.label_ == 'ORG'],
            'medias': [entity.text for entity in line.ents if entity.label_ == 'WORK_OF_ART'],
            'locations': [entity.text for entity in line.ents if entity.label_ == 'GPE'],
             'info': extract_info(line.text),
            'nouns': [w.text for w in line if w.tag_ == 'NN'],
            'action_verbs': [w.text for w in line if w.tag_ == 'VB'],
            'subject': [w.text for w in line if w.dep_ == 'nsubj'],
            'object': [w.text for w in line if w.dep_ == 'nobj']
        }

        return output

    def parse_doc(self, input):
        '''
            Assume input is a sequence of sentences.
            split multiple sentences and apply nlp parse
        '''
        doc = self.nlp(clean_text(input))
        return [self.parse_sentence(sent.text) for sent in doc.sents]
