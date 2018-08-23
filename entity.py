'''
Entity classification: (person, enthusiast_type, location)
Custom Named-Entity recognition: 
 - Rule-based training for spacy Entity Recognizer
 - ML-based (We can retrain the Entity Recognizer model for better accuracy and hook it to the pipeline) 

Entity Classification also takes word orders into account unlike Intent Classification.
'''

import spacy
nlp = spacy.load('en')

def add_entity_rules():
    '''
        Custom Named-Entity recognition: rule-based training 
        e.g. get me a flight from SFO to LAX leaving 20 December and arriving on January 5th
    '''
    nlp.matcher.add('LAX_Airport', 'AIRPORT', {}, [[{ORTH: 'LAX'}]])
    nlp.matcher.add('SFO_Airport', 'AIRPORT', {}, [[{ORTH: 'SFO'}]])

    nlp.matcher.add(
        "GoogleNow", # Entity ID: Not really used at the moment.
        "PRODUCT",   # Entity type: should be one of the types in the NER data
        {"wiki_en": "Google_Now"}, # Arbitrary attributes. Currently unused.
        [  # List of patterns that can be Surface Forms of the entity

            # This Surface Form matches "Google Now", verbatim
            [ # Each Surface Form is a list of Token Specifiers.
                { # This Token Specifier matches tokens whose orth field is "Google"
                    ORTH: "Google"
                },
                { # This Token Specifier matches tokens whose orth field is "Now"
                    ORTH: "Now"
                }
            ],
            [ # This Surface Form matches "google now", verbatim, and requires
              # "google" to have the NNP tag. This helps prevent the pattern from
              # matching cases like "I will google now to look up the time"
                {
                    ORTH: "google",
                    TAG: "NNP"
                },
                {
                    ORTH: "now"
                }
            ]
        ]
    )
    # ents = [(ent.label_, ent.text) for ent in doc.ents]
    return ''