import random
import json

def flatten(d, parent_key=''):
    """
    flatten data dict
    """
    items = []
    for key, val in d.items():
        new_key = parent_key + ":" + key if parent_key else key
        if isinstance(val, dict):       
            items.extend(flatten(val, new_key).items())
        if isinstance(val, list):
            items.append((new_key, val))
    return dict(items)

def vocab_csv(vocab="data/vocab.json"):
    categories=json.load(open(vocab))
    outputs={}
    for key, value in categories.items():
        output = []
        for word in value.split():
            phrase = "" + " ".join(word.split('_'))
            output.append(phrase.encode('ascii', 'ignore'))
        outputs[key] = output

    return outputs

def generate_bot_dataset(vocab="data/vocab.json"):
    # 3 axes: topics, sentiments and skill_level
    # sentiments: language, emotions, slang
    # a person can like|dislike love|hate a topic and be beginners|expert level
    categories=json.load(open(vocab))
    templates=[
            "I like ", 
            "I am good at", 
            "I want", 
            "I need help in", 
            "I am passionate about", 
            "I am fan of", 
            "I want to discuss ideas on", 
            "I am from", 
            "people who like", 
            "Connect me to people in", 
            "I am reading", 
            "I am watching" 
    ]
    outputs={}
    for key, value in categories.items():
        output = []
        for word in value.split():
            phrase = " ".join(word.split('_'))
            sentence=random.choice(templates) + " " + phrase
            output.append(sentence.encode('ascii', 'ignore'))
        outputs[key] = output

    return outputs

def json2file(data, fname="data/vocab.txt"):
    with open(fname, 'w') as outfile:
        json.dump(data, outfile)


"""
templates = {
       "food": ["i'd like something asian",
                "maybe korean",
                "what mexican options do i have",
                "what italian options do i have",
                "i want korean food",
                "i want german food",
                "i want vegetarian food",
                "i would like chinese food",
                "i would like indian food",
                "i would like sushi",
                "what japanese options do i have",
                "korean please",
                "what about indian",
                "i want some vegan food",
                "maybe thai",
                "do you have noodles",
                "i would like to have some nachos",
                "is it possible to order some sushi",
                "I want to order tacos",
                "Can I get pasta please",
                "I would like to order some lasagna",
                "i'd like something vegetarian",
                "show me french restaurants",
                "show me a cool malaysian spot"
                ],
        }
"""