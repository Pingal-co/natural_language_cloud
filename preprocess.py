import re
from autocorrect import spell
import spacy

# initialize spacy
class NLP:
    """
     Singleton instance
    """
    def __init__(self):
        self.nlp = None

    def getInstance(self):
        if self.nlp==None:
            self.nlp = spacy.load('en')
        return self.nlp

# prepare the singleton
spacy_nlp = NLP()

def tokenize(text):
    """ Tokenize a sentence with spaCy.
    """
    nlp = spacy_nlp.getInstance()   # lazy loading
    tokenizer = nlp(unicode(text))
    tokens = map(str, [token for token in tokenizer])
    lemmas = map(str, [token.lemma_ for token in tokenizer]) 
    return zip(tokens,lemmas)

# helper to clean all text before operation
def clean_text(text):
    # first clean out symbols
    #text = re.sub(r'[^\w]', ' ', text)  
    # Replace linebreaks with spaces
    text = text.replace("\n", " ").replace("\r", " ").replace("\t", " ")
    # Remove any leading or trailing whitespace
    text = text.strip()
    # Remove consecutive spaces
    text = re.sub(" +", " ", text)
    # tokenize
    text = text.split()
    # correct all spellings
    #text = map(spell, text)
    return " ".join(text)

def preprocessor(text, pipeline):
    """
        preprocess the text as per the given pipeline
        pipeline is a list of functions that process an input text and outputs text 
        (e.g., lemmatizing, removing punctuations etc.),
    """
    if len(pipeline)==0:
        return text
    else:
        return preprocessor(pipeline[0](text), pipeline[1:])

def run(pipeline):
    return lambda text: preprocessor(text, pipeline)

def pipeline_one():
    """
        text  
            |> remove special characters
            |> remove numerals,
            |> convert all alphabets to lower cases,
            |> filter words, e.g., stop words, and
            |> lemmatize the words 
    """
    pipeline = [
        lambda s: re.sub('[^\w\s]', '', s),
        lambda s: re.sub('[\d]', '', s),
        lambda s: s.lower(),
        lambda s: ' '.join([ lemma for word, lemma in tokenize(s)])
        ]
    return run(pipeline)

