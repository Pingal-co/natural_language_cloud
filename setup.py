#!/usr/bin/env python

from distutils.core import setup

setup(name='natural-language-cloud',
      version='1.0.0',
      description='nlp services',
      author='Pingal Team',
      author_email='help@pingal.ai',
      license='MIT',
      install_requires=[
        'Cython',
        'numpy',
        'sputnik',
        'gensim',
        'spacy',
        'sense2vec',
        'keras',
        'theano',
        'nanoservice',
        'parserator', 
        'probablepeople',
        'usaddress',
        'autocorrect'
      ]
)
