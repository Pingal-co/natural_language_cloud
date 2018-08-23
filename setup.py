#!/usr/bin/env python

from distutils.core import setup

setup(name='pingal-brain',
      version='1.0.0',
      description='Pingal nlp brain',
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