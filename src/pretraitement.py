import sys, re, inflect
import pandas as pd
from clean import clean_claimKG

import contractions as c
import nltk
from nltk.corpus import stopwords, wordnet
from nltk.stem.snowball import SnowballStemmer 
from nltk.stem.lancaster import LancasterStemmer
from nltk.stem import WordNetLemmatizer 
#nltk.download('all')

"""
    Classe qui permet de prétraiter les données. Pour activer une option, rendez sa valeur à True.

    @param lem, stem ne peuvent pas être à True en même temps. Si c'est le cas stem est ignorer.
"""
class TextPreTraitement:
    lowercase=False
    lem=False
    pos=False
    stem=False
    ponctuation=False
    contraction=False
    tokenize=False
    stopword=False
    number2words=False

    def __init__(self,lowercase=False,
            lem=False,
            pos=False,
            stem=False,
            ponctuation=False,
            contraction=False,
            tokenize=False,
            stopword=False,
            number2words=False):
        self.lowercase=lowercase
        self.lem=lem
        self.pos=pos
        self.stem=stem
        self.ponctuation=ponctuation
        self.contraction=contraction
        self.tokenize=tokenize
        self.stopword=stopword
        self.number2words=number2words

    def fit(self, X):
        return self

    def transform(self, X):
        transformed = X
        if self.lowercase:
            transformed = self.lowercase(transformed)
        
        if self.contraction:
            transformed = self.contraction(transformed)

        if self.number2words:
            transformed = self.number2words(transformed)

        if self.ponctuation:
            transformed = self.ponctuation(transformed)

        if self.tokenize:
            transformed = self.tokenize(transformed)

        pos_tags = []
        if self.pos:
            transformed = self.pos_tag(transformed)
            pos_tags = [p for t,p in transformed]
        
        if self.stopword:
            transformed = self.stopword(transformed)

        if self.lem:
            if len(pos_tags) != len(transformed):
                pos_tags = self.pos_tag(transformed)
                pos_tags = [p for t,p in pos_tags]
            transformed = self.lem(transformed,pos_tags)
        elif self.stem:
            transformed = self.stem(transformed)

        return transformed

    def fit_transform(self, X):
        return self.fit(X).transform(X)