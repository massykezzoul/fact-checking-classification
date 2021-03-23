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

    @param _lem, _stem ne peuvent pas être à True en même temps. Si c'est le cas _stem est ignorer.
    @param si _tokenize est False alors _pos_tag est ignorer
"""
class TextPreTraitement:
    _lowercase=False
    _lem=False
    _pos_tag=False
    _stem=False
    _ponctuation=False
    _contraction=False
    _tokenize=False
    _stopword=False
    _number2words=False
    _language='english'

    def __init__(self,lowercase=False,
            lem=False,
            pos_tag=False,
            stem=False,
            ponctuation=False,
            contraction=False,
            tokenize=False,
            stopword=False,
            number2words=False,
            language='english'):
        self._lowercase = lowercase
        self._lem = lem
        self._pos_tag = pos_tag
        self._stem = stem
        self._ponctuation = ponctuation
        self._contraction = contraction
        self._tokenize = tokenize
        self._stopword = stopword
        self._number2words = number2words
        self._language = language

    def fit(self, X):
        return self

    def transform(self, X):
        transformed = X
        if self._lowercase:
            transformed = self.lowercase(transformed)
        
        if self._contraction:
            transformed = self.contraction(transformed)

        if self._number2words:
            transformed = self.number2words(transformed)

        if self._ponctuation:
            transformed = self.ponctuation(transformed)

        transformed = self.tokenize(transformed)

        pos_tags = []
        text_tag = self.pos_tag(transformed)
        pos_tags = [[p for t,p in text] for text in text_tag]

        if self._stopword:
            text_tag = self.stopword(transformed,
                                            lst_pos_tags=pos_tags,
                                                language=self._language)
            pos_tags = [[p for t,p in text] for text in text_tag]
            transformed = [[t for t,p in text] for text in text_tag]
        del text_tag

        if self._lem:
            transformed = self.lem(transformed,pos_tags)

        elif self._stem:
            transformed = self.stem(transformed)

        if self._tokenize:
            if self._pos_tag:
                return [list(zip(text,pos)) for text, pos in zip(transformed, pos_tags)]
            else:
                return transformed
        else:
            return [" ".join(text) for text in transformed]

    def fit_transform(self, X):
        return self.fit(X).transform(X)

    """
        Définitions des méthodes de prétraitement
    """

    """
        Tokenize all element in @param lst
        @return a list of a list of tokens
    """
    @staticmethod
    def tokenize(lst):
        return [nltk.word_tokenize(e) for e in lst]

    """
        @return each elements of @param lst in lowercase
    """
    @staticmethod
    def lowercase(lst):
        return [e.lower() for e in lst]

    """
        @param text is a string that contain a number
        @return the number as words. eg: "15" -> "fifteen"
    """
    @staticmethod
    def number_to_words(text):
        try:
            float(text)
        except ValueError:
            return text
        else:
            return inflect.engine().number_to_words(text)

    """
        @param lst is a list of text. text should not be tokenized
        @return list of text such as all numbers are converted to words
    """
    @staticmethod
    def number2words(lst):
        tokens_lst = TextPreTraitement.tokenize(lst)
        result = []
        for tokens in tokens_lst:
            for i, word in enumerate(tokens):
                tokens[i] = TextPreTraitement.number_to_words(word)
            result.append(' '.join(tokens))
        return result

    @staticmethod
    def contraction(lst):
        return [c.fix(text) for text in lst]

    @staticmethod
    def ponctuation(lst):
        return [" ".join(re.sub(r'[^\w\s]', ' ', text).split()) for text in lst]

    """
        @param lst_tokenized should be a list of tokenized text
    """
    @staticmethod
    def stopword(lst_tokenized,lst_pos_tags=[],language='english'):
        stop_words = set(stopwords.words(language))
        if len(lst_pos_tags) == len(lst_tokenized):
            return [[(w, tag) for w, tag in zip(tokens,pos_tags) if not w in stop_words] 
                        for tokens, pos_tags in zip(lst_tokenized,lst_pos_tags)
                        ]
        else:
            return [[w for w in text if not w in stop_words] for text in lst_tokenized]

    """
        @param lst_tokenized should be a list of tokenized text
    """
    @staticmethod
    def stem(lst_tokenized, language='english',stemmer_name='snowball',verbose=False):
        if stemmer_name == 'snowball':
            if verbose:
                print('Snowball stemmer used!')
            stemmer = SnowballStemmer(language=language) 
        elif stemmer_name == 'lancaster':
            if language != 'english':
                print("LancasterStemmer do not suport "+language, file=sys.stderr)
                raise ValueError()
            stemmer = LancasterStemmer()

        return [[stemmer.stem(term) for term in text_tokenized] for text_tokenized in lst_tokenized]
    
    """
        @param lst_tokenized should be a list of tokenized text
    """
    @staticmethod
    def pos_tag(lst_tokenized):
        return [nltk.pos_tag(words) for words in lst_tokenized]

    @staticmethod
    def get_wordnet_pos(treebank_tag):
        if treebank_tag.startswith('J'):
            return wordnet.ADJ
        elif treebank_tag.startswith('V'):
            return wordnet.VERB
        elif treebank_tag.startswith('N'):
            return wordnet.NOUN
        elif treebank_tag.startswith('R'):
            return wordnet.ADV
        else:
            return None

    @staticmethod
    def lem(lst_tokenized, lst_pos_tag=[]):
        lemmatizer = WordNetLemmatizer()
        result = []
        if (len(lst_tokenized) == len(lst_pos_tag)):
            for words, pos_tag in zip(lst_tokenized,lst_pos_tag):
                if len(words) == len(pos_tag):
                    lem = []
                    pos_tag = [TextPreTraitement.get_wordnet_pos(p[1]) for p in pos_tag]
                    for w, pos in zip(words,pos_tag):
                        if pos is None:
                            lem.append(lemmatizer.lemmatize(w))
                        else:
                            lem.append(lemmatizer.lemmatize(w, pos=pos))
                    result.append(lem)
                else:
                    result.append([lemmatizer.lemmatize(w) for w in words])
            return result
        else:
            return [[lemmatizer.lemmatize(w) for w in words] for words in lst_tokenized]

if __name__ == "__main__":
    text = "More people have watched ‘Morning Joe’ than CNN and HLN 5 years in a row."

    pretraitement = TextPreTraitement(
        lowercase=True,
        lem=True,
        pos_tag=True,
        ponctuation=True,
        contraction=True,
        tokenize=True,
        stopword=True,
        number2words=True
    )

    X = pretraitement.fit_transform([text])

    print(X[0])


