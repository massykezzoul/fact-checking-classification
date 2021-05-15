from pretraitement import TextPreTraitement
from clean import clean_claimKG

import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.svm import SVC
from sklearn.pipeline import Pipeline
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.metrics import accuracy_score, confusion_matrix

def cut_data(df,col_name='ratingValue'):
    keep = df[col_name].unique()
    n = df[col_name].value_counts().min()

    new_df = df.loc[df[col_name] == keep[0]][:n]
    for i in keep[1:]:
        new_df = pd.concat([new_df , df.loc[df[col_name] == i][:n]])

    return new_df

def trueVSfalse(kg):
    translate = {'false': 0,
        'true': 1
    }
    # mettre toutes les valeurs en miniscule
    kg['rating_alternateName'] = kg['rating_alternateName'].apply(lambda x : str(x).lower())
    keep = ['false', 'true']
    rm_lst = kg.index[~kg['rating_alternateName'].isin(keep)].tolist()
    kg_clean = kg.drop(rm_lst)

    kg_clean['ratingValue'] = kg_clean['rating_alternateName'].apply(lambda x: translate[x])

    return kg_clean

def trueFalseVSmixture(kg):
    translate = {'false': 0,
        'true': 0,
        'mixture': 1,
        'mostly true': 1,
        'mostly false': 1,
        'half-true': 1,
    }

    # mettre toutes les valeurs en miniscule
    kg['rating_alternateName'] = kg['rating_alternateName'].apply(lambda x : str(x).lower())
    keep = ['false', 'true', 'mostly false', 'mostly true','half-true', 'mixture']
    rm_lst = kg.index[~kg['rating_alternateName'].isin(keep)].tolist()
    kg_clean = kg.drop(rm_lst)

    kg_clean['ratingValue'] = kg_clean['rating_alternateName'].apply(lambda x: translate[x])

    return kg_clean

def trueVSfalseVSmixture(kg):
    translate = {'false': 0,
        'true': 1,
        'mixture': 2,
        'mostly true': 2,
        'mostly false': 2,
        'half-true': 2,
    }
    
    # mettre toutes les valeurs en miniscule
    kg['rating_alternateName'] = kg['rating_alternateName'].apply(lambda x : str(x).lower())
    keep = ['false', 'true', 'mostly false', 'mostly true','half-true', 'mixture']
    rm_lst = kg.index[~kg['rating_alternateName'].isin(keep)].tolist()
    kg_clean = kg.drop(rm_lst)

    kg_clean['ratingValue'] = kg_clean['rating_alternateName'].apply(lambda x: translate[x])

    return kg_clean
