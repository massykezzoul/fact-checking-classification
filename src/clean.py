import pandas as pd 

def clean_claimKG(df, verbose=False, inplace=False):
    if verbose:
        print("Taille du dataframe:",df.shape)
    nb_assertion = len(df)
    manq = df.isnull().sum()
    col_manq = ['Unnamed: 0','claimReview_source']
    for k, v in manq.items():
        if v > nb_assertion/2 :
            col_manq.append(k)
    if verbose:
        print("Suppression des",len(col_manq),"columns suivant:")
        for col in col_manq:
            print("\t->",col)
    if inplace:
        df.drop(columns=col_manq,inplace=inplace)
    else:
        df = df.drop(columns=col_manq,inplace=inplace)

    d = [i for i, v in df.duplicated().items() if v]
    if verbose:
        print("Suppression de",len(d),"lignes en doubles.")
    
    if inplace:
        df.drop(d,inplace=inplace)
    else:
        df = df.drop(d,inplace=inplace)
    
    mask = df['claimReview_claimReviewed'].isin(["false","true"])
    rm = set(df[mask].index.values)
    mask = df['claimReview_claimReviewed'].isnull()
    rm = rm.union(set(df[mask].index.values))
    
    if verbose:
        print("Suppression de",len(rm),"lignes.")
    if inplace:
        df.drop(rm, inplace=inplace)
    else:
        df = df.drop(rm, inplace=inplace)

    if verbose:
        print("Taille finale:",df.shape)
    return df


if __name__ == '__main__':
    file_name = "../data/claimKG.csv"

    # Lecture du fichier
    kg = pd.read_csv(file_name)

    clean_claimKG(kg, verbose=True,inplace=True)