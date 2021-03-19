import pandas as pd 

def clean_claimKG(df, verbose=False):
    nb_assertion = len(df)
    manq = df.isnull().sum()
    col_manq = ['Unnamed: 0','claimReview_source']
    for k, v in manq.items():
        if v > nb_assertion/2 :
            col_manq.append(k)
    if verbose:
        print("Suppression des columns suivant:",col_manq)
    df = df.drop(columns=col_manq)

    d = [i for i, v in df.duplicated().items() if v]
    rm = df['claimReview_claimReviewed'].isin(["false","true"]) |\
         df['claimReview_claimReviewed'].isnull()
    for i, v in rm.items():
        if v:
            d.append(i)
    df = df.drop(d)

    return df


if __name__ == '__main__':
    file_name = "../data/claimKG.csv"

    # Lecture du fichier
    kg = pd.read_csv(file_name)

    print('initial size=', len(kg))
    kg = clean_claimKG(kg, verbose=True)

    print('size=', len(kg))
    print('columns=',kg.columns)