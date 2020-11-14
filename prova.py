import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import streamlit as st
from sklearn.preprocessing import StandardScaler
from sklearn.tree import DecisionTreeClassifier
from sklearn.manifold import Isomap
from sklearn.model_selection import train_test_split

if __name__ == "__main__":

    df_heart = pd.read_csv('heart.csv')
    df_heart.head()

    df_heart.info()
    # nessuna colonna del dataset presenta dati mancanti
    # creo la lista delle features numeriche e, per esclusione, ricavo la lista delle features da convertire in categoriche
    numerical_features = ['age', 'trestbps', 'chol', 'thalach', 'oldpeak']
    categorical_features = df_heart.columns.drop(numerical_features)

    # dichiaro il dizionario delle features categoriche e lo utilizzo come input nella funzione "astype" per la conversione
    dict_categorical = {c:'category' for c in categorical_features}
    df_heart = df_heart.astype(dict_categorical)
    df_heart.info()

    df_heart[categorical_features].describe()
    # il dataset è discretamente bilanciato sulla colonna target, che presenta circa il 54% dei valori pari a "1" e il
    # restante 46% pari a "0"

    pd.plotting.scatter_matrix(df_heart[numerical_features], c=df_heart['target'], figsize=(10,10))
    plt.show()

    df_heart[categorical_features].groupby('target').count()
    # su questo punto non sono sicurissimo!

    df_heart[numerical_features].describe()



    scaler = StandardScaler()
    scaled_df = pd.DataFrame(scaler.fit_transform(df_heart[numerical_features]), columns=numerical_features)
    scaled_df.describe()

    # Dopo la standardizzazione, le features numeriche presentano effettivamente valori di media e deviazione standard molto
    # prossimi a 0 e 1



    isomap = Isomap(n_neighbors=6, n_components=3)
    isomap_df = isomap.fit_transform(scaled_df)
    type(isomap_df)

    heart_iso = pd.DataFrame(isomap_df, columns=['ISO1', 'ISO2', 'ISO3'])
    heart_iso.head()

    pd.plotting.scatter_matrix(heart_iso, c=df_heart['target'], figsize=(10,10))

    # Utilizzo la funzione "train_test_split" del modulo sklearn.model_selection che permette di dividere facilmente il dataset
    # in train-set e test-set rispettando, al contempo, le proporzioni rispetto alla colonna target (di default la funzione
    # esegue lo shuffle del dataset)



    X = df_heart.drop('target', axis=1)
    y = df_heart['target']

    heart_train, heart_test, target_train, target_test = train_test_split(X, y, test_size=0.2, random_state = 19, stratify=y)



    tree_clf = DecisionTreeClassifier() # l'albero decisionale può essere istanziato anche senza iperparametri
    tree_clf.fit(heart_train, target_train)
    print('Decision Tree score on training set: {}'.format(tree_clf.score(heart_train, target_train)))

    print('Decision Tree score on test set: {}'.format(tree_clf.score(heart_test, target_test)))

    # A fronte di uno score praticamente perfetto sul training set, 

    feat_imp = {feature:round(tree_clf.feature_importances_[i]*100,2) for i,feature in enumerate(X.columns)}
    feat_imp

    # Su 13 features, solo 4 presentano un peso superiore ad una soglia arbitraria del 10% ('cp', 'trestbps', 'oldpeak' e 'ca').
    # La feature 'exang' presenta un peso pari a zero, non risultando perciò utile alla definizione del modello.
    

