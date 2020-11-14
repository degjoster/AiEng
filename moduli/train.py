import numpy as np
import pandas as pd
from sklearn.preprocessing import StandardScaler
from sklearn.tree import DecisionTreeClassifier
from sklearn.manifold import Isomap
from sklearn.model_selection import train_test_split

def splitHeart(df_heart):
    X = df_heart.drop('target', axis=1)
    y = df_heart['target']

    heart_train, heart_test, target_train, target_test = train_test_split(X, y, test_size=0.2, random_state = 19, stratify=y)
    return heart_train, heart_test, target_train, target_test 

def trainHeart(heart_train, heart_test, target_train, target_test):
    tree_clf = DecisionTreeClassifier() # l'albero decisionale può essere istanziato anche senza iperparametri
    tree_clf.fit(heart_train, target_train)
    print('Decision Tree score on training set: {}'.format(tree_clf.score(heart_train, target_train)))

    print('Decision Tree score on test set: {}'.format(tree_clf.score(heart_test, target_test)))

    return tree_clf

def feature_import(tree_clf):
    feat_imp = {feature:round(tree_clf.feature_importances_[i]*100,2) for i,feature in enumerate(X.columns)}
    feat_imp
    return
