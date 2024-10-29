import streamlit as st

import pandas as pd
import numpy as np

from sklearn.tree import DecisionTreeClassifier
from sklearn.ensemble import RandomForestClassifier
from sklearn.base import BaseEstimator, TransformerMixin
from sklearn.decomposition import PCA
from sklearn.pipeline import Pipeline
from sklearn.metrics import accuracy_score, confusion_matrix, classification_report

from sklearn.model_selection import train_test_split

import pickle
from io import BytesIO

st.set_page_config(layout='wide', page_title='EBAC - Projeto Final')
st.sidebar.write('# EBAC - Projeto Final')

# Pipeline
class NullImputer(BaseEstimator, TransformerMixin):
    def __init__(self):
        self.fill_values = {}

    def fit(self, X, y=None):
        for column in X.columns:
            if X[column].isnull().any():
                if X[column].dtype == np.number:
                    self.fill_values[column] = X[column].mean()
                else:
                    self.fill_values[column] = X[column].mode()[0]
        return self

    def transform(self, X, y=None):
        X = X.copy()
        for column, value in self.fill_values.items():
            X[column].fillna(value, inplace=True)
        return X

class OutlierRemover(BaseEstimator, TransformerMixin):
    def __init__(self, method='remove'):
        self.method = method
        self.outliers_indices = []

    def fit(self, X, y=None):
        for column in X.select_dtypes(include=[np.number]).columns:
            Q1 = X[column].quantile(0.25)
            Q3 = X[column].quantile(0.75)
            IQR = Q3 - Q1
            lower_bound = Q1 - 1.5 * IQR
            upper_bound = Q3 + 1.5 * IQR
            self.outliers_indices.extend(X[(X[column] < lower_bound) | (X[column] > upper_bound)].index)
        return self

    def transform(self, X, y=None):
        if self.method == 'remove':
            return X.drop(index=self.outliers_indices)
        elif self.method == 'cap':
            X = X.copy()
            for column in X.select_dtypes(include=[np.number]).columns:
                Q1 = X[column].quantile(0.25)
                Q3 = X[column].quantile(0.75)
                IQR = Q3 - Q1
                lower_bound = Q1 - 1.5 * IQR
                upper_bound = Q3 + 1.5 * IQR
                X[column] = np.where(X[column] < lower_bound, lower_bound, X[column])
                X[column] = np.where(X[column] > upper_bound, upper_bound, X[column])
            return X

class PCAReducer(BaseEstimator, TransformerMixin):
    def __init__(self, n_components=5):
        self.n_components = n_components
        self.pca = PCA(n_components=self.n_components)

    def fit(self, X, y=None):
        self.pca.fit(X)
        return self

    def transform(self, X, y=None):
        return pd.DataFrame(self.pca.transform(X))

class DummiesTransformer(BaseEstimator, TransformerMixin):
    def fit(self, X, y=None):
        return self
    
    def transform(self, X, y=None):
        return pd.get_dummies(X, drop_first=True)

def preprocessamento():
    pipeline = Pipeline(steps=[
        ('imputer', NullImputer()),                 # Substituição de nulos
        ('pca', PCAReducer(n_components=5)),        # Redução de dimensionalidade (PCA)
        ('dummies', DummiesTransformer())           # Criação de dummies (inclui 'posse_de_veiculo')
    ])
    return pipeline

def pre_df(df):
    df['data_ref'] = pd.to_datetime(df['data_ref'])
    df['mes'] = df['data_ref'].dt.month_name()

    df = df[df['mes'].isin(['December', 'November', 'October'])]
    df = df.drop(columns=['data_ref', 'index'])
    df = df.drop_duplicates()

    return df

#####################################################
#               Interface de uso
#####################################################

st.sidebar.write("## Suba o arquivo")
data_file_1 = st.sidebar.file_uploader("Bank Credit Dataset", type = ['csv','ftr'])

# Verifica se há conteúdo carregado na aplicação
if (data_file_1 is not None):
    df = pd.read_csv(data_file_1).drop(columns=['Unnamed: 0'])
    df = pre_df(df)
    df_ML = pd.get_dummies(df, drop_first=True)

    X = df_ML.drop('mau', axis=1)
    y = df_ML['mau']

    pipeline = preprocessamento()
    X_preprocessed = pipeline.fit_transform(X)

    with open('decision_tree_model.pkl', 'rb') as file:
        loaded_model = pickle.load(file)

    predictions = loaded_model.predict(X_preprocessed)
    df['predictions'] = predictions

    st.write(df.drop(columns='mau'))