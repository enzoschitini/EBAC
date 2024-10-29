import streamlit as st
import pandas as pd
import plotly.express as px

from sklearn import tree
from sklearn.tree import plot_tree
from sklearn.tree import DecisionTreeClassifier

from sklearn.metrics import accuracy_score
from sklearn.metrics import confusion_matrix
from sklearn.metrics import precision_score
from sklearn.metrics import recall_score

from sklearn.model_selection import train_test_split
from sklearn.model_selection import cross_val_score

from imblearn.under_sampling import RandomUnderSampler
from sklearn.metrics import classification_report

class Avvio(object):
    def __init__(self) -> None:
        pass

    def benvenuto(self):  # Add the 'self' parameter here
        st.write('Benvenuto')
    
    def guida(self, data_frame): # 14/02/2024
        righe, qnt_colonne = data_frame.shape
        quantita_righe = format(righe, ",").replace(',', '.')
        sequenza = list(range(qnt_colonne + 1))
        sequenza = sequenza[1:]

        colonne = data_frame.columns.to_list()
        types_list = [str(type(data_frame[col][0])).split("'")[1] for col in data_frame.columns]
        lista_categorie = [data_frame[col].nunique() for col in data_frame.columns]

        elementi_nulli = data_frame.isnull().sum()
        elementi_nulli = elementi_nulli.to_list()

        memory = (data_frame.memory_usage(deep=True) / (1024 ** 2)).round() # Mb
        memory_list = memory.to_list()
        memory_list = memory_list[1:]

        memory = data_frame.memory_usage(deep=True) # Totale Mb
        memory_totale = round(memory.sum() / (1024 ** 2), 2)

        percentuale_dati_nulli = round((data_frame.isnull().sum() / righe) * 100)
        percentuale_dati_nulli = percentuale_dati_nulli.to_list()

        data = pd.DataFrame({'Nome': colonne, 
                            'Tipo': types_list, 
                            'qunt_categorie': lista_categorie,
                            'Dati nulli' : elementi_nulli,
                            'Dati nulli %' : percentuale_dati_nulli,
                            'Memoria (Mb)': memory_list}, index=sequenza)
        
        # Intestazioni
        print('Teabella Esplorativa')
        print(f'In questi dati abbiamo {quantita_righe} righe e {qnt_colonne} colonne.')
        print(f'Consumassione di memoria: {memory_totale}Mb.')
        
        return data
    
    def grafico_barre(self, colonna, categoria, dataframe):
        contingency_table = pd.crosstab(dataframe[colonna], dataframe[categoria], normalize='index') * 100
        contingency_table = contingency_table.reset_index().melt(id_vars=colonna, value_name='Percentuale')

        fig = px.bar(
            contingency_table, 
            x=colonna, 
            y='Percentuale', 
            color=categoria, 
            barmode='group',
            labels={'Percentuale': 'Percentuale (%)', 'ESCMAE': 'Cetegorie'},
            title=f'Paragone tra {categoria} e {colonna}'
        )

        fig.update_layout(plot_bgcolor="white")
        st.plotly_chart(fig)
    
    def Variabile_Migliore(self, dataframe:pd.DataFrame, variabile_impostata):
        dataframe.dropna(inplace=True)
        dataframe.drop_duplicates(inplace=True)
        dataframe = pd.get_dummies(dataframe)

        X = dataframe.drop(columns=variabile_impostata)
        y = dataframe[variabile_impostata]

        st.write('Variabile Migliore')
        X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=.25, random_state=1729)
        clf = DecisionTreeClassifier(max_depth=4)
        clf.fit(X_train, y_train)

        features_names = X_train.columns

        features_importance = pd.DataFrame(clf.feature_importances_, index=features_names).sort_values(0, ascending=False)
        st.write(features_importance.head(25))
    
    def comprimere(self):
        def download_parquet(dataframe, file_name):
            parquet = dataframe.to_parquet(index=False)
            with open(file_name, "rb") as f:
                parquet_bytes = f.read()
            st.download_button(label="Clicca per scaricare", data=parquet_bytes, file_name=file_name, mime='application/octet-stream')
        
        uploaded_file = st.file_uploader("Carica un set di dati da comprimere", type=["csv"]) 
        file_name = "Data Frame.parquet"

        if uploaded_file:
            data = pd.read_csv(uploaded_file)
            data.to_parquet(file_name)
            download_parquet(data, file_name)