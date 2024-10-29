import streamlit as st
import time
import numpy as np
import pandas as pd
import seaborn as sns
import matplotlib.pyplot as plt

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

import analisi as pca

PCA = pca.Avvio() # Pacchetto delle analisi


st.set_page_config(layout='wide', page_title='Da Vinci',
                   page_icon="./Files/Vector.png")

st.image("./Files/Nome.png", width=250)
st.title('Benvenuto a Streamlit!')  
st.write('---')

st.sidebar.write('Bar')

pagine = ["---", "Benvenuto", "Analisi"]
pagina_impostata = st.sidebar.selectbox("Scegli cosa vuoi fare:", pagine)


def simulated_process():
    for percent_complete in range(100):
        time.sleep(0.01) 
        yield percent_complete + 1

def prestazioni(albero, teste_x, teste_y):
    predictions = albero.predict(teste_x)
    accuracy = accuracy_score(teste_y, predictions)
    precision_score_value = precision_score(teste_y, predictions)
    recall_score_value = recall_score(teste_y,predictions)
    
    st.write("Accuracy:  " + str((accuracy * 100).round()) + "%")
    st.write('\nConfusion Matrix:')
    #st.write(confusion_matrix(teste_y,predictions))
    st.write("\nRecall Score:  " + str((recall_score_value * 100).round()) + "%")
    st.write('\nPrecision Score Value  ' + str((precision_score_value * 100).round()) + "%")



if pagina_impostata == '---':
    st.write('')

elif pagina_impostata == 'Benvenuto':
    PCA.benvenuto()

elif pagina_impostata == 'Collegamento tra le variabili':
    st.write('#### Collegamento tra le variabili')

    testo_input = st.text_input("Inserisci il nome del progetto qui:")

    tipo_delle_colonne = ["object", "categoty"]
    tipo_impostato = st.selectbox("Imposta il tipo delle colonne categoriche", tipo_delle_colonne)

    uploaded_file = st.file_uploader("Carica un set di dati", type=["csv"])

    if uploaded_file:
        data = pd.read_csv(uploaded_file)
        #st.write(data_category)
        #colonne_categoriche = data.select_dtypes(tipo_impostato).columns.to_list()
        colonne_categoriche = data.columns.to_list()
        
        colonna_impostata = st.selectbox("Imposta la colonna da analizzare", colonne_categoriche)
        
        with st.expander("Vedere i dati"):
            st.write(data)
            righe, colonne = data.shape
            st.write("Ci sono " + str(righe) + ' righe e ' + str(colonne) + ' colonne')
    
    #tipo_impostato = st.selectbox("  ", tipo_delle_colonne)

    if testo_input and tipo_delle_colonne and uploaded_file:
        start_loading = st.button("Avviare il processo")

        if start_loading:
            progress_bar = st.progress(0)

            for percent_complete in simulated_process():
                progress_bar.progress(percent_complete)
            
            df = data.drop(columns='Unnamed: 0')
            df.drop_duplicates(inplace=True)
            df.dropna(inplace=True)

            #st.write(df)

            df = pd.get_dummies(df)
            y = df[colonna_impostata]
            X = df.drop(colonna_impostata, axis=1)

            with st.expander("Bilancio"):
                st.write("## Proporzione")
                fig, ax = plt.subplots()
                y.value_counts().plot.pie(autopct='%.2f', ax=ax)
                st.pyplot(fig)

                # Suddivisione dei dati -----------------------
                X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.3, random_state=100)

                # Bilancio -------------------------------------
                rus = RandomUnderSampler(random_state=42)
                X_train, y_train = rus.fit_resample(X_train, y_train)
                y_train.value_counts().plot.pie(autopct='%.2f');

                st.write("## Nuova proporzione")
                fig, ax = plt.subplots()
                y_train.value_counts().plot.pie(autopct='%.2f', ax=ax)
                st.pyplot(fig)

            # Modello --------------------------------------

            clf = DecisionTreeClassifier(random_state=100)
            path = clf.cost_complexity_pruning_path(X_train, y_train)
            ccp_alphas, impurities = path.ccp_alphas, path.impurities

            clfs = []
            for ccp_alpha in ccp_alphas:
                clf = DecisionTreeClassifier(random_state=0, ccp_alpha=ccp_alpha)
                clf.fit(X_train, y_train)
                clfs.append(clf)

            train_scores = [clf.score(X_train, y_train) for clf in clfs]
            test_score = [clf.score(X_test, y_test) for clf in clfs]

            with st.expander("Grafico del modello"):
                fig, ax = plt.subplots()
                ax.set_xlabel('Alpha')
                ax.set_ylabel("Accuratezza vs alpha dell'addestramento e teste")
                ax.plot(ccp_alphas, train_scores, marker='o', label='Addestramento',
                        drawstyle='steps-post')
                ax.plot(ccp_alphas, test_score, marker='o', label='Teste',
                        drawstyle='steps-post')
                ax.legend()
                st.pyplot(fig)

            ALPHA = pd.DataFrame({'Alphas':ccp_alphas.tolist(), 'Score': test_score})
            max_alpha = ALPHA.loc[ALPHA['Score'].idxmax(), 'Alphas']
            #st.write(max_alpha)

            clf = DecisionTreeClassifier(ccp_alpha=max_alpha, random_state=0)
            clf.fit(X_train, y_train) # 0.011505
            progress_bar.empty() #  ----------------------------- Nascondere la barra di caricamento

            predictions = clf.predict(X_test)

            # Calcola il numero totale di 0 e 1 all'interno dell'array
            count_0 = np.count_nonzero(predictions == 0)
            count_1 = np.count_nonzero(predictions == 1)

            # Calcola la proporzione di 0 e 1 rispetto alla lunghezza totale dell'array
            proportion_0 = (count_0 / len(predictions)) * 100
            proportion_1 = (count_1 / len(predictions)) * 100

            # Stampa i risultati
            #st.write("Proporzione di 0:" + str(proportion_0))
            #st.write("Proporzione di 1:" + str(proportion_1))
            
            with st.expander("Prestazioni del modello"):
                prestazioni(clf, X_test, y_test)
                report = classification_report(y_test, predictions, target_names=['0', '1'])
                st.text(report)
            
            with st.expander("Colonne con un alto collegamento"):
                feature_names = X.columns
                feature_importance = pd.DataFrame(clf.feature_importances_, index = feature_names).sort_values(0, ascending=False)
                st.write(feature_importance)
            
            with st.expander("Albero"):
                fig = plt.figure(figsize=(25,10))
                _ = tree.plot_tree(clf, 
                                feature_names=feature_names,  
                                class_names={0:'Paga', 1:'Não paga'},
                                filled=True,
                                fontsize=12)
                st.pyplot(fig)

elif pagina_impostata == 'Analisi':
    st.write('Analisi')
    provenienza = ["---", "Link"]
    provenienza_impostata = st.selectbox("Da dove vengono i dati?", provenienza)

    def caricando_i_dati(provenienza_impostata):
        if provenienza_impostata == '---':
            uploaded_file = st.file_uploader("Carica un set di dati da analizzare", type=["csv"])
            if uploaded_file:
                data = pd.read_csv(uploaded_file)
                data.columns.to_list()
        else:
            link_del_dataset = st.text_input("Inserisci il link dei dati qui:")
            if link_del_dataset:
                data = pd.read_csv(link_del_dataset)
        
        if 'Unnamed: 0' in data.columns:
            data = data.drop(columns='Unnamed: 0')
        return data
    
    def viwer_data(data):
        with st.expander("Vedere i dati"):
            st.write(data)
            righe, colonne = data.shape
            st.write("Ci sono " + str(righe) + ' righe e ' + str(colonne) + ' colonne')

        with st.expander("Tabella Esplorativa"):
            tabella = PCA.guida(data)
            st.write(tabella)
        
        with st.expander("Distrubuzione tramite un grafico di barre"):
            colonne_da_impostare = data.select_dtypes('object').columns.to_list()
            col1, col2 = st.columns(2)
            
            with col1:
                colonna_impostata_01 = st.selectbox("Scegli la prima colonna", colonne_da_impostare)
            with col2:
                colonna_impostata_02 = st.selectbox("Scegli la seconda colonna", colonne_da_impostare)

            button_clicked = st.button('Clicca qui')

            if button_clicked:
                PCA.grafico_barre(colonna_impostata_01, colonna_impostata_02, data)

    try:
        data = caricando_i_dati(provenienza_impostata)
        viwer_data(data)
    except UnboundLocalError:
        st.write('Non hai inserito nessun database al momento')