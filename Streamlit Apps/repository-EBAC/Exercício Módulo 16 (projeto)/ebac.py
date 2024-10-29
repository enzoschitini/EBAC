import pandas as pd
import numpy as np
import seaborn as sns
import matplotlib.pyplot as plt
import streamlit as st
import time

from sklearn.model_selection import train_test_split
from sklearn.linear_model import LinearRegression
from sklearn.metrics import mean_squared_error
from sklearn.metrics import mean_absolute_error

st.set_page_config(layout='wide', page_title='Projeto EBAC - Previsão de renda',
                   page_icon="./Projeto EBAC - Previsão de Renda/Group.png")
col1, col2 = st.columns(2)

with col1:
    st.title('Projeto EBAC - Previsão de Renda')

with col2:
    st.write(' ')
    st.write(' ')
    st.write('02 Abril')

st.write('---') 
st.sidebar.image("./Projeto EBAC - Previsão de Renda/Group 1321314386.png", width=200)

renda = pd.read_csv('./Projeto EBAC - Previsão de Renda/previsao_de_renda.csv').drop(columns='Unnamed: 0', axis=1)
def download_csv(dataframe, file_name):
    csv = dataframe.to_csv(index=False)
    csv = csv.encode('utf-8')
    st.download_button(label="Clique para fazer o download", data=csv, file_name=file_name, mime='text/csv')

def tratamento(dataframe):
    dataframe['data_ref'] = pd.to_datetime(dataframe['data_ref'])
    dataframe['sexo'] = dataframe['sexo'].astype(bool)

    dataframe['tipo_renda'] = dataframe['tipo_renda'].astype('category')
    dataframe['educacao'] = dataframe['educacao'].astype('category')
    dataframe['estado_civil'] = dataframe['estado_civil'].astype('category')
    dataframe['tipo_residencia'] = dataframe['tipo_residencia'].astype('category')

    dataframe['qt_pessoas_residencia'] = dataframe['qt_pessoas_residencia'].astype(int)

    dataframe = dataframe.dropna(subset=['tempo_emprego'])

    dataframe['data_ref'] = pd.to_datetime(dataframe['data_ref'].astype(str), format='%Y-%m-%d')

    dataframe['Dia'] = dataframe['data_ref'].dt.day_name()
    dataframe['Mês'] = dataframe['data_ref'].dt.month_name()
    dataframe['Ano'] = dataframe['data_ref'].dt.year

    mapping_day = {"Monday": "Segunda-feira", 
            "Tuesday": "Terça-feira", 
            "Wednesday": "Quarta-feira",
            "Thursday": "Quinta-feira",
            "Friday": "Sexta-feira",
            "Saturday": "Sábado",
            "Sunday": "Domingo"}

    mapping_month = {"January": "Janeiro",
                    "February": "Fevereiro",
                    "March": "Março",
                    "April": "Abril",
                    "May": "Maio", 
                    "June": "Junho", 
                    "July": "Julho",
                    "August": "Agosto",
                    "September": "Setembro",
                    "October": "Outubro",
                    "November": "Novembro",
                    "December": "Dezembro"}

    dataframe["Dia"] = dataframe["Dia"].replace(mapping_day)
    dataframe["Mês"] = dataframe["Mês"].replace(mapping_month)

    dataframe['Dia'] = dataframe['Dia'].astype('category')
    dataframe['Mês'] = dataframe['Mês'].astype('category')

    return dataframe

renda = tratamento(renda)

# Lista de métricas
valores = ["Machine Learning", "Análise Exploratória", "---"]
valor_selecionado = st.sidebar.selectbox("Selecione um tipo de métrica:", valores)
st.sidebar.write('Veja uma pequena Análise Exploratória dos dados ou previsões feita pelo modelo de Machine Learning')
st.sidebar.write('[Enzo Schitini]("https://www.linkedin.com/in/enzoschitini/") 😉')

if valor_selecionado == 'Análise Exploratória': #  --------------------------------------- Análise Exploratória
    #st.write(renda)
    st.write('## Análise Exploratória:')
    st.write('')
    st.write('')
    def main():
        with st.expander("Gráfico de barras"): # ------------------------------ Gráfico de barras
            plt.close('all')
            plt.rc('figure', figsize=(23, 8))
            fig, axes = plt.subplots(2, 2)	

            sns.countplot(ax = axes[0, 0], x='educacao', data=renda)
            sns.countplot(ax = axes[0, 1], x='posse_de_veiculo', data=renda)
            sns.countplot(ax = axes[1, 0], x='posse_de_imovel', data=renda)
            sns.countplot(ax = axes[1, 1], x='tipo_residencia', data=renda)

            st.pyplot(plt)

            plt.close('all')
            plt.rc('figure', figsize=(23, 8))

            plt.close('all')
            sns.countplot(x='Dia', data=renda)
            st.pyplot(plt)

            plt.close('all')
            sns.countplot(x='qtd_filhos', data=renda)
            st.pyplot(plt)

            plt.close('all')
            sns.countplot(x='tipo_renda', data=renda)
            st.pyplot(plt)

        with st.expander("Boxplot"): # ------------------------------ Boxplot
            plt.close('all')
            plt.rc('figure', figsize=(10, 23))
            fig, axes = plt.subplots(5, 1)

            sns.boxplot(ax=axes[0], x='qtd_filhos', data=renda)
            sns.boxplot(ax=axes[1], x='idade', data=renda)
            sns.boxplot(ax=axes[2], x='tempo_emprego', data=renda)
            sns.boxplot(ax=axes[3], x='qt_pessoas_residencia', data=renda)
            sns.boxplot(ax=axes[4], x='renda', data=renda)
            st.pyplot(plt)

        with st.expander("Describe method"): # ------------------------------ Describe
            st.write(renda.select_dtypes('number').describe().transpose())
            file_name = "Describe number.csv"
            download_csv(pd.DataFrame(renda.select_dtypes('number').describe().transpose()), file_name)
            st.write(' ')
            st.write(renda.select_dtypes('category').describe().transpose())
            file_name = "Describe category.csv"
            download_csv(pd.DataFrame(renda.select_dtypes('category').describe().transpose()), file_name)

    if __name__ == "__main__":
        main()

elif valor_selecionado == 'Machine Learning': # ---------------------------------------Machine Learning Box
    st.write('## Machine Learning:')

    def main():
        st.write("#### Carregue o CSV `previsao_de_renda.csv` para fazer as previsões")

        # Up file csv
        uploaded_file = st.file_uploader("Carregar um arquivo CSV", type=["csv"])

        if uploaded_file is not None:
            # Lendo o arquivo
            df = pd.read_csv(uploaded_file)
            df = tratamento(df)

            def simulated_process():
                for percent_complete in range(100):
                    time.sleep(0.01) 
                    yield percent_complete + 1

            start_loading = st.button("Inicie o processo")

            if start_loading:
                progress_bar = st.progress(0)

                for percent_complete in simulated_process():
                    progress_bar.progress(percent_complete)

                with st.expander("Machine Learning Model"): # --------------------------------------- Machine Learning
                    data = df.drop(columns='Unnamed: 0', axis=1)
                    data = tratamento(data)

                    data['data_ref'] = pd.to_datetime(data['data_ref'])

                    data = pd.get_dummies(data, columns=['tipo_renda', 'educacao', 'estado_civil', 'tipo_residencia', 'Dia', 'Mês'])

                    X = data.drop(columns=["renda", "data_ref", "id_cliente"])
                    y = data["renda"]
                    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)
                    
                    model = LinearRegression()
                    model.fit(X_train, y_train)


                    st.write('### Renda prevista:')
                    y_pred = model.predict(X_test)
                    renda_prevista = pd.DataFrame({'Renda prevista': y_pred.tolist()})
                    st.write(renda_prevista)
                    st.write(renda_prevista.shape)
                    file_name = "previsões.csv"
                    download_csv(renda_prevista, file_name)

                    st.write('### Dados de teste:')
                    _ , data_teste = train_test_split(df.drop(columns=['Unnamed: 0', 'renda'], axis=1), test_size=0.2, random_state=42)
                    st.write(data_teste)
                    st.write(data_teste.shape)

                with st.expander("Avaliação dos resultados"): # ------------------------------ Describe
                    st.write('### Mean Squared Error (MSE):')
                    mse = mean_squared_error(y_test, y_pred, squared=False)
                    st.write(mse)

                    st.write('### Erro Médio Absoluto (MAE):')
                    mae = mean_absolute_error(y_test, y_pred)
                    st.write("MAE:", mae)

                    st.write('### Erro percentual médio absoluto (MAPE):')
                    def mean_absolute_percentage_error(y_true, y_pred): 
                        return np.mean(np.abs((y_true - y_pred) / y_true)) * 100

                    mape = mean_absolute_percentage_error(y_test, y_pred)
                    st.write("MAPE:", mape)

                    st.write('Gráfico do desempenho do modelo')

                    teste = list(y_test)
                    predicao = list(y_pred)

                    plt.close('all')
                    plt.rc('figure', figsize=(23, 8))

                    plt.plot(teste, label='Teste')
                    plt.plot(predicao, label='Predição')

                    plt.title('Desempenho do modelo')
                    plt.ylabel('Valor')

                    plt.legend()

                    st.pyplot(plt)
            
            #st.write(df)
            #st.write(df.shape)

    if __name__ == "__main__":
        main()

# Enzo Schitini 😉