import pandas as pd
import seaborn as sns
import matplotlib.pyplot as plt
import os
import streamlit as st 

# Configurações da página
st.set_page_config(layout='wide', page_title='Projeto EBAC - Análise SINASC',
                   page_icon="./mapa.png")

# Importando os dados
sinasc = pd.read_csv('./input/SINASC_RO_2019.csv')

##################################################################################################
##################################################################################################

# Barra lateral
st.sidebar.image("./mapa.png", width=100)
st.sidebar.write('# Estado de Rondônia')
st.sidebar.write('Filtro do dataset - Escolha como quer ver os dados:')

sinasc.DTNASC = pd.to_datetime(sinasc.DTNASC)
max_data = sinasc.DTNASC.max()
min_data = sinasc.DTNASC.min()

data_inicial = st.sidebar.date_input('Data inicial',
              value=min_data,
              min_value=min_data,
              max_value=max_data)

data_final = st.sidebar.date_input('Data final',
              value=max_data,
              min_value=min_data,
              max_value=max_data)

st.sidebar.write('[Enzo Schitini]("https://www.linkedin.com/in/enzoschitini/") 😉')

##################################################################################################
##################################################################################################

# Filtrando o Dataset
total_nascidos, _ = sinasc.shape
sinasc = sinasc[(sinasc['DTNASC'] <= pd.to_datetime(data_final)) & (sinasc['DTNASC'] >= pd.to_datetime(data_inicial))]

# Página
st.title('Análise SINASC')
st.write('###### Periodo: ' + str(data_inicial) + ' - ' + str(data_final))
col1, col2 = st.columns(2)

with col1:
    st.write("# Bem Vindo(a)!!!")
    st.write("Nessa página você vai poder entender melhor sobre os dados do SINASC de Rondônia no ano de 2019. Fique à vontade para escolher como deseja ver esses dados, definindo a análise e a época do ano")

with col2:
    total_nascidos_no_periodo, _ = sinasc.shape
    percentual = round(((total_nascidos_no_periodo / total_nascidos) * 100), 2)
    st.write('## Percentual de nascidos: ' + str(percentual) + '%')
    st.write('Total de nascidos no estado no ano de 2019: ' + str(total_nascidos))
    st.write('Total de nascidos no período selecionado: ' + str(total_nascidos_no_periodo))

st.write('---')

##################################################################################################
##################################################################################################

def select_columns(dataframe, mostrar:None, dados:None):
    selected_columns = st.multiselect("Selecione algumas colunas **(opcional)**:", dataframe.columns)
    if mostrar != None:
        if selected_columns:
            filtered_df = dataframe[selected_columns]
            #st.write("Colunas selecionadas")
            #st.write(filtered_df)
            st.write("DataFrame filtrado")
            st.write(dataframe[selected_columns])
        #else:
            #st.warning("Selecione pelo menos uma coluna.") 
    if dados != None:
        dataframe = dataframe[selected_columns]
        return dataframe

##################################################################################################
##################################################################################################

# Gráfico

def plota_pivot_table(df, value, index, func, ylabel, xlabel, opcao='nada'):
    if opcao == 'nada':
        pd.pivot_table(df, values=value, index=index,aggfunc=func).plot(figsize=[15, 5])
    elif opcao == 'unstack':
        pd.pivot_table(df, values=value, index=index,aggfunc=func).unstack().plot(figsize=[15, 5])
    elif opcao == 'sort':
        pd.pivot_table(df, values=value, index=index,aggfunc=func).sort_values(value).plot(figsize=[15, 5])
    plt.ylabel(ylabel)
    plt.xlabel(xlabel)
    st.pyplot(plt)
    return None

##################################################################################################
##################################################################################################

def main():
    with st.expander("Quantidade de nascidos"):
        plota_pivot_table(sinasc, 'ORIGEM', 'DTNASC', 'count', 'Quantidade de nascidos', 'data nascimento')
    with st.expander("Média idade mãe por sexo"):
        plota_pivot_table(sinasc, 'IDADEMAE', ['DTNASC', 'SEXO'], 'mean', 'media idade mae','data de nascimento','unstack')
    with st.expander("Peso mediano por escolaridade mãe"):
        plota_pivot_table(sinasc, 'PESO', 'ESCMAE', 'median', 'PESO mediano','escolaridade mae','sort')
    with st.expander("Média do apgar1 por gestação"):
        plota_pivot_table(sinasc, 'APGAR1', 'GESTACAO', 'mean', 'apgar1 medio','gestacao','sort')

if __name__ == "__main__":
    main()

##################################################################################################
##################################################################################################


# Lista de métricas
valores = ["---", "Describe das colunas numéricas", "Describe das colunas categóricas"]
valor_selecionado = st.selectbox("Selecione um tipo de métrica:", valores)

if valor_selecionado == 'Describe das colunas numéricas':
    st.write('---')
    st.write('## Vamos analizar as colunas numéricas')
    sinasc_describe = sinasc.select_dtypes('number')
    sinasc_describe = select_columns(sinasc_describe, None, 1)
    try:
        st.write(sinasc_describe.describe())
    except ValueError:
        st.write(sinasc.select_dtypes('number').describe())

elif valor_selecionado == 'Describe das colunas categóricas':
    st.write('---')
    st.write('## Vamos analizar as colunas categóricas')
    sinasc_describe = sinasc.select_dtypes('object')
    sinasc_describe = select_columns(sinasc_describe, None, 1)
    try:
        st.write(sinasc_describe.describe())
    except ValueError:
        st.write(sinasc.select_dtypes('object').describe())


# Enzo Schitini 😉