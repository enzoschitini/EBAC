import os
import pandas as pd
import seaborn as sns
import matplotlib.pyplot as plt
import sys

# "analise.py" JAN FEV
lista = [sys.argv[1], sys.argv[2], sys.argv[3], sys.argv[4], sys.argv[5], sys.argv[6],
         sys.argv[7], sys.argv[8], sys.argv[9], sys.argv[10], sys.argv[11], sys.argv[12]]

#current_directory = os.getcwd() # Pasta atual para Jupyter
current_directory = os.path.dirname(__file__) # Pasta atual para .py
#save = r"C:\Users\Soldado\Desktop\Data Science\EBAC\Moduli\Modulo_14\Support_Exercise_M14"

sns.set_theme()  

def plota_pivot_table(df, value, index, func, ylabel, xlabel, opcao='nada'):
    if opcao == 'nada':
        pd.pivot_table(df, values=value, index=index,
                       aggfunc=func).plot(figsize=[15, 5])
    elif opcao == 'sort':
        pd.pivot_table(df, values=value, index=index,
                       aggfunc=func).sort_values(value).plot(figsize=[15, 5])
    elif opcao == 'unstack':
        pd.pivot_table(df, values=value, index=index,
                       aggfunc=func).unstack().plot(figsize=[15, 5])
    plt.ylabel(ylabel)
    plt.xlabel(xlabel)
    return None

def list_plot(data, max):
    loc = r'{current_directory}\output\{max}'.format(current_directory=current_directory, max=max)
    plota_pivot_table(data, 'IDADEMAE', 'DTNASC', 'count', 'quantidade de nascimento','data de nascimento') #'./output/'+max+'/quantidade de nascimento.png'
    plt.savefig(loc+'\quantidade de nascimento.png')

    plota_pivot_table(data, 'IDADEMAE', ['DTNASC', 'SEXO'], 'mean', 'media idade mae','data de nascimento','unstack')
    plt.savefig(loc+'\media idade mae por sexo.png') #'./output/'+max+'/media idade mae por sexo.png'

    plota_pivot_table(data, 'PESO', ['DTNASC', 'SEXO'], 'mean', 'media peso bebe','data de nascimento','unstack')
    plt.savefig(loc+'\media peso bebe por sexo.png') #'./output/'+max+'/media peso bebe por sexo.png'

    plota_pivot_table(data, 'PESO', 'ESCMAE', 'median', 'apgar1 medio','gestacao','sort')
    plt.savefig(loc+'\media apgar1 por escolaridade mae.png') # './output/'+max+'/media apgar1 por escolaridade mae.png'

    plota_pivot_table(data, 'APGAR1', 'GESTACAO', 'mean', 'apgar1 medio','gestacao','sort')
    plt.savefig(loc+'\media apgar1 por gestacao.png')  # './output/'+max+'/media apgar1 por gestacao.png'

    plota_pivot_table(data, 'APGAR5', 'GESTACAO', 'mean', 'apgar5 medio','gestacao','sort')
    plt.savefig(loc+'\media apgar5 por gestacao.png') # './output/'+max+'/media apgar5 por gestacao.png'


for periodo in lista:
    #local = f"{current_directory}\input\SINASC_RO_2019_{periodo}.csv"
    filename = r'{current_directory}\input\SINASC_RO_2019_{periodo}.csv'.format(current_directory=current_directory, periodo=periodo)
    sinasc = pd.read_csv(filename)
    # print(sinasc.DTNASC.min(), sinasc.DTNASC.max())

    max_data = sinasc.DTNASC.max()[:7]
    # print(max_data)
    
    #os.makedirs('./output/'+max_data, exist_ok=True)
    loc = r'{current_directory}\output\{max_data}'.format(current_directory=current_directory, max_data=max_data)
    os.makedirs(loc, exist_ok=True)
    list_plot(sinasc, max_data)
    #print(f"{current_directory}\input\SINASC_RO_2019_{periodo}.csv")