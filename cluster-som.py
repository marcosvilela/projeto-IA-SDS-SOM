import numpy as np
import pandas as pd
import somoclu

"""
Biblioteca utilizada: somoclu
Repositório original: https://github.com/peterwittek/somoclu
"""

dataset = pd.read_csv('./teste-colunas.csv')


"""Seleciona os ID's, que serão os rótulos dos neurônios no mapa"""
labels = dataset.iloc[:, 0]


"""Remove os ID's para não influenciarem no agrupamento"""
data = np.float32(dataset.iloc[:, 1:].values)
"""Se o np.float32() não for usado, será emitido o seguinte alerta durante
    a execução: Warning: data was not float32. A 32-bit copy was made
   e os dados serão transformados automaticamente para o tipo float32
   """                     


"""Os valores das linhas e colunas podem ser alterados"""
n_rows, n_columns = 100, 100


som = somoclu.Somoclu(n_columns, n_rows, initialization="pca")
   


som.train(data, epochs=10)

"""Treina o mapa usando os dados atuais no objeto Somoclu."""


som.cluster()

""" Classifica os neurônios, preenchendo a variável som.clusters, também seleciona
    as BMUs(neuônios que são exibidos no mapa) para cada entrada.
    
"""  
        
              
som.view_umatrix(bestmatches=True, labels=labels, filename='./mapa.png')
"""Plota a U-Matrix do mapa treinado. """


np.savetxt("./bmus.txt", som.bmus)
np.savetxt("./clusters.csv", som.clusters, delimiter=",")


clusters = pd.read_csv('./clusters.csv')


id_classes = np.empty((len(data),2), dtype=int)
id_class = pd.DataFrame()


""" som.bmus(arquivo bmus.txt) possui as coordenadas das BMUs, que são as 
    células que conseguimos ver no mapa.
    
    som.clusters(arquivo cluster.csv) é o resultado do método som.cluster() e 
    possui a classificação de cada neurônio do mapa(ao total: (n_row * n_columns) 
    neurônios).
    
    Como a localização dos neurônios após o treinamento é fixa no mapa, através 
    das coordenadas das BMUs em som.bmus, é possível extrair as suas respectivas 
    classificações em som.clusters apenas em função de suas coordenadas.
    """
i=-1
for linha, coluna in som.bmus:
    i=i+1
    id_classes[i][0] = labels[i]                      #id
    id_classes[i][1] = som.clusters[linha][coluna]    #classe
   
output = pd.DataFrame(id_classes, columns=['ID', 'Classe'])
output.to_csv('./classes.csv', sep=',', index=False)








        

     