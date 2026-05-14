import pandas as pd

data = pd.read_csv('Microarray_Clasificado_KEGG_9_Grupos.csv')

data = data[data['Grupo_Metabolico'].str.contains('[1-7].*', regex=True)]

data.drop(columns=['entrez_id', 'index'], inplace=True, errors='ignore')

data.sort_values(by='original_index', ascending=True, inplace=True)

data.reset_index(drop=True, inplace=True)

data.to_csv('Genes_EC_families.csv')

print(data)