import numpy as np
import pandas as pd

# read einstein csv
print('reading csv...')
EINSTEIN_EXAMS_FILE = 'einstein.in.csv'
e = pd.read_csv(f'{EINSTEIN_EXAMS_FILE}', sep='|')

print('preprocessing dataframe...')
# remove duplicates
e.drop_duplicates()

# lower case columns: de_analito, de_resultado, de_valor_referencia
e.de_analito = e.de_analito.str.lower()
e.de_resultado = e.de_resultado.str.lower()
e.de_valor_referencia = e.de_valor_referencia.str.lower()

# replace values
e = e.replace({
    'resultado covid-19:': 'pcr-result',

    'covid19 igm, teste rápido': 'igm-result',
    'covid igm interp': 'igm-result',
    'igm, covid19': 'igm-index',

    'covid19 igg, teste rápido': 'igg-result',
    'covid igg interp': 'igg-result',
    'igg, covid19': 'igg-index',

    'não reagente': 0,
    'reagente': 1,

    'não detectado': 0,
    'detectado': 1,

    'ausente': 0,
    'presente': 1,

    'ausentes': 0,
    'presentes': 1,

    '+': 1,
    '++': 2,
    '+++': 3,

    '1+ (~50 mg/dL)':  1,
    '2+ (~150 mg/dL)': 2,
    '3+ (~300 mg/dL)': 3,
    '4+ (>=500mg/dL)': 4
})

# remove useless analitos
e = e[
    (e.de_analito != 'leucócitos')
    & (e.de_analito != 'linfócitos')
    & (e.de_analito != 'monócitos')
    & (e.de_analito != 'eosinófilos')
    & (e.de_analito != 'basófilos')
    & (e.de_analito != 'neutrófilos')
    & (e.de_analito != 'segmentados')
    & (e.de_analito != 'bastonestes')
    & (e.de_analito != 'metamielócitos')
    & (e.de_analito != 'promielócitos')
    & (e.de_analito != 'mielócitos')
    & (e.de_analito != 'mieloblastos')
    & (e.de_analito != 'igg-index')
    & (e.de_analito != 'igm-index')
]

# get 100 most common analitos + pcr, igm, igg
analito_dict = e.de_analito.value_counts()
analito_keys = analito_dict.keys()[:103]
analito_to_index = {k: i for i, k in enumerate(analito_keys)}

# group by id and date
gb = e.groupby(['id_paciente', 'dt_coleta'])
groups = gb.groups

# create new database
n = len(analito_keys)
m = len(groups)
new_data = np.empty((m, n))
new_data[:] = np.NaN


def analito_indexes():
    if analito in analito_keys:
        yield analito_to_index[analito]


def digest_result(result, ref):
    try:
        # convert numerical data to float
        return float(result)
    except:
        # convert categorical data using reference values
        return 0 if result == ref else 1


def get_exams(groups):
    for key in list(groups.keys()):
        yield gb.get_group(key)


def exams_data(exams):
    for _, exam in exams.iterrows():
        analito = exam['de_analito']
        result = exam['de_resultado']
        ref = exam['de_valor_referencia']
        yield analito, result, ref


print('generating new database...')
for i, exams in enumerate(get_exams(groups)):
    if i % 1000 == 0:
        print(f'processing { i/m*100:3.0f}%')

    for analito, result, ref in exams_data(exams):
        for j in analito_indexes():
            new_data[i, j] = digest_result(result, ref)

# database to dataframe
df = pd.DataFrame(new_data)

pcr = analito_to_index['pcr-result']
igm = analito_to_index['igm-result']
igg = analito_to_index['igg-result']

# rows with at least one target exam not null
df = df[df[pcr].notnull() | df[igm].notnull() | df[igg].notnull()]

# rows with at least four exams
df = df.dropna(thresh=4)

# moving pcr, igm and igg columns to the beginning
pcr_col = df.pop(pcr)
igm_col = df.pop(igm)
igg_col = df.pop(igg)

df.insert(0, 'pcr', pcr_col)
df.insert(1, 'igm', igm_col)
df.insert(2, 'igg', igg_col)

# saving new database
df.to_csv('einstein.out.csv', index=False)
print('database saved at "einstein.out.csv"')
