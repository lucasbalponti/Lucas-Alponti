import pandas as pd
import requests
from datetime import date
from datetime import datetime
import urllib.parse

# Importando a base de dados das 25 empresas com maior valuation via API
base_25top = pd.DataFrame(requests.get('https://brapi.dev/api/quote/list?sortBy=market_cap_basic&sortOrder=desc&limit=25').json()['stocks'])

# Mudando os pontos por virgulas, para que os dados fiquemno formato adequado
base_25top.loc[:, 'close'] = [str(base_25top['close'][i]).replace('.',',') for i in range(len(base_25top))]
base_25top.loc[:, 'change'] = [str(base_25top['change'][i]).replace('.',',') for i in range(len(base_25top))]
base_25top.loc[:, 'market_cap'] = [str(base_25top['market_cap'][i]).replace('.',',') for i in range(len(base_25top))]

# Criando uma string com o nome de todas as ações importadas para importar o desempenho histórico das mesmas
str_stocks = base_25top['stock'].values[0]

for i in range(len(base_25top) - 1):
    str_stocks = str_stocks + ',' + base_25top['stock'].values[i+1]

# Importando a base de dados do desempenho histórico
json_resultados = requests.get('https://brapi.dev/api/quote/'+ urllib.parse.quote_plus(str_stocks) +'?range=5y&interval=1d&fundamental=true').json()

# Montando a base de dados com o desempenho histórico das ações
base_historico = pd.DataFrame()

for i in range(len(json_resultados['results'])):
    
    base_intermediaria = pd.DataFrame(json_resultados['results'][i]['historicalDataPrice'])
    base_intermediaria.loc[:, 'Nome'] = json_resultados['results'][i]['symbol']
    
    base_historico = pd.concat([base_historico, base_intermediaria], ignore_index = True)]

base_historico.loc[:, 'date'] = [datetime.fromtimestamp(base_historico['date'][i]).date() for i in range(len(base_historico))]
base_historico.loc[:, 'open'] = [str(base_historico['open'][i]).replace('.',',') for i in range(len(base_historico))]
base_historico.loc[:, 'high'] = [str(base_historico['high'][i]).replace('.',',') for i in range(len(base_historico))]
base_historico.loc[:, 'low'] = [str(base_historico['low'][i]).replace('.',',') for i in range(len(base_historico))]
base_historico.loc[:, 'close'] = [str(base_historico['close'][i]).replace('.',',') for i in range(len(base_historico))]
