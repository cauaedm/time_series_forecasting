import requests
from bs4 import BeautifulSoup
import pandas as pd

def scrap(HEADERS: str, URL: str) -> pd.DataFrame:
    
    webpages = requests.get(URL, headers=HEADERS)

    if webpages.status_code != 200:
        return
    
    soup = BeautifulSoup(webpages.content, 'html.parser')
    try:
        tabela = soup.find_all('table', attrs={'class':'table table-main orange'})
    except Exception as e:
        print(e)
        return
    
    if len(tabela) < 0:
        return
    
    tempo = soup.find_all('th', attrs={'class':'text-center'})
    if len(tempo) < 0:
        return
    
    tempo = tempo[3:]

    preços = soup.find_all('td', attrs={'class':'text-center'})
    if len(preços) < 0:
        return

    preços_estaduais = [preços[i] for i in range(len(preços)) if i % 2 == 0]
    preços_nacionais = [preços[i] for i in range(len(preços)) if i % 2 == 1]

    tempo = [i.text for i in tempo]
    preços_nacionais = [i.text for i in preços_nacionais]
    preços_estaduais = [i.text for i in preços_estaduais]

    dados = pd.DataFrame()
    dados.rename(columns=['time', 'precos_nacionais', 'precos_estaduais'])

    dados['time'] = tempo
    dados['precos_nacionais'] = preços_nacionais
    dados['precos_estaduais'] = preços_estaduais

    dados['precos_nacionais'] = dados['precos_nacionais'].str.replace(',', '.')
    dados['precos_nacionais'] = pd.to_numeric(dados['precos_nacionais'], errors='coerce')

    dados['precos_estaduais'] = dados['precos_estaduais'].str.replace(',', '.')
    dados['precos_estaduais'] = pd.to_numeric(dados['precos_estaduais'], errors='coerce')

    return dados