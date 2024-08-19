import pandas as pd
import matplotlib.pyplot as plt
from wordcloud import WordCloud, STOPWORDS
from dotenv import load_dotenv
import os
import requests
import json
import pickle

#Búsqueda todo, sobre la Greanpeace, se restringe a contenido de ecologia y sosteibilidad,
#con fecha desde el 2010 y se ordena por fecha de publición %%

# Cargar la API key desde el archivo .env
load_dotenv()
api_key = os.getenv('API_KEY')

# Definir los parámetros de búsqueda para la API de noticias
params = {
    'q': 'Greenpeace ecology sustainability',
    'from': '2010',
    'language': 'en',
    'sortBy': 'publishedAt',
    'apiKey': api_key
}

# Hacer la solicitud GET a la API de noticias
url = "https://newsapi.org/v2/everything"
response = requests.get(url, params=params)

# Verificar si la solicitud fue exitosa
if response.status_code == 200:
    # Convertir la respuesta en formato JSON
    data = response.json()
    
    # Procesar los datos según sea necesario
    print(data)
else:
    # Manejar el error si la solicitud no fue exitosa
    print(f"Error: {response.status_code} - {response.text}")

# Esta busqueda arroja 1953 resultados

news_content = response.content
print(news_content)
news = json.loads(news_content)
news.keys()

# Creando el DataFrame
news_df = pd.DataFrame(news['articles'])
news_df.head(6)

#Extrayendo datos del DF
news_df.tail(3).iloc[0]

news_df.info()

news_df.describe()


# Tablas de Frecuencia para source y author:
news_df['source'] = news_df['source'].apply(lambda x: x['name'] if isinstance(x, dict) else x)
source_counts = news_df['source'].value_counts()
author_counts = news_df['author'].value_counts()

print("Source Counts:")
print(source_counts)
print("\nAuthor Counts:")
print(author_counts)



# Convertir 'publishedAt' a formato de fecha
news_df['publishedAt'] = pd.to_datetime(news_df['publishedAt'])

# Extraer el año y el mes de la columna 'publishedAt'
news_df['year_month'] = news_df['publishedAt'].dt.to_period('M')

# Contar la cantidad de publicaciones por mes
publications_per_month = news_df['year_month'].value_counts().sort_index()



# Crear un gráfico de barras
plt.figure(figsize=(15, 7))
publications_per_month.plot(kind='bar')
plt.title('Publicaciones por Mes sobre Greenpeace y Sostenibilidad')
plt.xlabel('Mes')
plt.ylabel('Cantidad de Publicaciones')
plt.xticks(rotation=90)
plt.show()

# Calcular la longitud de los títulos, descripciones y contenidos
news_df['title_length'] = news_df['title'].apply(len)
news_df['description_length'] = news_df['description'].apply(len)
news_df['content_length'] = news_df['content'].apply(len)

# Visualizar la distribución de las longitudes de los títulos
plt.figure(figsize=(10, 5))
news_df['title_length'].hist(bins=20)
plt.title('Distribución de la Longitud de los Títulos')
plt.xlabel('Longitud (número de caracteres)')
plt.ylabel('Frecuencia')
plt.show()

# Visualizar la distribución de las longitudes de las descripciones
plt.figure(figsize=(10, 5))
news_df['description_length'].hist(bins=20)
plt.title('Distribución de la Longitud de las Descripciones')
plt.xlabel('Longitud (número de caracteres)')
plt.ylabel('Frecuencia')
plt.show()

# Visualizar la distribución de las longitudes de los contenidos
plt.figure(figsize=(10, 5))
news_df['content_length'].hist(bins=20)
plt.title('Distribución de la Longitud de los Contenidos')
plt.xlabel('Longitud (número de caracteres)')
plt.ylabel('Frecuencia')
plt.show()


# Contar la cantidad de publicaciones por fuente
publications_per_source = news_df['source'].value_counts()

# Visualizar la distribución de fuentes
plt.figure(figsize=(10, 5))
publications_per_source.plot(kind='bar')
plt.title('Distribución de Publicaciones por Fuente')
plt.xlabel('Fuente')
plt.ylabel('Cantidad de Publicaciones')
plt.xticks(rotation=90)
plt.show()

# Definir stopwords adicionales en español
stopwords_es = set(STOPWORDS)
stopwords_adicionales = {'en','los','y', 'una', 'por', 'un','en' , 'la', 'el', 'del', 'lo', 'de', 'que'}

# Combinar stopwords predeterminadas con las adicionales
stopwords_es.update(stopwords_adicionales)

# Unir todos los títulos en un solo string
all_titles = ' '.join(news_df['title'])

# Crear y mostrar la nube de palabras
wordcloud = WordCloud(width=800, height=400, background_color='white', stopwords=stopwords_es).generate(all_titles)
plt.figure(figsize=(10, 5))
plt.imshow(wordcloud, interpolation='bilinear')
plt.axis('off')
plt.title('Nube de Palabras de los Títulos')
plt.show()

