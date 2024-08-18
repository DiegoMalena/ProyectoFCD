import streamlit as st
import pickle
import pandas as pd
import matplotlib.pyplot as plt
from wordcloud import WordCloud, STOPWORDS
import re

# Cargar el vectorizador, el modelo y el DataFrame del EDA desde los archivos pickle
with open('vectorizer.pkl', 'rb') as vec_file:
    vectorizer = pickle.load(vec_file)

with open('modelo_sentimientos.pkl', 'rb') as model_file:
    model = pickle.load(model_file)

with open('news_data.pkl', 'rb') as eda_file:
    news_df = pickle.load(eda_file)

# Función para limpiar el texto
def clean_text(text):
    text = re.sub(r'\s+', ' ', text)  # Eliminar espacios adicionales
    text = re.sub(r'[^a-zA-Z0-9]', ' ', text)  # Eliminar caracteres especiales
    text = text.lower()  # Convertir a minúsculas
    stop_words = set(STOPWORDS)  # Usar stopwords predeterminadas
    text = ' '.join([word for word in text.split() if word not in stop_words])  # Eliminar stopwords
    return text

# Función para clasificar el sentimiento
def classify_sentiment(polarity):
    if polarity > 0:
        return 'positive'
    elif polarity < 0:
        return 'negative'
    else:
        return 'neutral'

# Streamlit app
st.header("Analizador de Sentimientos")

st.write("Vamos a indagar sobre las noticias respecto a la sostenibilidad y prácticas medioambientales de Greenpeace, con el fin de entender como es percibida su compromiso ambiental.")

with st.form(key='nlpForm'):
    text = st.text_area("Ingrese su texto a analizar")
    submit_button = st.form_submit_button(label="Analizar")

if submit_button:
    cleaned_text = clean_text(text)
    vectorized_text = vectorizer.transform([cleaned_text])
    sentimiento = model.predict(vectorized_text)
    sentimiento_analizado = classify_sentiment(sentimiento[0])
    
    st.info("Resultados")
    st.write(f"Sentimiento predicho: {sentimiento_analizado}")

# Visualización de gráficos del EDA
st.subheader("Visualización de Gráficas del EDA")


# Gráfico de publicaciones por mes
st.header("Publicaciones por Mes")
publications_per_month = news_df['year_month'].value_counts().sort_index()

plt.figure(figsize=(15, 7))
publications_per_month.plot(kind='bar')
plt.title('Publicaciones por Mes sobre Greenpeace y Sostenibilidad')
plt.xlabel('Mes')
plt.ylabel('Cantidad de Publicaciones')
plt.xticks(rotation=90)
st.pyplot(plt)

# Gráfico de publicaciones por fuente
st.header("Distribución de Publicaciones por Fuente")
publications_per_source = news_df['source'].value_counts()

# Visualizar la distribución de fuentes
plt.figure(figsize=(10, 5))
publications_per_source.plot(kind='bar')
plt.title('Distribución de Publicaciones por Fuente')
plt.xlabel('Fuente')
plt.ylabel('Cantidad de Publicaciones')
plt.xticks(rotation=90)
st.pyplot(plt)

st.header("Nube de Palabras de los Titulos")
# Definir stopwords adicionales en español
stopwords_en = set(STOPWORDS)

# Unir todos los títulos en un solo string
all_titles = ' '.join(news_df['title'])

# Crear y mostrar la nube de palabras
wordcloud = WordCloud(width=800, height=400, background_color='white', stopwords=stopwords_en).generate(all_titles)
plt.figure(figsize=(10, 5))
plt.imshow(wordcloud, interpolation='bilinear')
plt.axis('off')
plt.title('Nube de Palabras de los Títulos')
st.pyplot(plt)
