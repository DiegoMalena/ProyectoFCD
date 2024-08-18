import re
from textblob import TextBlob
import seaborn as sns
from sklearn.feature_extraction.text import CountVectorizer
from sklearn.model_selection import train_test_split
from sklearn.linear_model import LinearRegression
from sklearn.metrics import mean_squared_error
from sklearn.metrics import accuracy_score
from sklearn.metrics import confusion_matrix
import pickle

# Cargar el DataFrame desde el archivo Pickle
with open('news_data.pkl', 'rb') as f:
    news_df = pickle.load(f)

# Lista de stopwords en inglés
stop_words = [
    'i', 'me', 'my', 'myself', 'we', 'our', 'ours', 'ourselves', 'you', 'your', 'yours',
    'yourself', 'yourselves', 'he', 'him', 'his', 'himself', 'she', 'her', 'hers', 'herself',
    'it', 'its', 'itself', 'they', 'them', 'their', 'theirs', 'themselves', 'what', 'which',
    'who', 'whom', 'this', 'that', 'these', 'those', 'am', 'is', 'are', 'was', 'were', 'be',
    'been', 'being', 'have', 'has', 'had', 'having', 'do', 'does', 'did', 'doing', 'a', 'an',
    'the', 'and', 'but', 'if', 'or', 'because', 'as', 'until', 'while', 'of', 'at', 'by',
    'for', 'with', 'about', 'against', 'between', 'into', 'through', 'during', 'before',
    'after', 'above', 'below', 'to', 'from', 'up', 'down', 'in', 'out', 'on', 'off', 'over',
    'under', 'again', 'further', 'then', 'once', 'here', 'there', 'when', 'where', 'why',
    'how', 'all', 'any', 'both', 'each', 'few', 'more', 'most', 'other', 'some', 'such',
    'no', 'nor', 'not', 'only', 'own', 'same', 'so', 'than', 'too', 'very', 's', 't', 'can',
    'will', 'just', 'don', 'should', 'now'
]

# Función para limpiar texto
def clean_text(text):
    text = re.sub(r'\s+', ' ', text)  # Eliminar espacios adicionales
    text = re.sub(r'[^a-zA-Z0-9]', ' ', text)  # Eliminar caracteres especiales
    text = text.lower()  # Convertir a minúsculas
    text = ' '.join([word for word in text.split() if word not in stop_words])  # Eliminar stopwords
    return text

# Aplicar limpieza al contenido de las noticias
news_df['cleaned_content'] = news_df['content'].apply(lambda x: clean_text(x) if isinstance(x, str) else '')

# Mostrar las primeras filas del DataFrame limpio
print(news_df[['title', 'cleaned_content']].head())

# Función para obtener el sentimiento
def get_sentiment(text):
    if text:
        analysis = TextBlob(text)
        return analysis.sentiment.polarity
    return 0

# Aplicar análisis de sentimientos
news_df['sentiment'] = news_df['cleaned_content'].apply(get_sentiment)

# Clasificar el sentimiento
def classify_sentiment(polarity):
    if polarity > 0:
        return 'positive'
    elif polarity < 0:
        return 'negative'
    else:
        return 'neutral'

news_df['sentiment_label'] = news_df['sentiment'].apply(classify_sentiment)

# Mostrar las primeras filas con la clasificación de sentimiento
print(news_df[['title', 'cleaned_content', 'sentiment', 'sentiment_label']].head())

# Vectorizar el texto
vectorizer = CountVectorizer(max_df=0.95, min_df=2, stop_words='english')

X = vectorizer.fit_transform(news_df['cleaned_content'])

# Definir variable objetivo
y = news_df['sentiment']

# Dividir los datos en conjunto de entrenamiento y prueba
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.5, random_state=42)
# Entrenar el modelo de regresión lineal
model = LinearRegression()
model.fit(X_train, y_train)

# Predecir en el conjunto de prueba
y_pred = model.predict(X_test)

# Evaluar el modelo
mse = mean_squared_error(y_test, y_pred)
print(f'Mean Squared Error: {mse}')

# Guardar el modelo en un archivo .pkl
with open('modelo_sentimientos.pkl', 'wb') as file:
    pickle.dump(model, file)

# Guardar el vectorizador en un archivo .pkl
with open('vectorizer.pkl', 'wb') as file:
    pickle.dump(vectorizer, file)
