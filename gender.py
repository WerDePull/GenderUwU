import numpy as np
import pandas as pd
import re
import string
from nltk.corpus import stopwords
from nltk.tokenize import TweetTokenizer
from sklearn.model_selection import train_test_split
from sklearn.linear_model import LogisticRegression
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.metrics import classification_report
from sklearn.utils import resample

# Завантаження даних
file_path = r"D:\\blog\\blog\\blog-gender-dataset.xlsx"
df = pd.read_excel(file_path, header=None)  # Читаємо без заголовків

# Видалення рядків із порожніми текстами
df = df[df[0].notna()]  # Видаляємо NaN у текстах
df = df[df[0].str.strip() != '']  # Видаляємо рядки, які складаються лише з пробілів

# Перевірка балансу даних
print("Кількість текстів за гендером:")
print(df[1].value_counts())

# Балансування вибірки (Oversampling)
df_male = df[df[1] == 'M']
df_female = df[df[1] == 'F']

df_male_oversampled = resample(df_male, replace=True, n_samples=len(df_female), random_state=42)
df_balanced = pd.concat([df_female, df_male_oversampled])

# Токенізація, видалення стоп-слів та стемінг
stopwords_english = stopwords.words('english')
tokenizer = TweetTokenizer(preserve_case=False, strip_handles=True, reduce_len=True)

def process_tweet(tweet):
    if not isinstance(tweet, str):
        tweet = str(tweet)
    tweet = re.sub(r'\$\w*', '', tweet)  # remove stock market tickers like $GE
    tweet = re.sub(r'^RT[\s]+', '', tweet)  # remove old style retweet text "RT"
    tweet = re.sub(r'https?://[^\s\n\r]+', '', tweet)  # remove hyperlinks
    tweet = re.sub(r'#', '', tweet)  # remove hashtags
    tweet_tokens = tokenizer.tokenize(tweet)
    return [word for word in tweet_tokens if word not in stopwords_english and word not in string.punctuation]

# Підготовка даних
df_balanced['processed_text'] = df_balanced[0].apply(process_tweet)  # Текст блогу в стовпці 0
X = df_balanced['processed_text'].apply(lambda x: ' '.join(x))  # Перетворюємо список слів у рядок
y = df_balanced[1].apply(lambda x: 1 if x == 'M' else 0)  # Гендер в стовпці 1 (M - 1, F - 0)

# Розділення на тренувальну та тестову вибірку
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

# Векторизація текстів
vectorizer = TfidfVectorizer(max_features=5000, ngram_range=(1, 2))  # Використовуємо TF-IDF з біграмами
X_train_vec = vectorizer.fit_transform(X_train)
X_test_vec = vectorizer.transform(X_test)

# Навчання моделі
model = LogisticRegression(C=0.5, max_iter=200)
model.fit(X_train_vec, y_train)

# Оцінка моделі
accuracy = model.score(X_test_vec, y_test)
print(f'Accuracy: {accuracy:.4f}')

# Звіт про класифікацію
y_pred = model.predict(X_test_vec)
print(classification_report(y_test, y_pred, target_names=['Female', 'Male']))

# Перевірка важливих ознак
feature_names = vectorizer.get_feature_names_out()
coef = model.coef_[0]
top_features = sorted(zip(coef, feature_names), key=lambda x: abs(x[0]), reverse=True)[:10]
print("Top features influencing the model:")
for weight, feature in top_features:
    print(f"{feature}: {weight}")

# Функція для передбачення гендеру
def predict_gender(blog_text):
    processed_blog = process_tweet(blog_text)
    blog_vec = vectorizer.transform([' '.join(processed_blog)])
    prediction = model.predict(blog_vec)
    return 'Male' if prediction[0] == 1 else 'Female'

# Перевірка введеного тексту
while True:
    blog_input = input("Enter a blog text (or type 'exit' to quit): ")
    if blog_input.lower() == 'exit':
        break
    gender_prediction = predict_gender(blog_input)
    print(f"The predicted gender of the author is: {gender_prediction}")
