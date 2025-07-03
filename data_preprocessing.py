import pandas as pd
import re
import analyze_dataset

# Učitavanje i obeležavanje podataka
def load_data(fake_path='Fake.csv', true_path='True.csv'):
    fake = pd.read_csv(fake_path)
    true = pd.read_csv(true_path)
    fake['label'] = 0
    true['label'] = 1
    return pd.concat([fake, true], ignore_index=True)

# Čišćenje teksta
def clean_text(text):
    text = str(text).lower()
    text = re.sub(r'<.*?>', '', text)
    text = re.sub(r'[^a-z\s]', '', text)
    return text

# Glavna funkcija za obradu
def preprocess(df):
    df = df.drop_duplicates(subset=['title', 'text'])
    df = df.dropna(subset=['title', 'text'])

    df['title_clean'] = df['title'].apply(clean_text)
    df['text_clean'] = df['text'].apply(clean_text)

    # Koristi funkcije iz analyze_dataset.py
    df['title_len'] = df['title'].apply(analyze_dataset.title_length)
    df['title_word_count'] = df['title'].apply(analyze_dataset.word_count)
    df['title_upper_words'] = df['title'].apply(analyze_dataset.count_uppercase_words)
    df['title_exclam'] = df['title'].apply(lambda x: analyze_dataset.count_simbol(x, '!'))
    df['title_question'] = df['title'].apply(lambda x: analyze_dataset.count_simbol(x, '?'))
    df['title_all_upper'] = df['title'].apply(analyze_dataset.is_all_upper)
    # df['text_len'] = df['text'].apply(analyze_dataset.title_length)
    # df['text_word_count'] = df['text'].apply(analyze_dataset.word_count)
    # df['text_upper_words'] = df['text'].apply(analyze_dataset.count_uppercase_words)
    # df['text_exclam'] = df['text'].apply(lambda x: analyze_dataset.count_simbol(x, '!'))
    # df['text_question'] = df['text'].apply(lambda x: analyze_dataset.count_simbol(x, '?'))
    # df['text_all_upper'] = df['text'].apply(analyze_dataset.is_all_upper)

    df['combined_text'] = df['title_clean'] + ' ' + df['text_clean']
    return df

# Glavno izvođenje
df = load_data()
df = preprocess(df)
df.to_csv('data/cleaned_dataset.csv', index=False)

print("Preprocessing završen. Dataset sačuvan u 'data/cleaned_dataset.csv'")
print("Ukupan broj podataka: ", df.shape)
