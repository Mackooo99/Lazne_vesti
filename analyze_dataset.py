import string
import pandas as pd
import matplotlib.pyplot as plt
import re
from collections import Counter
import numpy as np

# Funkcije
def title_length(title):
    return len(str(title))

def word_count(title):
    return len(str(title).split())

def contains_number(title):
    return int(any(char.isdigit() for char in str(title)))

def count_punctuation(title):
    return sum(1 for char in str(title) if char in string.punctuation)

def count_uppercase_words(title):
    return sum(1 for word in str(title).split() if word == word.upper() and len(word) > 3)

def is_all_upper(title):
    return (str(title).isupper())

def count_simbol(title, simbol):
    return str(title).lower().count(simbol.lower())

def first_word(title):
    words = str(title).split()
    return words[0] if words else ''





fake_news = pd.read_csv("Fake.csv")
true_news = pd.read_csv("True.csv")


fake_tokens = fake_news['title'].str.lower().apply(lambda title: re.findall(r'\b[a-z]{4,}\b', str(title)))
true_tokens = true_news['title'].str.lower().apply(lambda title: re.findall(r'\b[a-z]{4,}\b', str(title)))

# Najčešće reči
fake_word_counts = Counter([word for tokens in fake_tokens for word in tokens])
true_word_counts = Counter([word for tokens in true_tokens for word in tokens])

# Najčešće prve reči
fake_first_word_counts = Counter([first_word(title) for title in fake_news['title'].dropna()])
true_first_word_counts = Counter([first_word(title) for title in true_news['title'].dropna()])

# Naslovi pisani velikim slovima
fake_upper_titles = fake_news[fake_news['title'].apply(is_all_upper)]
true_upper_titles = true_news[true_news['title'].apply(is_all_upper)]

# Reči pisane velikim slovima
fake_upper_words = []
fake_news['title'].dropna().apply(lambda title: fake_upper_words.extend(
    [word for word in title.split() if word == word.upper() and len(word) > 3]))

true_upper_words = []
true_news['title'].dropna().apply(lambda title: true_upper_words.extend(
    [word for word in title.split() if word == word.upper() and len(word) > 3]))

fake_upper_word_counts = Counter(fake_upper_words)
true_upper_word_counts = Counter(true_upper_words)

# Prosečne dužine naslova i teksta
fake_title_len = fake_news['title'].dropna().apply(len)
fake_text_len = fake_news['text'].dropna().apply(len)
true_title_len = true_news['title'].dropna().apply(len)
true_text_len = true_news['text'].dropna().apply(len)

# Interpunkcija
fake_exclam = fake_news['title'].dropna().apply(lambda x: count_simbol(x, '!'))
fake_question = fake_news['title'].dropna().apply(lambda x: count_simbol(x, '?'))
fake_dots = fake_news['title'].dropna().apply(lambda x: count_simbol(x, '...'))

true_exclam = true_news['title'].dropna().apply(lambda x: count_simbol(x, '!'))
true_question = true_news['title'].dropna().apply(lambda x: count_simbol(x, '?'))
true_dots = true_news['title'].dropna().apply(lambda x: count_simbol(x, '...'))

# Vizualizacije
fig, axs = plt.subplots(1, 2, figsize=(14, 5))
fig.suptitle("Dužina naslova i teksta")

axs[0].hist(fake_title_len, bins=30, color='red', alpha=0.5, edgecolor='black', label='Fake')
axs[0].hist(true_title_len, bins=30, color='blue', alpha=0.5, edgecolor='black', label='True')
axs[0].set_title("Dužina naslova")
axs[0].legend()
axs[0].grid(True)

axs[1].hist(fake_text_len, bins=30, color='red', alpha=0.5, edgecolor='black', label='Fake')
axs[1].hist(true_text_len, bins=30, color='blue', alpha=0.5, edgecolor='black', label='True')
axs[1].set_title("Dužina teksta")
axs[1].set_xlim(0, 30000)
axs[1].legend()
axs[1].grid(True)

plt.show()

# Reči u naslovima
fig, axs = plt.subplots(1, 2, figsize=(14, 5))
fig.suptitle("Najčešće reči u naslovima")

labels_fake, values_fake = zip(*fake_word_counts.most_common(30))
labels_true, values_true = zip(*true_word_counts.most_common(30))

axs[0].bar(labels_fake, values_fake, color='red')
axs[0].set_title("Fake")
axs[0].tick_params(axis='x', rotation=90)

axs[1].bar(labels_true, values_true, color='blue')
axs[1].set_title("True")
axs[1].tick_params(axis='x', rotation=90)

plt.tight_layout()
plt.show()

# Broj naslova pisanim velikim slovima
plt.bar(['Fake', 'True'], [len(fake_upper_titles), len(true_upper_titles)], color=['red', 'blue'])
plt.title("Naslovi pisani velikim slovima")
plt.ylabel("Broj vesti")
plt.show()

# Reči pisane velikim slovima
fig, axs = plt.subplots(1, 2, figsize=(14, 5))
fig.suptitle("Najčešće reči velikim slovima")

f_words, f_counts = zip(*fake_upper_word_counts.most_common(10))
t_words, t_counts = zip(*true_upper_word_counts.most_common(10))

axs[0].bar(f_words, f_counts, color='red')
axs[0].set_title("Fake")
axs[0].tick_params(axis='x', rotation=45)

axs[1].bar(t_words, t_counts, color='blue')
axs[1].set_title("True")
axs[1].tick_params(axis='x', rotation=45)

plt.tight_layout()
plt.show()

# Prve reči
fig, axs = plt.subplots(1, 2, figsize=(14, 5))
fig.suptitle("Najčešće prve reči")

fw_labels_fake, fw_values_fake = zip(*fake_first_word_counts.most_common(10))
fw_labels_true, fw_values_true = zip(*true_first_word_counts.most_common(10))

axs[0].bar(fw_labels_fake, fw_values_fake, color='red')
axs[0].set_title("Fake")
axs[0].tick_params(axis='x', rotation=45)

axs[1].bar(fw_labels_true, fw_values_true, color='blue')
axs[1].set_title("True")
axs[1].tick_params(axis='x', rotation=45)

plt.tight_layout()
plt.show()

# Interpunkcija
labels = ['!', '?', '...']
fake_means = [fake_exclam.mean(), fake_question.mean(), fake_dots.mean()]
true_means = [true_exclam.mean(), true_question.mean(), true_dots.mean()]

x = np.arange(len(labels))
width = 0.35

fig, ax = plt.subplots()
ax.bar(x - width/2, fake_means, width, label='Fake', color='red', alpha=0.6)
ax.bar(x + width/2, true_means, width, label='True', color='blue', alpha=0.6)
ax.set_ylabel('Prosečan broj po naslovu')
ax.set_title('Prosečna učestalost interpunkcija')
ax.set_xticks(x)
ax.set_xticklabels(labels)
ax.legend()

plt.tight_layout()
plt.show()
