import os
import pandas as pd
import re
from sklearn.linear_model import LogisticRegression
from sklearn.model_selection import train_test_split
from sklearn.metrics import classification_report, confusion_matrix, accuracy_score
import matplotlib.pyplot as plt
import seaborn as sns

# Čišćenje teksta
def clean_text(text):
    text = str(text).lower()
    text = re.sub(r'<.*?>', '', text)
    text = re.sub(r'[^a-z\s]', '', text)
    return text

# Dodavanje osobina iz naslova
def add_features(df):
    df['title_clean'] = df['title'].apply(clean_text)
    df['title_len'] = df['title'].apply(lambda x: len(str(x)))
    df['title_word_count'] = df['title'].apply(lambda x: len(str(x).split()))
    df['title_upper_words'] = df['title'].apply(lambda x: sum(1 for word in str(x).split() if word.isupper() and len(word) > 3))
    df['title_exclam'] = df['title'].apply(lambda x: str(x).count('!'))
    df['title_question'] = df['title'].apply(lambda x: str(x).count('?'))
    df['title_all_upper'] = df['title'].apply(lambda x: int(str(x).isupper()))
    return df

# === 1. Učitavanje i priprema trening podataka ===
if not os.path.exists("data/cleaned_dataset.csv"):
    print("Fajl 'cleaned_dataset.csv' ne postoji.")
    os.system("python3 data_preprocessing.py")

train_df = pd.read_csv("data/cleaned_dataset.csv")
train_df = add_features(train_df)

X = train_df[['title_len', 'title_word_count', 'title_upper_words',
              'title_exclam', 'title_question', 'title_all_upper']]
y = train_df['label']


X_train, X_val, y_train, y_val = train_test_split(X, y, test_size=0.15, stratify=y, random_state=42)


model = LogisticRegression(max_iter=400)
model.fit(X_train, y_train)

# Evaluacija na validacionom skupu
y_val_pred = model.predict(X_val)

print("\n Rezultati na VALIDACIONOM skupu:")
print("Tačnost:", accuracy_score(y_val, y_val_pred))
print("Matrica konfuzije:\n", confusion_matrix(y_val, y_val_pred))
print("Izveštaj klasifikacije:\n", classification_report(y_val, y_val_pred))



# Učitavanje i obrada test skupa
test_df = pd.read_csv("News_2025.csv")
test_df = test_df.dropna(subset=['title'])
test_df = add_features(test_df)

X_test = test_df[['title_len', 'title_word_count', 'title_upper_words',
                  'title_exclam', 'title_question', 'title_all_upper']]

# Predikcija na test skupu
test_df['prediction'] = model.predict(X_test)
test_df['predicted_label'] = test_df['prediction'].map({0: 'FAKE', 1: 'TRUE'})

# Grafički prikaz predikcija na test skupu
pred_counts = test_df['predicted_label'].value_counts()

plt.figure(figsize=(6, 4))
sns.barplot(x=pred_counts.index, y=pred_counts.values, hue=pred_counts.index, dodge=False, palette=['red', 'blue'], legend=False)
plt.title("Broj predikcija po klasi (test skup)")
plt.ylabel("Broj vesti")
plt.xlabel("Predikcija")
plt.tight_layout()
plt.show()

# Čuvanje rezultata test predikcije
test_df.to_csv("data/test_results.csv", index=False)
print("Predikcije su sačuvane u 'test_results.csv'")


from sklearn.metrics import roc_curve, auc

# ROC AUC kriva
y_val_probs = model.predict_proba(X_val)[:, 1]  # verovatnoće za klasu 1 (TRUE)
fpr, tpr, thresholds = roc_curve(y_val, y_val_probs)
roc_auc = auc(fpr, tpr)

plt.figure(figsize=(6, 5))
plt.plot(fpr, tpr, color='darkorange', lw=2, label=f'ROC kriva (AUC = {roc_auc:.2f})')
plt.plot([0, 1], [0, 1], color='navy', lw=2, linestyle='--')
plt.xlim([0.0, 1.0])
plt.ylim([0.0, 1.05])
plt.xlabel('False Positive Rate')
plt.ylabel('True Positive Rate')
plt.title('ROC AUC kriva (validacija)')
plt.legend(loc="lower right")
plt.grid(True)
plt.tight_layout()
plt.show()

