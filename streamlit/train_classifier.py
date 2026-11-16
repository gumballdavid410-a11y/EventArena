"""Train and save EventArena's synthetic-data intent classifier."""
import pandas as pd
import joblib
from sklearn.model_selection import train_test_split
from sklearn.pipeline import Pipeline
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import classification_report, accuracy_score

df = pd.read_csv("eventarena_synthetic_intents.csv")
X_train, X_test, y_train, y_test = train_test_split(
    df["text"], df["intent"], test_size=0.22, random_state=42, stratify=df["intent"]
)
model = Pipeline([
    ("tfidf", TfidfVectorizer(ngram_range=(1,2), lowercase=True, strip_accents="unicode")),
    ("classifier", LogisticRegression(max_iter=2000, class_weight="balanced", random_state=42))
])
model.fit(X_train, y_train)
pred = model.predict(X_test)
print("Accuracy:", accuracy_score(y_test, pred))
print(classification_report(y_test, pred, zero_division=0))
joblib.dump(model, "intent_classifier.joblib")
