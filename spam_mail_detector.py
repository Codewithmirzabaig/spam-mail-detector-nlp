import re
import sys
import urllib.request
import zipfile
from pathlib import Path

import pandas as pd
import matplotlib.pyplot as plt

from sklearn.model_selection import train_test_split
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.naive_bayes import MultinomialNB
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import accuracy_score, confusion_matrix, classification_report

import joblib


# -------------------------
# CONFIG
# -------------------------
DATA_URL = "https://archive.ics.uci.edu/ml/machine-learning-databases/00228/smsspamcollection.zip"
DATA_DIR = Path("data")
ZIP_PATH = DATA_DIR / "smsspamcollection.zip"
EXTRACT_DIR = DATA_DIR / "smsspamcollection"
DATA_FILE = EXTRACT_DIR / "SMSSpamCollection"

RANDOM_STATE = 42
TEST_SIZE = 0.2


# -------------------------
# HELPERS
# -------------------------
def download_and_extract_dataset() -> Path:
    """Downloads and extracts the SMS Spam Collection dataset; returns path to SMSSpamCollection file."""
    DATA_DIR.mkdir(exist_ok=True)

    # Download zip if missing
    if not ZIP_PATH.exists():
        print(f"Downloading dataset to: {ZIP_PATH}")
        urllib.request.urlretrieve(DATA_URL, ZIP_PATH)
        print("Download completed.")

    # Extract zip if data file missing
    if not DATA_FILE.exists():
        print(f"Extracting dataset to: {EXTRACT_DIR}")
        EXTRACT_DIR.mkdir(exist_ok=True)
        with zipfile.ZipFile(ZIP_PATH, "r") as z:
            z.extractall(EXTRACT_DIR)
        print("Extraction completed.")

    if not DATA_FILE.exists():
        raise FileNotFoundError(
            f"Expected dataset file not found: {DATA_FILE}. "
            f"Check extraction contents in {EXTRACT_DIR}."
        )

    return DATA_FILE


def clean_text(text: str) -> str:
    """
    Basic preprocessing:
    - lowercase
    - remove URLs
    - keep letters + digits (optional), remove punctuation
    - compress whitespace
    """
    text = text.lower()
    text = re.sub(r"http\S+|www\S+", " ", text)        # remove URLs
    text = re.sub(r"[^a-z0-9\s]", " ", text)           # keep letters/numbers
    text = re.sub(r"\s+", " ", text).strip()           # remove extra spaces
    return text


def print_evaluation(y_true, y_pred, title: str):
    print("\n" + "=" * 70)
    print(f"MODEL: {title}")
    print("Accuracy:", round(accuracy_score(y_true, y_pred), 4))

    cm = confusion_matrix(y_true, y_pred, labels=["ham", "spam"])
    print("\nConfusion Matrix (rows=true, cols=pred) [ham, spam]:\n", cm)

    print("\nClassification Report:\n", classification_report(y_true, y_pred))


def predict_message(message: str, vectorizer: TfidfVectorizer, model) -> str:
    msg_clean = clean_text(message)
    msg_vec = vectorizer.transform([msg_clean])
    return model.predict(msg_vec)[0]


# -------------------------
# MAIN
# -------------------------
def main():
    # 1) Load dataset
    try:
        dataset_path = download_and_extract_dataset()
    except Exception as e:
        print("ERROR while downloading/extracting dataset:", e)
        sys.exit(1)

    df = pd.read_csv(dataset_path, sep="\t", header=None, names=["label", "message"])
    print("\nDataset loaded successfully.")
    print("Shape:", df.shape)
    print(df.head())

    # 2) Quick class distribution
    print("\nClass distribution:")
    print(df["label"].value_counts())

    # 3) Visual (optional) - bar chart
    try:
        df["label"].value_counts().plot(kind="bar")
        plt.title("Spam vs Ham Count")
        plt.xlabel("Label")
        plt.ylabel("Count")
        plt.tight_layout()
        plt.show()
    except Exception:
        # In case matplotlib backend issues in some terminals; ignore
        pass

    # 4) Clean text
    df["clean_message"] = df["message"].astype(str).apply(clean_text)

    # 5) Split
    X = df["clean_message"]
    y = df["label"]
    X_train, X_test, y_train, y_test = train_test_split(
        X, y, test_size=TEST_SIZE, random_state=RANDOM_STATE, stratify=y
    )
    print("\nTrain size:", X_train.shape[0], "| Test size:", X_test.shape[0])

    # 6) TF-IDF vectorization
    vectorizer = TfidfVectorizer(
        stop_words="english",
        max_df=0.95,
        min_df=2,
        ngram_range=(1, 2)  # unigrams + bigrams often helps spam detection
    )
    X_train_vec = vectorizer.fit_transform(X_train)
    X_test_vec = vectorizer.transform(X_test)
    print("\nTF-IDF matrix (train):", X_train_vec.shape)

    # 7) Train Naive Bayes (primary model)
    nb_model = MultinomialNB()
    nb_model.fit(X_train_vec, y_train)
    y_pred_nb = nb_model.predict(X_test_vec)
    print_evaluation(y_test, y_pred_nb, "Multinomial Naive Bayes (Primary)")

    # 8) Train Logistic Regression (comparison)
    lr_model = LogisticRegression(max_iter=2000, random_state=RANDOM_STATE)
    lr_model.fit(X_train_vec, y_train)
    y_pred_lr = lr_model.predict(X_test_vec)
    print_evaluation(y_test, y_pred_lr, "Logistic Regression (Comparison)")

    # 9) Custom message tests
    test_messages = [
        "Congratulations! You won a free iPhone. Click the link now to claim.",
        "Hi, are we meeting at 6 pm today?",
        "URGENT! Your account has been suspended. Verify now.",
        "Can you send me the report by tonight?",
        "Win cash now!!! Reply YES to claim your prize"
    ]

    print("\n" + "=" * 70)
    print("CUSTOM MESSAGE PREDICTIONS (Using Naive Bayes):")
    for msg in test_messages:
        print(f"- {msg}\n  => {predict_message(msg, vectorizer, nb_model)}")

    # 10) Save artifacts (model + vectorizer)
    artifacts_dir = Path("artifacts")
    artifacts_dir.mkdir(exist_ok=True)

    joblib.dump(vectorizer, artifacts_dir / "tfidf_vectorizer.joblib")
    joblib.dump(nb_model, artifacts_dir / "naive_bayes_model.joblib")
    joblib.dump(lr_model, artifacts_dir / "logistic_regression_model.joblib")

    print("\nSaved artifacts to:", artifacts_dir.resolve())
    print(" - tfidf_vectorizer.joblib")
    print(" - naive_bayes_model.joblib")
    print(" - logistic_regression_model.joblib")

    print("\nDONE. Use the printed metrics in your report (accuracy, confusion matrix, precision/recall/F1).")


if __name__ == "__main__":
    main()