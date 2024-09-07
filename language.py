import tkinter as tk
from tkinter import messagebox
import nltk
import joblib
import string
from nltk.tokenize import word_tokenize
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.naive_bayes import MultinomialNB
from sklearn.pipeline import Pipeline

# Download NLTK resources (if not already downloaded)
nltk.download('punkt')

# Sample data for training (individual words with language labels)
data_words = {
    'english': ['hello', 'how', 'are', 'you', 'this', 'is', 'a', 'sample', 'text'],
    'kannada': ['ಹಲೋ', 'ನೀವು', 'ಹೇಗಿದ್ದೀರಿ', 'ಈ', 'ಒಂದು', 'ನಮೂನೆ', 'ಪಠ್ಯ'],
    'hindi': ['नमस्ते', 'आप', 'कैसे', 'हैं', 'यह', 'एक', 'नमूना', 'पाठ', 'है'],
    'telugu': ['హలో', 'మీరు', 'ఎలా', 'ఉన్నారు', 'ఇది', 'ఒక', 'నమూనా', 'టెక్స్ట్'],
    'tamil': ['வணக்கம்', 'நீங்கள்', 'எப்படி', 'இருக்கின்றீர்கள்', 'இது', 'ஒரு', 'மாதிரி', 'உரையாடல்']
}

# Preprocess text using NLTK (remove punctuation and lowercase)
def preprocess_text(text):
    return text.translate(str.maketrans('', '', string.punctuation)).lower()

# Tokenize text into words
def tokenize_text(text):
    return word_tokenize(text)

# Create combined dataset
X_train = []
y_train = []
for lang, words in data_words.items():
    X_train.extend(words)
    y_train.extend([lang] * len(words))

# Preprocess training data
X_train_processed = [preprocess_text(word) for word in X_train]

# Create pipeline with TF-IDF vectorizer and Multinomial Naive Bayes classifier
pipeline = Pipeline([
    ('tfidf', TfidfVectorizer(tokenizer=tokenize_text)),
    ('clf', MultinomialNB()),
])

# Train the model
pipeline.fit(X_train_processed, y_train)

# Save the model
joblib.dump(pipeline, 'language_identifier_model_wordlevel.joblib')

# Load the trained model
model = joblib.load('language_identifier_model_wordlevel.joblib')

def predict_language(input_text):
    # Tokenize input text into words
    input_words = word_tokenize(input_text)
    # Preprocess each word
    processed_words = [preprocess_text(word) for word in input_words]
    # Predict language for each word
    predictions = [model.predict([word])[0] for word in processed_words]
    return predictions

def on_predict_click():
    user_input = entry.get()
    predictions = predict_language(user_input)
    result_text = "Predicted Languages for Each Word:\n\n"
    for word, lang in zip(word_tokenize(user_input), predictions):
        result_text += f"{word}: {lang}\n"
    result_label.config(text=result_text, fg="black", font=("Arial", 12))

# Create Tkinter GUI
window = tk.Tk()
window.title("Word-Level Language Identifier")

# Center the window on the screen
window_width = 600
window_height = 400
screen_width = window.winfo_screenwidth()
screen_height = window.winfo_screenheight()
x_position = int((screen_width - window_width) / 2)
y_position = int((screen_height - window_height) / 2)
window.geometry(f"{window_width}x{window_height}+{x_position}+{y_position}")

# Add graphical design and widgets
label = tk.Label(window, text="Enter your text:", font=("Arial", 14))
label.pack(pady=10)

entry = tk.Entry(window, width=50, font=("Arial", 12))
entry.pack()

predict_button = tk.Button(window, text="Predict Languages", command=on_predict_click, font=("Arial", 12))
predict_button.pack(pady=10)

result_label = tk.Label(window, text="", fg="black", font=("Arial", 12), wraplength=500, justify="left")
result_label.pack(pady=20, padx=10)

window.mainloop()