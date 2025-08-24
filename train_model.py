# train_model.py

# ==============================================================================
# 1. SETUP AND IMPORTS
# ==============================================================================
import pandas as pd
import os
import re
import pickle
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import LabelEncoder
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import accuracy_score, classification_report # NEW IMPORTS

print("--- Starting Model Training and Evaluation Pipeline ---")

# ==============================================================================
# 2. DEFINE FILE PATHS AND MAPPINGS
# ==============================================================================
BASE_DIRECTORY = '.' # Assumes all files are in the same folder
INSTACART_PRODUCTS_FILE = os.path.join(BASE_DIRECTORY, 'products.csv')
INSTACART_DEPARTMENTS_FILE = os.path.join(BASE_DIRECTORY, 'departments.csv')
KAGGLE_RETAIL_FILE = os.path.join(BASE_DIRECTORY, 'train.csv')
FINAL_TRAINING_DATA_FILE = os.path.join(BASE_DIRECTORY, 'final_training_dataset.csv')

# --- Mappings ---
instacart_department_map = {
    'produce': 'Food & Beverage', 'meat seafood': 'Food & Beverage', 'dairy eggs': 'Food & Beverage',
    'bakery': 'Food & Beverage', 'frozen': 'Food & Beverage', 'pantry': 'Food & Beverage',
    'canned goods': 'Food & Beverage', 'international': 'Food & Beverage', 'bulk': 'Food & Beverage',
    'dry goods pasta': 'Food & Beverage', 'snacks': 'Food & Beverage', 'breakfast': 'Food & Beverage',
    'beverages': 'Food & Beverage', 'alcohol': 'Food & Beverage', 'deli': 'Food & Beverage',
    'personal care': 'Health & Beauty', 'household': 'Home & Garden', 'babies': 'Babies',
    'pets': 'Pets', 'other': 'Miscellaneous', 'missing': 'Miscellaneous'
}
kaggle_category_map = {
    'Clothing, Shoes & Jewelry': 'Apparel & Accessories', 'Electronics': 'Electronics & Media',
    'Cell Phones & Accessories': 'Electronics & Media', 'Automotive': 'Automotive & Fuel',
    'Health & Personal Care': 'Health & Beauty', 'Beauty': 'Health & Beauty',
    'Tools & Home Improvement': 'Home & Garden', 'Patio, Lawn & Garden': 'Home & Garden',
    'Grocery & Gourmet Food': 'Food & Beverage', 'Pet Supplies': 'Pets', 'Baby': 'Babies'
}

# ==============================================================================
# 3. LOAD, MAP, AND COMBINE DATASETS
# ==============================================================================
print("Step 1: Loading and combining datasets...")
# (This section creates the final_training_dataset.csv file)
try:
    products_df = pd.read_csv(INSTACART_PRODUCTS_FILE)
    departments_df = pd.read_csv(INSTACART_DEPARTMENTS_FILE)
    instacart_df = pd.merge(products_df, departments_df, on='department_id')
    instacart_df['category'] = instacart_df['department'].map(instacart_department_map)
    instacart_df.rename(columns={'product_name': 'description'}, inplace=True)
    instacart_labeled = instacart_df[['description', 'category']].dropna()

    kaggle_df = pd.read_csv(KAGGLE_RETAIL_FILE, usecols=['title', 'categories'])
    kaggle_df['category'] = kaggle_df['categories'].map(kaggle_category_map)
    kaggle_df.rename(columns={'title': 'description'}, inplace=True)
    kaggle_labeled = kaggle_df[['description', 'category']].dropna()

    combined_df = pd.concat([instacart_labeled, kaggle_labeled], ignore_index=True)
    combined_df.drop_duplicates(subset=['description'], inplace=True)
    combined_df.to_csv(FINAL_TRAINING_DATA_FILE, index=False)
    print(f"-> Combined dataset created with {len(combined_df)} items.")
except Exception as e:
    print(f"❌ ERROR: Failed to create the dataset. Make sure all raw CSV files are in the folder. Details: {e}")
    exit()

# ==============================================================================
# 4. TRAIN THE MODEL
# ==============================================================================
print("\nStep 2: Training the model...")
df = pd.read_csv(FINAL_TRAINING_DATA_FILE)

def clean_text(text):
    if not isinstance(text, str): text = ''
    text = text.strip('"').lower()
    text = re.sub(r'[^a-z\s]', '', text)
    return ' '.join(text.split())

df['cleaned_description'] = df['description'].apply(clean_text)
df.dropna(subset=['cleaned_description', 'category'], inplace=True)
df = df[df['cleaned_description'] != '']

le = LabelEncoder()
df['category_encoded'] = le.fit_transform(df['category'])

X = df['cleaned_description']
y = df['category_encoded']
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42, stratify=y)

tfidf = TfidfVectorizer(ngram_range=(1, 2))
X_train_tfidf = tfidf.fit_transform(X_train)

model = LogisticRegression(random_state=42, max_iter=2000)
model.fit(X_train_tfidf, y_train)
print("-> Model training complete.")

# --- NEW SECTION: EVALUATE THE MODEL ---
print("\nStep 3: Evaluating the model...")
# Vectorize the test data
X_test_tfidf = tfidf.transform(X_test)
# Make predictions on the test set
y_pred = model.predict(X_test_tfidf)
# Calculate and print the accuracy
accuracy = accuracy_score(y_test, y_pred)
print(f"-> Overall Model Accuracy: {accuracy:.2%}")
# Print the full classification report
print("\n--- Classification Report: ---")
report = classification_report(y_test, y_pred, target_names=le.classes_)
print(report)
# --- END NEW SECTION ---

# ==============================================================================
# 5. SAVE THE MODEL FILES
# ==============================================================================
print("\nStep 4: Saving model files...")
with open('tfidf_vectorizer.pkl', 'wb') as f:
    pickle.dump(tfidf, f)
with open('category_classifier.pkl', 'wb') as f:
    pickle.dump(model, f)
with open('label_encoder.pkl', 'wb') as f:
    pickle.dump(le, f)
print("✅ Success! Your three .pkl model files have been saved to this folder.")
print("--- Model Training and Evaluation Pipeline Finished ---")
