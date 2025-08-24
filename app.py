# ==============================================================================
# 1. SETUP AND IMPORTS
# ==============================================================================
import streamlit as st
import pandas as pd
import os
import shutil
import json
import re
import pickle
import cv2
import subprocess
import time
import numpy as np
from PIL import Image, ImageOps
from paddleocr import PaddleOCR
import ollama
import plotly.express as px
import stat

# ==============================================================================
# 2. HELPER FUNCTIONS
# ==============================================================================
def convert_numpy_types(obj):
    if isinstance(obj, dict): return {k: convert_numpy_types(v) for k, v in obj.items()}
    elif isinstance(obj, list): return [convert_numpy_types(elem) for elem in obj]
    elif isinstance(obj, np.integer): return int(obj)
    elif isinstance(obj, np.floating): return float(obj)
    elif isinstance(obj, np.ndarray): return obj.tolist()
    else: return obj

@st.cache_resource
def manage_ollama():
    """
    Manually installs and starts Ollama.
    This is cached and runs only ONCE. All UI calls have been removed.
    """
    # 1. Define local directory for the Ollama binary
    ollama_dir = os.path.join(os.getcwd(), "ollama_bin")
    ollama_path = os.path.join(ollama_dir, "ollama")
    os.makedirs(ollama_dir, exist_ok=True)

    # 2. Add the local directory to the system's PATH
    os.environ["PATH"] = f"{ollama_dir}:{os.environ['PATH']}"

    # 3. Install Ollama if not present
    if not os.path.exists(ollama_path):
        print("Ollama not found locally. Starting one-time download...")
        try:
            ollama_url = "https://github.com/ollama/ollama/releases/download/v0.2.5/ollama-linux-amd64"
            download_path = os.path.join(ollama_dir, "ollama-linux-amd64")
            
            subprocess.run(["curl", "-L", ollama_url, "-o", download_path], check=True)
            
            os.rename(download_path, ollama_path)
            print("Ollama downloaded. Setting permissions...")
            
            current_permissions = os.stat(ollama_path).st_mode
            os.chmod(ollama_path, current_permissions | stat.S_IEXEC)
            
            print("Ollama installed locally successfully!")
        except Exception as e:
            print(f"Failed to install Ollama. Error: {e}")
            st.stop()

    # 4. Start the Ollama server
    print("Starting local LLM server...")
    subprocess.Popen(['ollama', 'serve'])
        
    # 5. Wait for the server to be ready
    client = None
    max_wait_time = 60
    start_time = time.time()
    
    print("Waiting for Ollama server to start...")
    while time.time() - start_time < max_wait_time:
        try:
            client = ollama.Client(host='127.0.0.1:11434', timeout=2)
            client.list()
            print("Ollama server is running.")
            break
        except Exception:
            time.sleep(1)
            client = None
            
    if not client:
        print("Ollama server failed to start within the time limit.")
        st.stop()

    # 6. Pull the model if it's not available
    try:
        # Change the model name to the smaller, faster phi3:mini
        model_name = 'phi3:mini'
        models_response = client.list()
        local_models = [m.get('name') for m in models_response.get('models', []) if m.get('name')]
        
        if model_name not in local_models:
            print(f"Model '{model_name}' not found. Downloading now...")
            
            long_timeout_client = ollama.Client(host='127.0.0.1:11434', timeout=600)
            long_timeout_client.pull(model_name)

            print(f"Model '{model_name}' downloaded successfully!")
    except Exception as e:
        print(f"Failed to pull the LLM model. Error: {e}")
        st.stop()
    
    print("Ollama setup complete.")

@st.cache_resource
def load_model_assets(vectorizer_path, model_path, encoder_path):
    with open(vectorizer_path, 'rb') as f: vectorizer = pickle.load(f)
    with open(model_path, 'rb') as f: model = pickle.load(f)
    with open(encoder_path, 'rb') as f: encoder = pickle.load(f)
    return vectorizer, model, encoder

@st.cache_resource
def load_ocr_model():
    return PaddleOCR(lang='en', use_textline_orientation=True)

# ==============================================================================
# 3. PIPELINE FUNCTIONS
# ==============================================================================
def perform_ocr_and_generate_lines(ocr_model, img_path):
    img = cv2.imread(img_path)
    if img is None: return [], None
    image_dims = img.shape
    ocr_result = ocr_model.predict(img_path)
    words_with_boxes = []
    if ocr_result and ocr_result[0] and ocr_result[0].get('rec_texts'):
        texts = ocr_result[0].get('rec_texts', [])
        boxes = ocr_result[0].get('rec_polys', [])
        for box, text in zip(boxes, texts):
            words_with_boxes.append({'bbox': box.tolist(), 'text': text})
    return words_with_boxes, image_dims

def ocr_task_processor(ocr_model, image_path, output_root):
    try:
        filename_without_ext = os.path.splitext(os.path.basename(image_path))[0]
        subfolder_path = os.path.join(output_root, f'output_{filename_without_ext}')
        os.makedirs(subfolder_path, exist_ok=True)
        
        words, image_dims = perform_ocr_and_generate_lines(ocr_model, image_path)
        
        if not words:
            return subfolder_path, "Success (no text found)"
        
        full_output_filename = os.path.join(subfolder_path, 'full_ocr_output.json')
        with open(full_output_filename, 'w', encoding='utf-8') as f: json.dump(words, f, indent=4)
        
        return subfolder_path, "Success"
    except Exception as e:
        return None, f"Error on {os.path.basename(image_path)}: {e}"

def parse_text_with_llm(ocr_data_json):
    system_prompt = "You are an expert financial analyst specializing in parsing unstructured documents like receipts. Your task is to extract key information from the provided OCR data, which contains text lines and their bounding boxes. Focus on identifying the store name, transaction date, total amount, and a list of purchased items. For each item, extract its description, quantity, and total price. Be meticulous and only include data that is clearly present. If any information is missing, use a null value. Your output MUST be a valid JSON object."
    user_prompt = f"""
    **OCR TEXT DATA:**
    {json.dumps(ocr_data_json, indent=2)}

    **JSON SCHEMA TO POPULATE:**
    {{
      "store_name": "string | null",
      "date": "string | null",
      "total": "number | null",
      "items": [
        {{
          "description": "string | null",
          "quantity": "number | null",
          "total_price": "number | null"
        }}
      ]
    }}

    **INSTRUCTIONS:**
    Extract the requested information from the OCR data and return it as a single JSON object that strictly adheres to the schema provided.
    * Find the name of the store. Look for common brand names or headers.
    * Find the transaction date. Look for common date formats.
    * Find the overall total price of the receipt. This is usually at the bottom.
    * Find the list of purchased items. For each item:
        * Extract a concise description.
        * Identify the quantity (if available).
        * Find the total price for that specific item.
    * If any value is not found, use a null value. Do not guess or hallucinate values.
    """
    try:
        client = ollama.Client(host='127.0.0.1:11434', timeout=300)
        
        # Change the model name to the smaller, faster phi3:mini
        response = client.chat(model='phi3:mini', messages=[{'role': 'system', 'content': system_prompt}, {'role': 'user', 'content': user_prompt}], format='json')
        
        raw_response = response['message']['content']
        
        if raw_response.strip().startswith('{'):
            parsed_json = json.loads(raw_response)
            return True, parsed_json
        else:
            return False, f"LLM did not return a valid JSON object. Raw output: {raw_response}"
    except Exception as e:
        return False, str(e)
    
def clean_text(text):
    if not isinstance(text, str): return ''
    text = text.strip('"').lower()
    text = re.sub(r'[^a-z\s]', '', text)
    return ' '.join(text.split())

def categorize_items(items_list, vectorizer, model, encoder):
    if not items_list: return []
    descriptions = [clean_text(item.get('description', '')) for item in items_list]
    X_tfidf = vectorizer.transform(descriptions)
    predictions_numeric = model.predict(X_tfidf)
    for i, item in enumerate(items_list):
        item['category'] = int(predictions_numeric[i])
    return items_list

# ==============================================================================
# 4. STREAMLIT UI APPLICATION
# ==============================================================================
st.set_page_config(layout="wide", page_title="Receipt Dashboard")
st.title("🧾 Receipt Processing & Spending Tracker")
manage_ollama()

# --- Define Paths ---
BASE_DIRECTORY = '.' 
MODEL_VECTORIZER_PATH = os.path.join(BASE_DIRECTORY, 'tfidf_vectorizer.pkl')
MODEL_CLASSIFIER_PATH = os.path.join(BASE_DIRECTORY, 'category_classifier.pkl')
LABEL_ENCODER_PATH = os.path.join(BASE_DIRECTORY, 'label_encoder.pkl')
UPLOAD_FOLDER = os.path.join(BASE_DIRECTORY, 'uploaded_receipts')
OUTPUT_FOLDER = os.path.join(BASE_DIRECTORY, 'pipeline_outputs')
ALL_RECEIPTS_DB = os.path.join(BASE_DIRECTORY, 'all_receipts.json')
os.makedirs(UPLOAD_FOLDER, exist_ok=True)
os.makedirs(OUTPUT_FOLDER, exist_ok=True)

# --- Load Models ---
try:
    vectorizer, model, encoder = load_model_assets(MODEL_VECTORIZER_PATH, MODEL_CLASSIFIER_PATH, LABEL_ENCODER_PATH)
    ocr_model = load_ocr_model()
except FileNotFoundError:
    st.error("Model files not found! Please run the `train_model.py` script once to create them.")
    st.stop()

# --- Load Database ---
all_receipts_data = []
if os.path.exists(ALL_RECEIPTS_DB):
    try:
        with open(ALL_RECEIPTS_DB, 'r') as f:
            if os.path.getsize(ALL_RECEIPTS_DB) > 0:
                all_receipts_data = json.load(f)
    except json.JSONDecodeError:
        st.warning("Could not read the receipt database. Starting fresh.")
        all_receipts_data = []

# --- Sidebar for Uploading ---
st.sidebar.header("Upload New Receipt")
uploaded_file = st.sidebar.file_uploader("Choose an image...", type=["png", "jpg", "jpeg"])
if uploaded_file:
    # Check image size and stop if it's too large
    MAX_SIDE_LENGTH = 2048 
    try:
        with Image.open(uploaded_file) as img:
            width, height = img.size
            if max(width, height) > MAX_SIDE_LENGTH:
                st.sidebar.error(f"Image is too large. Maximum side length is {MAX_SIDE_LENGTH} pixels.")
                st.stop()
    except Exception as e:
        st.sidebar.error(f"Could not read image file: {e}")
        st.stop()
    
    file_path = os.path.join(UPLOAD_FOLDER, uploaded_file.name)
    with open(file_path, "wb") as f: f.write(uploaded_file.getbuffer())
    st.sidebar.image(file_path, caption="Uploaded Receipt")
    if st.sidebar.button("Process Receipt"):
        
        # Use a status container to show progress
        status_placeholder = st.empty()
        
        with status_placeholder:
            with st.spinner('Running OCR...'):
                subfolder_path, ocr_status = ocr_task_processor(ocr_model, file_path, OUTPUT_FOLDER)
            
            if not subfolder_path or "Error" in ocr_status or "Skipped" in ocr_status:
                status_placeholder.error(f"OCR Pipeline Failed: {ocr_status}")
            elif "no text found" in ocr_status:
                status_placeholder.warning("OCR was successful, but no text was detected.")
            else:
                with st.spinner('Extracting data with LLM...'):
                    ocr_input_path = os.path.join(subfolder_path, 'full_ocr_output.json')
                    with open(ocr_input_path, 'r') as f: ocr_data = json.load(f)
                    
                    is_success, extracted_data = parse_text_with_llm(ocr_data)
                
                if not is_success:
                    status_placeholder.error(f"LLM data extraction failed: {extracted_data}")
                else:
                    with st.spinner('Categorizing items...'):
                        items = extracted_data.get("items", [])
                        extracted_data["items"] = categorize_items(items, vectorizer, model, encoder)
                    
                    final_data_to_save = convert_numpy_types(extracted_data)
                    all_receipts_data.append(final_data_to_save)
                    
                    with open(ALL_RECEIPTS_DB, 'w') as f:
                        json.dump(all_receipts_data, f, indent=4)
                    
                    status_placeholder.empty() # Clear the status message
                    st.success("Receipt processed successfully!")
                    st.rerun()
                    
# --- Main Dashboard ---
st.header("Spending Dashboard")
if not all_receipts_data:
    st.info("Upload a receipt to see your spending analysis.")
else:
    all_items = []
    for receipt in all_receipts_data:
        for item in receipt.get("items", []):
            try:
                price = float(item.get('total_price', 0.0))
            except (ValueError, TypeError):
                price = 0.0
            
            category_numeric = item.get('category', -1)
            category_text = "Miscellaneous" 
            if isinstance(category_numeric, (int, np.integer)):
                try:
                    category_text = encoder.inverse_transform([category_numeric])[0]
                except ValueError:
                    pass
            elif isinstance(category_numeric, str):
                category_text = category_numeric

            all_items.append({
                'store': receipt.get('store_name', 'N/A'),
                'date': receipt.get('date', 'N/A'),
                'description': item.get('description', 'N/A'),
                'category': category_text,
                'price': price
            })

    df = pd.DataFrame(all_items)
    
    if not df.empty:
        st.subheader("Total Spending by Category")
        category_spending = df.groupby('category')['price'].sum().sort_values(ascending=False)
        col1, col2 = st.columns(2)
        col1.dataframe(category_spending)
        fig = px.pie(category_spending, values='price', names=category_spending.index)
        col2.plotly_chart(fig, use_container_width=True)
    else:
        st.warning("No categorized items found. Please upload a receipt with valid items to see the analysis.")

    st.subheader("Last Processed Receipt")
    last_receipt = all_receipts_data[-1]
    col1, col2 = st.columns(2)
    col1.text(f"Store: {last_receipt.get('store_name', 'N/A')}")
    col1.text(f"Date: {last_receipt.get('date', 'N/A')}")
    col1.text(f"Total: ${last_receipt.get('total', 'N/A')}")
    if last_receipt.get("items"):
        last_receipt_df_items = []
        for item in last_receipt["items"]:
            category_numeric = item.get('category', -1)
            category_text = "Miscellaneous"
            if isinstance(category_numeric, (int, np.integer)):
                try:
                    category_text = encoder.inverse_transform([category_numeric])[0]
                except ValueError:
                    pass
            elif isinstance(category_numeric, str):
                category_text = category_numeric
            
            last_receipt_df_items.append({
                'description': item.get('description'),
                'category': category_text,
                'total_price': item.get('total_price')
            })

        last_receipt_df = pd.DataFrame(last_receipt_df_items)
        if not last_receipt_df.empty:
            col2.dataframe(last_receipt_df)
        else:
            col2.info("No items with categories to display for the last receipt.")