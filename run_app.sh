#!/bin/bash

# Start Ollama in the background
/usr/local/bin/ollama serve &

# Wait for Ollama to start and download the model
sleep 5
echo "🟢 Pulling Llama 3 model..."
/usr/local/bin/ollama pull llama3:8b

# Start the Streamlit application
streamlit run app.py --server.port=8501 --server.address=0.0.0.0