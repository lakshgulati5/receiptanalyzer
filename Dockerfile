# Use a base image that includes Ollama and Python
FROM ollama/ollama:latest

# Set the working directory for your application
WORKDIR /app

# Copy the requirements file and install dependencies
COPY requirements.txt requirements.txt
RUN apt-get update && apt-get install -y python3 python3-pip && apt-get clean
RUN pip3 install --no-cache-dir -r requirements.txt

# Copy the rest of your application files
COPY . .

# Expose the ports for Streamlit and Ollama
EXPOSE 8501
EXPOSE 11434

# Use a custom entrypoint script to run both services
COPY run_app.sh .
RUN chmod +x run_app.sh
ENTRYPOINT ["/app/run_app.sh"]