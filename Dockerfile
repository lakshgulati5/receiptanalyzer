# Use a standard, clean base image
FROM ubuntu:22.04

# Set environment variables for the app
ENV DEBIAN_FRONTEND=noninteractive
ENV PATH="/usr/local/bin:$PATH"

# Install system dependencies
RUN apt-get update && apt-get install -y \
    python3 python3-pip curl git \
    && rm -rf /var/lib/apt/lists/*

# Install Ollama and add it to the PATH
RUN curl -fsSL https://ollama.com/install.sh | sh

# Set the working directory for your application
WORKDIR /app

# Copy the requirements file and install Python dependencies
COPY requirements.txt requirements.txt
RUN pip3 install --no-cache-dir -r requirements.txt

# Copy the rest of your application files
COPY . .

# Expose the ports for Streamlit and Ollama
EXPOSE 8501
EXPOSE 11434

# Use a custom entrypoint script to run both services
ENTRYPOINT ["/app/run_app.sh"]