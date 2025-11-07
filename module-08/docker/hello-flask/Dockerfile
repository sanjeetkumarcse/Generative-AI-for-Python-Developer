# Use a small Python image
FROM python:3.10-slim

# Set working directory
WORKDIR /app

# Copy dependency file and install Flask
COPY requirements.txt .
RUN pip install --no-cache-dir -r requirements.txt

# Copy application code
COPY . .

# Expose Flask app port
EXPOSE 8000

# Command to start the Flask app
CMD ["python", "app.py"]
