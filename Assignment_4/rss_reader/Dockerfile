FROM python:3.9-slim

WORKDIR /app

# Install dependencies
COPY requirements.txt .
RUN pip install --no-cache-dir -r requirements.txt

# Copy application code
COPY rss_reader.py .

# Make the script executable
RUN chmod +x rss_reader.py

# Set the entrypoint
CMD ["python", "rss_reader.py"]