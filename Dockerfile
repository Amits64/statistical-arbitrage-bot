# Use the official Python image from the Docker registry
FROM python:3.9-slim

# Set the working directory inside the container
WORKDIR /app

# Copy the current directory contents into the container at /app
COPY . /app

# Install dependencies from requirements.txt
RUN pip install --no-cache-dir -r requirements.txt

# Expose the port that your application will run on
EXPOSE 8080

# Run the trading script
CMD ["python", "pairs_trader.py"]

