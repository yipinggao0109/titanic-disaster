# Use a lightweight Python base image
FROM python:3.11-slim

# Set the working directory inside the container
WORKDIR /app

# Copy requirements.txt into the container
COPY requirements.txt /app/requirements.txt

# Install dependencies (from your requirements.txt)
RUN pip install --no-cache-dir -r requirements.txt

# Copy your code into the container
COPY src /app/src

# Command to run your Python script (replace with your actual file later)
CMD ["python", "src/titanic_model/titanic_main.py"]