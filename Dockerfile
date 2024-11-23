FROM tensorflow/tensorflow:2.17.0-gpu-jupyter

# Install dependencies from requirements.txt
COPY requirements.txt .
RUN pip install --no-cache-dir -r requirements.txt
