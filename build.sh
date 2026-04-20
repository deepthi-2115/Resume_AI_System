#!/bin/bash
echo "Installing dependencies..."
pip install -r requirements.txt

echo "Downloading NLTK data..."
python -c "
import nltk
nltk.download('punkt')
nltk.download('stopwords')
print('NLTK data downloaded successfully')
"

echo "Creating required directories..."
mkdir -p models
mkdir -p uploads
mkdir -p data
mkdir -p logs

echo "Training model..."
python train.py

echo "Setup complete!"
