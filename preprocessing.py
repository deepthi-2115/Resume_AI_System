"""
Preprocessing module for resume text extraction and cleaning
"""
import re
import nltk
from nltk.corpus import stopwords
from nltk.tokenize import word_tokenize

# Download required NLTK data
try:
    nltk.data.find('tokenizers/punkt')
except LookupError:
    nltk.download('punkt')

try:
    nltk.data.find('corpora/stopwords')
except LookupError:
    nltk.download('stopwords')


def extract_text_from_pdf(pdf_path):
    """Extract text from PDF file"""
    try:
        import PyPDF2
        text = ""
        with open(pdf_path, 'rb') as file:
            pdf_reader = PyPDF2.PdfReader(file)
            for page_num in range(len(pdf_reader.pages)):
                page = pdf_reader.pages[page_num]
                text += page.extract_text()
        return text
    except Exception as e:
        print(f"Error extracting PDF: {e}")
        return ""


def clean_text(text):
    """Clean and normalize text"""
    # Convert to lowercase
    text = text.lower()
    # Remove special characters and digits
    text = re.sub(r'[^a-zA-Z\s]', '', text)
    # Remove extra whitespace
    text = re.sub(r'\s+', ' ', text).strip()
    return text


def remove_stopwords(text):
    """Remove stopwords from text"""
    stop_words = set(stopwords.words('english'))
    tokens = word_tokenize(text)
    filtered_tokens = [word for word in tokens if word.lower() not in stop_words]
    return ' '.join(filtered_tokens)


def preprocess_resume(text):
    """Complete preprocessing pipeline"""
    # Clean text
    cleaned = clean_text(text)
    # Remove stopwords
    processed = remove_stopwords(cleaned)
    return processed


def extract_keywords(text, num_keywords=20):
    """Extract top keywords from text using simple frequency"""
    words = text.lower().split()
    word_freq = {}
    
    for word in words:
        if len(word) > 3:  # Only consider words longer than 3 characters
            word_freq[word] = word_freq.get(word, 0) + 1
    
    # Sort by frequency
    sorted_words = sorted(word_freq.items(), key=lambda x: x[1], reverse=True)
    return [word for word, freq in sorted_words[:num_keywords]]


def calculate_keyword_match(resume_text, job_keywords):
    """Calculate keyword match percentage"""
    resume_words = set(resume_text.lower().split())
    matched = sum(1 for keyword in job_keywords if keyword.lower() in resume_words)
    match_percentage = (matched / len(job_keywords)) * 100 if job_keywords else 0
    return match_percentage
