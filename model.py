"""
Machine Learning models for resume classification
"""
import pickle
import numpy as np
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.linear_model import LogisticRegression
from sklearn.ensemble import RandomForestClassifier
from sklearn.pipeline import Pipeline
from sklearn.metrics import accuracy_score, classification_report
import pandas as pd


class ResumeClassifier:
    """Resume Classification Model using TF-IDF and ML algorithms"""
    
    def __init__(self, model_type='logistic'):
        """
        Initialize classifier
        model_type: 'logistic' or 'random_forest'
        """
        self.model_type = model_type
        self.vectorizer = TfidfVectorizer(max_features=100, stop_words='english')
        
        if model_type == 'logistic':
            self.classifier = LogisticRegression(max_iter=200, random_state=42)
        else:
            self.classifier = RandomForestClassifier(n_estimators=100, random_state=42)
        
        self.pipeline = Pipeline([
            ('tfidf', self.vectorizer),
            ('classifier', self.classifier)
        ])
        self.job_roles = None
        self.is_trained = False
    
    def train(self, resumes, labels):
        """Train the model"""
        self.job_roles = list(set(labels))
        self.pipeline.fit(resumes, labels)
        self.is_trained = True
        
        # Calculate training accuracy
        predictions = self.pipeline.predict(resumes)
        accuracy = accuracy_score(labels, predictions)
        print(f"Model trained with {self.model_type} classifier")
        print(f"Training Accuracy: {accuracy:.2%}")
        
        return accuracy
    
    def predict(self, resume_text):
        """Predict job role for a resume"""
        if not self.is_trained:
            raise ValueError("Model is not trained yet")
        
        prediction = self.pipeline.predict([resume_text])[0]
        probability = self.pipeline.predict_proba([resume_text])[0]
        confidence = max(probability) * 100
        
        return prediction, confidence
    
    def predict_with_probabilities(self, resume_text):
        """Predict with all class probabilities"""
        if not self.is_trained:
            raise ValueError("Model is not trained yet")
        
        prediction = self.pipeline.predict([resume_text])[0]
        probabilities = self.pipeline.predict_proba([resume_text])[0]
        
        results = {}
        for i, role in enumerate(self.job_roles):
            results[role] = probabilities[i] * 100
        
        return prediction, results
    
    def save_model(self, filepath):
        """Save trained model to file"""
        with open(filepath, 'wb') as f:
            pickle.dump(self.pipeline, f)
        # Also save job_roles for reference
        roles_file = filepath.replace('.pkl', '_roles.txt')
        with open(roles_file, 'w') as f:
            f.write(','.join(self.job_roles))
        print(f"Model saved to {filepath}")
    
    def load_model(self, filepath):
        """Load trained model from file"""
        with open(filepath, 'rb') as f:
            self.pipeline = pickle.load(f)
        self.is_trained = True
        
        # Try to load job_roles from saved file
        roles_file = filepath.replace('.pkl', '_roles.txt')
        try:
            with open(roles_file, 'r') as f:
                self.job_roles = f.read().strip().split(',')
        except:
            # Fallback: extract from pipeline if available
            if hasattr(self.pipeline.named_steps['classifier'], 'classes_'):
                self.job_roles = list(self.pipeline.named_steps['classifier'].classes_)
            else:
                self.job_roles = []
        
        print(f"Model loaded from {filepath}")


class ATSScorer:
    """ATS (Applicant Tracking System) Scoring System"""
    
    def __init__(self):
        """Initialize ATS scorer"""
        self.job_keywords = {
            'Software Engineer': ['python', 'java', 'javascript', 'coding', 'development', 'api', 'database', 'sql'],
            'Data Scientist': ['machine learning', 'python', 'statistics', 'data analysis', 'pandas', 'numpy', 'tensorflow'],
            'DevOps Engineer': ['docker', 'kubernetes', 'aws', 'ci/cd', 'jenkins', 'linux', 'terraform', 'cloud'],
            'Product Manager': ['product', 'roadmap', 'stakeholder', 'agile', 'analytics', 'strategy', 'user', 'market'],
            'UX Designer': ['ui', 'ux', 'figma', 'design', 'prototyping', 'user research', 'wireframe', 'design system'],
        }
    
    def calculate_ats_score(self, resume_text, job_role):
        """
        Calculate ATS score based on keyword matching
        Returns score out of 100
        """
        resume_lower = resume_text.lower()
        
        # Get keywords for the job role
        keywords = self.job_keywords.get(job_role, [])
        
        if not keywords:
            return 50  # Default score if role not found
        
        # Count matched keywords
        matched_keywords = []
        for keyword in keywords:
            if keyword.lower() in resume_lower:
                matched_keywords.append(keyword)
        
        # Calculate score
        score = (len(matched_keywords) / len(keywords)) * 100
        
        return score, matched_keywords
    
    def add_job_keywords(self, job_role, keywords):
        """Add custom keywords for a job role"""
        self.job_keywords[job_role] = keywords
    
    def get_missing_keywords(self, resume_text, job_role):
        """Get list of missing keywords from resume"""
        resume_lower = resume_text.lower()
        keywords = self.job_keywords.get(job_role, [])
        
        missing = []
        for keyword in keywords:
            if keyword.lower() not in resume_lower:
                missing.append(keyword)
        
        return missing


def create_sample_training_data():
    """Create sample training data for demonstration"""
    sample_resumes = [
        "Python expert with 5 years of software development experience. Strong in Java, JavaScript, and database design. Created multiple APIs using Flask and Django.",
        "Experienced Data Scientist with expertise in Machine Learning, Python, Statistics, and Data Analysis. Skilled in Pandas, NumPy, and TensorFlow implementations.",
        "DevOps professional with extensive Docker and Kubernetes experience. AWS certified. Strong in CI/CD pipelines using Jenkins. Linux and Terraform expertise.",
        "Product Manager with 7 years of experience. Excellent in roadmap development, stakeholder management, and agile methodologies. Data-driven decision making.",
        "Creative UX Designer with 6 years of experience. Proficient in UI design, Figma, and prototyping. User-centered design approach with design system expertise.",
        "Full-stack developer with expertise in JavaScript, Python, and React. Built scalable APIs and databases. Strong understand of software architecture.",
        "ML Engineer with passion for deep learning. Experience with TensorFlow, PyTorch, and Python. Data analysis and statistics background.",
        "Infrastructure engineer specializing in cloud deployment using AWS. Docker containerization and Kubernetes orchestration expert. CI/CD setup.",
        "Strategic Product Lead with focus on market research and user analytics. Experience in agile team management and product roadmap planning.",
        "UI/UX Designer focused on user research and design systems. Prototyping in Figma and wireframing for web applications."
    ]
    
    sample_labels = [
        "Software Engineer",
        "Data Scientist",
        "DevOps Engineer",
        "Product Manager",
        "UX Designer",
        "Software Engineer",
        "Data Scientist",
        "DevOps Engineer",
        "Product Manager",
        "UX Designer"
    ]
    
    return sample_resumes, sample_labels
