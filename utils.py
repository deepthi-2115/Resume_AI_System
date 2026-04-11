"""
Utility functions for the Resume Screening System
"""
import os
import pickle
from datetime import datetime


def ensure_directories():
    """Ensure required directories exist"""
    directories = [
        'models',
        'uploads',
        'data',
        'logs'
    ]
    
    for directory in directories:
        if not os.path.exists(directory):
            os.makedirs(directory)
            print(f"Created directory: {directory}")


def log_event(event_type, details):
    """Log events to file"""
    ensure_directories()
    
    log_file = 'logs/events.log'
    timestamp = datetime.now().strftime("%Y-%m-%d %H:%M:%S")
    
    with open(log_file, 'a') as f:
        f.write(f"[{timestamp}] {event_type}: {details}\n")


def save_analysis_result(filename, result_data):
    """Save analysis result to file"""
    ensure_directories()
    
    filepath = os.path.join('data', f"{filename}_{datetime.now().strftime('%Y%m%d_%H%M%S')}.pkl")
    
    with open(filepath, 'wb') as f:
        pickle.dump(result_data, f)
    
    return filepath


def load_analysis_result(filepath):
    """Load analysis result from file"""
    with open(filepath, 'rb') as f:
        result = pickle.load(f)
    
    return result


def format_percentage(value, decimals=1):
    """Format value as percentage"""
    return f"{value:.{decimals}f}%"


def get_confidence_level(confidence_score):
    """Get confidence level based on score"""
    if confidence_score >= 90:
        return "Very High", "🟢"
    elif confidence_score >= 75:
        return "High", "🟡"
    elif confidence_score >= 60:
        return "Medium", "🟠"
    else:
        return "Low", "🔴"


def get_ats_recommendation(ats_score):
    """Get ATS score recommendation"""
    if ats_score >= 85:
        return "Excellent - Resume highly optimized for ATS"
    elif ats_score >= 70:
        return "Good - Resume is ATS friendly"
    elif ats_score >= 50:
        return "Fair - Some ATS optimization needed"
    else:
        return "Poor - Significant ATS optimization required"


if __name__ == "__main__":
    ensure_directories()
    print("Utility module initialized")
