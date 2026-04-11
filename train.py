"""
Training script for the Resume Classification Model
Can be run independently to train with custom data
"""
import os
import pickle
from model import ResumeClassifier, create_sample_training_data
from utils import log_event, ensure_directories


def train_and_save_model(model_type='logistic', save_path='models/resume_model.pkl'):
    """
    Train the model and save it
    
    Args:
        model_type: 'logistic' or 'random_forest'
        save_path: Path to save the trained model
    
    Returns:
        Trained classifier object
    """
    ensure_directories()
    
    print("="*50)
    print(f"Training Resume Classification Model ({model_type})")
    print("="*50)
    
    # Create classifier
    classifier = ResumeClassifier(model_type=model_type)
    
    # Get training data
    print("\nLoading training data...")
    resumes, labels = create_sample_training_data()
    print(f"Loaded {len(resumes)} training resumes")
    print(f"Job Roles: {set(labels)}")
    
    # Train model
    print("\nTraining model...")
    accuracy = classifier.train(resumes, labels)
    
    # Save model
    os.makedirs(os.path.dirname(save_path), exist_ok=True)
    classifier.save_model(save_path)
    
    # Log event
    log_event("MODEL_TRAINING", f"Trained {model_type} model with accuracy {accuracy:.2%}")
    
    print(f"\n✅ Model training completed!")
    print(f"Model saved to: {save_path}")
    
    return classifier


def make_predictions_on_samples():
    """Make predictions on sample resumes"""
    print("\n" + "="*50)
    print("Making Predictions on Sample Resumes")
    print("="*50)
    
    # Load or train model
    model_path = 'models/resume_model.pkl'
    
    if os.path.exists(model_path):
        classifier = ResumeClassifier()
        classifier.load_model(model_path)
    else:
        classifier = train_and_save_model()
    
    # Get sample data
    resumes, true_labels = create_sample_training_data()
    
    # Make predictions
    print("\nMaking predictions on sample resumes...\n")
    
    correct = 0
    for i, (resume, true_label) in enumerate(zip(resumes, true_labels), 1):
        predicted_role, probabilities = classifier.predict_with_probabilities(resume)
        
        is_correct = predicted_role == true_label
        correct += is_correct
        
        status = "✅" if is_correct else "❌"
        print(f"{status} Sample {i}:")
        print(f"   True Label: {true_label}")
        print(f"   Predicted: {predicted_role}")
        print(f"   Confidence: {probabilities[predicted_role]:.2f}%\n")
    
    accuracy = (correct / len(resumes)) * 100
    print(f"Overall Accuracy on Test Data: {accuracy:.2f}%")
    
    log_event("PREDICTIONS", f"Made {len(resumes)} predictions with accuracy {accuracy:.2f}%")


if __name__ == "__main__":
    import sys
    
    if len(sys.argv) > 1 and sys.argv[1] == 'predict':
        make_predictions_on_samples()
    else:
        # Train with Logistic Regression
        train_and_save_model('logistic')
        
        # Optionally train with Random Forest
        # train_and_save_model('random_forest', 'models/resume_model_rf.pkl')
        
        # Make predictions
        print("\n")
        make_predictions_on_samples()
