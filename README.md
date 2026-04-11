# 🚀 AI Resume Screening & Job Prediction System

An intelligent machine learning-powered system for automatic resume screening, job role classification, and ATS (Applicant Tracking System) scoring using Natural Language Processing.

## 📋 Features

- **📄 PDF Resume Processing**: Automatically extract and process text from PDF resumes
- **🤖 AI-Powered Classification**: Machine learning models for accurate job role prediction
- **📊 TF-IDF Vectorization**: Advanced text feature extraction technique
- **🎯 Job Role Prediction**: Automatic identification of suitable job positions (5 roles)
- **📈 Confidence Scoring**: Probability-based confidence metrics
- **🔍 ATS Scoring System**: Keyword relevance evaluation
- **🌐 Interactive Dashboard**: User-friendly web interface built with Streamlit
- **💾 Model Persistence**: Save and load trained models

## 🛠️ Technology Stack

- **Python 3.8+**: Core programming language
- **scikit-learn**: Machine Learning algorithms
- **Pandas & NumPy**: Data manipulation and analysis
- **NLTK**: Natural Language Processing
- **Streamlit**: Interactive web interface
- **PyPDF2**: PDF text extraction
- **TF-IDF**: Text vectorization

## 📦 Installation

### Prerequisites
- Python 3.8 or higher
- pip (Python package manager)
- Git

### Setup Instructions

1. **Clone the repository**
```bash
git clone https://github.com/deepthi-2115/Resume_AI_System.git
cd Resume_AI_System
```

2. **Create a virtual environment (recommended)**
```bash
python -m venv venv
source venv/bin/activate  # On Windows: venv\Scripts\activate
```

3. **Install dependencies**
```bash
pip install -r requirements.txt
```

## 🚀 Usage

### 1. Train the Model

First, train the model with sample data:

```bash
python train.py
```

This will:
- Create necessary directories
- Train the Logistic Regression model
- Save the trained model to `models/resume_model.pkl`
- Test predictions on sample resumes

### 2. Run the Streamlit Application

Launch the interactive web interface:

```bash
streamlit run app.py
```

The application will open in your browser at `http://localhost:8501`

### 3. Upload and Analyze Resumes

1. Navigate to the "Upload Resume" section
2. Upload a PDF resume
3. The system will:
   - Extract text from the PDF
   - Preprocess and vectorize the text
   - Predict the job role
   - Calculate ATS score
   - Display keyword analysis

## 📊 Supported Job Roles

The system can classify resumes into the following job roles:

1. **Software Engineer** - Development, coding, API design
2. **Data Scientist** - ML, statistics, data analysis
3. **DevOps Engineer** - Infrastructure, CI/CD, cloud
4. **Product Manager** - Product strategy, roadmap, analytics
5. **UX Designer** - UI/UX, design systems, prototyping

## 📁 Project Structure

```
Resume_AI_System/
├── app.py                 # Main Streamlit application
├── model.py              # ML models and ATS scorer
├── preprocessing.py      # Text processing and extraction
├── train.py             # Model training script
├── utils.py             # Utility functions
├── requirements.txt     # Python dependencies
├── README.md           # This file
├── .gitignore          # Git ignore file
├── models/             # Saved trained models
├── data/               # Training/analysis data
├── logs/               # Event logs
└── uploads/            # Uploaded resume files
```

## 🔧 Configuration

### Modifying Job Keywords for ATS

Edit the `job_keywords` dictionary in `model.py`:

```python
self.job_keywords = {
    'Your Role': ['keyword1', 'keyword2', 'keyword3', ...],
}
```

### Changing ML Algorithm

In `app.py`, modify the model initialization:

```python
classifier = ResumeClassifier(model_type='random_forest')  # or 'logistic'
```

## 📈 Model Performance

- **Algorithm**: Logistic Regression with TF-IDF
- **Features**: 100 most important text features
- **Training Samples**: 10 diverse resumes
- **Expected Accuracy**: 85%+

## 🎯 How It Works

```
Resume (PDF)
    ↓
Text Extraction (PyPDF2)
    ↓
Text Preprocessing (Cleaning, Tokenization)
    ↓
TF-IDF Vectorization
    ↓
ML Model Prediction
    ↓
ATS Keyword Matching
    ↓
Results & Visualization
```

## 🌐 deployment

### Local Deployment

```bash
streamlit run app.py
```

### Cloud Deployment

#### Streamlit Cloud
1. Push code to GitHub
2. Go to [Streamlit Cloud](https://streamlit.io/cloud)
3. Create new app from repository
4. Select `app.py` as main file

#### AWS / Azure / Heroku
Refer to platform-specific documentation for Python Streamlit app deployment.

## 📝 Example Usage

```python
from model import ResumeClassifier, ATSScorer
from preprocessing import preprocess_resume

# Initialize classifier
classifier = ResumeClassifier(model_type='logistic')

# Train (or load existing model)
classifier.load_model('models/resume_model.pkl')

# Analyze resume
resume_text = "Python expert with 5 years experience..."
predicted_role, probabilities = classifier.predict_with_probabilities(resume_text)

# Calculate ATS score
ats_scorer = ATSScorer()
score, matched = ats_scorer.calculate_ats_score(resume_text, predicted_role)

print(f"Predicted Role: {predicted_role}")
print(f"ATS Score: {score:.1f}%")
```

## 🧪 Testing

Run predictions on sample resumes:

```bash
python train.py predict
```

## 📋 API Endpoints (for API extension)

Future versions can integrate FastAPI:

```
POST /predict - Predict job role from resume
POST /train - Train new model
GET /roles - Get supported job roles
```

## 🐛 Troubleshooting

### Issue: PDF extraction returns empty text
- Ensure PDF contains readable text (not scanned images)
- Try a different PDF reader: Use online PDF to text converter

### Issue: Model not found error
- Run `python train.py` to train and save the model first

### Issue: Streamlit app not starting
- Check Python and Streamlit versions
- Run `pip install --upgrade streamlit`

## 🚀 Future Enhancements

- [x] TF-IDF vectorization
- [x] Multiple ML algorithms support
- [x] ATS scoring system
- [ ] OCR for scanned resumes
- [ ] Batch resume processing
- [ ] REST API interface
- [ ] Database integration
- [ ] Advanced NLP (BERT, transformers)
- [ ] Resume ranking and scoring
- [ ] Candidate matching with job requirements

## 📄 License

This project is open source and available under the MIT License.

## 👨‍💻 Author

**Deepthi-2115**

- GitHub: [@deepthi-2115](https://github.com/deepthi-2115)
- Email: deepthi.contact@example.com

## 🤝 Contributing

Contributions are welcome! Please feel free to submit a Pull Request.

## 📞 Support

For issues, questions, or suggestions:
1. Open an Issue on GitHub
2. Check existing documentation
3. Contact the author

## ⭐ Acknowledgments

- scikit-learn for ML algorithms
- Streamlit for the web framework
- NLTK for NLP tools
- PyPDF2 for PDF processing

---

**Made with ❤️ by Deepthi-2115**

If you find this project helpful, please give it a ⭐ on GitHub!
