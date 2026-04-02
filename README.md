# ScreenerPro AI: Automated Resume Screening & Skill Matching

## 🚀 Project Overview
ScreenerPro AI is an advanced, automated system designed to streamline recruitment by processing and analyzing candidate resumes against job descriptions. Leveraging **Sentence Transformers (SBERT)** and custom NLP pipelines, the system provides objective, data-driven insights into candidate suitability through semantic similarity matching and strict skill verification.

## ✨ Key Features
- **Semantic Matching**: Uses `paraphrase-multilingual-MiniLM-L12-v2` for high-accuracy contextual matching across 50+ languages.
- **Weighted Scoring (60/30/10)**: Sophisticated ranking based on Semantic Similarity (60%), Skill Alignment (30%), and Education Compatibility (10%).
- **Strict Skill Verification**: Optional 100% skill match requirement for "Suitable" status, ensuring high-quality talent selection.
- **Advanced Education Extraction**: Consumption-based algorithm to accurately identify degrees (B.Tech, MBA, PhD, etc.) while avoiding pattern overlap (e.g., MBA vs BA).
- **Explainable AI (XAI)**: Human-readable explanations for every match decision, improving recruiter trust.
- **Interactive Analytics Dashboard**: Premium Streamlit UI with real-time metrics, score breakdowns, and candidate deep-dives.
- **Personalized Development Paths**: Automatically identifies skill gaps and recommends learning areas for candidates.
- **Headless API**: CLI interface for seamless integration into automated batch processing workflows.

## 🌟 Project Highlights
- **High-Performance NLP**: Optimized for fast processing of large-scale resume batches.
- **Production-Ready UI**: Clean, intuitive dashboard designed for professional HR use.
- **Modular Architecture**: Easy to extend with new parsers, models, or ranking logic.
- **Robust Error Handling**: Handles malformed documents and model initialization edge cases gracefully.

## 🛠️ Tech Stack
- **Language**: Python 3.10+
- **Deep Learning**: PyTorch, Sentence-Transformers (SBERT)
- **NLP & Linguistics**: SpaCy (en_core_web_sm)
- **Frontend/Dashboard**: Streamlit
- **Data Science**: Pandas, Numpy, Scikit-learn
- **Document Parsing**: PyPDF2, python-docx, pdfplumber

## 📋 Prerequisites
- Python 3.10 or higher
- Git

## ⚙️ Setup Instructions

1. **Clone the Repository**
   ```bash
   git clone https://github.com/defender07/claysis.git
   cd claysis/resume-screening-system
   ```

2. **Create a Virtual Environment**
   ```bash
   python -m venv venv
   source venv/bin/activate  # On Windows: venv\Scripts\activate
   ```

3. **Install Dependencies**
   ```bash
   pip install -r requirements.txt
   python -m spacy download en_core_web_sm
   ```

## 🚀 Usage

### Running the Dashboard
```bash
streamlit run app.py
```

### Running the CLI API
```bash
python api.py --jd path/to/jd.txt --resumes path/to/resume1.pdf path/to/resume2.docx
```

## 🧠 Solution Approach
1. **Ingestion**: Modular support for PDF, DOCX, and TXT files using `pdfplumber` and `python-docx`.
2. **Preprocessing**: 
   - **Cleaning**: Lowercasing, noise removal, and whitespace normalization.
   - **Normalization**: Lemmatization and stop-word removal via SpaCy.
   - **Entity Extraction**: Advanced regex-based extraction for Experience and Education (Consumption-based matching).
   - **Synonym Mapping**: Standardizes industry terms (e.g., "ML" -> "Machine Learning").
3. **Embedding**: Generates 384-dimensional dense vectors using a multilingual SBERT model to capture semantic intent beyond simple keywords.
4. **Ranking (Multi-Criteria)**: 
   - **Semantic Score (60%)**: Cosine similarity between JD and Resume.
   - **Skill Score (30%)**: Jaccard similarity between required and extracted skills.
   - **Education Score (10%)**: Hierarchical matching of degrees.
5. **Human-in-the-Loop**: Generates detailed explanations and recommendations to support recruiter decision-making.

## 📄 License
This project is for demonstration and recruitment evaluation purposes.
