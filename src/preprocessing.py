import re
import spacy

try:
    nlp = spacy.load("en_core_web_sm")
except OSError:
    print("Downloading en_core_web_sm...")
    import subprocess
    import sys
    subprocess.run([sys.executable, "-m", "spacy", "download", "en_core_web_sm"])
    
    # Force a reload of the module paths so it identifies the newly installed model
    import importlib
    importlib.invalidate_caches()
    
    # Try one more time, if it fails they may just need to restart the script
    try:
        nlp = spacy.load("en_core_web_sm")
    except OSError:
        import spacy.util
        nlp = spacy.load(spacy.util.get_package_path("en_core_web_sm"))

def clean_text(text):
    """
    Cleans the input text by:
    - Lowercasing
    - Removing special characters (keeping only letters, numbers, spaces, and basic punctuation)
    - Normalizing whitespace
    """
    if not text:
        return ""
    
    # Handle common abbreviations before lowercasing or removing punctuation if needed
    # But for now, simple cleaning is fine
    text = str(text).lower()
    
    # Standardize common synonyms/abbreviations
    synonyms = {
        r'\bnlp\b': 'natural language processing',
        r'\bml\b': 'machine learning',
        r'\bai\b': 'artificial intelligence',
        r'\baws\b': 'amazon web services',
        r'\bgcp\b': 'google cloud platform',
        r'\bazure\b': 'microsoft azure',
        r'\bjs\b': 'javascript',
        r'\bts\b': 'typescript',
        r'\bcv\b': 'computer vision',
        r'\bdl\b': 'deep learning',
    }
    for pattern, replacement in synonyms.items():
        text = re.sub(pattern, replacement, text)

    # Replace newlines and tabs with space
    text = re.sub(r'[\n\t\r]', ' ', text)
    # Replace special characters with space to avoid joining words (e.g. B.Tech/MBA -> b.tech mba)
    # We keep letters, numbers, spaces, and certain characters (.,;+#-)
    text = re.sub(r'[^a-z0-9\s.,;+#-]', ' ', text)
    # Normalize whitespace
    text = re.sub(r'\s+', ' ', text).strip()
    return text

def normalize_text(text):
    """
    Tokenizes and lemmatizes text using spaCy.
    Removes stop words and punctuation.
    Returns a space-separated string of lemmas.
    """
    doc = nlp(text)
    tokens = [
        token.lemma_ for token in doc 
        if not token.is_stop and not token.is_punct and token.text.strip()
    ]
    return " ".join(tokens)

def preprocess(text):
    """
    Applies text cleaning followed by NLP normalization.
    """
    cleaned = clean_text(text)
    normalized = normalize_text(cleaned)
    return normalized

def extract_skills(text, known_skills=None):
    """
    Extracts skills from text.
    Uses basic string matching and regex against a known skills list.
    """
    if known_skills is None:
        known_skills = [
            'python', 'java', 'c++', 'c#', 'sql', 'machine learning', 'natural language processing', 
            'artificial intelligence', 'aws', 'amazon web services', 'docker', 'kubernetes', 'spacy', 'tensorflow', 
            'pytorch', 'scikit-learn', 'react', 'javascript', 'typescript', 'html', 'css', 
            'data science', 'pandas', 'numpy', 'flask', 'django', 'fastapi', 'git', 'linux', 
            'azure', 'microsoft azure', 'gcp', 'google cloud platform', 'deep learning', 'computer vision', 'tableau', 'power bi',
            'spark', 'hadoop', 'kafka', 'mongodb', 'postgresql', 'redis', 'elasticsearch',
            'jenkins', 'terraform', 'ansible', 'graphql', 'rest api', 'microservices',
            'text embedding', 'sentence transformer', 'semantic similarity', 'entity extraction',
            'performance analysis', 'software engineering', 'scalable system design', 'problem-solving',
            'data preprocessing', 'model evaluation', 'ranking metrics', 'classification metrics',
            'dashboard', 'explainability', 'active learning', 'multilingual support',
            'precision', 'recall', 'f1-score', 'mrr', 'ndcg'
        ]
    
    synonyms = {
        'ml': 'machine learning',
        'nlp': 'natural language processing',
        'ai': 'artificial intelligence',
        'dl': 'deep learning',
        'cv': 'computer vision',
        'js': 'javascript',
        'ts': 'typescript',
        'aws': 'amazon web services',
        'gcp': 'google cloud platform',
        'bi': 'business intelligence',
        'sbert': 'sentence transformer'
    }
    
    extracted = set()
    cleaned = clean_text(text)
    
    # Check for each skill
    for skill in known_skills:
        pattern = r'\b' + re.escape(skill.lower()) + r'\b'
        if re.search(pattern, cleaned):
            extracted.add(skill.lower())
    
    # Check for synonyms
    for abbr, full_name in synonyms.items():
        pattern = r'\b' + re.escape(abbr.lower()) + r'\b'
        if re.search(pattern, cleaned):
            extracted.add(full_name)
            
    return list(extracted)

def extract_experience(text):
    """
    Extracts years of experience from text using regex.
    """
    cleaned = clean_text(text)
    # Match digit followed by optional +, then 'year(s)' or 'yr(s)'
    patterns = [
        r'(\d+)\s*[\+]?\s*year[s]?\s*(?:of\s*)?experience',
        r'(\d+)\s*year[s]?\s*exp',
        r'experience\s*(?:of\s*)?(\d+)\s*year[s]?',
        r'(\d+)\s*\+\s*year[s]?'
    ]
    
    years = []
    for pattern in patterns:
        matches = re.findall(pattern, cleaned)
        for match in matches:
            try:
                years.append(int(match))
            except ValueError:
                continue
    
    if years:
        return max(years)
    return 0

def extract_education(text):
    """
    Extracts educational degrees using 'Consumption-Based Pattern Matching'.
    Ensures that 'M.B.A.' is not matched as 'B.A.'.
    """
    degree_map = {
        'PhD': [r'ph\.?d', r'doctor of philosophy'],
        'MBA': [r'm\.?b\.?a'],
        'Masters': [r'master[\']?s', r'm\.?\s?s\.?', r'm\.?\s?tech', r'm\.?\s?c\.?\s?a', r'm\.?\s?sc', r'm\.?\s?com', r'm\.?\s?a\.?', r'm\.?e\.?'],
        'Bachelors': [r'bachelor[\']?s', r'b\.?\s?s\.?', r'b\.?\s?tech', r'b\.?\s?e\.?', r'b\.?\s?sc', r'b\.?\s?c\.?\s?a', r'b\.?\s?com', r'b\.?\s?a\.?'],
        'Diploma': [r'diploma']
    }
    
    # Priority list for consumption (longest/most specific first)
    # Using word boundaries and handling both dots and no dots
    priority_order = [
        (r'ph\.?d', 'PhD'),
        (r'doctor of philosophy', 'PhD'),
        (r'm\.?b\.?a', 'MBA'),
        (r'master of business administration', 'MBA'),
        
        # Masters
        (r'm\.?\s?tech', 'Masters'),
        (r'master of technology', 'Masters'),
        (r'm\.?\s?s\.?c', 'Masters'),
        (r'm\.?s\.?c\.?i', 'Masters'),
        (r'master of science', 'Masters'),
        (r'm\.?\s?s\.?\b', 'Masters'),
        (r'm\.?\s?c\.?\s?a', 'Masters'),
        (r'master of computer applications', 'Masters'),
        (r'm\.?\s?e\.?\b', 'Masters'),
        (r'master of engineering', 'Masters'),
        (r'm\.?\s?com', 'Masters'),
        (r'master of commerce', 'Masters'),
        (r'm\.?a\.?\b', 'Masters'),
        (r'master of arts', 'Masters'),
        (r'master[\']?s degree', 'Masters'),
        (r'master[\']?s', 'Masters'),
        
        # Bachelors
        (r'b\.?\s?tech', 'Bachelors'),
        (r'bachelor of technology', 'Bachelors'),
        (r'b\.?\s?s\.?c', 'Bachelors'),
        (r'bachelor of science', 'Bachelors'),
        (r'b\.?\s?s\.?\b', 'Bachelors'),
        (r'b\.?\s?c\.?\s?a', 'Bachelors'),
        (r'bachelor of computer applications', 'Bachelors'),
        (r'b\.?\s?e\.?\b', 'Bachelors'),
        (r'bachelor of engineering', 'Bachelors'),
        (r'b\.?\s?com', 'Bachelors'),
        (r'bachelor of commerce', 'Bachelors'),
        (r'b\.?a\.?\b', 'Bachelors'),
        (r'bachelor of arts', 'Bachelors'),
        (r'bachelor[\']?s degree', 'Bachelors'),
        (r'bachelor[\']?s', 'Bachelors'),
        
        # Diploma
        (r'diploma', 'Diploma'),
        (r'associate degree', 'Bachelors') # Often categorized as early undergrad
    ]
    
    found_degrees = []
    # Consumption text
    consumption_text = clean_text(text)
    
    for pattern, label in priority_order:
        # Use word boundaries for abbreviations
        # Handle cases where the pattern might contain dots which are not word characters
        p = r'\b' + pattern
        if not pattern.endswith(r'\b'):
             p += r'\b'
             
        match = re.search(p, consumption_text)
        if match:
            found_degrees.append(label)
            # CONSUME: Replace with spaces to keep offset structure but remove the text
            consumption_text = consumption_text[:match.start()] + " " * (match.end() - match.start()) + consumption_text[match.end():]
            
    return list(set(found_degrees))
