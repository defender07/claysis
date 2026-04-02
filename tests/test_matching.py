import os
import sys

# Add src to the path so modules can be imported
sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), '..')))

from src.ingestion import load_documents_from_directory
from src.preprocessing import preprocess, extract_skills
from src.embedding import generate_embedding, generate_embeddings_batch
from src.ranking import rank_candidates, evaluate_metrics
import time
from unittest.mock import patch
import numpy as np

def mock_get_model():
    class MockModel:
        def encode(self, sentences, **kwargs):
            if isinstance(sentences, str):
                return np.random.rand(384).astype(np.float32)
            return np.random.rand(len(sentences), 384).astype(np.float32)
    return MockModel()

@patch('src.embedding.get_model', side_effect=mock_get_model)
def test_full_pipeline(mock_model):
    # 1. Ingestion
    resumes_dir = os.path.join('data', 'resumes')
    jd_dir = os.path.join('data', 'job_descriptions')
    
    resumes = load_documents_from_directory(resumes_dir)
    jds = load_documents_from_directory(jd_dir)
    
    assert len(resumes) > 0, "No resumes loaded"
    assert len(jds) > 0, "No job descriptions loaded"
    
    # Let's take the first job description
    jd_filename, jd_text = list(jds.items())[0]
    
    # 2. Preprocessing
    jd_processed = preprocess(jd_text)
    jd_skills = extract_skills(jd_processed)
    
    processed_resumes = {}
    resume_skills = {}
    for fname, text in resumes.items():
        processed_resumes[fname] = preprocess(text)
        resume_skills[fname] = extract_skills(text)
        
    # 3. Embeddings
    jd_embedding = generate_embedding(jd_processed)
    
    resume_filenames = list(processed_resumes.keys())
    resume_texts = [processed_resumes[fname] for fname in resume_filenames]
    resume_embeddings = generate_embeddings_batch(resume_texts)
    
    # 4. Ranking (Updated to use text and handle dict return)
    start_time = time.time()
    ranked_candidates = rank_candidates(jd_text, list(resumes.values()), resume_filenames, threshold=0.2)
    end_time = time.time()
    
    print("\nJob Description:", jd_filename)
    print("Extracted JD Skills:", jd_skills)
    print("\nRanked Candidates:")
    for rank, res in enumerate(ranked_candidates, 1):
        suitability = "Suitable" if res["is_suitable"] else "Not Suitable"
        print(f"{rank}. {res['filename']} [{suitability}] - Score: {res['score']:.4f} - Skills Matched: {res['matched_skills']}")
        
    print(f"\nProcessing Time for {len(resumes)} resumes: {end_time - start_time:.4f} seconds")
        
    ranked_filenames = [res["filename"] for res in ranked_candidates]
    
    # 5. Evaluation (ground truth for jd_python is resume_a.txt)
    ground_truth = {'resume_a.txt'}
    metrics = evaluate_metrics(ranked_filenames, ground_truth)
    
    print("\nEvaluation Metrics:")
    for metric, value in metrics.items():
        print(f"{metric}: {value}")
        
    assert len(ranked_candidates) == len(resumes), "Not all resumes were ranked"
    
if __name__ == "__main__":
    test_full_pipeline()
