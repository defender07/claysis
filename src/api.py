from fastapi import FastAPI, HTTPException
from pydantic import BaseModel
from typing import List

from src.preprocessing import preprocess, extract_skills
from src.embedding import generate_embedding, generate_embeddings_batch
from src.ranking import rank_candidates

app = FastAPI(
    title="Resume Screening and Skill Matching API",
    description="API interface for external recruitment platforms to rank resumes against a Job Description.",
    version="1.0.0"
)

class MatchRequest(BaseModel):
    job_description: str
    resumes: List[str]

@app.post("/api/v1/match")
async def match_resumes(request: MatchRequest):
    """
    Takes a Job Description and a list of resume texts, calculates semantic embeddings, 
    and returns a ranked list of candidates based on matching score.
    """
    if not request.job_description or not request.resumes:
        raise HTTPException(status_code=400, detail="Job description and resumes must be provided.")
        
    try:
        # Preprocess Job Description
        jd_processed = preprocess(request.job_description)
        jd_embedding = generate_embedding(jd_processed)

        # Preprocess Resumes
        processed_resumes = [preprocess(resume) for resume in request.resumes]
        resume_embeddings = generate_embeddings_batch(processed_resumes)

        # Create dummy identifiers if texts are provided
        resume_ids = [f"Candidate_{i+1}" for i in range(len(request.resumes))]
        
        # Calculate scores and sort
        ranked = rank_candidates(jd_embedding, resume_embeddings, resume_ids)
        
        # Extract skills for extra insight
        results = []
        for rank, (fname, score, is_suitable) in enumerate(ranked, 1):
            original_idx = int(fname.split("_")[1]) - 1
            skills = extract_skills(request.resumes[original_idx])
            results.append({
                "candidate_id": fname,
                "similarity_score": round(score, 4),
                "is_suitable": is_suitable,
                "extracted_skills": skills,
                "rank": rank
            })
            
        return {"job_description_skills": extract_skills(request.job_description), "rankings": results}

    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))
