import numpy as np
from src.similarity import calculate_similarity, calculate_batch_similarity
from src.preprocessing import extract_skills, extract_experience, extract_education

def rank_candidates(job_description, resumes_text, resume_filenames, threshold=0.3, model=None):
    """
    Ranks candidates based on a weighted score:
    - 60% Semantic Similarity (SBERT)
    - 30% Skill Match (Jaccard-like overlap)
    - 10% Education Match
    """
    from src.embedding import generate_embedding, generate_embeddings_batch
    from src.preprocessing import preprocess, extract_skills, extract_experience, extract_education
    
    # 1. Analyze Job Description
    jd_processed = preprocess(job_description)
    jd_embed = generate_embedding(jd_processed, model=model)
    jd_skills = set(extract_skills(job_description)) # Extract from raw text for better coverage
    jd_edu = set(extract_education(job_description))
    jd_exp = extract_experience(job_description)
    
    # 2. Analyze all Resumes
    res_processed = [preprocess(r) for r in resumes_text]
    res_embeddings = generate_embeddings_batch(res_processed, model=model)
    
    # Calculate Semantic Similarities
    scores = calculate_batch_similarity(jd_embed, res_embeddings)
    
    results = []
    for i, (name, s_score) in enumerate(zip(resume_filenames, scores)):
        res_text = resumes_text[i]
        cand_skills = set(extract_skills(res_text))
        cand_exp = extract_experience(res_text)
        cand_edu = set(extract_education(res_text))
        
        # Skill Match Score
        skill_score = 0.0
        if jd_skills:
            overlap = cand_skills.intersection(jd_skills)
            skill_score = len(overlap) / len(jd_skills)
        else:
            skill_score = 1.0 # No skills requested
            
        # Education Match Score
        edu_score = 0.0
        if jd_edu:
            # Hierarchical education matching
            edu_hierarchy = {
                'PhD': 4,
                'MBA': 3,
                'Masters': 3,
                'Bachelors': 2,
                'Diploma': 1
            }
            
            # Find the max level in JD and Candidate
            jd_max_level = max([edu_hierarchy.get(e, 0) for e in jd_edu]) if jd_edu else 0
            cand_max_level = max([edu_hierarchy.get(e, 0) for e in cand_edu]) if cand_edu else 0
            
            # If candidate level is >= JD level, it's a match (1.0)
            if cand_max_level >= jd_max_level and jd_max_level > 0:
                edu_score = 1.0
            elif jd_max_level == 0:
                edu_score = 1.0
        else:
            edu_score = 1.0 # No education requested

        # Final Weighted Score
        final_score = (0.6 * s_score) + (0.3 * skill_score) + (0.1 * edu_score)
        
        is_suitable = final_score >= threshold
        
        # Multi-Tier Suitability & Selection Logic
        if skill_score == 1.0:
            status = "✅ Selected (Suitable)"
            is_suitable = True
        elif skill_score >= 0.5:
            status = "⚠️ Potential Match"
            is_suitable = False
        else:
            status = "❌ Poor Match"
            is_suitable = False
            
        # HARD CONSTRAINT: Experience Check (overrides Selected status)
        if jd_exp > 0 and cand_exp < jd_exp:
            is_suitable = False
            status = "❌ Not Suitable (Exp. Mismatch)"
            
        # Match explanation for interpretability
        explanation = get_match_explanation(jd_skills, cand_skills, jd_exp, cand_exp, jd_edu, cand_edu)
        
        results.append({
            "filename": name,
            "full_text": res_text,
            "status": status,
            "score": float(final_score),
            "semantic_score": float(s_score),
            "skill_score": float(skill_score),
            "edu_score": float(edu_score),
            "is_suitable": is_suitable,
            "matched_skills": list(cand_skills.intersection(jd_skills)) if jd_skills else [],
            "missing_skills": list(jd_skills - cand_skills) if jd_skills else [],
            "experience": cand_exp,
            "education": list(cand_edu),
            "explanation": explanation
        })
        
    # Sort descending
    results.sort(key=lambda x: x['score'], reverse=True)
    return results

def get_match_explanation(jd_skills, res_skills, jd_exp, res_exp, jd_edu, res_edu):
    """
    Generates a human-readable explanation of why a candidate matched or didn't.
    """
    matched = res_skills.intersection(jd_skills)
    missing = jd_skills - res_skills
    
    explanations = []
    
    if matched:
        explanations.append(f"Matched key skills: {', '.join(list(matched)[:5])}.")
    
    if jd_exp > 0:
        if res_exp >= jd_exp:
            explanations.append(f"Meets experience requirement ({res_exp} years).")
        else:
            explanations.append(f"Does not meet experience requirement ({res_exp}/{jd_exp} years).")
        
    if any(edu in res_edu for edu in jd_edu) and jd_edu:
        explanations.append("Meets educational requirements.")
        
    if not matched and jd_skills:
        explanations.append("Missing critical technical skills requested.")
        
    return " ".join(explanations)

def evaluate_metrics(ranked_candidates, ground_truth_relevant):
    """
    Evaluates ranking using standard metrics based on binary relevance.
    """
    k = len(ground_truth_relevant)
    if k == 0:
         return {"Error": "No relevant ground truth candidates provided."}
         
    top_k = ranked_candidates[:k]
    
    relevant_retrieved = len(set(top_k) & set(ground_truth_relevant))
    total_retrieved_k = len(top_k)
    total_relevant = len(ground_truth_relevant)
    
    precision_k = relevant_retrieved / total_retrieved_k if total_retrieved_k > 0 else 0.0
    recall = len(set(ranked_candidates) & set(ground_truth_relevant)) / total_relevant if total_relevant > 0 else 0.0
    
    f1 = 2 * (precision_k * recall) / (precision_k + recall) if (precision_k + recall) > 0 else 0.0
    
    # MRR calculation
    mrr = 0.0
    for rank, candidate in enumerate(ranked_candidates, 1):
        if candidate in ground_truth_relevant:
            mrr = 1.0 / rank
            break
            
    # NDCG calculation (binary relevance)
    dcg = 0.0
    for rank, candidate in enumerate(ranked_candidates, 1):
        if candidate in ground_truth_relevant:
            dcg += 1.0 / np.log2(rank + 1)
            
    idcg = sum([1.0 / np.log2(i + 1) for i in range(1, total_relevant + 1)])
    ndcg = dcg / idcg if idcg > 0 else 0.0
    
    metrics = {
        "Precision@K": precision_k,
        "Recall": recall,
        "F1 Score": f1,
        "MRR": mrr,
        "NDCG": ndcg
    }
    return metrics
