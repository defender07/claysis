import argparse
import sys
import os
import json
from src.ranking import rank_candidates
from src.ingestion import read_file

def main():
    parser = argparse.ArgumentParser(description="ScreenerPro AI - CLI Interface")
    parser.add_argument("--jd", type=str, required=True, help="Path to Job Description file")
    parser.add_argument("--resumes", type=str, nargs="+", required=True, help="Paths to Resume files")
    parser.add_argument("--output", type=str, default="results.json", help="Output JSON file path")
    parser.add_argument("--threshold", type=float, default=0.25, help="Similarity threshold")
    
    args = parser.parse_args()
    
    if not os.path.exists(args.jd):
        print(f"Error: JD file not found at {args.jd}")
        sys.exit(1)
        
    jd_text = read_file(args.jd)
    
    resumes_text = []
    resume_names = []
    for r_path in args.resumes:
        if os.path.exists(r_path):
            resumes_text.append(read_file(r_path))
            resume_names.append(os.path.basename(r_path))
        else:
            print(f"Warning: Resume file not found at {r_path}, skipping.")
            
    if not resumes_text:
        print("Error: No resumes loaded.")
        sys.exit(1)
        
    print(f"Analyzing {len(resumes_text)} resumes against {os.path.basename(args.jd)}...")
    results = rank_candidates(jd_text, resumes_text, resume_names, threshold=args.threshold)
    
    with open(args.output, "w") as f:
        json.dump(results, f, indent=4)
        
    print(f"\nDone! Results saved to {args.output}")
    print("\nTop Candidates:")
    for i, r in enumerate(results[:3], 1):
        print(f"{i}. {r['filename']} - Score: {r['score']:.4f} ({r['status']})")

if __name__ == "__main__":
    main()
