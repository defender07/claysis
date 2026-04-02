import streamlit as st
import pandas as pd
import sys
import os
import tempfile
import time

# Add current folder to path to ensure 'src' is found relative to this file
sys.path.append(os.path.abspath(os.path.dirname(__file__)))

import src.preprocessing as preprocessing
import src.embedding as embedding
from src.ranking import rank_candidates
from src.ingestion import read_file

# --- Model Caching ---
@st.cache_resource
def get_sbert_model():
    """
    Caches the SentenceTransformer model to prevent re-initialization 
    on every Streamlit rerun, which can cause 'client closed' errors.
    """
    return embedding.get_model()

# Initialize the model once at the start
sbert_model = get_sbert_model()

# Map functions for backward compatibility
preprocess = preprocessing.preprocess
extract_skills = preprocessing.extract_skills
extract_experience = preprocessing.extract_experience
extract_education = preprocessing.extract_education
generate_embedding = embedding.generate_embedding
generate_embeddings_batch = embedding.generate_embeddings_batch

# --- Page Configuration ---
st.set_page_config(
    page_title="ScreenerPro AI - Advanced Resume Analysis",
    page_icon="🎯",
    layout="wide",
    initial_sidebar_state="expanded"
)

# --- Custom Styling ---
st.markdown("""
    <style>
    @import url('https://fonts.googleapis.com/css2?family=Inter:wght@400;600;700&display=swap');
    
    html, body, [class*="css"] {
        font-family: 'Inter', sans-serif;
    }
    
    .main {
        background-color: #f8f9fa;
    }
    
    .stButton>button {
        width: 100%;
        border-radius: 8px;
        height: 3em;
        background-color: #007bff;
        color: white;
        font-weight: 600;
        border: none;
        transition: all 0.3s ease;
    }
    
    .stButton>button:hover {
        background-color: #0056b3;
        box-shadow: 0 4px 12px rgba(0,123,255,0.3);
    }
    
    .metric-card {
        background-color: white;
        padding: 20px;
        border-radius: 12px;
        box-shadow: 0 2px 8px rgba(0,0,0,0.05);
        border: 1px solid #e9ecef;
    }
    
    .status-suitable {
        color: #28a745;
        font-weight: 700;
    }
    
    .status-unsuitable {
        color: #dc3545;
        font-weight: 700;
    }
    </style>
""", unsafe_allow_html=True)

# --- Sidebar Inputs ---
with st.sidebar:
    st.image("https://img.icons8.com/clouds/100/000000/resume.png", width=100)
    st.title("ScreenerPro AI")
    st.markdown("---")
    
    st.subheader("📋 Job Configuration")
    jd_input = st.text_area("Job Description", "Enter the job requirements here...", height=300)
    
    st.subheader("📁 Candidate Source")
    uploaded_files = st.file_uploader(
        "Upload Resumes (PDF, DOCX, TXT)", 
        type=["pdf", "docx", "txt"], 
        accept_multiple_files=True
    )
    
    st.markdown("---")
    analyze_btn = st.button("🚀 Analyze Candidates", use_container_width=True, type="primary")

# --- Main App Logic ---
if not analyze_btn and not st.session_state.get('results'):
    # Welcome State
    st.title("Resume Screening & Skill Matching System")
    st.info("👈 Please enter the job description and upload resumes in the sidebar to begin analysis.")
    
    col_x, col_y, col_z = st.columns(3)
    with col_x:
        st.markdown("### 🔍 Semantic Search")
        st.write("Beyond keyword matching. We use SBERT embeddings to understand the true context of resumes.")
    with col_y:
        st.markdown("### 🎯 Skill Extraction")
        st.write("Automatically extracts technical skills, education, and years of experience.")
    with col_z:
        st.markdown("### 🏆 Strict Ranking")
        st.write("Candidates are ranked based on suitability and a 100% skill match requirement for selection.")

if analyze_btn:
    if not uploaded_files or jd_input.strip() in ["Enter the job requirements here...", ""]:
        st.error("Please provide both a Job Description and at least one Resume.")
    else:
        with st.spinner("🧠 AI is analyzing resumes..."):
            start_time = time.time()
            
            resumes_text = []
            resume_names = []
            for file in uploaded_files:
                ext = file.name.split('.')[-1]
                with tempfile.NamedTemporaryFile(delete=False, suffix=f".{ext}") as tmp:
                    tmp.write(file.getvalue())
                    tmp_path = tmp.name
                
                text = read_file(tmp_path)
                os.unlink(tmp_path)
                resumes_text.append(text)
                resume_names.append(file.name)
            
            # Perform Analysis
            ranked_results = rank_candidates(jd_input, resumes_text, resume_names, threshold=0.25, model=sbert_model)
            processing_time = time.time() - start_time
            
            # Store in session state
            st.session_state['results'] = ranked_results
            st.session_state['total_count'] = len(uploaded_files)
            st.session_state['suitable_count'] = sum(1 for r in ranked_results if r['is_suitable'])
            st.session_state['proc_time'] = processing_time
            st.session_state['jd_text'] = jd_input

if st.session_state.get('results'):
    ranked_results = st.session_state['results']
    
    # --- Dashboard Header ---
    st.title("🎯 Analysis Dashboard")
    
    m_col1, m_col2, m_col3, m_col4 = st.columns(4)
    with m_col1:
        st.metric("Total Candidates", st.session_state['total_count'])
    with m_col2:
        st.metric("Suitable Matches", st.session_state['suitable_count'], 
                  delta=f"{st.session_state['suitable_count']/st.session_state['total_count']*100:.1f}%")
    with m_col3:
        best_score = max([r['score'] for r in ranked_results]) if ranked_results else 0
        st.metric("Top Match Score", f"{best_score:.2f}")
    with m_col4:
        st.metric("Analysis Time", f"{st.session_state['proc_time']:.2f}s")
    
    st.divider()
    
    # --- Results Organization ---
    tab_rank, tab_viz, tab_deep = st.tabs(["📊 Ranked Candidates", "📈 Score Analytics", "🔍 Candidate Deep Dive"])
    
    with tab_rank:
        st.subheader("Candidate Suitability Table")
        data = []
        for rank, res in enumerate(ranked_results, 1):
            data.append({
                "Rank": rank,
                "Candidate": res.get("filename", "Unknown"),
                "Overall Score": round(res.get('score', 0), 3),
                "Status": res.get("status", "Unknown"),
                "Matched Skills": ", ".join(res.get("matched_skills", [])),
                "Experience": f"{res.get('experience', 0)} yrs",
                "Education": ", ".join(res.get("education", []))
            })
        
        df = pd.DataFrame(data)
        st.dataframe(df, use_container_width=True, hide_index=True)

    with tab_viz:
        st.subheader("Match Score Breakdown")
        # Visualizing the components of the score
        viz_data = pd.DataFrame({
            'Candidate': [r['filename'] for r in ranked_results],
            'Semantic Similarity (60%)': [r['semantic_score'] * 0.6 for r in ranked_results],
            'Skill Match (30%)': [r['skill_score'] * 0.3 for r in ranked_results],
            'Education Match (10%)': [r['edu_score'] * 0.1 for r in ranked_results]
        })
        
        st.bar_chart(viz_data.set_index('Candidate'), stack=True)
        st.caption("The final score is a weighted combination of semantic context, skill overlap, and education alignment.")

    with tab_deep:
        st.subheader("Detailed Candidate Insights")
        
        all_names = [r['filename'] for r in ranked_results]
        selected_cand = st.selectbox("Select a candidate for detailed analysis", all_names)
        
        if selected_cand:
            cand_res = next(r for r in ranked_results if r["filename"] == selected_cand)
            
            c_col_a, c_col_b = st.columns(2)
            with c_col_a:
                st.write(f"### {selected_cand}")
                st.write(f"**Suitability Status:** {'✅ Suitable' if cand_res['is_suitable'] else '❌ Not Suitable'}")
                st.write(f"**Overall Match Score:** `{cand_res['score']:.4f}`")
                
                # Progress bars for breakdown
                st.write("**Score Components:**")
                st.write(f"Semantic Relevance ({int(cand_res['semantic_score']*100)}%)")
                st.progress(cand_res['semantic_score'])
                
                st.write(f"Skill Alignment ({int(cand_res['skill_score']*100)}%)")
                st.progress(cand_res['skill_score'])
                
                st.write(f"Education Compatibility ({int(cand_res['edu_score']*100)}%)")
                st.progress(cand_res['edu_score'])
                
                st.write(f"**Experience Level:** {cand_res['experience']} years")
                st.write(f"**Education Detected:** {', '.join(cand_res['education']) if cand_res['education'] else 'None documented'}")
            
            with c_col_b:
                st.write("### 🧠 AI Match Explanation")
                st.info(cand_res['explanation'])
                
                st.write("---")
                st.write("### 🛠️ Skill Analysis")
                if cand_res['matched_skills']:
                    st.write("**Matched Key Skills:**")
                    # Use pills/tags if available in newer streamlit, otherwise just bullet points
                    st.write(", ".join([f"✅ {s}" for s in cand_res['matched_skills']]))
                
                if cand_res['missing_skills']:
                    st.write("**Missing Requested Skills:**")
                    st.write(", ".join([f"❌ {s}" for s in cand_res['missing_skills']]))
            
            # --- Personalized Skill Recommendations ---
            if cand_res['missing_skills']:
                st.divider()
                st.write("#### 🚀 Personalized Skill Development Path")
                st.write(f"To improve the match for this role, the candidate should focus on:")
                m_cols = st.columns(min(len(cand_res['missing_skills']), 4))
                for i, skill in enumerate(sorted(cand_res['missing_skills'])):
                    with m_cols[i % len(m_cols)]:
                        st.warning(f"📚 **{skill.title()}**")
            
            st.divider()
            # Feed back from earlier version if available, or just full text
            with st.expander("📄 View Candidate Source Text"):
                # Note: 'full_text' was in old version, if missing we show a placeholder
                if 'full_text' in cand_res:
                    st.text_area("Extracted Content", cand_res['full_text'], height=400)
                else:
                    st.write("Source text not cached in session. Re-run analysis to view.")
            
            st.markdown("---")
            st.write("#### 🛡️ Recruiter Decision Support")
            col_dec1, col_dec2 = st.columns(2)
            with col_dec1:
                feedback = st.radio("Decision", ("Accept", "Reject", "Hold"), key=f"feed_{selected_cand}")
            with col_dec2:
                if st.button("✅ Confirm Decision", key=f"btn_{selected_cand}"):
                    st.toast(f"Decision for {selected_cand} saved! 📈")
