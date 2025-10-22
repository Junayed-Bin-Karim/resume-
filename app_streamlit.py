import streamlit as st
import docx2txt
import PyPDF2
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.metrics.pairwise import cosine_similarity
import os
import uuid
import pandas as pd
from io import BytesIO
import re
import plotly.express as px
import plotly.graph_objects as go
from datetime import datetime
import base64
import nltk
from nltk.corpus import stopwords
from nltk.tokenize import word_tokenize
import spacy
from collections import Counter
import zipfile

# Download NLTK data (run once)
try:
    nltk.data.find('tokenizers/punkt')
except LookupError:
    nltk.download('punkt')
try:
    nltk.data.find('corpora/stopwords')
except LookupError:
    nltk.download('stopwords')

# ---------------- Page Setup ----------------
st.set_page_config(
    page_title="Resume Matching Pro",
    layout="wide",
    page_icon="📄"
)

# ---------------- Custom CSS ----------------
st.markdown("""
<style>
    .main-header {
        font-size: 3rem;
        color: #1F77B4;
        text-align: center;
        margin-bottom: 1rem;
    }
    .feature-card {
        background-color: #f8f9fa;
        padding: 1.5rem;
        border-radius: 10px;
        border-left: 4px solid #1F77B4;
        margin-bottom: 1rem;
    }
    .match-score-high {
        background-color: #d4edda;
        color: #155724;
        padding: 0.25rem 0.5rem;
        border-radius: 15px;
        font-weight: bold;
    }
    .match-score-medium {
        background-color: #fff3cd;
        color: #856404;
        padding: 0.25rem 0.5rem;
        border-radius: 15px;
        font-weight: bold;
    }
    .match-score-low {
        background-color: #f8d7da;
        color: #721c24;
        padding: 0.25rem 0.5rem;
        border-radius: 15px;
        font-weight: bold;
    }
</style>
""", unsafe_allow_html=True)

# ---------------- Header ----------------
st.markdown("""
<div style="text-align: center;">
    <h1 class="main-header">Resume Matching Pro</h1>
    <h3 style='color:gray;'>Advanced Resume Screening & Analysis</h3>
    <p style='color:gray;'>by Md Junayed Bin Karim</p>
</div>
""", unsafe_allow_html=True)

st.markdown("---")

# ---------------- Sidebar for Settings ----------------
with st.sidebar:
    st.header("⚙️ Settings")
    
    st.subheader("Matching Algorithm")
    algorithm = st.selectbox(
        "Select matching method:",
        ["TF-IDF + Cosine Similarity", "Keyword-Based", "Hybrid Approach"]
    )
    
    st.subheader("Weight Settings")
    skills_weight = st.slider("Skills Weight", 0.0, 1.0, 0.4)
    experience_weight = st.slider("Experience Weight", 0.0, 1.0, 0.3)
    education_weight = st.slider("Education Weight", 0.0, 1.0, 0.2)
    keywords_weight = st.slider("Keywords Weight", 0.0, 1.0, 0.1)
    
    st.subheader("Filters")
    min_similarity = st.slider("Minimum Similarity Score", 0, 100, 50)
    required_skills = st.text_input("Required Skills (comma-separated)")
    
    st.markdown("---")
    st.info("💡 **Pro Tip**: Use the hybrid approach for most accurate results combining semantic matching and keyword analysis.")

# ---------------- Instructions with Feature Cards ----------------
col1, col2, col3 = st.columns(3)

with col1:
    st.markdown("""
    <div class="feature-card">
        <h4>🚀 Smart Matching</h4>
        <p>Advanced AI-powered resume matching with multiple algorithms</p>
    </div>
    """, unsafe_allow_html=True)

with col2:
    st.markdown("""
    <div class="feature-card">
        <h4>📊 Analytics Dashboard</h4>
        <p>Comprehensive insights and visualizations</p>
    </div>
    """, unsafe_allow_html=True)

with col3:
    st.markdown("""
    <div class="feature-card">
        <h4>🔧 Advanced Features</h4>
        <p>Batch processing, custom weights, and filters</p>
    </div>
    """, unsafe_allow_html=True)

# ---------------- Job Description Input ----------------
st.subheader("📋 Job Description")
job_description = st.text_area(
    "Enter Job Description Here:",
    height=150,
    placeholder="Write the job description for the position...",
    help="Include skills, experience requirements, and qualifications for better matching"
)

# ---------------- Resume Upload with Batch Processing ----------------
st.subheader("📂 Upload Resumes")
col1, col2 = st.columns([2, 1])

with col1:
    resume_files = st.file_uploader(
        "Upload Resumes (.pdf, .docx, .txt)",
        type=["pdf", "docx", "txt"],
        accept_multiple_files=True,
        help="You can upload multiple files at once"
    )

with col2:
    st.info(f"""
    **Upload Status:**
    - Total Files: {len(resume_files) if resume_files else 0}
    - Supported formats: PDF, DOCX, TXT
    """)

# ---------------- Advanced Options ----------------
with st.expander("🔧 Advanced Options"):
    col1, col2, col3 = st.columns(3)
    
    with col1:
        top_n = st.number_input(
            "Number of top matches to display:",
            min_value=1,
            max_value=len(resume_files) if resume_files else 10,
            value=5
        )
    
    with col2:
        auto_download = st.checkbox("Auto-download report", value=True)
        show_analytics = st.checkbox("Show analytics dashboard", value=True)
    
    with col3:
        export_format = st.selectbox(
            "Export format:",
            ["Excel", "CSV", "JSON"]
        )

# ---------------- Enhanced Text Extraction Functions ----------------
def extract_text_from_pdf(file):
    text = ""
    try:
        reader = PyPDF2.PdfReader(file)
        for page in reader.pages:
            page_text = page.extract_text()
            if page_text:
                text += page_text + "\n"
    except Exception as e:
        st.error(f"Error reading PDF {file.name}: {str(e)}")
    return text

def extract_text_from_docx(file):
    try:
        temp_path = f"temp_{uuid.uuid4().hex}_{file.name}"
        with open(temp_path, "wb") as f:
            f.write(file.read())
        text = docx2txt.process(temp_path)
        os.remove(temp_path)
        return text
    except Exception as e:
        st.error(f"Error reading DOCX {file.name}: {str(e)}")
        return ""

def extract_text_from_txt(file):
    try:
        return file.read().decode("utf-8")
    except:
        return file.read().decode("latin-1")

def extract_text(file):
    if file.type == "application/pdf":
        return extract_text_from_pdf(file)
    elif file.type == "application/vnd.openxmlformats-officedocument.wordprocessingml.document":
        return extract_text_from_docx(file)
    elif file.type == "text/plain":
        return extract_text_from_txt(file)
    else:
        return ""

# ---------------- Enhanced Contact Info Extraction ----------------
def extract_contact_info(text):
    contact = {}

    # Extract emails
    emails = re.findall(r"[a-zA-Z0-9._%+-]+@[a-zA-Z0-9.-]+\.[a-zA-Z]{2,}", text)
    contact['Email'] = emails[0] if emails else "N/A"

    # Extract phone numbers (enhanced pattern)
    phones = re.findall(r"(\+?(\d{1,3}[-.\s]?)?\(?\d{3}\)?[-.\s]?\d{3}[-.\s]?\d{4})", text)
    contact['Phone'] = phones[0][0] if phones else "N/A"

    # Extract LinkedIn profile URLs
    linkedin = re.findall(r"https?://(www\.)?linkedin\.com/in/[A-Za-z0-9_-]+", text)
    contact['LinkedIn'] = linkedin[0] if linkedin else "N/A"

    # Extract potential name (simple heuristic)
    lines = text.split('\n')
    for line in lines[:10]:  # Check first 10 lines
        if re.match(r'^[A-Z][a-z]+ [A-Z][a-z]+$', line.strip()):
            contact['Name'] = line.strip()
            break
    if 'Name' not in contact:
        contact['Name'] = "N/A"

    return contact

# ---------------- Skills and Experience Extraction ----------------
def extract_skills_experience(text):
    # Common skills database
    technical_skills = [
        'python', 'java', 'javascript', 'c++', 'sql', 'html', 'css', 'react', 'angular',
        'node.js', 'docker', 'kubernetes', 'aws', 'azure', 'gcp', 'machine learning',
        'data analysis', 'project management', 'agile', 'scrum'
    ]
    
    education_keywords = ['bachelor', 'master', 'phd', 'degree', 'university', 'college']
    experience_pattern = r'(\d+)\+?\s*years?'
    
    skills_found = []
    for skill in technical_skills:
        if re.search(r'\b' + re.escape(skill) + r'\b', text, re.IGNORECASE):
            skills_found.append(skill)
    
    experience_matches = re.findall(experience_pattern, text, re.IGNORECASE)
    experience_years = max([int(match) for match in experience_matches]) if experience_matches else 0
    
    education_level = "N/A"
    for edu in education_keywords:
        if edu in text.lower():
            education_level = edu.capitalize()
            break
    
    return {
        'skills': skills_found,
        'experience_years': experience_years,
        'education': education_level
    }

# ---------------- Enhanced Keyword Highlight Function ----------------
def highlight_keywords(text, keywords, job_description):
    # Combine explicit keywords with important words from job description
    all_keywords = set(keywords)
    job_words = set(re.findall(r'\b[a-zA-Z]{4,}\b', job_description.lower()))
    important_words = job_words - set(stopwords.words('english'))
    all_keywords.update(list(important_words)[:15])  # Top 15 important words
    
    highlighted_text = text
    for kw in all_keywords:
        if len(kw) > 3:  # Only highlight words longer than 3 characters
            pattern = re.compile(re.escape(kw), re.IGNORECASE)
            highlighted_text = pattern.sub(f"<mark style='background-color: yellow;'>{kw}</mark>", highlighted_text)
    
    return highlighted_text

# ---------------- Advanced Matching Algorithm ----------------
def advanced_similarity(job_desc, resume_text, skills_info):
    # TF-IDF similarity
    vectorizer = TfidfVectorizer(stop_words='english')
    tfidf_matrix = vectorizer.fit_transform([job_desc, resume_text])
    tfidf_sim = cosine_similarity(tfidf_matrix[0:1], tfidf_matrix[1:2])[0][0]
    
    # Skills matching
    job_skills = set(re.findall(r'\b[a-zA-Z]{4,}\b', job_desc.lower()))
    resume_skills = set(skills_info['skills'])
    skills_match = len(job_skills.intersection(resume_skills)) / len(job_skills) if job_skills else 0
    
    # Experience matching
    exp_pattern = r'(\d+)\+?\s*years?'
    job_exp_matches = re.findall(exp_pattern, job_desc, re.IGNORECASE)
    job_exp = max([int(match) for match in job_exp_matches]) if job_exp_matches else 0
    exp_match = min(skills_info['experience_years'] / job_exp, 1) if job_exp > 0 else 0.5
    
    # Combined score
    final_score = (
        tfidf_sim * 0.4 +
        skills_match * 0.3 +
        exp_match * 0.3
    )
    
    return final_score * 100

# ---------------- Analytics Dashboard ----------------
def create_analytics_dashboard(results, job_description):
    st.subheader("📊 Analytics Dashboard")
    
    # Create columns for metrics
    col1, col2, col3, col4 = st.columns(4)
    
    with col1:
        avg_score = sum([r['Similarity (%)'] for r in results]) / len(results)
        st.metric("Average Score", f"{avg_score:.1f}%")
    
    with col2:
        top_score = max([r['Similarity (%)'] for r in results])
        st.metric("Top Score", f"{top_score:.1f}%")
    
    with col3:
        qualified = len([r for r in results if r['Similarity (%)'] >= 70])
        st.metric("Highly Qualified", qualified)
    
    with col4:
        total_skills = sum([len(r['Skills']) for r in results])
        st.metric("Total Skills Found", total_skills)
    
    # Visualization
    col1, col2 = st.columns(2)
    
    with col1:
        # Score distribution
        scores = [r['Similarity (%)'] for r in results]
        fig = px.histogram(x=scores, nbins=10, title="Score Distribution")
        fig.update_layout(xaxis_title="Similarity Score", yaxis_title="Count")
        st.plotly_chart(fig, use_container_width=True)
    
    with col2:
        # Top skills chart
        all_skills = []
        for r in results:
            all_skills.extend(r['Skills'])
        skill_counts = Counter(all_skills)
        top_skills = dict(skill_counts.most_common(10))
        
        fig = px.bar(x=list(top_skills.keys()), y=list(top_skills.values()),
                    title="Top Skills in Candidate Pool")
        fig.update_layout(xaxis_title="Skills", yaxis_title="Count")
        st.plotly_chart(fig, use_container_width=True)

# ---------------- Match Button ----------------
if st.button("🔍 Match Resumes", type="primary"):
    if not job_description.strip():
        st.error("⚠️ Please enter a job description.")
    elif not resume_files:
        st.error("⚠️ Please upload at least one resume.")
    else:
        with st.spinner("⏳ Processing resumes with advanced analysis..."):
            progress_bar = st.progress(0)
            results = []
            
            for i, file in enumerate(resume_files):
                # Update progress
                progress = (i + 1) / len(resume_files)
                progress_bar.progress(progress)
                
                # Extract text and information
                text = extract_text(file)
                contact_info = extract_contact_info(text)
                skills_info = extract_skills_experience(text)
                
                # Calculate similarity
                similarity_score = advanced_similarity(job_description, text, skills_info)
                
                # Filter by minimum similarity
                if similarity_score >= min_similarity:
                    results.append({
                        "Resume": file.name,
                        "Similarity (%)": round(similarity_score, 2),
                        "Name": contact_info['Name'],
                        "Email": contact_info['Email'],
                        "Phone": contact_info['Phone'],
                        "LinkedIn": contact_info['LinkedIn'],
                        "Skills": skills_info['skills'],
                        "Experience (Years)": skills_info['experience_years'],
                        "Education": skills_info['education'],
                        "Raw Text": text
                    })
            
            # Sort by similarity score
            results.sort(key=lambda x: x['Similarity (%)'], reverse=True)
            top_results = results[:top_n]

        st.success(f"✅ Analysis complete! Processed {len(resume_files)} resumes, found {len(results)} qualified candidates.")

        # ---------------- Display Results ----------------
        if top_results:
            st.subheader("🏆 Top Matching Resumes")
            
            # Score color function
            def get_score_color(score):
                if score >= 80:
                    return "match-score-high"
                elif score >= 60:
                    return "match-score-medium"
                else:
                    return "match-score-low"
            
            for i, result in enumerate(top_results, 1):
                with st.container():
                    score_color = get_score_color(result['Similarity (%)'])
                    
                    st.markdown(f"""
                    <div style='
                        background-color:#E8F0FE;
                        padding:20px;
                        border-radius:10px;
                        margin-bottom:15px;
                        border-left: 5px solid #1F77B4;
                    '>
                        <div style='display: flex; justify-content: space-between; align-items: center;'>
                            <h3 style='color:black; margin:0;'>#{i} {result['Resume']}</h3>
                            <span class='{score_color}'>{result['Similarity (%)']}% Match</span>
                        </div>
                        <p style='color:black; margin:5px 0;'><b>Name:</b> {result['Name']}</p>
                        <p style='color:black; margin:5px 0;'><b>Contact:</b> {result['Email']} | {result['Phone']}</p>
                        <p style='color:black; margin:5px 0;'><b>LinkedIn:</b> {result['LinkedIn']}</p>
                        <p style='color:black; margin:5px 0;'><b>Experience:</b> {result['Experience (Years)']} years | <b>Education:</b> {result['Education']}</p>
                        <p style='color:black; margin:5px 0;'><b>Key Skills:</b> {', '.join(result['Skills'][:8])}</p>
                    </div>
                    """, unsafe_allow_html=True)
                    
                    with st.expander("📄 Resume Preview & Analysis"):
                        col1, col2 = st.columns([2, 1])
                        
                        with col1:
                            highlighted_text = highlight_keywords(result['Raw Text'], result['Skills'], job_description)
                            st.markdown(f"<div style='background-color: white; padding: 15px; border-radius: 5px; max-height: 300px; overflow-y: auto;'>{highlighted_text}</div>", unsafe_allow_html=True)
                        
                        with col2:
                            st.write("**Skills Analysis:**")
                            for skill in result['Skills'][:10]:
                                st.write(f"✅ {skill}")
            
            # ---------------- Analytics Dashboard ----------------
            if show_analytics:
                create_analytics_dashboard(results, job_description)
            
            # ---------------- Enhanced Download Options ----------------
            st.subheader("📥 Export Results")
            
            col1, col2, col3 = st.columns(3)
            
            with col1:
                # Excel Report
                df = pd.DataFrame(top_results)
                excel_buffer = BytesIO()
                df.to_excel(excel_buffer, index=False)
                
                st.download_button(
                    "💾 Download Excel Report",
                    data=excel_buffer,
                    file_name=f"resume_analysis_{datetime.now().strftime('%Y%m%d_%H%M')}.xlsx",
                    mime="application/vnd.openxmlformats-officedocument.spreadsheetml.sheet"
                )
            
            with col2:
                # CSV Report
                csv_buffer = BytesIO()
                df.to_csv(csv_buffer, index=False)
                
                st.download_button(
                    "📊 Download CSV Report",
                    data=csv_buffer,
                    file_name=f"resume_analysis_{datetime.now().strftime('%Y%m%d_%H%M')}.csv",
                    mime="text/csv"
                )
            
            with col3:
                # Summary Report
                summary = f"""
                Resume Analysis Report
                Generated: {datetime.now().strftime('%Y-%m-%d %H:%M')}
                Total Resumes: {len(resume_files)}
                Qualified Candidates: {len(results)}
                Top Candidate: {top_results[0]['Name']} ({top_results[0]['Similarity (%)']}%)
                Average Score: {sum([r['Similarity (%)'] for r in results]) / len(results):.1f}%
                """
                
                st.download_button(
                    "📄 Download Summary",
                    data=summary,
                    file_name=f"analysis_summary_{datetime.now().strftime('%Y%m%d_%H%M')}.txt",
                    mime="text/plain"
                )
        else:
            st.warning("No resumes met the minimum similarity criteria. Try adjusting the filters or job description.")

# ---------------- Footer ----------------
st.markdown("---")
st.markdown("""
<div style='text-align: center; color: gray;'>
    <p>© 2025 Md Junayed Bin Karim | Resume Matching Pro v2.0</p>
    <p style='font-size: 0.8rem;'>Advanced AI-powered resume screening solution</p>
</div>
""", unsafe_allow_html=True)