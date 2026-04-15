import streamlit as st
from utils import extract_text, compute_similarity, rank_resumes

st.set_page_config(page_title="Resume Ranker", layout="wide")

# ---- Header ----
st.title("📄 Resume Ranker")
st.markdown("Upload resumes and rank them based on a job description using BERT")

st.divider()

# ---- Job Description ----
st.subheader("📌 Step 1: Provide Job Description")

jd_text = st.text_area("Paste Job Description", height=200)

jd_file = st.file_uploader("Or upload Job Description (.txt)", type=["txt"])

# ---- Resume Upload ----
st.subheader("📂 Step 2: Upload Resumes")

resume_files = st.file_uploader(
    "Upload multiple resumes (PDF/TXT)",
    type=["pdf", "txt"],
    accept_multiple_files=True
)

st.divider()

# ---- Run Button ----
if st.button("🚀 Rank Resumes"):

    # ---- Validation ----
    if not jd_text and not jd_file:
        st.error("⚠️ Please provide a job description")
        st.stop()

    if not resume_files:
        st.error("⚠️ Please upload at least one resume")
        st.stop()

    # ---- Processing ----
    with st.spinner("Analyzing resumes... ⏳"):

        # Job Description
        if jd_file:
            job_desc = jd_file.read().decode("utf-8")
        else:
            job_desc = jd_text

        resumes_text = []
        resume_names = []

        for file in resume_files:
            text = extract_text(file)

            if not text.strip():
                st.warning(f"⚠️ Could not read {file.name}")
                continue

            resumes_text.append(text)
            resume_names.append(file.name)

        if not resumes_text:
            st.error("❌ No valid resumes found")
            st.stop()

        scores = compute_similarity(job_desc, resumes_text)
        results = rank_resumes(resume_names, scores)

    # ---- Results ----
    st.success("✅ Ranking Complete!")

    st.subheader("🏆 Results")

    for i, res in enumerate(results):
        with st.container():
            col1, col2 = st.columns([4, 1])

            with col1:
                st.markdown(f"**{i+1}. {res['Resume']}**")

            with col2:
                st.metric("Score", f"{res['Score']}%")

            st.progress(float(res["Score"]) / 100)