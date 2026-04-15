# 📄 Resume Ranker using BERT & Cosine Similarity

An AI-powered web application that ranks resumes based on their relevance to a given job description using Natural Language Processing (NLP) techniques.

---

## 🚀 Overview

The Resume Ranker is designed to automate the resume screening process by comparing multiple resumes against a job description. It leverages semantic similarity using BERT embeddings and compares it with a traditional TF-IDF cosine similarity approach.

This project helps recruiters quickly identify the most relevant candidates, reducing manual effort and improving decision-making.

---

## 🧠 Features

* Upload multiple resumes (PDF/TXT)
* Paste or upload a job description
* Semantic similarity using BERT (`all-MiniLM-L6-v2`)
* Baseline comparison using TF-IDF + cosine similarity
* Resume ranking with scores (0–100)
* Clean and interactive UI using Streamlit
* Evaluation metrics:

  * Accuracy
  * Precision
  * Recall
  * F1 Score

---

## 🏗️ Tech Stack

* **Frontend/UI:** Streamlit
* **Backend:** Python
* **ML/NLP:** Sentence Transformers (BERT)
* **Libraries:**

  * scikit-learn
  * PyPDF2
  * numpy

---

## 📂 Project Structure

```
resume-ranker/
│
├── app.py                  # Streamlit UI
├── utils.py                # Text extraction & similarity logic
├── eval.py           # BERT evaluation metrics
├── requirements.txt        # Dependencies
└── README.md
```

---

## ⚙️ Installation

1. Clone the repository:

```
git clone <your-repo-link>
cd resume-ranker
```

2. Create virtual environment (recommended):

```
python -m venv venv
venv\Scripts\activate   # Windows
```

3. Install dependencies:

```
pip install -r requirements.txt
```

---

## ▶️ Usage

Run the Streamlit app:

```
streamlit run app.py
```

Then:

1. Paste or upload a job description
2. Upload multiple resumes
3. Click **"Rank Resumes"**
4. View ranked results with scores

---

## 📊 Model Evaluation

### BERT Model

```
python eval.py
```

---

## 🔍 How It Works

1. Extract text from resumes (PDF/TXT)
2. Convert text into embeddings using BERT
3. Compute cosine similarity with job description
4. Rank resumes based on similarity scores

---

## 📈 Results

* BERT model outperforms TF-IDF in:

  * Semantic understanding
  * Context-based matching
* Provides more accurate ranking of resumes

---

## 🎯 Applications

* Automated resume screening
* Recruitment systems
* Job matching platforms
* HR analytics tools

---

## 🔮 Future Scope

* Resume vs Job Description highlighting
* Skill extraction & recommendation
* Integration with job portals
* Real-time deployment (Streamlit Cloud)
* Advanced ranking using fine-tuned models

---
