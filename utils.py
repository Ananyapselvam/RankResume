import PyPDF2
from sentence_transformers import SentenceTransformer
from sklearn.metrics.pairwise import cosine_similarity

# Load model once (important for performance)
model = SentenceTransformer('all-MiniLM-L6-v2')


def extract_text_from_pdf(file):
    text = ""
    try:
        reader = PyPDF2.PdfReader(file)
        for page in reader.pages:
            text += page.extract_text() or ""
    except Exception as e:
        print("Error reading PDF:", e)
    return text


def extract_text(file):
    if file.name.endswith(".pdf"):
        return extract_text_from_pdf(file)
    elif file.name.endswith(".txt"):
        return file.read().decode("utf-8")
    else:
        return ""


def compute_similarity(job_desc, resumes):
    texts = [job_desc] + resumes
    
    embeddings = model.encode(texts)
    
    job_embedding = embeddings[0]
    resume_embeddings = embeddings[1:]
    
    scores = cosine_similarity([job_embedding], resume_embeddings)[0]
    
    return scores


def rank_resumes(resume_names, scores):
    results = []
    
    for name, score in zip(resume_names, scores):
        results.append({
            "Resume": name,
            "Score": round(score * 100, 2)
        })
    
    # Sort descending
    results = sorted(results, key=lambda x: x["Score"], reverse=True)
    
    return results