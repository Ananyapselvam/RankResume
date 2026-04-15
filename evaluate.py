from utils import compute_similarity
from sklearn.metrics import accuracy_score, precision_score, recall_score

# ---- Step 1: Define Job Description ----
job_desc = """
Looking for a Python developer with ML, NLP, and API experience.
"""

# ---- Step 2: Add Resume Texts ----
resumes = [
    "Python developer with NLP and ML experience",
    "Frontend developer with React skills",
    "Data scientist with machine learning and APIs"
]

# ---- Step 3: Ground Truth Labels ----
# 1 = relevant, 0 = not relevant
true_labels = [1, 0, 1]

# ---- Step 4: Compute Similarity ----
scores = compute_similarity(job_desc, resumes)

# ---- Step 5: Convert to Predictions ----
# You can tune threshold (important!)
threshold = 0.5

predictions = [1 if score >= threshold else 0 for score in scores]

# ---- Step 6: Metrics ----
accuracy = accuracy_score(true_labels, predictions)
precision = precision_score(true_labels, predictions)
recall = recall_score(true_labels, predictions)

# ---- Step 7: Print Results ----
print("\n--- Model Evaluation ---")
print("Scores:", scores)
print("Predictions:", predictions)
print("Actual:", true_labels)

print(f"\nAccuracy: {accuracy:.2f}")
print(f"Precision: {precision:.2f}")
print(f"Recall: {recall:.2f}")