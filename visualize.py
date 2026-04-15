import matplotlib.pyplot as plt
from sklearn.metrics import precision_score, recall_score, confusion_matrix
import numpy as np

from utils import compute_similarity

# ---- Sample Data (replace with your own later) ----
job_desc = "Looking for Python developer with ML and NLP experience"

resumes = [
    "Python developer with machine learning",
    "Frontend developer with React",
    "Data scientist with NLP and APIs",
    "Java backend engineer"
]

true_labels = [1, 0, 1, 0]

# ---- Compute Scores ----
scores = compute_similarity(job_desc, resumes)

# ---- Try multiple thresholds ----
thresholds = np.linspace(0.1, 0.9, 9)

precisions = []
recalls = []

for t in thresholds:
    preds = [1 if s >= t else 0 for s in scores]
    
    precisions.append(precision_score(true_labels, preds))
    recalls.append(recall_score(true_labels, preds))

# ---- Plot Precision vs Recall ----
plt.figure()
plt.plot(recalls, precisions, marker='o')
plt.xlabel("Recall")
plt.ylabel("Precision")
plt.title("Precision vs Recall")
plt.show()


# ---- Confusion Matrix (for one threshold) ----
threshold = 0.5
preds = [1 if s >= threshold else 0 for s in scores]

cm = confusion_matrix(true_labels, preds)

plt.figure()
plt.imshow(cm)
plt.title("Confusion Matrix")
plt.xlabel("Predicted")
plt.ylabel("Actual")

for i in range(len(cm)):
    for j in range(len(cm)):
        plt.text(j, i, cm[i][j], ha='center', va='center')

plt.show()


# ---- Score Distribution ----
plt.figure()
plt.hist(scores, bins=5)
plt.xlabel("Similarity Score")
plt.ylabel("Frequency")
plt.title("Score Distribution")
plt.show()