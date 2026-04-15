from utils import compute_similarity
from sklearn.metrics import accuracy_score, precision_score, recall_score, f1_score

def evaluate_model(job_desc, resumes, true_labels, threshold=0.5):
    """
    Evaluate BERT-based resume ranking model
    """

    # ---- Step 1: Compute similarity scores ----
    scores = compute_similarity(job_desc, resumes)

    # ---- Step 2: Convert scores to binary predictions ----
    predictions = [1 if score >= threshold else 0 for score in scores]

    # ---- Step 3: Compute metrics ----
    accuracy = accuracy_score(true_labels, predictions)
    precision = precision_score(true_labels, predictions, zero_division=0)
    recall = recall_score(true_labels, predictions, zero_division=0)
    f1 = f1_score(true_labels, predictions, zero_division=0)

    # ---- Step 4: Print results ----
    print("\n📊 ----- Evaluation Results (BERT Model) -----\n")

    for i in range(len(resumes)):
        print(f"Resume {i+1}:")
        print(f"Score      : {scores[i]:.4f}")
        print(f"Prediction : {predictions[i]}")
        print(f"Actual     : {true_labels[i]}")
        print("-" * 30)

    print("\n📈 Metrics:")
    print(f"Accuracy  : {accuracy:.2f}")
    print(f"Precision : {precision:.2f}")
    print(f"Recall    : {recall:.2f}")
    print(f"F1 Score  : {f1:.2f}")

    return {
        "scores": scores,
        "predictions": predictions,
        "accuracy": accuracy,
        "precision": precision,
        "recall": recall,
        "f1": f1
    }


# ---- Example Usage ----
if __name__ == "__main__":

    job_desc = """
    Looking for a Python developer with machine learning,
    NLP, and API development experience.
    """

    resumes = [
        "Python developer with NLP and ML experience",
        "Frontend developer with React and CSS",
        "Data scientist with machine learning and APIs",
        "Java backend engineer"
    ]

    true_labels = [1, 0, 1, 0]

    evaluate_model(job_desc, resumes, true_labels, threshold=0.5)