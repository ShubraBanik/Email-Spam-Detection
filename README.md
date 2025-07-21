# Spam Email Classification

This project implements a spam email classifier using **Logistic Regression (TF-IDF)** and **Multinomial Naïve Bayes (Bag-of-Words)** models. It compares the performance of both models on the same dataset and provides metrics such as accuracy, precision, recall, F1-score, and confusion matrices.

---

## 📌 Dataset Description
- **Source:** `mail_data.csv`
- **Total Rows:** 5572  
- **Class Distribution:**  
  - **Ham:** 4825  
  - **Spam:** 747  
- **Null Handling:** Replaced null/NaN text fields with empty strings.

---

## ⚙️ Project Workflow
1. **Data Preprocessing:** Text cleaning, tokenization, stopword removal, and feature extraction (TF-IDF / CountVectorizer).
2. **Model Training:**  
   - Logistic Regression (TF-IDF features)  
   - Multinomial Naïve Bayes (CountVectorizer features)
3. **Evaluation:** Accuracy, Precision, Recall, F1-score, Confusion Matrices.
4. **Comparison:** Observed performance difference and trade-offs.

---

## 📊 Evaluation Metrics

### Logistic Regression (TF-IDF)
- **Training Accuracy:** 0.8769
- **Test Accuracy:** 0.9686
- **Precision (Spam):** 1.00
- **Recall (Spam):** 0.76
- **F1-Score (Spam):** 0.86
- **Confusion Matrix:**

### Multinomial Naïve Bayes (Counts)
- **Training Accuracy:** 0.9949
- **Test Accuracy:** 0.9612
- **Precision (Spam):** 0.99
- **Recall (Spam):** 0.81
- **F1-Score (Spam):** 0.86
- **Confusion Matrix:**

  ## 🛠️ Technologies Used
- Python
- Scikit-learn
- Pandas
- Numpy
- Matplotlib / Seaborn
- NLTK

