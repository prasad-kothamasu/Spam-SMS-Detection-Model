

# Spam SMS Detection | Machine Learning Project

This project builds a machine learning system to classify SMS messages as **Spam** or **Not Spam**. It applies natural language processing techniques and multiple classifiers to achieve high accuracy, helping users filter unwanted messages effectively.

---

## Objective

- Preprocess raw SMS text data.
- Transform text into numerical features using **TF-IDF Vectorization**.
- Train and evaluate multiple models:
  - Logistic Regression
  - Multinomial Naive Bayes
  - Support Vector Machine (SVM)
- Save trained models for future predictions.

---

## Dataset

- [SMS Spam Collection Dataset - Kaggle](https://www.kaggle.com/datasets/uciml/sms-spam-collection-dataset?resource=download)
- Contains 5,500+ labeled SMS messages.

---

## Technologies Used

- Python
- Google Colab
- Pandas, NumPy
- NLTK (Natural Language Toolkit)
- Scikit-learn
- Matplotlib, Seaborn (for visualization)
- Pickle (for saving models)

---

## Project Workflow

1. **Loaded Data**  
   Uploaded the SMS dataset into Colab for processing.

2. **Data Preprocessing**  
   - Lowercasing text  
   - Removing punctuation and stopwords  

3. **Feature Extraction**  
   - Applied **TF-IDF Vectorization** to convert text into numerical vectors.

4. **Training The Models**  
   - Trained and compared **Logistic Regression**, **Naive Bayes**, and **SVM** models.

5. **Model Evaluation**  
   - Evaluated using **accuracy score**, **confusion matrix**, and **classification report**.
   - Cross-validation performed for robustness.

6. **Saving Models**  
   - Saved the best model and vectorizer as `.pkl` files.

7. **Predict Function**  
   - Added a custom function to predict if a user-input SMS is spam or not.

---

## Results

| Model | Accuracy |
|:-----|:---------|
| Logistic Regression | ~97% |
| Naive Bayes | ~96% |
| SVM | ~97% |

- Logistic Regression chosen as final model due to simplicity and strong performance.
- Cross-validation confirmed model stability.

---


## How to Run the Project

1. Upload the `spam.csv` dataset.
2. Run all notebook cells sequentially.
3. Test the `predict_message()` function with your own SMS text.
4. Models and vectorizer will be saved and downloadable.

---

## Author

- **Name:** *Kothamasu Venkata Naga Jaya Prasad*  
- **Email:** *prasadkothamasudup@gmail.com*

---



