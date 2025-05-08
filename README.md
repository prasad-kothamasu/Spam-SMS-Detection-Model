# Spam SMS Detection Model | Machine Learning

This project builds a machine learning system to classify SMS messages as **Spam** or **Not Spam** using Natural Language Processing. 
It trains a Logistic Regression model on preprocessed text and deploys it using a Streamlit web interface.

---

## Objective

- Clean and preprocess SMS message text
- Convert text into numerical features using **TF-IDF Vectorization**
- Train a **Logistic Regression** model to classify messages
- Save the model and vectorizer using `pickle`
- Build an interactive UI using **Streamlit** for predictions

---

## Dataset

- **Source:** [SMS Spam Collection Dataset – Kaggle](https://www.kaggle.com/datasets/uciml/sms-spam-collection-dataset?resource=download)
- **Size:** ~5,500 labeled SMS messages
- **Labels:** `spam` or `ham` (not spam)

---

## Technologies Used

- **Python**
- **Google Colab** – Model training
- **Scikit-learn** – ML model
- **NLTK** – Text preprocessing
- **TF-IDF** – Feature extraction
- **Pickle** – Model saving
- **Streamlit** – Web app frontend
- **Pandas, NumPy**

---

## Project Workflow

1. **Data Loading**
   - Loaded and read `spam.csv` in Colab

2. **Text Preprocessing**
   - Lowercasing
   - Removing punctuation
   - Removing stopwords (using NLTK)

3. **Feature Extraction**
   - Applied **TF-IDF Vectorization** to convert text to vectors

4. **Model Training**
   - Trained a **Logistic Regression** model for binary classification

5. **Model Evaluation**
   - Evaluated with **accuracy**, **confusion matrix**, and **classification report**
   - Cross-validation used for robustness

6. **Saving for Inference**
   - Saved:
     - `spam_model.pkl` (trained model)
     - `vectorizer.pkl` (TF-IDF transformer)

7. **Frontend Integration**
   - Built a **Streamlit app** to let users test SMS inputs interactively

---

## Results

| Model Used          | Accuracy |
|---------------------|----------|
| Logistic Regression | ~97%     |

Logistic Regression was selected due to its strong performance, simplicity, and real-time prediction speed.

---

## How to Run

### 1. Open Command Prompt (CMD)

You can do this by pressing `Win + R`, typing `cmd`, and hitting Enter.

---

### 2. Navigate to Your Project Folder
Use the `cd` command to change to your project directory. Example:
```bash
cd "C:\Users\YourName\Desktop\spam_sms_project" 

3.Create Virtual Environment
python -m venv spamSmsDetector

4. Activate Virtual Environment
spamSmsDetector\Scripts\activate

5. Install All Required Packages
pip install -r requirements.txt
Note : Run it only for the first time of execution

6.Run the streamlit App
streamlit run app.py

Your app will open in the browser at http://localhost:8501

7.To Stop The App
Press Ctrl + C in the terminal

8.To Deactivate the Environment
deactivate

Outputs:
## Streamlit Interface:
![Streamlit interface](https://github.com/user-attachments/assets/5ebfdd7b-b18f-43f3-9eb8-1219c69921f4)

----

![Safe message output](https://github.com/user-attachments/assets/05c1d54c-a6d2-4ef6-81fa-45e5c3cab6ba)

----

![Spam message output](https://github.com/user-attachments/assets/8f9f7838-d4b7-4ba2-8259-929cab54b9fb)

