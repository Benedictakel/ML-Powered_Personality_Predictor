# 🧠 ML-Powered Personality Predictor

This repository contains a **machine learning-based text classification system** that predicts a person’s **MBTI personality type** from their writing samples.



## 📑 Table of Contents

* [Introduction](#introduction)
* [Dataset](#dataset)
* [Objectives](#objectives)
* [Technologies Used](#technologies-used)
* [Installation](#installation)
* [Usage](#usage)
* [Model Overview](#model-overview)
* [Evaluation Metrics](#evaluation-metrics)
* [Sample Predictions](#sample-predictions)
* [Project Structure](#project-structure)
* [Future Work](#future-work)
* [Contributing](#contributing)
* [License](#license)
* [Contact](#contact)



## 📝 Introduction

The **Myers-Briggs Type Indicator (MBTI)** categorizes people into 16 personality types based on preferences in four dimensions:

1. **Introversion (I) / Extraversion (E)**
2. **Intuition (N) / Sensing (S)**
3. **Thinking (T) / Feeling (F)**
4. **Judging (J) / Perceiving (P)**

This project uses **Natural Language Processing (NLP)** and **machine learning algorithms** to analyze textual data and predict a person’s MBTI type.



## 📚 Dataset

* **Source:** [Kaggle - MBTI Personality Dataset]()
* **Data:**

  * \~8,600 posts labeled with MBTI types
  * Each sample includes multiple posts from a single user
* **Classes:** 16 MBTI personality types (e.g., INTP, ENFJ, ISTJ)



## 🎯 Objectives

✔️ Preprocess and clean user posts text data

✔️ Convert text to numerical features using NLP vectorization techniques

✔️ Train a classifier model to predict MBTI personality type

✔️ Evaluate model performance with suitable metrics

✔️ *(Optional)* Predict on custom user inputs in a web interface



## ✨ Features

* **Text cleaning and preprocessing:** Lowercasing, stopword removal, lemmatization
* **Vectorization:** TF-IDF
* **Classification models:** Multinomial Naive Bayes, Logistic Regression, Random Forest
* **Evaluation metrics:** Accuracy, precision, recall, F1-score
*  Web interface for personality prediction from user input



## 🛠️ Technologies Used

* **Python 3**
* **Pandas & NumPy**
* **Scikit-learn**
* **NLTK / spaCy**
* **Matplotlib / Seaborn** *(visualizations)*
* *(Optional)* Flask / Streamlit *(web app deployment)*



## ⚙️ Installation

1. **Clone the repository**

```bash
git clone https://github.com/yourusername/ML-Powered_Personality_Predictor.git
cd ML-Powered_Personality_Predictor
```

2. **Create and activate a virtual environment (optional)**

```bash
python -m venv venv
source venv/bin/activate  # macOS/Linux
venv\Scripts\activate     # Windows
```

3. **Install dependencies**

```bash
pip install -r requirements.txt
```



## ▶️ Usage

### 🧪 Train the Model

```bash
python train_model.py
```

### 🔍 Predict Personality Type from Text

```bash
python predict_personality.py --text "I love working on creative projects and exploring new ideas."
```





## 🧠 Model Overview

### 💻 Text Preprocessing

* Lowercasing
* Removing URLs, punctuation, and stopwords
* Lemmatization using **spaCy**

### 🔢 Feature Extraction

* **TF-IDF Vectorizer** for converting text data to numerical features

### 🔮 Classification Models

* Multinomial Naive Bayes (baseline)
* Logistic Regression
* Random Forest

### ⚖️ Model Selection

* Train-test split (80-20)
* Cross-validation for hyperparameter tuning



## 📊 Evaluation Metrics

| Model               | Accuracy | Precision | Recall | F1-Score |
| ------------------- | -------- | --------- | ------ | -------- |
| Naive Bayes         | 65%      | 64%       | 63%    | 63.5%    |
| Logistic Regression | 68%      | 67%       | 66%    | 66.5%    |
| Random Forest       | 66%      | 65%       | 64%    | 64.5%    |





## 🔍 Sample Predictions

```plaintext
Input Text: "I love exploring theories, reading philosophy, and coding new ML models."
Predicted MBTI Type: INTP
Confidence: 88.5%
```



## 📁 Project Structure

```
ML-Powered_Personality_Predictor/
 ┣ data/
 ┃ ┗ mbti_1.csv
 ┣ models/
 ┃ ┗ personality_classifier.pkl
 ┣ src/
 ┃ ┣ preprocess.py
 ┃ ┣ train_model.py
 ┃ ┗ predict_personality.py
 ┣ app.py  # (optional web app)
 ┣ requirements.txt
 ┗ README.md
```



## 💡 Future Work

* Implement **transformer-based models (BERT)** for improved accuracy
* Build a **Streamlit web app** for interactive predictions
* Extend to **Big Five (OCEAN) personality traits prediction**
* Analyze **feature importance** to interpret text patterns linked to MBTI types



## 🤝 Contributing

Contributions are welcome to enhance model performance, add deployment features, or expand dataset analysis.

1. Fork the repository
2. Create your feature branch (`git checkout -b feature/YourFeature`)
3. Commit your changes (`git commit -m 'Add YourFeature'`)
4. Push to your branch (`git push origin feature/YourFeature`)
5. Open a pull request



## 📄 License

This project is licensed under the **MIT License**. See [LICENSE](LICENSE) for full details.



## 📬 Contact

**Ugama Benedicta Kelechi**
[LinkedIn](www.linkedin.com/in/ugama-benedicta-kelechi-codergirl-103041300) | [Email](mailto:ugamakelechi501@gmail.com) | [Portfolio](#)



### ⭐ If you find this project insightful, please give it a star and share with NLP and psychology AI enthusiasts.


