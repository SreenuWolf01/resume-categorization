# Resume Categorization and Sorting with NLP 📄✨

## Project Overview

This project implements a Natural Language Processing (NLP) pipeline to categorize and sort resumes. The objective is to efficiently process a collection of resumes and classify them into predefined categories (e.g., based on skills, roles, or departments) using text-based features and machine learning algorithms.

## Problem Statement

In recruitment, manually sorting through a large volume of resumes can be time-consuming and inefficient. This project addresses the challenge of automating resume categorization to streamline the hiring process, enabling faster and more accurate matching of candidates to job requirements.

## Data

The project utilizes a dataset of resumes, likely in a textual format.

* **Source:** The notebook indicates data loading from a Google Drive path: `/content/drive/MyDrive/resume_sorting/gpt_dataset.csv`. This suggests a custom or privately sourced dataset.
* **Description:** The dataset presumably contains resume text and corresponding labels/categories.
* **Key Features (Inferred):** Raw resume text, which undergoes preprocessing to extract features relevant for classification.

## Methodology

The project employs a standard NLP and machine learning pipeline:

### 1. Data Preprocessing

* **Text Cleaning:** Removal of punctuation, stopwords (`nltk.corpus.stopwords`), and conversion to lowercase.
* **Tokenization & Lemmatization/Stemming:** Breaking text into words and reducing them to their base forms (e.g., using `nltk`).
* **Word Cloud Generation:** Visualization of common terms in the cleaned resume data to gain initial insights (`wordcloud`).

### 2. Feature Extraction

* **TF-IDF Vectorization:** Transforms the cleaned text data into numerical feature vectors using `TfidfVectorizer`. TF-IDF (Term Frequency-Inverse Document Frequency) quantifies the importance of words in a document relative to a corpus.

### 3. Model Training

* **Model Selection:** The project explores two common supervised machine learning classification algorithms:
    * **Logistic Regression:** A linear model often used for binary or multi-class classification.
    * **Random Forest Classifier:** An ensemble learning method that builds multiple decision trees and merges their predictions for improved accuracy and robustness.
* **Data Splitting:** The dataset is split into training and testing sets (`train_test_split`) to evaluate model performance on unseen data.

### 4. Model Evaluation

* **Metrics:** Model performance is assessed using standard classification metrics:
    * `accuracy_score`: Measures the proportion of correctly classified instances.
    * `classification_report`: Provides precision, recall, f1-score, and support for each class.

## Expected Outcomes / Results

The project aims to demonstrate:

* Effective text preprocessing techniques for unstructured resume data.
* The application of TF-IDF for converting text into meaningful numerical features.
* The ability of Logistic Regression and Random Forest models to classify resumes into relevant categories with reasonable accuracy.
* Insights into common keywords and phrases found in the resume dataset.

## Tools and Technologies

* **Language:** Python
* **Libraries:**
    * `numpy`: Fundamental package for numerical computing.
    * `pandas`: For data manipulation and analysis.
    * `nltk`: Natural Language Toolkit for text processing (stopwords, tokenization, lemmatization).
    * `wordcloud`: For generating visual representations of word frequency.
    * `matplotlib.pyplot`: For data visualization.
    * `sklearn.feature_extraction.text.TfidfVectorizer`: For text feature extraction.
    * `sklearn.model_selection.train_test_split`: For splitting data.
    * `sklearn.linear_model.LogisticRegression`: For logistic regression modeling.
    * `sklearn.ensemble.RandomForestClassifier`: For random forest modeling.
    * `sklearn.metrics.accuracy_score`, `sklearn.metrics.classification_report`: For model evaluation.
* **Environment:** Jupyter Notebook (for interactive development and presentation).
* **Data Source Interaction:** Google Colab / Google Drive for data storage and access.

## Files in this Repository

* `Resume_sorting.ipynb`: The Jupyter Notebook containing the complete Python code for data loading, preprocessing, model training, and evaluation.
* `README.md`: This file, providing an overview of the project.
* **(Important)** `gpt_dataset.csv`: **(You should ideally upload this dataset or provide clear instructions/a link to it, as the notebook directly references it from Google Drive.)**
