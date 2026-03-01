# computational-linguistics-toxic-analysis
# Detecting Toxic Comments: Statistical vs. Neural Paradigms
**A Comparative Analysis of Machine Learning and Deep Learning in NLP**

## 📌 Project Overview
This project addresses the challenge of identifying and classifying "toxic" online comments to foster healthier digital environments. Using a subset of the **Jigsaw Toxic Comment Classification** dataset, I conducted a rigorous comparative study between traditional statistical methods and modern neural network architectures.

The project demonstrates a complete end-to-end NLP pipeline: from advanced text preprocessing and vectorization to model evaluation and error analysis.

## 📊 Key Analytical Features
* **Data Exploration (EDA):** Analyzed class imbalance across multiple labels (toxic, severe_toxic, obscene, threat, insult, identity_hate).
* **Text Preprocessing:** Implemented custom pipelines for tokenization, lemmatization, and removing noise (URLs, punctuation, stopwords) specific to social media text.
* **Model Comparison:**
    * **Baseline:** Statistical models using TF-IDF vectorization (e.g., Logistic Regression).
    * **Advanced:** Neural paradigms using word embeddings (Word2Vec/GloVe) and sequence models (LSTM/GRU).
* **Performance Metrics:** Evaluated models using ROC-AUC, F1-Score, and Precision-Recall curves, focusing on performance under high-class imbalance.

## 🛠️ Tech Stack
* **Language:** Python 3.x
* **NLP & ML:** `Scikit-learn`, `NLTK`, `Gensim`
* **Deep Learning:** `TensorFlow/Keras` (or `PyTorch`)
* **Data Science:** `Pandas`, `NumPy`, `Matplotlib`, `Seaborn`

## 📈 Key Insights & Results
* **Statistical Efficiency:** Traditional models provide strong baselines with significantly lower computational costs for clear-cut toxicity.
* **Neural Superiority:** Sequence models excel in capturing context and subtle linguistic nuances, leading to a higher ROC-AUC on multi-label classification.
* **Class Imbalance:** Identified that "Threat" and "Identity Hate" categories require specialized sampling techniques due to extreme data sparsity.

## 📂 File Structure
* `Computational_Linguistics_Final_Project_r1077191.ipynb`: Full implementation and comparative analysis.
* `data/`: Training and testing subsets (CSV).
* `visualizations/`: Exported performance charts and confusion matrices.
* 
## 📂 Data Source
The dataset used in this project is the **Jigsaw Toxic Comment Classification** dataset. 
Due to GitHub's file size limitations, the raw data is not included in this repository. 
You can download it from Kaggle here: [Link to Kaggle Dataset].


**Author:** Liangyu Gan  
