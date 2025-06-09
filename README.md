# SENTIMENT ANALYSIS using Random Forest Algorithm

## 🔎Overview:
This project is focused on building a machine learning model that can classify the sentiment of a given piece of text as Positive, Negative, or Neutral. It uses a Random Forest Classifier, a robust ensemble learning method, to perform the classification. The project includes all essential steps of the machine learning pipeline — from data preprocessing to model evaluation and prediction.

The model is trained on a labeled dataset of text samples, and the input sentences are first cleaned and transformed into numerical features using TF-IDF vectorization. The trained Random Forest model is then able to predict the sentiment of new, unseen text inputs.

📌Language: Python<br>
💻Environment: Jupyter Notebook<br>
📝Algorithm Used: Random Forest Classifier<br>
📚Libraries: pandas, numpy, scikit-learn, nltk, matplotlib, seaborn, re, nltk

## 📊Dataset
The model is trained on the dataset train_csv with:<br>
•Format: CSV with two columns: text and label<br>
•Labels: Positive, Negative, Neutral

## 🧩Features:
1. 🔤 Text Preprocessing Pipeline<br>
 • Automatic cleaning of input text<br>
 • Lowercasing, punctuation and number removal<br>
 • Tokenization and stopword removal<br>
 • Stemming using NLTK's PorterStemmer<br>

2. 📚 Feature Extraction with TF-IDF<br>
 • Converts text into numerical features using TF-IDF (Term Frequency–Inverse Document Frequency)<br>
 • Handles sparse, high-dimensional data effectively<br>

3. 🧠 Machine Learning with Random Forest<br>
 • Uses Random Forest Classifier (ensemble-based model) for robust sentiment classification<br>
 • Handles non-linear relationships and reduces overfitting<br>

4. 🎯 Model Evaluation<br>
 • Performance metrics include Accuracy, Confusion Matrix, Precision, Recall, and F1-Score<br>
 • Visualizations (optional) of confusion matrix and class distribution<br>

5. 💬 Real-Time Sentiment Prediction<br>
 • Interactive input allows users to enter custom text in the notebook<br>
 • Model instantly predicts whether the sentiment is Positive, Negative, or Neutral<br>

 ## 🔄Project Workflow
The end-to-end workflow of the sentiment analysis project includes the following stages:<br>

1. 📥 Data Collection<br>
 • Load the labeled dataset (text + sentiment labels)<br>
 • Example format: CSV file with columns like Text and Sentiment<br>

2. 🧹 Text Preprocessing<br>
 • Remove punctuation, numbers, and special characters<br>
 • Convert text to lowercase<br>
 • Tokenize text into individual words<br>
 • Remove stopwords using NLTK<br>
 • Apply stemming using PorterStemmer<br>

3. 📐 Feature Extraction<br>
 • Convert the cleaned text into numerical vectors using TF-IDF Vectorizer<br>
 • Each text is now represented as a sparse matrix suitable for ML algorithms<br>
 
 ## 🌲Model Training:<br>
   • Random Forest algorithm is used for classification purpose<br>
   • Splitting of dataset done on 70:30 basis(70% training,30% testing)<br>
   • Train the model using the training data<br>
   • Predict on the test set<br>
   • Build multiple decision trees and average the results for final prediction<br>

  ## 📈Evaluation Metrics:<br>
  • Accuracy<br>
  • Precision<br>
  • Recall<br>
  • f1-score<br>
  • Confusion Matrix<br>

  ## 💬How to Use:<br>
• Clone the repository and install the dependencies:<br>

git clone https://github.com/debaratighosh/SENTIMENT_ANALYSIS.git<br>
cd SENTIMENT_ANALYSIS<br>
pip install -r requirements.txt<br>

• Open the notebook and run all cells:<br>
jupyter notebook Sentiment_Analysis.ipynb<br>
• Enter your own sentence in the input prompt at the bottom cell to get sentiment prediction<br>

## 🚀 Sample Prediction:<br>

Enter a sentence: I love it<br>
Predicted Sentiment: Positive
