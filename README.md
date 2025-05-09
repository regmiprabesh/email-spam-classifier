# Email Spam Filter

A machine learning project that uses Naive Bayes classification to identify spam emails with high accuracy.

## Overview

This project implements a spam email detection system using the Multinomial Naive Bayes algorithm. The model analyzes the content of emails and classifies them as either spam (unwanted, possibly malicious emails) or ham (legitimate emails).

## Features

- **High Accuracy**: Achieves approximately 98% accuracy in spam detection
- **Efficient Processing**: Uses CountVectorizer for text processing
- **Simple Implementation**: Leverages scikit-learn's Naive Bayes implementation
- **Ready for Use**: Pre-trained model ready for classifying new emails

## How It Works

The system works using the following steps:

1. **Data Processing**: Emails are processed and converted into numerical features using CountVectorizer
2. **Classification**: A Multinomial Naive Bayes classifier analyzes the word frequencies
3. **Prediction**: New emails are classified as spam (1) or ham (0)

## Mathematical Basis

The project is based on Bayes' Theorem:

P(A|B) = (P(B|A) * P(A)) / P(B)

Where in our context:
- P(A|B) is the probability of an email being spam given its content
- P(B|A) is the probability of specific content appearing in spam emails
- P(A) is the prior probability of any email being spam
- P(B) is the probability of the specific content appearing in any email

## Requirements

- Python 3.x
- pandas
- numpy
- scikit-learn

## Usage

1. Clone the repository
2. Install the required packages
3. Run the Jupyter notebook

```python
# Example usage
from sklearn.feature_extraction.text import CountVectorizer
from sklearn.naive_bayes import MultinomialNB

# Load the pre-trained model and vectorizer
# (assuming you've saved them using joblib or pickle)

# Classify a new email
email = ["Hello, are we still meeting tomorrow?"]
email_count = vectorizer.transform(email)
result = model.predict(email_count)

if result[0] == 0:
    print("This is a legitimate email (ham)")
else:
    print("This is spam")
```

## Model Performance

- Training accuracy: ~98%
- Test accuracy: ~99%

## Dataset

The model is trained on a labeled dataset containing:
- 4825 legitimate emails (ham)
- 747 spam emails

## Future Improvements

- Implement TF-IDF vectorization for better feature extraction
- Add more features like email headers and metadata
- Create a web interface for real-time email classification
- Deploy as a service that can integrate with email clients

## Author

Prabesh Regmi

## Acknowledgments

- Thomas Bayes for the probability theorem
- scikit-learn team for the ML implementation tools
