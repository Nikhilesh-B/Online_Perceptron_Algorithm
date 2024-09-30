# Restaurant Reviews Classification

This project aims to classify restaurant reviews from Pittsburgh as either positive (rating of 4 or 5) or negative (rating below 4). The dataset consists of training (`reviews_tr.csv`) and test data (`reviews_te.csv`), with each review preprocessed to remove non-alphanumeric symbols and capitalization. The first column of each file is the label (0 or 1), and the second column is the review text.

## Data Representations

Three different representations of the review data are explored:

1. **Unigram Representation**:
    - This representation uses term frequency (tf), where each word in a review becomes a feature. The value of each feature is the number of times the word appears in the review.
2. **TF-IDF Representation**:
    - A refinement of the unigram representation, where each word's frequency is weighted by its inverse document frequency (IDF) across the training data. This reduces the influence of common words and highlights rarer, more informative terms.
3. **Bigram Representation**:
    - This representation considers pairs of consecutive words (bigrams) as features, counting their frequency in each review to capture more contextual information.

## Classifier: Online Perceptron with Online-to-Batch Conversion

The Online Perceptron algorithm is implemented with two passes through the training data, using random shuffling between passes. The final classifier is an average of the last `n + 1` weight vectors, where `n` is the number of training examples. The Perceptron algorithm updates the weight vector only when a prediction mistake occurs.

## Usage

1. **Dependencies**:
    - Python 3.x
    - Required libraries: `numpy`, `pandas`, `scikit-learn`, `matplotlib`
2. **Running the Perceptron Algorithm**:
    - Run the script `perceptron.py` to train the Perceptron algorithm on the training data and evaluate its performance on the test data. The script allows for selecting different data representations (unigram, tf-idf, or bigram) using command-line arguments.
3. **Performance Evaluation**:
    - The model's performance is evaluated based on classification accuracy, precision, recall, and F1-score. Performance graphs are provided to compare the effectiveness of unigram, bigram, and tf-idf representations.

## Key Results

- **Unigram Representation**: Fast and simple, but suffers from high dimensionality and sparse data issues. Common words may dominate.
- **TF-IDF Representation**: Generally performs better by giving more weight to rare, informative words and mitigating the impact of frequent words.
- **Bigram Representation**: Provides more contextual information but increases the feature space and may overfit on smaller datasets.

## Files

- `reviews_tr.csv` - Training data (labels and review text).
- `reviews_te.csv` - Test data (labels and review text).
- `perceptron.py` - Python script for implementing the Online Perceptron algorithm with online-to-batch conversion
