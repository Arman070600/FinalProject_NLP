# Sentiment Analysis Project

## Overview

This project performs sentiment analysis on given train and test data. It has a complete workflow from data exploration and preprocessing to model training and inference.

## Data Science (DS) Part Report

### Conclusions from EDA

- The dataset is balanced with an equal number of positive and negative reviews, minimizing bias in model training.
- Review lengths vary, with most reviews being moderately lengthy, indicating sufficient textual content for analysis.
- Common words and phrases identified in the EDA phase were removed during preprocessing to focus the analysis on more meaningful terms.

### Description of Feature Engineering

- **Tokenization** and **stop-words filtering** were applied to clean the text data.
- **TF-IDF Vectorization** was used to transform text data into a format suitable for machine learning, emphasizing important but less frequent words.

### Reasonings on Model Selection

- A **Logistic Regression** model was chosen as the baseline for its simplicity and interpretability.
- Additional models tested include **Multinomial Naive Bayes** and **Support Vector Machines (SVM)** for their strong performance in text classification tasks.
- **Logistic Regression** was selected for the final model due to its performance near 86% accuracy in handling high-dimensional data and its ability to model complex decision boundaries.

### Overall Performance Evaluation

- The Logistic Regression model achieved the highest accuracy and F1-score among the tested models, making it the best choice for our sentiment analysis task.
- A detailed performance evaluation revealed the model's strengths in correctly classifying positive sentiments and its comparative weakness in handling negative sentiments, guiding future improvement efforts.

### Potential Business Applications and Value for Business

- This sentiment analysis model can help businesses automatically categorize customer feedback, enhancing customer service efficiency.
- Content platforms can use the model to automatically tag and filter reviews, improving user experience by highlighting more relevant and positive content.
- Marketing departments can analyze sentiment in social media mentions to gauge brand perception and guide marketing strategies.

## Machine Learning (ML) Part

### How to Run the Solution

Ensure Docker is installed on your machine. For installation instructions, refer to [Docker's official documentation](https://docs.docker.com/get-docker/).

#### Training the Model

1. Navigate to the project root directory.
2. Build the Docker image for training:

docker build -t sentiment_analysis_training -f src/train/Dockerfile .

3. Run the Docker container to train the model and save outputs:

docker run -v $(pwd)/outputs:/app/outputs sentiment_analysis_training



#### Running Inference

1. Build the Docker image for inference:

docker build -t sentiment_analysis_inference -f src/inference/Dockerfile .

2. Run the Docker container to perform inference:

docker run -v $(pwd)/outputs:/app/outputs sentiment_analysis_inference



### Quickstart Instructions

To quickly start working with the sentiment analysis project, follow these steps:

1. Clone the repository:

git clone https://github.com/Arman070600/FinalProject_NLP.git

2. Navigate to the project directory:

cd FinalProject_NLP

3. Follow the steps in the "Training the Model" and "Running Inference" sections above.

## Dependencies

Please refer to `requirements.txt` for a list of necessary Python packages.

## Contributing

Contributions to the project are welcome! 


