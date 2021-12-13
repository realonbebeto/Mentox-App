# Startup Classification-Recommender System: Project Overview

* Created a sentiment analysis app that identifies emotions in a text to help a pyschologists and psychiatrists evaluate emotions.
* Processed and transformed Go emotions data
* Matched 28 emotions to from 7 labels.
* Preprocess data by removing numbers, stop words, punctuations and processed emojis. 
* Trained and Implemented an NLP model based on Glove twitter embedddings using tensorflow.
* Built a client facing API using Streamlit framework

## Code and Resources Used 
**Python Version:** 3.8  
**Packages:** pandas, numpy, tensorflow, streamlit, pickle  
**For Project Framework Requirements:**  ```pip install -r requirements.txt```  
**Data on Kaggle:** https://github.com/google-research/google-research/tree/master/goemotions   
**Streamlit Productionization:** https://towardsdatascience.com/deploying-a-basic-streamlit-app-ceadae286fd0

## Data Cleaning and Processing
After collecting the data, I cleaned it up so that it was usable for our model and matched over 20 emotions to 7 labels.

## Model Building 

First, I transformed the categorical variables into dummy variables. I also split the data into train and tests sets with a test size of 30%.   

I tried four different models and evaluated them using accuracy score. I chose accuracy score because it is relatively easy to interpret based on the correct classifying.   

I tried four different models:
*	**Multi-layer Perceptron Classifier** – Experiment of a neural network based on LSTM

## Model performance

The NLP model achived a score of over 60% on the test data with:
*   **MLP Classifier**: Score = 0.6127

## Productionization

In this step, I built a Streamlit API endpoint that was hosted on a local webserver by following along with the productization tutorial in the reference section above together with the official [Streamlit documentation](https://docs.streamlit.io/en/stable/). The API endpoint takes in a request with text as the input and returns a graph of potential emotions in the text by probability.