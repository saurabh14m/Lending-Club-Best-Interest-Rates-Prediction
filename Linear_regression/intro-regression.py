#!/usr/bin/env python
# -*- coding: utf-8 -*-


# coding: utf-8

## Introduction to Regression & Classification

# In[ ]:




# **Note: This notebook requires GraphLab Create 1.2 or higher.**
# 
# Creating regression models is easy with GraphLab Create! The regression/classificaiton toolkit contains several models including (but not restricted to) **linear regression**, **logistic regression**, and **gradient boosted trees**. All models are built to work with millions of features and billions of examples. The models differ in how they make predictions, but conform to the same API. Like all  GraphLab Create toolkits, you can call *create()* to create a model, *predict()* to make predictions on the returned model, and *evaluate()* to measure performance of the predictions.
# 
# Be sure to check out our notebook on [feature engineering](feature-engineering.html) which discusses **advanced features**  including the use of **categorical variables, dictionary features, and text data**. All of these feature types are easy and intuitive to use with GraphLab Create.

### Overview 

# In this notebook, we will go over how GraphLab Create can be used for basic tasks in regression analysis. Specifically, we will go over:
# 
# * [Training, prediction, and evaluation of models](#Training,-Predicting,-and-Evaluating-Models)
# * [Interpreting the results of the model](#Interpreting-Results)
# * [Binary classification](#Binary-Classification)
# * [Multiclass classification](#Multiclass-Classification)
# * [Handling Imbalanced Classes](#Imbalanced-Classes)
# 
# We will start by importing GraphLab Create!

# In[1]:

import graphlab as gl
gl.canvas.set_target('ipynb')


### Data Overview

# In this notebook, we will use a subset of the data from the [Yelp Dataset Challenge](http://www.yelp.com/dataset_challenge) for this tutorial. The task is to **predict the 'star rating' for a restaurant for a given user.** The dataset comprises three tables that cover 11,537 businesses, 8,282 check-ins, 43,873 users, and 229,907 reviews. The entire dataset as well as details about the dataset are available on the [Yelp website](http://www.yelp.com/dataset_challenge).
# 
# #### Review Data
# 
# The review table includes information about each review. Specifically, it contains:
# 
# * *business_id*: An encrypted business ID for the business being reviewed.
# * *user_id*: An encrypted user ID for the user who provided the review.
# * *stars*: A star rating (on a scale of 1-5)
# * *text*: The raw review text.
# * *date*: Date, formatted like '2012-03-14'
# * *votes*: The number of 'useful', 'funny' or 'cool' votes provided by other users for this review.
# 
# #### User Data
# 
# The user table consists of details about each user:
# 
# * *user_id*: The encrypted user ID (cross referenced in the Review table)
# * *name*: First name
# * *review_count*: Total number of reviews made by the user.
# * *average_stars*: Average rating (on a scale of 1-5) made by the user.
# * *votes*: For each review type i.e ('useful', 'funny', 'cool') the total number of votes for reviews made by this user.
# 
# #### Business Data
# 
# The business table contains details about each business:
# 
# * *business_id*: Encrypted business ID (cross referenced in the Review table)
# * *name*: Business name.
# * *neighborhoods*: Neighborhoods served by the business.
# * *full_address*: Address (text format)
# * *city*: City where the business is located.
# * *state*: State where the business is located.
# * *latitude*: Latitude of the business.
# * *longitude*: Longitude of the business.
# * *stars*: A star rating (rounded to half-stars) for this business.
# * *review_count*: The total number of reviews about this business.
# * *categories*: Category tags for this business.
# * *open*: Is this business still open? (True/False)
# 
# 
# Let us take a closer look at the data.

# In[2]:

business = gl.SFrame('http://s3.amazonaws.com/dato-datasets/regression/business.csv')
user = gl.SFrame('http://s3.amazonaws.com/dato-datasets/regression/user.csv')
review = gl.SFrame('http://s3.amazonaws.com/dato-datasets/regression/review.csv')


# The schema and the first few entries of the *review* are shown below. For the sake of brevity, we will skip the business and user tables.

# In[3]:

review.show()


#### Preparing the data

# In this section, we will go through some basic steps to prepare the dataset for regression models.
# 
# First, we use an SFrame **[join operation](https://graphlab.com/products/create/docs/generated/graphlab.SFrame.join.html#graphlab.SFrame.join)** to merge the *business* and *review* tables, using the *business_id* column to "match" the rows of the two tables. The output of the join is a single table with both business and review information. For clarity we rename some of the business columns to have more meaningful descriptions.

# In[4]:

review_business_table = review.join(business, how='inner', on='business_id')
review_business_table = review_business_table.rename({'stars.1': 'business_avg_stars', 
                              'type.1': 'business_type',
                              'review_count': 'business_review_count'})


# Now, join *user table* to the result, using the *user_id* column to match rows. Now we have *review*, *business*, and *user* information in a single table.

# In[5]:

user_business_review_table = review_business_table.join(user, how='inner', on="user_id")
user_business_review_table = user_business_review_table.rename({'name.1': 'user_name', 
                                   'type.1': 'user_type', 
                                   'average_stars': 'user_avg_stars',
                                   'review_count': 'user_review_count'})


# Now we're good to go! Let's take a look at what the final dataset looks like:

# In[6]:

user_business_review_table.head(5)


### Training, Predicting, and Evaluating Models

# It's now time to do some data science! First, let us split our data into training and testing sets, using SFrame's **[random_split](https://graphlab.com/products/create/docs/generated/graphlab.SFrame.random_split.html#graphlab.SFrame.random_split)** function.

# In[7]:

train_set, test_set = user_business_review_table.random_split(0.8, seed=1)


# Let's start out with a simple model. The target is the **star rating** for each review and the features are: 
# 
# * Average rating of a given business
# * Average rating made by a user
# * Number of reviews made by a user
# * Number of reviews that concern a business

# In[8]:

model = gl.linear_regression.create(train_set, target='stars', 
                                    features = ['user_avg_stars','business_avg_stars', 
                                                'user_review_count', 'business_review_count'])


# Much of the summary output is self-explanatory. We will explain below what the terms *'Strongest positive coefficients'* and '*Strongest negative coefficients*' mean.

#### Making Predictions

# GraphLab Create easily allows you to make predictions using the created model with the **[predict](https://graphlab.com/products/create/docs/generated/graphlab.linear_regression.LinearRegressionModel.predict.html#graphlab.linear_regression.LinearRegressionModel.predict)** function. The predict function returns an SArray with a prediction for each example in the test dataset.

# In[9]:

predictions = model.predict(test_set)
predictions.head(5)


#### Evaluating Results

# We can also [evaluate](https://graphlab.com/products/create/docs/generated/graphlab.linear_regression.LinearRegressionModel.evaluate.html#graphlab.linear_regression.LinearRegressionModel.evaluate) our predictions by comparing them to known ratings. The results are evaluated using two metrics: [root-mean-square error (RMSE)](http://en.wikipedia.org/wiki/Root-mean-square_deviation) is a global summary of the differences between predicted values and the values actually observed, while [max-error](http://en.wikipedia.org/wiki/Maximum_norm) measures the worst case performance of the model on a single observation. In this example, our model made predictions which were about *1* star away from the true rating (on average) but there were a few cases where we were off by almost *4* stars.

# In[10]:

model.evaluate(test_set)


# Let's go further in analyzing how well our model performed at predicting ratings. We perform a *groupby-aggregate* to calculate the average predicted rating (on the test set) for each value of the actual rating (1-5). This will help us understand when the model performs well and when it does not.

# In[11]:

sf = gl.SFrame()
sf['Predicted-Rating'] = predictions
sf['Actual-Rating'] = test_set['stars']
predict_count = sf.groupby('Actual-Rating', [gl.aggregate.COUNT('Actual-Rating'), gl.aggregate.AVG('Predicted-Rating')])
predict_count.topk('Actual-Rating', k=5, reverse=True)    


# It looks like our model does well on ratings that were between 3 and 5 but not too well on ratings 1 and 2. One reason why this could happen is that we have a lot more reviews with 4 and 5 star ratings. In fact, the number of 4 and 5 star reviews is more than twice the number of reviews with 1-3 stars.

### Interpreting Results

# In addition to making predictions about new data, GraphLab's regression toolkit can provide valuable insight about the relationships between the target and feature columns in your data, revealing why your model returns the predictions that it does. Let's briefly venture into some mathematical details to explain. Linear regression models the target $Y$ as a linear combination of the feature variables $X_j$, random noise $\epsilon$, and a bias term ($\alpha_0$) (also known as the intercept or global offset): 
# 
# $$Y = \alpha_0 + \sum_{j} \alpha_j X_j + \epsilon$$
# 
# The *coefficients* ($\alpha_j$) are what the training procedure learns. Each model coefficient describes the expected change in the target variable associated with a unit change in the feature. The bias term indicates the "inherent" or "average" target value if all feature values were set to zero.
# 
# The coefficients often tell an interesting story of how much each feature matters in predicting target values. The magnitude (absolute value) of the coefficient for each feature indicates the strength of the feature's association to the target variable, *holding all other features constant*. The sign on the coefficient (positive or negative) gives the direction of the association.
# 
# For a trained model, we can access the coefficients as follows. The **name** is the name of the feature, the **index** refers to a category for categorical variables, and the **value** is the value of the coefficient.

# In[12]:

coefs = model['coefficients']
coefs


# Not surpisingly, high ratings are associated with 1. users who give a lot of high ratings on average, and 2. businesses that receive high ratings on average. More interestingly, the *number* of reviews submitted by a user or recieved by a business appears to have a very weak association with ratings.

### Binary Classification

# Logistic regression is a model that is popularly used for classification tasks. In logistic regression, the probability that a  **binary target is True** is modeled as a [logistic function](http://en.wikipedia.org/wiki/Logistic_function) of the features.
# 
# First, let's construct a binary target variable. In this example, we will predict **if a restaurant is good or bad**, with 1 and 2 star ratings indicating a bad business and 3-5 star ratings indicating a good one.

# In[13]:

user_business_review_table['is_good'] = user_business_review_table['stars'] >= 3


# First, let's create a train-test split:

# In[14]:

train_set, test_set = user_business_review_table.random_split(0.8, seed=1)


# We will use the same set of features that we used for the linear regression model. Note that the API is very similar to the linear regression API.

# In[15]:

model = gl.logistic_classifier.create(train_set, target="is_good", 
                                      features = ['user_avg_stars','business_avg_stars', 
                                                'user_review_count', 'business_review_count'])


#### Making Predictions (Probabilities, Classes, or Margins)

# Logistic regression predictions can take one of three forms:
# 
# * **Classes** (default) : Thresholds the probability estimate at 0.5 to predict a class label i.e. **True/False**.
# * **Probabilities** : A probability estimate (in the range [0,1]) that the example is in the **True** class.
# * **Margins** : Distance to the linear decision boundary learned by the model. The larger the distance, the more confidence we have that it belongs to one class or the other.
# 
# GraphLab's logistic regression model can return [predictions](https://graphlab.com/products/create/docs/generated/graphlab.logistic_classifier.LogisticClassifier.predict.html#graphlab.logistic_classifier.LogisticClassifier.predict) for any of these types:

# In[16]:

# Probability
predictions = model.predict(test_set)
predictions.head(5)


# In[17]:

predictions = model.predict(test_set, output_type = "margin")
predictions.head(5)


# In[19]:

predictions = model.predict(test_set, output_type = "probability")
predictions.head(5)


#### Evaluating Results

# We can also evaluate our predictions by comparing them to known ratings. The results are evaluated using two metrics:
# 
# * [Classification Accuracy](http://en.wikipedia.org/wiki/Accuracy_and_precision): Fraction of test set examples with correct class label predictions.
# * [Confusion Matrix](http://en.wikipedia.org/wiki/Confusion_matrix): Cross-tabulation of predicted and actual class labels.
# 

# In[20]:

result = model.evaluate(test_set)
print "Accuracy         : %s" % result['accuracy']
print "Confusion Matrix : \n%s" % result['confusion_matrix']


# GraphLab Create's [evaluation toolkit](https://graphlab.com/products/create/docs/graphlab.toolkits.evaluation.html) contains more detail on evaluation metrics for both regression and classification. You are now good to go with regression! Be sure to check out our notebook on [feature engineering](feature-engineering.html) to learn new tricks that can help you make better classifiers and predictors!

### Multiclass Classification

# Logistic Regression can also be used for multiclass classficiation. Multiclass classification allows each observation to be assigned to one of many categories (for example: ratings may be 1, 2, 3, 4, or 5). In this example, we will predict **the rating of the restaurant**.

# In[76]:

model = gl.logistic_classifier.create(train_set, target="stars", 
                                      features = ['user_avg_stars','business_avg_stars', 
                                                'user_review_count', 'business_review_count'])


# Statistics about the training data including the number of classes, the set of classes registered in the dataset, as well as the number of examples in each class are stored in the model.

# In[22]:

print "This model has %s classes" % model['num_classes']
print "The set of classes in the training set are %s" % model['classes']


#### Top-k predictions with multiclass classfication

# While training models for multiclass classificaiotn, the top-k predictions can be of the following type.
# 
# * **Probabilities** (default): A probability estimate (in the range [0,1]) that the example is in the predicted class.
# * **Margins** : A score that reflects the confidence we have that the example belongs to the predicted class. The larger the score, the greater the confidence.
# * **Rank** : A rank (from 1-k) that the example belongs to the predicted class.
# 
# In the following example, we calculate the top-

# In[23]:

predictions = model.predict_topk(test_set, output_type = 'probability', k = 2)
predictions.head(5)


# In[24]:

predictions = model.predict_topk(test_set, output_type = 'margin', k = 2)
predictions.head(5)


# In[25]:

predictions = model.predict_topk(test_set, output_type = 'rank', k = 2)
predictions.head(5)


#### Evaluation

# We can also evaluate our predictions by comparing them to known ratings. The results are evaluated using two metrics:
# 
# * [Classification Accuracy](http://en.wikipedia.org/wiki/Accuracy_and_precision): Fraction of test set examples with correct class label predictions.
# * [Confusion Matrix](http://en.wikipedia.org/wiki/Confusion_matrix): Cross-tabulation of predicted and actual class labels.

# In[78]:

result = model.evaluate(test_set)
print "Confusion Matrix : \n%s" % result['confusion_matrix']


### Imbalanced Datasets

# Many difficult **real-world** problems have imbalanced data, where at least one class is under-represented. GraphLab Create models can improve prediction quality for some unbalanced scenarios by assigning different costs to misclassification errors for different classes.
# 
# Let us see the distribution of examples for each class in the dataset.

# In[34]:

review['stars'].astype(str).show()


# In[36]:

model = gl.logistic_classifier.create(train_set, target="stars", 
                                      features = ['user_avg_stars','business_avg_stars', 
                                                'user_review_count', 'business_review_count'], 
                                      class_weights = 'auto')


# In[40]:

result = model.evaluate(test_set)
print "Confusion Matrix : \n%s" % result['confusion_matrix']

