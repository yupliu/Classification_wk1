from __future__ import division #to make 2/5 = 0.4 not 0
import graphlab
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
try:
    import seaborn
except ImportError:
    pass

import math
import string

products = graphlab.SFrame('C:\\Machine_Learning\\Classification_wk1\\amazon_baby.gl\\')
def remove_punctuation(text):
    return text.translate(None,string.punctuation)

review_no_puctuation = products['review'].apply(remove_punctuation)
products['word_count'] = graphlab.text_analytics.count_words(review_no_puctuation)
products = products[products['rating']!=3]
len(products)
products['sentiment'] = products['rating'].apply(lambda x:+1 if x > 3 else -1)
train_data, test_data = products.random_split(.8, seed=1)
print len(train_data)
print len(test_data)
sentiment_model = graphlab.logistic_classifier.create(train_data,target='sentiment',features=['word_count'],validation_set=None)
weights = sentiment_model.coefficients
sentiment_model['coefficients'].print_rows(num_rows=12)
print weights.column_names()
num_positive_weights = len(weights[weights['value'] >=0])
num_negative_weights = len(weights[weights['value'] < 0])

print "Number of positive weights: %s " % num_positive_weights
print "Number of negative weights: %s " % num_negative_weights

sample_test_data = test_data[10:13]
print sample_test_data['rating']
sample_test_data

scores = sentiment_model.predict(sample_test_data, output_type='margin')
print scores

def pred_by_score(test_data):
    scores = sentiment_model.predict(test_data, output_type='margin')
    scores = scores.apply(lambda x:+1 if x>0 else -1)
    return scores

pred_score = pred_by_score(sample_test_data)
print pred_score

print "Class predictions according to GraphLab Create:" 
print sentiment_model.predict(sample_test_data)

def prob_pred(test_data):
    scores = sentiment_model.predict(test_data, output_type='margin')
    #scores = scores.apply(lambda x: 1 / (1 + np.exp(-x)))
    scores = 1 / (1 + np.exp(-scores))
    return scores
   
prob_by_score = prob_pred(sample_test_data)
print prob_by_score

print "Class probability according to GraphLab Create:" 
print sentiment_model.predict(sample_test_data,output_type='probability')

test_data['pred_prob'] = sentiment_model.predict(test_data,output_type='probability')
pred_top20_postive = test_data.topk('pred_prob',20)
pred_top20_postive.print_rows(num_rows=20)
pred_top20_negative = test_data.topk('pred_prob',20,reverse=True)
pred_top20_negative.print_rows(num_rows=20)


def get_classification_accuracy(model,data,labels):
    print len(labels)
    pred_labels = model.predict(data)
    #print pred_labels
    diff = pred_labels - labels
    #print diff
    correct = diff[diff==0]
    print len(correct)
    return len(correct) / len(labels)

accuracy = get_classification_accuracy(sentiment_model, test_data, test_data['sentiment'])
print accuracy

significant_words = ['love', 'great', 'easy', 'old', 'little', 'perfect', 'loves', 
      'well', 'able', 'car', 'broke', 'less', 'even', 'waste', 'disappointed', 
      'work', 'product', 'money', 'would', 'return']
    
len(significant_words)
train_data['word_count_subset'] = train_data['word_count'].dict_trim_by_keys(significant_words, exclude=False)
test_data['word_count_subset'] = test_data['word_count'].dict_trim_by_keys(significant_words, exclude=False)

print train_data[0]['review']
print train_data[0]['word_count']
print train_data[0]['word_count_subset']

simple_model = graphlab.logistic_classifier.create(train_data,
                                                   target = 'sentiment',
                                                   features=['word_count_subset'],
                                                   validation_set=None)
simple_model
simple_acc = get_classification_accuracy(simple_model, test_data, test_data['sentiment'])
print simple_acc

simple_model.coefficients.sort('value', ascending=False).print_rows(num_rows=21)
simple_coff = simple_model.coefficients.sort('value', ascending=False)
sentiment_coeff = sentiment_model.coefficients.sort('value', ascending=False)
sentiment_coeff = sentiment_coeff.filter_by(simple_coff['index'],'index')
sentiment_coeff.print_rows(num_rows=21)

simple_train_acc = get_classification_accuracy(simple_model,train_data,train_data['sentiment'])
print simple_train_acc
sentiment_train_acc = get_classification_accuracy(sentiment_model,train_data,train_data['sentiment'])
print sentiment_train_acc

num_positive  = (train_data['sentiment'] == +1).sum()
num_negative = (train_data['sentiment'] == -1).sum()
print num_positive
print num_negative

#majority class is 1
def get_maj_accuracy(data,labels):
    diff = labels - 1
    #print diff
    correct = diff[diff==0]
    return len(correct) / len(labels)

maj_train_acc = get_maj_accuracy(train_data,train_data['sentiment'])
print maj_train_acc
maj_test_acc = get_maj_accuracy(test_data,test_data['sentiment'])
print maj_test_acc

