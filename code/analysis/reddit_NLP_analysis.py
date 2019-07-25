import pandas as pd
import string
import numpy as np
import time

import seaborn as sns
import matplotlib.pyplot as plt
sns.set_style('whitegrid')

import nltk
from nltk.corpus import stopwords
from nltk.stem import PorterStemmer, WordNetLemmatizer

from sklearn.feature_extraction.text import CountVectorizer, TfidfTransformer
from sklearn.naive_bayes import MultinomialNB
from sklearn.model_selection import train_test_split
from sklearn.metrics import accuracy_score, classification_report, confusion_matrix
from sklearn.pipeline import Pipeline
from sklearn.svm import SVC


reddit = pd.read_csv('../../datasets/reddit_train.csv')
reddit.drop(['num', 'X'], axis = 1, inplace = True) # These columns provide no information for classification so we remove them
reddit['REMOVED'] = reddit['REMOVED'].map({1:'Yes', 0:'No'}) # To make the understanding of the dataset clearer for now

print(reddit.head(), '\n\n')

print('Description of the data: \n')
print(reddit.describe(), '\n\n')

print('Description of the data grouped by REMOVED: \n')
print(reddit.groupby('REMOVED').describe(), '\n\n')

print('Getting length of each comment...')
reddit['LENGTH'] = reddit['BODY'].apply(len) # find the length of each comment
print('done\n')

reddit.hist(column = 'LENGTH', bins = 100, by = 'REMOVED', sharex = True)

print('From this plot we can see that the length of the comment does not seem to be a good indicator of whether a message was removed or not, this is as expected but it is always worth exploring the data.\n')
plt.show()

def extract_punc(text):
    punc = [char for char in text if char in string.punctuation]
    return len(punc)

print('Getting number of punctuation points used in comments...')
reddit['NUM_PUNC'] = reddit['BODY'].apply(extract_punc)
print('done\n')

reddit.hist(column = 'NUM_PUNC', bins = 100, by = 'REMOVED', sharex = True)

print('Again this shows that there is little to no difference between the distribution of number of punctuation points used in comments that were removed and not removed meaning it is not a good indicator to add to our classifier.\n')
plt.show()



# It is important to preprocess your text data into a simpler form, for example removed words that carry no weight, e.g. 'I' or 'an' so as to not drown out the important words that do carry meaning, we will also remove punctuation from our comments, however this could be an important feature in some datasets.

def text_process(text):
    no_punc = [char for char in text if char not in string.punctuation]
    no_punc = ''.join(no_punc)
    return [word for word in no_punc.split() if word.lower() not in stopwords.words('english')]


print('Splitting data into train and test set...')
X_train, X_test, y_train, y_test = train_test_split(reddit['BODY'], reddit['REMOVED'])
print('done\n')

text_pipeline = Pipeline([('bow', CountVectorizer(analyzer=text_process)),
                            ('tfidf', TfidfTransformer()),
                            ('NaiveBayes', MultinomialNB())
                        ])
print('Fitting NaiveBayes model...')
remove_analysis = text_pipeline.fit(X_train, y_train)
print('done\nPredicting values...')
pred = remove_analysis.predict(X_test)
print('done\n')

print(accuracy_score(y_test, pred), '\n\n')
print(classification_report(y_test, pred), '\n\n')
print(confusion_matrix(y_test, pred), '\n\n')

print('We can see that although the accuracy score of this model seemed decent at around 68%, the actual model is terrible, this could be due to the significant class imbalance. Following, we create a model where the classes are evenly balanced\n')

reddit_no = reddit[reddit['REMOVED'] == 'No']
reddit_yes = reddit[reddit['REMOVED'] == 'Yes']

to_select = min(len(reddit_no), len(reddit_yes))

reddit = pd.concat([reddit_no.iloc[:to_select,:], reddit_yes.iloc[:to_select,:]])
sns.countplot(reddit['REMOVED'])
plt.show()

print('Splitting data into train and test set...')
X_train, X_test, y_train, y_test = train_test_split(reddit['BODY'], reddit['REMOVED'])
print('done\n')

print('Fitting NaiveBayes model with balanced dataset...')
remove_analysis_balanced = text_pipeline.fit(X_train, y_train)
print('done\nPredicting values...')
pred_balanced = remove_analysis_balanced.predict(X_test)
print('done\n')

print(accuracy_score(y_test, pred_balanced), '\n\n')
print(classification_report(y_test, pred_balanced), '\n\n')
print(confusion_matrix(y_test, pred_balanced), '\n\n')


print('Although the accuracy of the model is approximitely equal to that of the previous one, the overall performance metrics are much better as the model predicts more evenly about the classes. This is demonstrative as to why you need to explore your data before blindly fitting a model to it and looking at only the accuracy score as this can paint a very false picture of how good your model is. The phrase \'Garbage in garbage out comes to mind here\'\n')
print('To make the predictions better, another thing that might work is stemming. This is the process of attempting to reduce a word down to it\'s base word, e.g. running would become run. Lets try this and see if it has an impact on our predictions.(we will use the same balanced dataset as the previous model)\n')

def text_process_with_stem(text):
    ps = PorterStemmer()
    no_punc = [char for char in text if char not in string.punctuation]
    no_punc = ''.join(no_punc)
    no_stops = [word for word in no_punc.split() if word.lower() not in stopwords.words('english')]
    words_ps = [ps.stem(word.lower()) for word in no_stops]
    return words_ps

text_pipeline_stem = Pipeline([('bow', CountVectorizer(analyzer=text_process_with_stem)),
                            ('tfidf', TfidfTransformer()),
                            ('NaiveBayes', MultinomialNB())
                        ])

print('Splitting data into train and test set...')
X_train, X_test, y_train, y_test = train_test_split(reddit['BODY'], reddit['REMOVED'])
print('done\n')

print('Fitting NaiveBayes model with balanced and stemmed dataset...')
remove_analysis_balanced_stem = text_pipeline_stem.fit(X_train, y_train)
print('done\nPredicting values...')
pred_balanced_stem = remove_analysis_balanced_stem.predict(X_test)
print('done\n')


print(accuracy_score(y_test, pred_balanced_stem), '\n\n')
print(classification_report(y_test, pred_balanced_stem), '\n\n')
print(confusion_matrix(y_test, pred_balanced_stem), '\n\n')

print('This model performs very simmilar to the previous model, it is possibly marginally better but this will likely depend on how the data is split. Instead of stemming, we could try Lemmatizing the words.\n')


def text_process_with_lemma(text):
    lemma = WordNetLemmatizer()
    no_punc = [char for char in text if char not in string.punctuation]
    no_punc = ''.join(no_punc)
    no_stops = [word for word in no_punc.split() if word.lower() not in stopwords.words('english')]
    words_lemma = [lemma.lemmatize(word.lower()) for word in no_stops]
    return words_lemma

text_pipeline_lemma = Pipeline([('bow', CountVectorizer(analyzer=text_process_with_lemma)),
                            ('tfidf', TfidfTransformer()),
                            ('NaiveBayes', MultinomialNB())
                        ])

print('Splitting data into train and test set...')
X_train, X_test, y_train, y_test = train_test_split(reddit['BODY'], reddit['REMOVED'])
print('done\n')

print('Fitting NaiveBayes model with balanced and Lemmatized dataset...')
remove_analysis_balanced_lemma = text_pipeline_lemma.fit(X_train, y_train)
print('done\nPredicting values...')
pred_balanced_lemma = remove_analysis_balanced_lemma.predict(X_test)
print('done\n')


print(accuracy_score(y_test, pred_balanced_stem), '\n\n')
print(classification_report(y_test, pred_balanced_stem), '\n\n')
print(confusion_matrix(y_test, pred_balanced_stem), '\n\n')

print('This model appears to be no better than chance at predicting whether a comment should be removed, therefore we abandon this approach.')
