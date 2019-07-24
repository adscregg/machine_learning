import pandas as pd
import nltk
from nltk.corpus import stopwords
import numpy as np
import seaborn as sns
sns.set_style('whitegrid')
import matplotlib.pyplot as plt
from sklearn.feature_extraction.text import CountVectorizer, TfidfTransformer
import string
from sklearn.naive_bayes import MultinomialNB
from sklearn.model_selection import train_test_split
from sklearn.metrics import accuracy_score, classification_report, confusion_matrix
from sklearn.pipeline import Pipeline


reddit = pd.read_csv('../../datasets/reddit_train.csv')
reddit.drop(['num', 'X'], axis = 1, inplace = True) # These columns provide no information for classification so we remove them
reddit['REMOVED'] = reddit['REMOVED'].map({1:'Yes', 0:'No'}) # To make the understanding of the dataset clearer for now

print(reddit.head(), '\n\n')

print('Description of the data: \n')
print(reddit.describe(), '\n\n')

print('Description of the data grouped by REMOVED: \n')
print(reddit.groupby('REMOVED').describe(), '\n\n')

print('Getting length of each comment...\n')
reddit['LENGTH'] = reddit['BODY'].apply(len) # find the length of each comment
print('done\n')

reddit.hist(column = 'LENGTH', bins = 100, by = 'REMOVED', sharex = True)
plt.show()

print('From the plot that was shown, the length of the comment does not seem to be a good indicator of whether a message was removed or not, this is as expected but it is always worth exploring the data.\n')

# It is important to preprocess your text data into a simpler form, for example removed words that carry no weight, e.g. 'I' or 'an' so as to not drown out the important words that do carry meaning, we will also remove punctuation from our comments, however this could be an important feature in some datasets.

def text_process(text):
    no_punc = [char for char in text if char not in string.punctuation]
    no_punc = ''.join(no_punc)
    return [word for word in no_punc.split() if word.lower() not in stopwords.words('english')]

# print('Fitting and transforming using CountVectorizer...\n')
#
# bow_transformer = CountVectorizer(analyzer=text_process).fit(reddit['BODY'])
# reddit_bow = bow_transformer.transform(reddit['BODY'])
#
# print('done\n')
#
# print('Shape: ', reddit_bow.shape, '\n\n')
#
# print('Fitting TfidfTransformer...')
# tfidf_transformer = TfidfTransformer().fit(reddit_bow)
# print('Trasforming using TfidfTransformer...')
# reddit_tfidf = tfidf_transformer.transform(reddit_bow)

# reddit['REMOVED'] = reddit['REMOVED'].map({'No': 0, 'Yes': 1})

X_train, X_test, y_train, y_test = train_test_split(reddit['BODY'], reddit['REMOVED'])
#
text_pipeline = Pipeline([('bow', CountVectorizer(analyzer=text_process)),
                            ('tfidf', TfidfTransformer()),
                            ('NaiveBayes', MultinomialNB())
                        ])
#
remove_analysis = text_pipeline.fit(X_train, y_train)
pred = remove_analysis.predict(X_test)

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

X_train, X_test, y_train, y_test = train_test_split(reddit['BODY'], reddit['REMOVED'])

remove_analysis_balanced = text_pipeline.fit(X_train, y_train)
pred_balanced = remove_analysis_balanced.predict(X_test)

print(accuracy_score(y_test, pred_balanced), '\n\n')
print(classification_report(y_test, pred_balanced), '\n\n')
print(confusion_matrix(y_test, pred_balanced), '\n\n')


print('Although the accuracy of the model is approximitely equal to that of the previous one, the overall performance metrics are much better as the model predicts more evenly about the classes. This is demonstrative as to why you need to explore your data before blindly fitting a model to it and looking at only the accuracy score as this can paint a very false picture of how good your model is. The phrase \'Garbage in garbage out comes to mind here\'\n')
