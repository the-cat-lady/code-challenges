
"""
==========================================================
Sample pipeline for text feature extraction and evaluation
==========================================================

The dataset used in this example is the 20 newsgroups dataset which will be
automatically downloaded and then cached and reused for the document
classification example.

You can adjust the number of categories by giving their names to the dataset
loader or setting them to None to get the 20 of them.

Here is a sample output of a run on a quad-core machine when passing in
verbose=True in main()::

  Loading 20 newsgroups dataset for categories:
  ['alt.atheism', 'talk.religion.misc']
  1427 documents
  2 categories

  Performing grid search...
  pipeline: ['vect', 'tfidf', 'clf']
  parameters:
  {'clf__alpha': (1.0000000000000001e-05, 9.9999999999999995e-07),
   'clf__n_iter': (10, 50, 80),
   'clf__penalty': ('l2', 'elasticnet'),
   'tfidf__use_idf': (True, False),
   'vect__max_n': (1, 2),
   'vect__max_df': (0.5, 0.75, 1.0),
   'vect__max_features': (None, 5000, 10000, 50000)}
  done in 1737.030s

  Best score: 0.940
  Best parameters set:
      clf__alpha: 9.9999999999999995e-07
      clf__n_iter: 50
      clf__penalty: 'elasticnet'
      tfidf__use_idf: True
      vect__max_n: 2
      vect__max_df: 0.75
      vect__max_features: 50000

"""

# Author: Olivier Grisel <olivier.grisel@ensta.org>
#         Peter Prettenhofer <peter.prettenhofer@gmail.com>
#         Mathieu Blondel <mathieu@mblondel.org>
# License: BSD 3 clause

from __future__ import print_function

# Suppress FutureWarning for some environments
import warnings
warnings.simplefilter(action='ignore', category=FutureWarning)

from pprint import pprint
from time import time
import logging

from sklearn.datasets import fetch_20newsgroups
from sklearn.feature_extraction.text import CountVectorizer
from sklearn.feature_extraction.text import TfidfTransformer
from sklearn.linear_model import SGDClassifier
from sklearn.model_selection import GridSearchCV
from sklearn.pipeline import Pipeline

print(__doc__)

# Display progress logs on stdout
logging.basicConfig(level=logging.INFO,
                    format='%(asctime)s %(levelname)s %(message)s')

class TextClassifier():

    def __init__(self):
        ''' Initialize TextClassifier Object '''
        self.categories = None
        self.pipeline = None
        self.data = None
        self.parameters = None


    def set_categories(self, categories_list=None):
        '''
        Define categories to include from dataset
        categories_list : list of string categories
        '''
        if categories_list == None:
            self.categories = [
                'alt.atheism',
                'talk.religion.misc',
            ]
        else:
            # TODO: Add validation
            self.categories = categories_list

    def load_data(self, data=None):
        '''
        load our dataset
        data : of type <class 'sklearn.utils.Bunch'>
        '''

        if self.categories == None:
            self.set_categories()

        print("Loading 20 newsgroups dataset for categories:")
        print(self.categories)

        if data == None:
            self.data = fetch_20newsgroups(subset='train', categories=self.categories)
        else:
            self.data = data
        print("%d documents" % len(self.data.filenames))
        print("%d categories" % len(self.data.target_names))
        print()
        print(type(self.data))


    def define_pipeline(self, steps=None):
        '''
        define a pipeline combining a text feature extractor with a simple
        classifier

        steps : List of (name, transform) tuples (implementing fit/transform)
        that are chained, in the order in which they are chained, with the last
        object an estimator.
        '''
        if steps == None:
            self.pipeline = Pipeline([
                ('vect', CountVectorizer()),
                ('tfidf', TfidfTransformer()),
                ('clf', SGDClassifier()),
            ])
        else:
            # TODO: add validation
            self.pipeline = steps


    def set_parameters(self, parameters=None):
        '''
        Define parameters for the pipeline steps.
        Uncommenting more parameters will give better exploring power but will
        increase processing time in a combinatorial way
        '''
        if parameters == None:
            self.parameters = {
                'vect__max_df': (0.5, 0.75, 1.0),
                #'vect__max_features': (None, 5000, 10000, 50000),
                'vect__ngram_range': ((1, 1), (1, 2)),  # unigrams or bigrams
                #'tfidf__use_idf': (True, False),
                #'tfidf__norm': ('l1', 'l2'),
                'clf__alpha': (0.00001, 0.000001),
                'clf__penalty': ('l2', 'elasticnet'),
                #'clf__n_iter': (10, 50, 80),
            }
        else:
            # TODO: Add validation
            self.parameters = parameters


def main(verbose=False):
    '''categories = [
        'alt.atheism',
        'talk.religion.misc',
    ]'''
    text_classifier = TextClassifier()
    text_classifier.set_categories()
    text_classifier.load_data()
    text_classifier.define_pipeline()
    text_classifier.set_parameters()

    # multiprocessing requires the fork to happen in a __main__ protected
    # block

    # find the best parameters for both the feature extraction and the
    # classifier
    grid_search = GridSearchCV(text_classifier.pipeline, text_classifier.parameters,
                               n_jobs=-1, verbose=1)

    if verbose:
        print("Performing grid search...")
        print("pipeline:", [name for name, _ in text_classifier.pipeline.steps])
        print("parameters:")
        pprint(text_classifier.parameters)

    t0 = time()
    grid_search.fit(text_classifier.data.data, text_classifier.data.target)
    if verbose:
        print("done in %0.3fs" % (time() - t0))
    print()

    print("Best score: %0.3f" % grid_search.best_score_)
    print("Best parameters set:")
    best_parameters = grid_search.best_estimator_.get_params()
    for param_name in sorted(text_classifier.parameters.keys()):
        print("\t%s: %r" % (param_name, best_parameters[param_name]))


if __name__ == '__main__':
    main()
