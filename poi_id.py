#!/usr/bin/python

import sys
import pickle
import matplotlib.pyplot as plt
import matplotlib.patches as mpatches
import numpy as np

sys.path.append("tools/")

from feature_format import featureFormat, targetFeatureSplit
from tester import dump_classifier_and_data


### Load the dictionary containing the dataset
with open("final_project_dataset.pkl", "r") as data_file:
    data_dict = pickle.load(data_file)


### explore long names and clean data
print
print "explore long names:"
for point in data_dict:
    if len(point) >= 20:
        print point
print

data_dict.pop('THE TRAVEL AGENCY IN THE PARK', 0)

### remove point with all features are "NaN":
for point in data_dict:
    n = 0
    for v in data_dict[point]:
        if data_dict[point][v] == "NaN":
            n += 1
    if n == 20:
        print "all features are NaN:", point
print

data_dict.pop('LOCKHART EUGENE E', 0)


### Task 2: Remove outliers

# read in data dictionary, convert to numpy array
features = ["salary", "bonus"]
data = featureFormat(data_dict, features, sort_keys=True)

# scatter plot of "bonus by salary"
for point in data:
    salary = point[0]
    bonus = point[1]
    plt.scatter(salary, bonus)

plt.xlabel("salary")
plt.ylabel("bonus")
plt.title('bonus by salary with biggest outlier')
plt.show()



# find the biggest outlier
biggest_salary = 0

for point in data_dict:
    if data_dict[point]["salary"] != 'NaN' and data_dict[point]["salary"] > biggest_salary:
        biggest_salary = data_dict[point]["salary"]

for point in data_dict:
    if data_dict[point]["salary"] == biggest_salary:
        print "biggest outlier name:", point
print

# remove the biggest outlier
data_dict.pop('TOTAL', 0)

# read in data dictionary, convert to numpy array
data = featureFormat(data_dict, features, sort_keys=True)

# scatter plot of "bonus by salary"
for point in data:
    salary = point[0]
    bonus = point[1]
    plt.scatter(salary, bonus)

plt.xlabel("salary")
plt.ylabel("bonus")
plt.title('bonus by salary without biggest outlier')
plt.show()

# find another two outlier whose bonus > 5e6 & salary >1e6
print "another two outliers names:"
for point in data_dict:
    if data_dict[point]["salary"] != 'NaN' and data_dict[point]["salary"] > 1e6:
        if data_dict[point]["bonus"] != 'NaN' and data_dict[point]["bonus"] > 5e6:
            print point
print


### Task 3: Create new feature(s)
features = ['from_this_person_to_poi', 'from_poi_to_this_person',
            'to_messages', 'from_messages', 'poi']
data = featureFormat(data_dict, features, sort_keys=True)

# scatter plot of "from_poi_to_this_person by from_this_person_to_poi"
for point in data:
    from_this_person_to_poi = point[0]
    from_poi_to_this_person = point[1]
    poi = point[4]
    if poi == 1:
        plt.scatter(from_this_person_to_poi,
                    from_poi_to_this_person, color='r')
    else:
        plt.scatter(from_this_person_to_poi,
                    from_poi_to_this_person, color='b')

red_patch = mpatches.Patch(color='red', label='poi')
blue_patch = mpatches.Patch(color='blue', label='non poi')
plt.legend(handles=[red_patch, blue_patch])
plt.xlabel("from_this_person_to_poi")
plt.ylabel("from_poi_to_this_person")
plt.show()

def computeFraction(poi_messages, all_messages):
    """ given a number messages to/from POI (numerator)
        and number of all messages to/from a person (denominator),
        return the fraction of messages to/from that person
        that are from/to a POI
    """
    fraction = 0.
    if poi_messages != "NaN" or all_messages != "NaN":
        fraction = float(poi_messages) / all_messages
    return fraction

for point in data_dict:

    from_poi_to_this_person = data_dict[point]["from_poi_to_this_person"]
    to_messages = data_dict[point]["to_messages"]
    fraction_from_poi = computeFraction(from_poi_to_this_person, to_messages)
    data_dict[point]["fraction_from_poi"] = fraction_from_poi

    from_this_person_to_poi = data_dict[point]["from_this_person_to_poi"]
    from_messages = data_dict[point]["from_messages"]
    fraction_to_poi = computeFraction(from_this_person_to_poi, from_messages)
    data_dict[point]["fraction_to_poi"] = fraction_to_poi

features = ['fraction_to_poi', 'fraction_from_poi', 'poi']
data = featureFormat(data_dict, features, sort_keys=True)

# scatter plot of "fraction_from_poi by fraction_to_poi"
for point in data:
    fraction_to_poi = point[0]
    fraction_from_poi = point[1]
    poi = point[2]
    if poi == 1:
        plt.scatter(fraction_to_poi,
                    fraction_from_poi, color='r')
    else:
        plt.scatter(fraction_to_poi,
                    fraction_from_poi, color='b')

plt.xlabel("fraction_to_poi")
plt.ylabel("fraction_from_poi")
red_patch = mpatches.Patch(color='red', label='poi')
blue_patch = mpatches.Patch(color='blue', label='non poi')
plt.legend(handles=[red_patch, blue_patch])
plt.show()

# explore scatter plot
print "explore why some poi has small poi_messages:"
for point in data_dict:
    if data_dict[point]["poi"] == True:
        if data_dict[point]["fraction_to_poi"] < 0.1:
            print "names:", point, "poi:", data_dict[point]["poi"], \
                "fraction_to_poi:", data_dict[point]["fraction_to_poi"], \
                "fraction_from_poi", data_dict[point]["fraction_from_poi"], \
                "from_poi_to_this_person:", \
                data_dict[point]["from_poi_to_this_person"], \
                "from_this_person_to_poi:", \
                data_dict[point]["from_this_person_to_poi"]
print



### Store to my_dataset for easy export below.
my_dataset = data_dict
### Extract features and labels from dataset for local testing
# features_list = ['poi', 'to_messages']
# data = featureFormat(my_dataset, features_list, )
# labels, features = targetFeatureSplit(data)


### Task 1: Select what features you'll use.
### features_list is a list of strings, each of which is a feature name.
### The first feature must be "poi".(because it is the target, \
# targetFeatureSplit will split the first one out)

### Extract features and labels from dataset for local testing
features_extracted = ['poi', 'salary', 'deferral_payments', 'total_payments', \
                'loan_advances', 'bonus', 'restricted_stock_deferred', \
                'deferred_income', 'total_stock_value', 'expenses', \
                'exercised_stock_options', 'other', 'long_term_incentive', \
                 'restricted_stock', 'director_fees', \
                 'shared_receipt_with_poi', 'fraction_to_poi', 'fraction_from_poi']
# print len(features_extracted)
data = featureFormat(my_dataset, features_extracted, sort_keys=True)
labels, features = targetFeatureSplit(data)

# select 5 best features
from sklearn.feature_selection import SelectKBest
freatures_select_number = 5
selector = SelectKBest(k=freatures_select_number)
features_selected = selector.fit_transform(
    features, labels)


# match each features to its score and store them into a list
score = list(selector.scores_)
features_scores = []
for i, v in zip(features_extracted[1:], score):
    features_scores.append((i, v))

# sort the score in descending way
features_scores_sorted = sorted(features_scores,
                                    key = lambda x: x[1], reverse=True)

print freatures_select_number, "best features & scores:", \
        features_scores_sorted[0:freatures_select_number]
print

features_list = ["poi"]
for i, v in features_scores_sorted[0:freatures_select_number]:
    features_list.append(i)


### split data into training and testing sets
from sklearn.cross_validation import train_test_split
features_train, features_test, labels_train, labels_test = \
    train_test_split(features_selected, labels,
                     test_size=0.3, random_state=42)



### Task 4: Try a varity of classifiers
### Please name your classifier clf for easy export below.
### Note that if you want to do PCA or other multi-stage operations,
### you'll need to use Pipelines. For more info:
### http://scikit-learn.org/stable/modules/pipeline.html


### Task 5: Tune your classifier to achieve better than .3 precision and recall
### using our testing script. Check the tester.py script in the final project
### folder for details on the evaluation method, especially the test_classifier
### function. Because of the small size of the dataset, the script uses
### stratified shuffle split cross validation. For more info:
### http://scikit-learn.org/stable/modules/generated/sklearn.cross_validation.StratifiedShuffleSplit.html


# decision tree algorithm & validation
from sklearn.metrics import recall_score
from sklearn.metrics import precision_score
from sklearn.model_selection import GridSearchCV
from sklearn.tree import DecisionTreeClassifier
clf = DecisionTreeClassifier()
parameters = {"min_samples_split": range(5, 80, 5)}
clf = GridSearchCV(clf, parameters)
clf.fit(features_train, labels_train)
clf = clf.best_estimator_
print "Decision Tree best_estimator_"
print clf
print
pred = clf.predict(features_test)
print "Decision Tree algorithm performance:"
print "recall score:", recall_score(labels_test, pred, average="micro")
print "precision score:", precision_score(labels_test, pred, average="micro")
print



# naivebays algorithm & validation
from sklearn.naive_bayes import GaussianNB
clf = GaussianNB()
clf.fit(features_train, labels_train)
pred = clf.predict(features_test)
print "Naive Bays algorithm performance:"
print "recall score:", recall_score(labels_test, pred, average="micro")
print "precision score:", precision_score(labels_test, pred, average="micro")
print






### Task 6: Dump your classifier, dataset, and features_list so anyone can
### check your results. You do not need to change anything below, but make sure
### that the version of poi_id.py that you submit can be run on its own and
### generates the necessary .pkl files for validating your results.

dump_classifier_and_data(clf, my_dataset, features_list)
