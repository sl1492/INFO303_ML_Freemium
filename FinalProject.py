# Import the modules needed
import pandas as pd
import pymongo as pym
import numpy as np
import warnings
import matplotlib
import matplotlib.pyplot as plt
import seaborn as sns
import researchpy as rp
import os

# Turn off warnings just in case our Regression algorithm throws a warning that will cause our program to pause.
warnings.filterwarnings('ignore')
# Change the max_rows and max_columns to view more of the data in our output window.
pd.set_option('display.max_columns', 15)
pd.set_option('display.max_rows', 100)

# Connect to the Mongo server, create a variable for the database, and another variable for the collection.
try:
    client = pym.MongoClient('172.28.8.65', 27017)
    client.server_info()
except:
    print('Something went wrong connecting to Mongo')

db = client["project"]
customers = db.customers

# Let's verify that the net_users are unique to prevent any type of data leakage with
# this sample of customers.
val = list(customers.aggregate([
    {"$group": {"_id": "$net_user",
                "uniqueIds": {"$addToSet": "$net_user"},
                "uniques": {"$sum": 1}
                }
    },
    {"$match": {"uniques": {"$gt": 1}}}]))
print(len(val))

# Pull out the data from mongo server to a pandas DataFrame
index = []
my_data = []
my_columns = ['age', 'male', 'friends', 'songsListened', 'lovedTracks', 'posts',
              'playlists', 'shouts', 'delta1', 'adopter', 'tenure', 'good_country', 'delta2']
for c in customers.find({}):
    index.append(c.get('net_user'))
    my_list = []
    for col in my_columns:
        my_list.append(c.get(col))
    my_data.append(my_list)
#print(my_data)
my_df = pd.DataFrame(my_data, index=index, columns=my_columns)

# check the DataFrame we constructed
print(my_df.sample(3))
print(my_df.info())

# Check for nulls.
print(my_df.isnull().sum())

# We see that we have 50859 nulls in the "age" field, 38950 nulls in the "male" field,
# 1927 nulls in the "shouts" field, 32 nulls in the "tenure" field, 39155 nulls in the "good_country" field.
# After consultation with the team, we decide to replace nulls in "age" with the average value, which is 24
# and make nulls in "shouts" and "tenure" to be 0.
my_df['shouts'].fillna(0, inplace=True)
my_df['tenure'].fillna(0, inplace=True)
#print(my_df.isnull().sum())

my_df1 = my_df.drop(['age', 'male', 'good_country'], 1)
#my_df2 = my_df.dropna(subset=['age', 'male', 'good_country'])
#print(my_df1.isnull().sum())
#print(my_df2.isnull().sum())

# Build a new Delta1 pandas DataFrame #
index = []
my_data = []
my_columns = ['delta1_friend_cnt', 'delta1_avg_friend_age', 'delta1_avg_friend_male',
              'delta1_friend_country_cnt', 'delta1_subscriber_friend_cnt', 'delta1_songsListened',
              'delta1_lovedTracks', 'delta1_posts','delta1_playlists', 'delta1_shouts', 'delta1_good_country']
for c in customers.find({}):
    index.append(c.get('net_user'))
    my_list = []
    for col in my_columns:
        my_list.append(c.get('delta1')[col])
    my_data.append(my_list)
#print(my_data)
d1df = pd.DataFrame(my_data, index=index, columns=my_columns)
#print(d1df.isnull().sum())
#print(d1df.sample(10))
#print(d1df.dtypes)

# Change all the 'NULL' values to 0
for col in my_columns:
    d1df[col] = d1df[col].replace(to_replace='NULL', value=0)
    d1df[col] = d1df[col].apply(pd.to_numeric)
print(d1df.describe())
#print(d1df.info())

# Question 1

#######################
# Logistic Regression #
#######################
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import classification_report
from sklearn.metrics import confusion_matrix
from sklearn.metrics import accuracy_score
from sklearn.model_selection import train_test_split

# Create features and targets
my_features = d1df[['delta1_friend_cnt', 'delta1_avg_friend_age',
                      'delta1_subscriber_friend_cnt', 'delta1_songsListened','delta1_lovedTracks',
                      'delta1_posts','delta1_playlists', 'delta1_shouts']].values
my_targets = my_df1[['adopter']].values

# Make train and test data
f_train, f_test, t_train, t_test = train_test_split(my_features, my_targets, stratify=my_targets,
                                                    test_size=0.20, random_state=77)

# over-sampler：smote
from imblearn.over_sampling import SMOTE
over_sampler = SMOTE(random_state=43)
smote_features, smote_targets = over_sampler.fit_resample(f_train, t_train)
my_classifier = LogisticRegression(C=1, random_state=92, solver='liblinear')
smote_model = my_classifier.fit(smote_features, smote_targets)
score_test = 100 * smote_model.score(f_test, t_test)
print(f'Logistic regression prediction accuracy using SMOTE over-sampler = {score_test:.3f}%.')

# Let's check for over-fitting
score_train = 100 * smote_model.score(f_train, t_train)
print(f'Logistic regression prediction accuracy using over-sampler (SMOTE) = {score_train:.3f}%.')

predicted = smote_model.predict(f_test)
print(confusion_matrix(t_test.reshape(t_test.shape[0]), predicted))

# under-sampler: near miss
from imblearn.under_sampling import NearMiss
under_sampler = NearMiss()
nm_features, nm_targets = under_sampler.fit_resample(f_train, t_train)
my_classifier = LogisticRegression(C=1, random_state=92, solver='liblinear')
nm_model = my_classifier.fit(nm_features, nm_targets)
score_test = 100 * nm_model.score(f_test, t_test)
print(f'Logistic regression prediction accuracy using under-sampler (Near Miss) = {score_test:.1f}%.')

# Let's check for over-fitting
score_train = 100 * nm_model.score(f_train, t_train)
print(f'Logistic regression prediction accuracy using under-sampler (Near Miss) = {score_train:.1f}%.')

predicted = nm_model.predict(f_test)
print(confusion_matrix(t_test.reshape(t_test.shape[0]), predicted))


#############
# KNN Model #
#############
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler
from sklearn import neighbors

# Create features and targets
my_features = d1df[['delta1_friend_cnt', 'delta1_avg_friend_age',
                      'delta1_subscriber_friend_cnt', 'delta1_songsListened','delta1_lovedTracks',
                      'delta1_posts','delta1_playlists', 'delta1_shouts']].values
my_targets = my_df1[['adopter']].values

# Randomly split our data into training and testing data.
f_train, f_test, t_train, t_test = train_test_split(my_features, my_targets, test_size=0.20, stratify=my_targets, random_state=77)

# Create and fit the StandardScaler object to both the training and the testing data
sc_train = StandardScaler().fit(f_train)
f_train_sc = sc_train.transform(f_train)
sc_test = StandardScaler().fit(f_test)
f_test_sc = sc_test.transform(f_test)

# Create a variable that holds the algorithm with the hyper-parameters set
# and train the model by passing in the training features and targets.
num_neighbors = 5
knn = neighbors.KNeighborsClassifier(n_neighbors=num_neighbors, metric='euclidean', weights='uniform')
knn.fit(f_train_sc, t_train)

# over-sampler：smote
from imblearn.over_sampling import SMOTE
over_sampler = SMOTE(random_state=12)
smote_feature, smote_target = over_sampler.fit_resample(f_train, t_train)
my_classifier = knn
smote_model = my_classifier.fit(smote_feature, smote_target)
# Get the scores for the model
score_test = 100 * smote_model.score(f_test, t_test)
print(f'KNN prediction accuracy using SMOTE over-sampler = {score_test:.3f}%.')
# Get the confusion Matrix
smote_predictions= smote_model.predict(f_test)
print(confusion_matrix(t_test, smote_predictions))

# under-sampler: nearmiss
from imblearn.under_sampling import NearMiss
under_sampler = NearMiss()
nm_feature, nm_target = under_sampler.fit_resample(f_train, t_train)
my_classifier = knn
nm_model = my_classifier.fit(nm_feature, nm_target)
# Get the scores for the model
score_test = 100 * nm_model.score(f_test, t_test)
print(f'KNN prediction accuracy using under-sampler (Near Miss) = {score_test:.3f}%.')
# print confusion matrix
from sklearn.metrics import confusion_matrix
nm_predictions= nm_model.predict(f_test)
print(confusion_matrix(t_test, nm_predictions))

#################
# Decision Tree #
#################
from sklearn.tree import DecisionTreeClassifier

my_features = d1df[['delta1_friend_cnt', 'delta1_avg_friend_age',
                      'delta1_subscriber_friend_cnt', 'delta1_songsListened','delta1_lovedTracks',
                      'delta1_posts','delta1_playlists', 'delta1_shouts']].values
my_targets = my_df1[['adopter']].values

# Split our data into train and test
from sklearn.model_selection import train_test_split
f_train, f_test, t_train, t_test = train_test_split(my_features, my_targets, stratify=my_targets, test_size=0.20, random_state=77)

# Fit the Decision Tree Model
from sklearn.tree import DecisionTreeClassifier
dtr_estimator = DecisionTreeClassifier(min_samples_leaf=5, min_samples_split=5, max_depth=10)
trained_dt_model = dtr_estimator.fit(f_train, t_train)

# over-sampler: smote
from imblearn.over_sampling import SMOTE
over_sampler = SMOTE(random_state=12)
smote_feature, smote_target = over_sampler.fit_resample(f_train, t_train)
my_classifier = dtr_estimator
smote_model = my_classifier.fit(smote_feature, smote_target)
# Get the scores
score_test = 100 * smote_model.score(f_test, t_test)
print(f'Decision tree prediction accuracy using SMOTE over-sampler = {score_test:.3f}%.')
# print confusion matrix
predicted_labels = smote_model.predict(f_test)
target_names=["Not Premium", 'Get Premium']
from sklearn.metrics import confusion_matrix
print(confusion_matrix(t_test, predicted_labels))

def confusion(test, predict, title, labels, categories):
    pts, xe, ye = np.histogram2d(test, predict, bins=categories)
    pd_pts = pd.DataFrame(pts.astype(int), index=labels, columns=labels)
    hm = sns.heatmap(pd_pts, annot=True, fmt="d")
    hm.axes.set_title(title, fontsize=20)
    hm.axes.set_xlabel('True Target', fontsize=18)
    hm.axes.set_ylabel('Predicted Label', fontsize=18)
plt.close('all')
confusion(t_test.flatten(), predicted_labels, f'Decision Tree Model', target_names, 2)
plt.savefig(os.path.join(os.getcwd(), 'confusion_DecisionTree_SMOTE.png'))
plt.close('all')

# under-sampler: nearmiss
from imblearn.under_sampling import NearMiss
under_sampler = NearMiss()
nm_feature, nm_target = under_sampler.fit_resample(f_train, t_train)
my_classifier = dtr_estimator
nm_model = my_classifier.fit(nm_feature, nm_target)
# Get the scores
score_test = 100 * nm_model.score(f_test, t_test)
print(f'Decision tree prediction accuracy using under-sampler (Near Miss) = {score_test:.3f}%.')
# print confusion matrix
predicted_labels = nm_model.predict(f_test)
target_names=["Not Premium", 'Get Premium']
from sklearn.metrics import confusion_matrix
print(confusion_matrix(t_test, predicted_labels))

# Let's now display the relative importance of each feature on the Decision Tree model.
# Which features are the most important in our decision tree?
feature_names = ['delta1_friend_cnt', 'delta1_avg_friend_age', 'delta1_subscriber_friend_cnt',
                 'delta1_songsListened','delta1_lovedTracks', 'delta1_posts','delta1_playlists',
                 'delta1_shouts']
for name, val in zip(feature_names, smote_model.feature_importances_):
    print(f'{name} importance = {100.0*val:5.2f}%')

#################
# Random Forest #
#################
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import accuracy_score
from sklearn.metrics import classification_report
from sklearn.metrics import mean_absolute_error
from sklearn.metrics import mean_squared_error
from sklearn.metrics import confusion_matrix
from sklearn.metrics import median_absolute_error
from sklearn.model_selection import train_test_split

# Random Forest for Delta1
my_features = d1df[['delta1_friend_cnt', 'delta1_avg_friend_age',
                      'delta1_subscriber_friend_cnt', 'delta1_songsListened','delta1_lovedTracks',
                      'delta1_posts','delta1_playlists', 'delta1_shouts']].values
my_targets = my_df1[['adopter']].values

# Randomly split our data into training and testing data.
f_train, f_test, t_train, t_test = train_test_split(my_features, my_targets, test_size=0.20, random_state=77)

# Create the hyper parameters and fit the model
hyper_params = {'bootstrap': True, 'max_samples': 1000, 'oob_score': True, 'max_features': 'auto',
                'criterion': 'gini', 'n_estimators': 5000, 'random_state': 1986}
my_classifier = RandomForestClassifier(**hyper_params)
print(my_classifier.get_params())
rf_model = my_classifier.fit(f_train, t_train)

score_test = 100 * rf_model.score(f_test, t_test)
print(f'Random forest prediction accuracy with testing data = {score_test:.1f}%.')
score_train = 100 * rf_model.score(f_train, t_train)
print(f'Random forest prediction accuracy with training data = {score_train:.1f}%.')
# Calculate the prediction so we can use them to calculate the different performance metrics.
model_predictions = rf_model.predict(f_test)



# Build a new Delta2 pandas DataFrame For Engagement#
index = []
my_data = []
my_columns = ['delta2_friend_cnt', 'delta2_avg_friend_age', 'delta2_avg_friend_male',
              'delta2_friend_country_cnt', 'delta2_subscriber_friend_cnt', 'delta2_songsListened',
              'delta2_lovedTracks', 'delta2_posts','delta2_playlists', 'delta2_shouts', 'delta2_good_country']
for c in customers.find({}):
    index.append(c.get('net_user'))
    my_list = []
    for col in my_columns:
        my_list.append(c.get('delta2')[col])
    my_data.append(my_list)
#print(my_data)
d2df = pd.DataFrame(my_data, index=index, columns=my_columns)
#print(d2df.isnull().sum())
#print(d2df.sample(10))
#print(d2df.dtypes)

# Change all the 'NULL' values to 0
for col in my_columns:
    d2df[col] = d2df[col].replace(to_replace='NULL', value=0)
    d2df[col] = d2df[col].apply(pd.to_numeric)
#print(d2df.describe())


# Create the Engagement Score
d2df['engagement'] = d2df['delta2_lovedTracks'] + d2df['delta2_songsListened'] + d2df['delta2_playlists'] \
                     + d2df['delta2_posts'] + d2df['delta2_shouts'] + d2df['delta2_friend_cnt']
#print(d2df.sample(5))



# Question 3.
# Predictions for Engagement Score #
my_features = my_df1[['songsListened', 'lovedTracks', 'posts',
              'playlists', 'shouts', 'tenure']].values
my_targets = d2df[['engagement']].values

# Make train and test data
from sklearn.model_selection import train_test_split
f_train, f_test, t_train, t_test = train_test_split(my_features, my_targets,
                                                    test_size=0.20, random_state=77)

#####################
# Linear Regression #
#####################
from sklearn.linear_model import LinearRegression
# Define the estimator object
lgr = LinearRegression(fit_intercept=True)
# Now lets actually train the model with our data.
model = lgr.fit(f_train, t_train)

# Display the regression equation that best fit our training data
print(f'y = {model.intercept_[0]:5.2f} '
      f'+ {model.coef_[0][0]:5.2f}*songsListened + {model.coef_[0][1]:5.2f}*lovedTracks '
      f'+ {model.coef_[0][2]:5.2f}*posts + {model.coef_[0][3]:5.2f}*playlists + {model.coef_[0][4]:5.2f}*shouts'
      f'+ {model.coef_[0][5]:5.2f}*tenure')

# Compute model predictions for test data
results = model.predict(f_test)

# Compute the score and display the results (Coefficiet of Determination)
score = 100.0 * model.score(f_test, t_test)
print(f'Multivariate Linear Regression Model Score = {score:5.2f}%')

#############################
# KNN model with engagement #
#############################
from sklearn import neighbors
from sklearn.preprocessing import StandardScaler
sc_train = StandardScaler().fit(f_train)
f_train_sc = sc_train.transform(f_train)
sc_test = StandardScaler().fit(f_test)
f_test_sc = sc_test.transform(f_test)

num_neighbors = 450
knn = neighbors.KNeighborsRegressor(n_neighbors=num_neighbors, metric='euclidean', weights='uniform')
knn.fit(f_train_sc, t_train)

score_test = 100 * knn.score(f_test_sc, t_test)
print(f'KNN ({num_neighbors} neighbors) prediction accuracy with test data = {score_test:.3f}%')
score_train = 100 * knn.score(f_train_sc, t_train)
print(f'KNN ({num_neighbors} neighbors) prediction accuracy with test data = {score_train:.3f}%')

#################################
# Random Forest with engagement #
#################################
from sklearn.ensemble import RandomForestRegressor
hyper_params = {'bootstrap': True, 'max_samples': 1000, 'oob_score': True,
                'max_features': 'auto', 'n_estimators': 5000, 'random_state': 1986}
my_regressor = RandomForestRegressor(**hyper_params)
rf_model = my_regressor.fit(f_train, t_train)
# Get the scores
score_test = 100 * rf_model.score(f_test, t_test)
print(f'Random forest prediction accuracy with testing data = {score_test:.1f}%.')
score_train = 100 * rf_model.score(f_train, t_train)
print(f'Random forest prediction accuracy with training data = {score_train:.1f}%.')

# Calculate the prediction so we can use them to calculate the different performance metrics.
model_predictions = rf_model.predict(f_test)
# Print the classification report
print(classification_report(t_test, model_predictions))
# Calculate and display a few other metrics
print(f'Accuracy                 = {accuracy_score(t_test, model_predictions):4.2f}')
print(f'Mean Absolute Error      = {mean_absolute_error(t_test, model_predictions):4.2f}')
print(f'Mean Squared Error       = {mean_squared_error(t_test, model_predictions):4.2f}')
print(f'Median Absolute Error    = {median_absolute_error(t_test, model_predictions):4.2f}')

# Print the confusion matrix!
print(confusion_matrix(t_test, model_predictions))


################################
# Decison Tree with engagement #
################################
dtr_estimator = DecisionTreeClassifier(min_samples_leaf=5, min_samples_split=5, max_depth=10)
trained_dt_model = dtr_estimator.fit(f_train, t_train)
print('Test Score = {:,.1%}'.format(trained_dt_model.score(f_test, t_test)))
#print('Train Score = {:,.1%}'.format(trained_dt_model.score(f_train, t_train)))
print(f'Difference between training and testing to determine potential overfitting '
      f'is {trained_dt_model.score(f_train, t_train) - trained_dt_model.score(f_test, t_test):,.3%} points')

feature_names = ['songsListened', 'lovedTracks', 'posts', 'playlists', 'shouts', 'tenure']
for name, val in zip(feature_names, trained_dt_model.feature_importances_):
    print(f'{name} importance = {100.0*val:5.2f}%')

