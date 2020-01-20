import numpy as np
import pandas as pd
import math
from sklearn.model_selection import train_test_split
from sklearn.svm import LinearSVC
from sklearn.linear_model import LogisticRegression
from sklearn.tree import DecisionTreeClassifier

from sklearn.feature_selection import SelectPercentile
from imblearn.over_sampling import RandomOverSampler



def preprocessing(data):
    # Gender
    replace_gender = {'Male': 1, 'Female': 2, 'Other': 3}
    new_dict = dict(zip(replace_gender.keys(), replace_gender.values()))
    data.gender = data.gender.map(new_dict)

    # Married
    replace_married = {'Yes': 1, 'No': 0}
    new_dict = dict(zip(replace_married.keys(), replace_married.values()))
    data.ever_married = data.ever_married.map(new_dict)

    # Location
    replace_residence_type= {'Urban': 1, 'Rural': 0}
    new_dict = dict(zip(replace_residence_type.keys(), replace_residence_type.values()))
    data.Residence_type = data.Residence_type.map(new_dict)

    # BMI
    mean_bmi = np.mean(data['bmi'])
    data.bmi = data.bmi.fillna(mean_bmi)

    # Smoking
    # Introducing additional feature for data with missing values
    ones = np.ones(len(data))
    data['missing_smoke'] = ones
    data.missing_smoke[pd.isnull(data.smoking_status)] = 0


    replace_smoking_type = {'never smoked': 1, 'formerly smoked': 2, 'smokes': 3}
    new_dict = dict(zip(replace_smoking_type.keys(), replace_smoking_type.values()))
    data.smoking_status = data.smoking_status.map(new_dict)
    data.smoking_status = data.smoking_status.fillna(0)

    # Working
    data = pd.concat([data, pd.get_dummies(data['work_type'])], axis=1)
    data = data.drop("work_type", axis=1)

    # Dropping id
    data = data.drop("id", axis=1)

    return data


def data_analysis(X, y):
    ros = RandomOverSampler(random_state=0)
    X, y = ros.fit_resample(X, y)

    X_train, X_test, y_train, y_test = train_test_split(X, y, random_state=25)

    select = SelectPercentile(percentile=80).fit(X_train, y_train)
    X_train = select.transform(X_train)
    X_test = select.transform(X_test)

    # SVM
    linear_svm = LinearSVC().fit(X_train, y_train)
    y_est = linear_svm.predict(X_test)
    print("SVM Efficiency: {}".format(np.mean(y_est == y_test)))

    # Logistic regression
    linear_r = LogisticRegression().fit(X_train, y_train)
    y_est = linear_r.predict(X_test)
    print("LR Efficiency: {}".format(np.mean(y_est == y_test)))

    # Decision tree
    decision_t = DecisionTreeClassifier().fit(X_train, y_train)
    y_est = decision_t.predict(X_test)
    print("DT Efficiency: {}".format(np.mean(y_est == y_test)))

    cor_coef = np.corrcoef(y.astype('float64'), X[:, 8].astype('float64'))
    print("Correlation coefficient between smoking and having heart attack: {}".format(cor_coef[1][0]))


# Loading data
data = pd.read_csv("/Users/aida/Downloads/healthcare-dataset-stroke-data/train_2v.csv")
features = data.head()
data = pd.DataFrame(data)


# Preprocessing
data = preprocessing(data)

# overall data
y = data.stroke
X = data.drop("stroke", axis=1)

data_analysis(X, y)

# Using only without missing values
data_smoke = data[data.missing_smoke == 1]
data_smoke = data_smoke.drop("missing_smoke", axis=1)

y = data_smoke.stroke
X = data_smoke.drop("stroke", axis=1)

print("For the data with no missing values:")
data_analysis(X, y)





