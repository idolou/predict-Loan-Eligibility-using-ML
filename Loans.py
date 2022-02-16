###########################################################################
import numpy as np
import pandas as pd
from sklearn.preprocessing import LabelEncoder
from sklearn.model_selection import KFold  # For K-fold cross validation
from sklearn.tree import DecisionTreeClassifier, export_graphviz
from sklearn import metrics
import statistics as stat
from sklearn.linear_model import LogisticRegression
# from sklearn.metrics import accuracy_score
from sklearn.model_selection import train_test_split
from sklearn.metrics import classification_report
###########################################################################

###########################################################################
#1. Load File
print("\n########---Q1---########")
Dataframe = pd.read_csv("train_Loan.csv")
print("Dataframe Loaded.")
print("__________________________________________")
# print(Dataframe)

###########################################################################
'''all the values in the program'''
#all the columns in the csv
features = list(Dataframe.columns)
#only the colums that include numerical
NumValue = Dataframe.select_dtypes(include=np.number).columns.tolist()
#Changing all categorical values to numerical:
cat_var = [key for key in dict(Dataframe.dtypes)
           if dict(Dataframe.dtypes)[key] in ['object']]  # Categorical Variable
cat_var = cat_var[1:]

###########################################################################
#2. Frequency Distribution:
print("\n########---Q2---########")
def Values_Freq_Dist(dataframe, features, NumValue):
    """
    Parameters
        ----------
        dataframe : dataframe
    Print Frequency Distribution of all Values from
     a given Data Frame"""

    for f in features:
        if f not in NumValue:
            print(f"\n{f} Frequency Distribution:")
            print(dataframe[f].value_counts())
        else:
            print(f"\n{f} numerical variables:")
            print(dataframe[f].describe())
        print("__________________________________________")

Values_Freq_Dist(Dataframe, features, NumValue)

###########################################################################
#3. Data Types:
print("\n########---Q3---########")
print('\nData Types:')
print(Dataframe.dtypes)
print("__________________________________________")

###########################################################################
#4. Missing Values:
print("\n########---Q4---########")
for insertNumCat in cat_var:
    Dataframe[insertNumCat].fillna(stat.mode(Dataframe[insertNumCat]), inplace=True)

for insertNum in NumValue:
    if Dataframe[insertNum].isin([0, 1, None]).all():
        Dataframe[insertNum].fillna(stat.mode(Dataframe[insertNum]), inplace=True)
    else:
        Dataframe[insertNum].fillna(Dataframe[insertNum].mean(), inplace=True)

print('\nMissing data:')
print(Dataframe.apply(lambda x: sum(x.isnull()), axis=0))
print("__________________________________________")

###########################################################################
#5. Discretization:
print("\n########---Q5---########")

def binning(col, cut_points, labels=None):
    """
    Parameters
        ----------
        dataframe : column of dataframe
        cut_points: bin
        labels    : group_names
    Discretize continuous values into bins for further processing, using the cut function."""

    #Define min and max values:
    min_val = col.min()
    max_val = col.max()

    #create list by adding min and max to cut_points
    break_points = [min_val] + cut_points + [max_val]

    #if no labels provided, use default lables 0 ... (n-1)
    if not labels:
        labels = range(len(cut_points) + 1)

    #Binning using cut function of pandas
    colBin = pd.cut(col, bins=break_points, labels=labels, include_lowest=True)
    return colBin

# Define bins as 0<=x<50, 50<=x<150, 150<=x<300, x>=300
bins = [50, 150, 300]
group_names = ["Lower", "Middle", "Higher", "Extreme"]

# Discretization the values in LoanAmount attribute
Dataframe["Bin_LoanAmount"] = binning(Dataframe["LoanAmount"], bins, group_names)

#Count the number of observations which each value
print("\nBin_LoanAmount Frequency Distribution:")
print(pd.value_counts(Dataframe["Bin_LoanAmount"], sort=False))
print("__________________________________________")

###########################################################################
#6. Outlier Detection:
print("\n########---Q6---########")
# Keep only the ones that are within +3 to -3 standard deviations in the column 'LoanAmount'.
print("\nRecords that are within +3 to -3 standard deviations in the column LoanAmount:")
Dataframe = Dataframe[(np.abs(Dataframe.LoanAmount - Dataframe.LoanAmount.mean()) <= (3 * Dataframe.LoanAmount.std()))]
print(Dataframe[(np.abs(Dataframe.LoanAmount - Dataframe.LoanAmount.mean()) <= (3 * Dataframe.LoanAmount.std()))])
print("__________________________________________")

###########################################################################
#7. Normalized_Income:
print("\n########---Q7---########")
square_root = lambda x: np.sqrt(x) * 0.5
Dataframe['Normalized_Income'] = Dataframe['ApplicantIncome'].apply(square_root)
print("\nNormalized_Income Applied:")
print(Dataframe.iloc[:, 12:])
print("__________________________________________")

###########################################################################
#8. Dummy Variable:
print("\n########---Q8---########")
education_df = pd.get_dummies(Dataframe['Education'])
Dataframe = Dataframe.join(education_df)
print("\nDummy Variable Graduate/Not Graduate was added:")
print(Dataframe.iloc[:, 15:])
print("__________________________________________")

###########################################################################
#9. Convert Categorial to Numerical
print("\n########---Q9---########")
le = LabelEncoder()
for i in cat_var:
    Dataframe[i] = le.fit_transform(Dataframe[i])
print("\nDataframe represented by only numbers:")
print(Dataframe)
print("__________________________________________")

###########################################################################
#10. Export Dataframe To csv File (train_Loan_updated):
print("\n########---Q10---########")
Dataframe.to_csv('train_Loan_updated.csv', index=True)
print("\ncsv File created. We have {0} columns & {1} rows in the new file".format(len(Dataframe.columns), len(Dataframe.index)))
print("__________________________________________")

###########################################################################
#11 + 12. Decision Tree Classifier to predict Loan_Status based on Accuracy and Cross Validation Score:
print("\n########---Q11---########")

# Generic function for making a classification model and accessing performance:
def classification_model(model, data, predictors, outcome):
    """
    Parameters
        ----------
        model         : myModel
        data          : Dataframe
        predictors    : predictor_var
        outcome       : outcome_var
    supervised learning method used for classification and regression.
    The goal is to create a model that predicts the value of a target
    variable by learning simple decision rules inferred from the data features."""

    # Fit the model:
    model.fit(data[predictors], data[outcome])

    # Make predictions on training set:
    predictions = model.predict(data[predictors])

    # Print accuracy
    accuracy = metrics.accuracy_score(predictions, data[outcome])
    print("\nTraining accuracy: %s" % "{0:.3%}".format(accuracy))

    # Perform k-fold cross-validation with 10 folds
    kf = KFold(n_splits=10)
    accuracy = []
    for train, test in kf.split(data):
        # Filter training data
        train_predictors = (data[predictors].iloc[train, :])

        # The target we're using to train the algorithm.
        train_target = data[outcome].iloc[train]

        # Training the algorithm using the predictors and target.
        model.fit(train_predictors, train_target)
        # print(accuracy)
        # Record accuracy from each cross-validation run
        accuracy.append(model.score(data[predictors].iloc[test, :], data[outcome].iloc[test]))
    # Print Cross-Validation
    print("Cross-Validation Score: %s" % "{0:.3%}".format(np.mean(accuracy)))

    # Finding K-Group with highest accuracy
    best_k = max(accuracy)
    print("|\n|")
    print(f"Best accuracy was {best_k:.3%} with {accuracy.index(best_k)}K's")
    # Fit the model again so that it can be referred outside the function:
    model.fit(data[predictors], data[outcome])
print("\nDecisionTreeClassifier fitted")
print("__________________________________________")

#12. Accuracy And Cross Validation Score:
print("\n########---Q12---########")
# Apply DecisionTreeClassifier On necessary features:
print("\nDecisionTreeClassifier with All Variables:")
all_col = Dataframe.columns.tolist()  #getting A list of all columns names
unwanted_var = ['Bin_LoanAmount', 'ApplicantIncome', 'Loan_Status', 'Education', 'Loan_ID']
predictor_var = [ele for ele in all_col if
                 ele not in unwanted_var]  # Remove Unnecessary features include Loan_Status which is the outcome
outcome_var = 'Loan_Status'
myModel = DecisionTreeClassifier()
classification_model(myModel, Dataframe, predictor_var, outcome_var)
print("\n******************************************")

# Apply DecisionTreeClassifier On Credit_History, Gender, Married, Education features:
print("\nDecisionTreeClassifier with Credit_History, Gender, Married, Education Variables:")
outcome_var = 'Loan_Status'
predictor_var = ['Credit_History', 'Married', 'Education', 'Self_Employed']
myModel = DecisionTreeClassifier()
classification_model(myModel, Dataframe, predictor_var, outcome_var)
print("__________________________________________")

###########################################################################
#13. Creating Decision Tree Graph using GraphViz Library:
print("\n########---Q13---########")

def visualize_tree(tree, features_names):
    """Create tree png using graphviz.
    Open the generate 'tree.dot' file in notepad and copy it's contents to http://webgraphviz.com/.Args
    ----
    tree -- scikit learn DecisionTree.
    features_names -- list of features names"""

    dotfile = 'tree.dot'
    export_graphviz(tree, out_file=dotfile, feature_names=features_names, class_names=['No', 'Yes'])


(visualize_tree(myModel, predictor_var))
print("\nDecisionTree Graph Created.\nSee png tree on Word File")
print("__________________________________________")

###########################################################################
#14.
print("\n########---Q14---########")
print("\nSee Answer on Word File")
print("__________________________________________")

###########################################################################
#15.
print("\n########---Q15---########")
'''Logistic regression is the appropriate regression analysis to conduct when the dependent variable is dichotomous (binary).
    Like all regression analyses, the logistic regression is a predictive analysis.'''

X = Dataframe[['Credit_History', 'Gender', 'Married', 'Education']]  #Give the model 4 features to train on: Credit_History,
y = Dataframe.Loan_Status

X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.3, random_state=0)
log_reg = LogisticRegression()

log_reg.fit(X_train, y_train)

Loan_Status_pred = log_reg.predict(X_test)


print("\nAccuracy of logistic regression classifier on test set: {:.3%}%".format(log_reg.score(X_test, y_test)))
print("\nClassification Report:")
print(classification_report(y_test, Loan_Status_pred))
print("\nSee Answer on Word File")
print("__________________________________________")



