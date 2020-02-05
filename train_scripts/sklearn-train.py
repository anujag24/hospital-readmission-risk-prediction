import argparse
import os

# Importing libs into the python environment. These functions will be referenced later in the notebook code.
import os
import pandas as pd
import itertools
import pickle
import numpy as np
import csv 
import json
import matplotlib.pyplot as plt
import s3fs
import pyarrow.parquet as pq
from datetime import datetime
from datetime import date
from pyarrow.filesystem import S3FSWrapper
from sklearn.externals import joblib
from sklearn.model_selection import learning_curve

# Put this when it's called
from sklearn.model_selection import train_test_split
from sklearn.model_selection import learning_curve
from sklearn.model_selection import validation_curve
from sklearn.model_selection import cross_val_score

from sklearn.ensemble import RandomForestClassifier


from sklearn.model_selection import train_test_split
from sklearn.model_selection import cross_val_score
from sklearn.model_selection import cross_val_predict

from sklearn.metrics import roc_auc_score
from sklearn.metrics import accuracy_score
from sklearn.metrics import precision_recall_curve
from sklearn.metrics import precision_score, recall_score
from sklearn.metrics import precision_recall_curve, average_precision_score
from sklearn.metrics import f1_score
from sklearn.metrics import roc_curve, auc
from sklearn.metrics import confusion_matrix, classification_report


# Create table for missing data analysis
def draw_missing_data_table(df):
    total = df.isnull().sum().sort_values(ascending=False)
    percent = (df.isnull().sum()/df.isnull().count()).sort_values(ascending=False)
    missing_data = pd.concat([total, percent], axis=1, keys=['Total', 'Percent'])
    return missing_data*100

# Plot learning curve
def plot_learning_curve(estimator, title, X, y, ylim=None, cv=None,
                        n_jobs=1, train_sizes=np.linspace(.1, 1.0, 5)):
    plt.figure()
    plt.title(title)
    if ylim is not None:
        plt.ylim(*ylim)
    plt.xlabel("Training examples")
    plt.ylabel("Score")
    train_sizes, train_scores, test_scores = learning_curve(
        estimator, X, y, cv=cv, n_jobs=n_jobs, train_sizes=train_sizes)
    train_scores_mean = np.mean(train_scores, axis=1)
    train_scores_std = np.std(train_scores, axis=1)
    test_scores_mean = np.mean(test_scores, axis=1)
    test_scores_std = np.std(test_scores, axis=1)
    plt.grid()

    plt.fill_between(train_sizes, train_scores_mean - train_scores_std,
                     train_scores_mean + train_scores_std, alpha=0.1,
                     color="r")
    plt.fill_between(train_sizes, test_scores_mean - test_scores_std,
                     test_scores_mean + test_scores_std, alpha=0.1, color="g")
    plt.plot(train_sizes, train_scores_mean, 'o-', color="r",
             label="Training score")
    plt.plot(train_sizes, test_scores_mean, 'o-', color="g",
             label="Validation score")
    plt.legend(loc="best")
    return plt


# Plot validation curve
def plot_validation_curve(estimator, title, X, y, param_name, param_range, ylim=None, cv=None,
                        n_jobs=1, train_sizes=np.linspace(.1, 1.0, 5)):
    train_scores, test_scores = validation_curve(estimator, X, y, param_name, param_range, cv)
    train_mean = np.mean(train_scores, axis=1)
    train_std = np.std(train_scores, axis=1)
    test_mean = np.mean(test_scores, axis=1)
    test_std = np.std(test_scores, axis=1)
    plt.plot(param_range, train_mean, color='r', marker='o', markersize=5, label='Training score')
    plt.fill_between(param_range, train_mean + train_std, train_mean - train_std, alpha=0.15, color='r')
    plt.plot(param_range, test_mean, color='g', linestyle='--', marker='s', markersize=5, label='Validation score')
    plt.fill_between(param_range, test_mean + test_std, test_mean - test_std, alpha=0.15, color='g')
    plt.grid() 
    plt.xscale('log')
    plt.legend(loc='best') 
    plt.xlabel('Parameter') 
    plt.ylabel('Score') 
    plt.ylim(ylim)
    
#plot Confusion Matrix function

def plot_confusion_matrix(cm, classes,
                          normalize=False,
                          title='Confusion matrix',
                          cmap=plt.cm.Blues):
    """
    This function prints and plots the confusion matrix.
    Normalization can be applied by setting `normalize=True`.
    """
    if normalize:
        cm = cm.astype('float') / cm.sum(axis=1)[:, np.newaxis]
        print("Normalized confusion matrix")
    else:
        print('Confusion matrix, without normalization')

    print(cm)

    plt.imshow(cm, interpolation='nearest', cmap=cmap)
    plt.title(title)
    plt.colorbar()
    tick_marks = np.arange(len(classes))
    plt.xticks(tick_marks, classes, rotation=45)
    plt.yticks(tick_marks, classes)

    fmt = '.2f' if normalize else 'd'
    thresh = cm.max() / 2.
    for i, j in itertools.product(range(cm.shape[0]), range(cm.shape[1])):
        plt.text(j, i, format(cm[i, j], fmt),
                 horizontalalignment="center",
                 color="white" if cm[i, j] > thresh else "black")

    plt.ylabel('True label')
    plt.xlabel('Predicted label')
    plt.tight_layout()
    plt.show()
    plt.close()


# end Confusion Matrix function


# ROC Curve Plotting function

def plot_roc_curve(fpr,tpr):
    plt.title('Receiver Operating Characteristic')
    plt.plot(fpr, tpr, 'b',
             label='AUC = %0.2f' % roc_auc)
    plt.legend(loc='lower right')
    plt.plot([0, 1], [0, 1], 'r--')
    plt.xlim([-0.1, 1.2])
    plt.ylim([-0.1, 1.2])
    plt.ylabel('True Positive Rate')
    plt.xlabel('False Positive Rate')
    plt.show()
    plt.close()

#end ROC plotting function


# Precision Recall (PR) function

def plot_precision_recall_vs_threshold(precisions, recalls, thresholds):
    plt.plot(thresholds, precisions[:-1], "b--", label="Precision", linewidth=2)
    plt.plot(thresholds, recalls[:-1], "g-", label="Recall", linewidth=2)
    plt.xlabel("Threshold", fontsize=16)
    plt.legend(loc="upper left", fontsize=16)
    plt.ylim([0, 1])
    plt.show( )
    plt.close()

#end PR function

# inference functions ---------------
def model_fn(model_dir):
    clf = joblib.load(os.path.join(model_dir, "model.joblib"))
    return clf
    

def input_fn(request_body, request_content_type):
    """An input_fn that loads a pickled numpy array"""
    if request_content_type == "text/csv":
        df_first_merge = p_dataset.read().to_pandas()
        ## Selecting data - Ignoring wellness (since patient not admitted)
        df_first_merge = df_first_merge[df_first_merge['encounters_encounterclass'].isin(['inpatient','ambulatory','urgentcare','outpatient','emergency'])] 
        df_first_merge["encounters_stop"] = pd.to_datetime(df_first_merge["encounters_stop"]).dt.tz_localize(None)
        df_first_merge["encounters_start"] = pd.to_datetime(df_first_merge["encounters_start"]).dt.tz_localize(None)
        ## There are entries in which procedures_code is missing, reason could be that no procedures were performed 
        ## so setting the value as zero for those outliers 
        df_first_merge['procedures_code'] = df_first_merge['procedures_code'].fillna(0, inplace=False)

        ## Setting the procedures base cost to zero since there was no procedure performed and no charge shown
        df_first_merge['procedures_base_cost'] = df_first_merge['procedures_base_cost'].fillna(0, inplace=False) 

        df_first_merge['providers_utilization'] = df_first_merge['procedures_base_cost'].fillna(0, inplace=False)
        df_first_merge['organizations_revenue'] = df_first_merge['procedures_base_cost'].fillna(0, inplace=False)
        df_first_merge['patient_birthdate']= pd.to_datetime(df_first_merge['patient_birthdate'])
        def calculate_age(born):
            today = date.today()
            return today.year - born.year - ((today.month, today.day) < (born.month, born.day))
        ## Convert BIRTH DATE INTO AGE factor
        df_first_merge['age'] = df_first_merge['patient_birthdate'].apply(calculate_age)
        ## Dropping the columns which are not required to do feature selection
        df_first_merge=df_first_merge.drop(columns=['medications_totalcost','medications_base_cost',
                                          'patient_birthdate','encounters_id','patient_id','organizations_id',
                                         'encounters_start','encounters_stop',
                                            'encounters_payer'])
        ## Identifying categorical variables in the data set i.e. enumerated values for variables such as Gender, Organization State,Encounter Class, Encounter Code, Encounter Cost, Marital Status
## We are identifying unique values for a column, if unique values are less than 25 we consider it as categorical variable
        df_first_merge['encounters_encounterclass'] = pd.Categorical(df_first_merge['encounters_encounterclass'])
        df_first_merge['patient_gender'] = pd.Categorical(df_first_merge['patient_gender'])
        df_first_merge['patient_marital'] = pd.Categorical(df_first_merge['patient_marital'])
        df_first_merge['patient_ethnicity'] = pd.Categorical(df_first_merge['patient_ethnicity'])
        df_first_merge['patient_race'] = pd.Categorical(df_first_merge['patient_race'])
        df_first_merge['providers_speciality'] = pd.Categorical(df_first_merge['providers_speciality'])
        df_first_merge['providers_state'] = pd.Categorical(df_first_merge['providers_state'])
        df_first_merge['encounters_reasoncode'] = pd.Categorical(df_first_merge['encounters_reasoncode'])
        df_first_merge['encounters_code'] = pd.Categorical(df_first_merge['encounters_code'])
        df_first_merge['procedures_code'] = pd.Categorical(df_first_merge['procedures_code'])

        # Transform categorical variables into dummy variables
        df_first_merge = pd.get_dummies(df_first_merge, drop_first=True)  # To avoid dummy trap
        ## As a next step, we can use trusted pickle to check for signature on pickle dump when it is read to avoid spoofing
        with open('readmission-predict-input-data.pkl', 'wb') as handle:
            pickle.dump(df_first_merge, handle, protocol=pickle.HIGHEST_PROTOCOL)
        s3.Bucket('readmission-data').upload_file('readmission-predict-input-data.pkl','batch-transform/processed-data/readmission-predict-input-data.pkl',ExtraArgs={"ServerSideEncryption": "aws:kms","SSEKMSKeyId":"3a90a5d2-2ba8-4942-b9df-9a27ff7bf412" })

        return 'readmission-predict-input-data.pkl'
    else:
        # Handle other content-types here or raise an Exception
        # if the content type is not supported.
        pass
        
        
if __name__ =='__main__':

    parser = argparse.ArgumentParser()

    # hyperparameters sent by the client are passed as command-line arguments to the script.
    parser.add_argument('--epochs', type=int, default=50)
    parser.add_argument('--batch-size', type=int, default=64)
    parser.add_argument('--learning-rate', type=float, default=0.05)
    parser.add_argument('--estimators', type=float, default=100)

    # Data, model, and output directories
    parser.add_argument('--output-data-dir', type=str, default=os.environ['SM_OUTPUT_DATA_DIR'])
    parser.add_argument('--model-dir', type=str, default=os.environ.get('SM_MODEL_DIR'))
    parser.add_argument('--train', type=str, default=os.environ.get('SM_CHANNEL_TRAIN'))
    parser.add_argument('--test', type=str, default=os.environ.get('SM_CHANNEL_TEST'))

    args, _ = parser.parse_known_args()
    print ('Reading the training input file')
    # Take the set of files and read them all into a single pandas dataframe
    input_files = [ os.path.join(args.train, file) for file in os.listdir(args.train) ]
    if len(input_files) == 0:
        raise ValueError(('There are no files in {}.\n' +
                          'This usually indicates that the channel ({}) was incorrectly specified,\n' +
                          'the data specification in S3 was incorrectly specified or the role specified\n' +
                          'does not have permission to access the data.').format(args.train, "train"))
    raw_data = [ pickle.load(open(file, 'rb')) for file in input_files ]
    df_ft_select_category = pd.concat(raw_data)
    print ('Training input load complete')
    # shuffle the samples
    df_ft_select_category = df_ft_select_category.sample(n = len(df_ft_select_category), random_state = 42)
    df_ft_select_category = df_ft_select_category.reset_index(drop = True)

    # Save 30% of the data as validation and test data 
    df_valid_test=df_ft_select_category.sample(frac=0.30,random_state=42)

    df_test = df_valid_test.sample(frac = 0.5, random_state = 42)
    df_valid = df_valid_test.drop(df_test.index)

    # use the rest of the data as training data
    df_train_all=df_ft_select_category.drop(df_valid_test.index)

    print('Test prevalence(n = %d):'%len(df_test),df_test.readmission.sum()/ len(df_test))
    print('Valid prevalence(n = %d):'%len(df_valid),df_valid.readmission.sum()/ len(df_valid))
    print('Train all prevalence(n = %d):'%len(df_train_all), df_train_all.readmission.sum()/ len(df_train_all))
    print('all samples (n = %d)'%len(df_ft_select_category))
    assert len(df_ft_select_category) == (len(df_test)+len(df_valid)+len(df_train_all)),'math didnt work'
    
    
    # split the training data into positive and negative
    rows_pos = df_train_all.readmission == 1
    df_train_pos = df_train_all.loc[rows_pos]
    df_train_neg = df_train_all.loc[~rows_pos]

    # merge the balanced data
    df_train = pd.concat([df_train_pos, df_train_neg.sample(n = len(df_train_pos), random_state = 42)],axis = 0)

    # shuffle the order of training samples 
    df_train = df_train.sample(n = len(df_train), random_state = 42).reset_index(drop = True)


    print('Train prevalence (n = %d):'%len(df_train), df_train.readmission.sum()/ len(df_train))
    
    
    ## Random Forest Model with overfitting data
    ## This is giving incorrect results. It's predicting that certain procedures require readmission but when validated 
    ## with the data, it was not the case. For example 

    X_train = df_train[df_train.columns.difference(['readmission'])]
    Y_train = df_train['readmission']
    x_train, x_test, y_train, y_test = train_test_split(X_train, Y_train, test_size=0.30, random_state=0)

    from sklearn.ensemble import RandomForestClassifier
    # Get the hyperparameter value from input channel 
    estimators = int(args.estimators)


    RF = RandomForestClassifier(random_state=42, n_estimators=estimators)
    print ('starting to fit the model')
    RF.fit(x_train, y_train)
    print ('Model fit complete')
    cross_val_score(RF, x_train, y_train, cv=3, scoring="accuracy")
    y_test_pred_RF = cross_val_predict(RF, x_test, y_test, cv=3)
    predictions = RF.predict(x_test)
    score = RF.score(x_test, y_test)
    y_test_pred = y_test_pred_RF
    print("RF Accuracy",accuracy_score(y_test, y_test_pred))
    roc = roc_auc_score(y_test, y_test_pred)
    print("roc score", roc)
    RF_CR = classification_report(y_test, y_test_pred)
    print("Classification Report")
    print("="*50)
    print(RF_CR)
    # Plot   confusion matrix
    cnf_matrix = confusion_matrix(y_test, y_test_pred)
    np.set_printoptions(precision=2)

    plot_confusion_matrix(cnf_matrix, classes=['Class 0', 'Class 1'],
                          title='Confusion matrix, without normalization')


    # Compute micro-average ROC curve and ROC area
    fpr, tpr, thrsehold = roc_curve(y_test , y_test_pred)

    roc_auc = roc_auc_score(y_test_pred, RF.predict(x_test))
    fpr, tpr, thresholds = roc_curve(y_test, RF.predict_proba(x_test)[:,1])
    plt.figure()
    plt.plot(fpr, tpr, label='Random Forest ' )
    plt.plot([0, 1], [0, 1],'r--')
    plt.xlim([0.0, 1.0])
    plt.ylim([0.0, 1.05])
    plt.xlabel('False Positive Rate')
    plt.ylabel('True Positive Rate')
    plt.title('Receiver operating characteristic')
    plt.legend(loc="lower right")
    plt.savefig("randonforest-roc-auc-curve.png")
    plt.show()


    ## Precision Recall Curve 
    y_score_rf = RF.predict_proba(x_test)[:,-1]
    average_precision = average_precision_score(y_test, y_score_rf)
    print('Average precision-recall score RF: {}'.format(average_precision))
    precision, recall, thresholds = precision_recall_curve(y_test, y_score_rf)

    plt.step(recall, precision, color='b', alpha=0.2,
             where='post')
    plt.fill_between(recall, precision, step='post', alpha=0.2,
                     color='b')

    plt.xlabel('Recall')
    plt.ylabel('Precision')
    plt.ylim([0.0, 1.05])
    plt.xlim([0.0, 1.0])
    plt.title('2-class Precision-Recall curve: AP={0:0.2f}'.format(
              average_precision))
    plt.show()

    # decision thresholds plot
    plot_precision_recall_vs_threshold(precision, recall, thresholds)
    
    ## After analysis of the different models, it is found that Randon Forest Model fits best on the current data set 
    ## As a next step, Identifying the significant features in the model
    feature_list = list(x_train.columns)


    # Get numerical feature importances
    importances = list(RF.feature_importances_)
    # List of tuples with variable and importance
    feature_importances = [(feature, round(importance, 4)) for feature, importance in zip(feature_list, importances)]
    # Sort the feature importances by most important first
    feature_importances = sorted(feature_importances, key = lambda x: x[1], reverse = True)
    print ("List of significant features correlated to readmission within 30 days")
    for pair in feature_importances:
        if pair[1] > 0:
            print (pair)
         
    

    #with open('significant_features.csv', 'w') as feature_file:
    #    fieldnames = ['significant_feature','importance']
    #    file_writer = csv.DictWriter(feature_file,fieldnames=fieldnames)
    #    item_length = len(feature_importances[0])
    #    file_writer.writeheader()
    #    for pair in feature_importances:
    #        if pair[1] > 0:
    #            print (pair)
    #            file_writer.writerow(pair)
    #    s3 = boto3.client('s3')
    #    s3.Bucket(s3_bucket).upload_file(feature_file,s3_prefix+'/significant_features.csv')
        
    with open(os.path.join(args.model_dir, "significant_features.csv"), 'w') as feature_file:
        file_writer = csv.writer(feature_file)
        item_length = len(feature_importances[0])
        file_writer.writerow(['significant_feature','importance'])
        for pair in feature_importances:
            if pair[1] > 0:
                print (pair)
                file_writer.writerow(pair)

    ## Dump the model to Sagemaker filesystem model directory 
    joblib.dump(RF, os.path.join(args.model_dir, "model.joblib"))
