import subprocess    
import acquire as ac
import prepare as pr

import os
import numpy as np
import pandas as pd
import numpy.ma as ma

import env
import scipy
from scipy import stats
from sklearn.model_selection import train_test_split
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import classification_report, confusion_matrix, ConfusionMatrixDisplay

import matplotlib.pyplot as plt
#%matplotlib inline #this is giving incorrect syntaxt when imported in ac, so hopefully keeping it in Presentation will work
import seaborn as sns
from matplotlib import rcParams
import matplotlib

# ignore warnings
import warnings
warnings.filterwarnings("ignore")
import pandas as pd
import numpy as np
from scipy import stats

from env import get_db_url


from sklearn.impute import SimpleImputer
import sklearn.preprocessing
from sklearn.tree import DecisionTreeClassifier
from sklearn.neighbors import KNeighborsClassifier
from sklearn.linear_model import LogisticRegression



#IMPORTS above pull in all of the libraries and stats things that we'll need to
#crunch these numbers


#We begin acquiring data below \/ \/ \/

def new_telco_data():
    '''
    This function reads the telco data from the Codeup db into a df.
    '''
    sql_query = """
                select * from customers
                join contract_types using (contract_type_id)
                join internet_service_types using (internet_service_type_id)
                join payment_types using (payment_type_id)
                """
    
    # Read in DataFrame from Codeup db.
    df = pd.read_sql(sql_query, get_db_url('telco_churn'))
    
    return df



def get_telco_data():
    '''
    This function reads in telco data from Codeup database, writes data to
    a csv file if a local file does not exist, and returns a df.
    '''
    if os.path.isfile('telco.csv'):
        
        # If csv file exists read in data from csv file.
        df = pd.read_csv('telco.csv', index_col=0)
        
    else:
        
        # Read fresh data from db into a DataFrame
        df = new_telco_data()
        
        # Cache data
        df.to_csv('telco.csv')
        
    return df




######################PREPARE   PREPARE     PREPARE#######################################
# ------------------- TELCO DATA -------------------

def split_telco_data(df):
    '''
    This function performs split on telco data, stratify churn.
    Returns train, validate, and test dfs.
    '''
    train_validate, test = train_test_split(df, test_size=.2, 
                                        random_state=123, 
                                        stratify=df.churn)
    train, validate = train_test_split(train_validate, test_size=.3, 
                                   random_state=123, 
                                   stratify=train_validate.churn)
    return train, validate, test

def xysplit(train, validate, test):
    x_train = train.drop(columns=['churn'])
    y_train = train.churn

    x_validate = validate.drop(columns=['churn'])
    y_validate = validate.churn

    x_test = test.drop(columns=['churn'])
    y_test = test.churn



def prep_telco_data(df):
     # Drop duplicate columns
    df.drop(columns=['payment_type_id', 'internet_service_type_id', 'contract_type_id', 'customer_id'], inplace=True)
       
     # Drop null values stored as whitespace    
    df['total_charges'] = df['total_charges'].str.strip()
    df = df[df.total_charges != '']
    
    # Convert to correct datatype
    df['total_charges'] = df.total_charges.astype(float)

    #trying to drop nulls if any in monthly charges
    df['monthly_charges'] = df['monthly_charges'].str.strip()
    df = df[df.total_charges != '']
    
    #converted to float
    df['monthly_charges'] = df.total_charges.astype(float)
    
    
#I was getting something out of place so I split them out differently
#Each line replaces the values which aren't numbers, with numbers so they can be
#processed/analyzed

    df['gender'].replace(['Female', 'Male'],[1,0], inplace=True)
    df['contract_type'].replace(['Two year', 'One year', 'Month-to-month'],[2,1,0], inplace=True)
    df['partner'].replace(['Yes', 'No'],[1,0], inplace=True)
    df['dependents'].replace(['Yes', 'No'],[1,0], inplace=True)
    df['paperless_billing'].replace(['Yes', 'No'],[1,0], inplace=True)
    df['phone_service'].replace(['Yes', 'No'],[1,0], inplace=True)
    df['multiple_lines'].replace(['Yes', 'No', 'No phone service'],[1,0,00], inplace=True)
    df['churn'].replace(['Yes', 'No'],[1,0], inplace=True)
    df['online_security'].replace(['Yes', 'No', 'No internet service'],[1,0,00], inplace=True)
    df['online_backup'].replace(['Yes', 'No','No internet service'],[1,0,00], inplace=True)	

    df['device_protection'].replace(['Yes', 'No','No internet service'],[1,0,00], inplace=True)	
    df['tech_support'].replace(['Yes', 'No','No internet service'],[1,0,00], inplace=True)	
    df['streaming_tv'].replace(['Yes', 'No','No internet service'],[1,0,00], inplace=True)	
    df['streaming_movies'].replace(['Yes', 'No','No internet service'],[1,0,00], inplace=True)	

  # df['monthly_charges'].replace(['Yes', 'No'],[1,0], inplace=True)	
    df['internet_service_type'].replace(['DSL', 'Fiber optic','None'],[2,1,0], inplace=True)	
    df['payment_type'].replace(['Mailed check', 'Electronic check', "Bank transfer (automatic)", "Credit card (automatic)" ],[0,1,2,3], inplace=True)	

    
    
    
    # Get dummies for non-binary categorical variables
    dummy_df = pd.get_dummies(df[[#'multiple_lines', \
                              'online_security', \
                              'online_backup', \
                              'device_protection', \
                              'tech_support', \
                              'streaming_tv', \
                              'streaming_movies', \
                              'contract_type', \
                              'internet_service_type', \
                              'payment_type']], dummy_na=False, \
                              drop_first=True)
    
    # Concatenate dummy dataframe to original 
    df = pd.concat([df, dummy_df], axis=1)
    
    df = df.T.drop_duplicates().T
    
    
    return df 
######################################################

#This creates a decision tree classifier and runs the math for our training data

def get_tree(x_train, x_validate, y_train, y_validate):
    '''get decision tree accuracy on train and validate data'''

    # create classifier object
    clf = DecisionTreeClassifier(max_depth=5, random_state=123)

    #fit model on training data
    clf = clf.fit(x_train, y_train)

    # print result
    print(f"Accuracy of Decision Tree on train data is {clf.score(x_train, y_train)}")
    print(f"Accuracy of Decision Tree on validate data is {clf.score(x_validate, y_validate)}")
    

#This creates a random forest classifier and runs the math for our training data

def get_forest(train_X, validate_X, train_y, validate_y):
    '''get random forest accuracy on train and validate data'''

    # create model object and fit it to training data
    rf = RandomForestClassifier(max_depth=4, random_state=123)
    rf.fit(train_X,train_y)

    # print result
    print(f"Accuracy of Random Forest on train is {rf.score(train_X, train_y)}")
    print(f"Accuracy of Random Forest on validate is {rf.score(validate_X, validate_y)}")

#This creates a  KNN classifier and runs the math for our training data

def get_knn(x_train, x_validate, y_train, y_validate):
    '''get KNN accuracy on train and validate data'''

    # create model object and fit it to the training data
    knn = KNeighborsClassifier(n_neighbors=5, weights='uniform')
    knn.fit(x_train, y_train)

    # print results
    print(f"Accuracy of KNN on train is {knn.score(x_train, y_train)}")
    print(f"Accuracy of KNN on validate is {knn.score(x_validate, y_validate)}")

#This creates a linear regression classifier and runs the math for our training data
    
def get_reg(x_train, x_validate, y_train, y_validate):
    '''get logistic regression accuracy on train and validate data'''

    # create model object and fit it to the training data
    logit = LogisticRegression(solver='liblinear')
    logit.fit(x_train, y_train)

    # print result
    print(f"Accuracy of Logistic Regression on train is {logit.score(x_train, y_train)}")
    print(f"Accuracy of Logistic Regression on validate is {logit.score(x_validate, y_validate)}")


# #expected values of churn X internet service type, manually calculated for chi2
# #169.148420	467.501818
# #335.956461	928.535138
# #262.990454	726.867709
# #EXPECTED VALUES manually calculated, already multiplied by N


# def chi2ISTchurn(train):
    
#     index = ['No Churn', 'Churn']
#     columns = ['No Svc', 'DSL', 'FiberOp']

#     observed = pd.DataFrame([[803, 987, 1101], [64, 735, 247]], index=index, columns=columns)
#     n = train.shape[0]

#     expected = pd.DataFrame([[169.148, 335.956, 262.99], [467.5, 928.535, 726.867]], index=index, columns=columns)

#     chi2 = ((observed - expected)**2 / expected).values.sum()


#     degrees_of_freedom = 2  #(2-1)(3-1)=2

#     p = stats.chi2(degrees_of_freedom).sf(chi2)

#     print('Observed')
#     print(observed)
#     print('---\nExpected')
#     print(expected)
#     print('---\n')
#     print(f'chi^2 = {chi2:.4f}')
#     print(f'p     = {p:.4f}')
    
    
    
    
    #this code gets us a scatter plot showing total charges, tenure, and churn status
def getfirstplot(train):
    rcParams['figure.figsize']=10,10
    sns.scatterplot(train.tenure,train.total_charges, hue=train.churn) 
    #rcParams['figure.figsize']=10,10  #may need to run it a few times to get the correct figsize
    
    
    
    
    
    #this code gets us a  plot showing payment type, tenure, and internet svc type
def getsecondplot(train):
    rcParams['figure.figsize']=8,8  
    PTT = sns.scatterplot(train.payment_type, train.tenure,        hue=train.internet_service_type, palette='bright')
    PTT.legend(loc='center left', bbox_to_anchor=(1, 0.5), ncol=1,labels=['No Svc', 'FiberOp','DSL'])
    PTT.set(xlabel='Mailed Check - 0           Electronic Check - 1           Bank Draft (auto) - 2           Credit Card (auto) - 3')
        
        
        
        
#this code gets us a scatter plot contract type, tenure, and churn status       
def getthirdplot(train):
    TD = sns.relplot(x=train.total_charges, y=train.contract_type,size=train.tenure, hue=train.churn, height=4, aspect=3)
    TD.set(ylabel='Monthly   |   One-Year   |   Two-Year')
    
    
    
    
    
#this code gets us a scatter plot showing internet svc type, tenure, and total charges vs churn  
def getfourthplot(train):
    #rcParams['figure.figsize']=30,30
    INTC = sns.relplot(data=train, col=train.internet_service_type, x=train.tenure, y=train.total_charges, hue=train.churn, height = 5, aspect = 1)  
    INTC.set_titles('')
    INTC.fig.suptitle("0-None                                                                    1-FiberOptic                                                                    2-DSL")
    INTC.fig.subplots_adjust(top=1)
    #I couldn't find a real way to change what would've been the {col_name} variable within the plot's available documentation
    #The most simple way is to make an overall title and space out manually; what I ended up with to label each plt
        
        
    
    
#this code gets us a scatter plot showing monthly charge, total, and churn status      
def getfifthplot(train):
    rcParams['figure.figsize']=8,8
    sns.scatterplot(data=train, x=train.monthly_charges, y=train.total_charges, hue=train.churn)
    #rcParams['figure.figsize']=10,10
    
    
    
    
    
    
def getchifirst(train):
    '''get rusults of chi-square for billing rate and churn'''

    observed = pd.crosstab(train.total_charges, train.churn)

    chi2, p, degf, expected = stats.chi2_contingency(observed)

    print(f'chi^2 = {chi2:.4f}')
    print(f'p     = {p:.4f}')   
    
    
    
    
    
def getchisecond(train):
    '''get rusults of chi-square for payment type and services'''

    observed = pd.crosstab(train.payment_type, train.internet_service_type)

    chi2, p, degf, expected = stats.chi2_contingency(observed)

    print(f'chi^2 = {chi2:.4f}')
    print(f'p     = {p:.4f}')   
                     
                 
                 
    
    
def getchithird(train):
    '''get rusults of chi-square for contract type and churn status'''

    observed = pd.crosstab(train.contract_type, train.churn)

    chi2, p, degf, expected = stats.chi2_contingency(observed)

    print(f'chi^2 = {chi2:.4f}')
    print(f'p     = {p:.4f}')   
                     
    
def getchifourth(train):
    '''get rusults of chi-square for internet service type and churn status'''

    observed = pd.crosstab(train.contract_type, train.churn)

    chi2, p, degf, expected = stats.chi2_contingency(observed)

    print(f'chi^2 = {chi2:.4f}')
    print(f'p     = {p:.4f}')  
    
def getchififth(train):
    '''get rusults of chi-square for monthly charges and churn status'''

    observed = pd.crosstab(train.monthly_charges, train.churn)

    chi2, p, degf, expected = stats.chi2_contingency(observed)

    print(f'chi^2 = {chi2:.4f}')
    print(f'p     = {p:.4f}')  
    

    
    
    
    
    
    
def get_reg_test(x_train, x_test, y_train, y_test):
    '''get logistic regression accuracy on train and validate data'''

    # create model object and fit it to the training data
    logit = LogisticRegression(solver='liblinear')
    logit.fit(x_train, y_train)

    # print result
    print(f"Accuracy of Logistic Regression on test is {logit.score(x_test, y_test)}")
    