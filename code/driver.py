# -- coding: utf-8 --
"""Mehrbod.ipynb

Automatically generated by Colaboratory.

Original file is located at
    https://colab.research.google.com/drive/1nyx5VDBTP2QXsau_VP4Wa3_v4_frkVyO

"""
import pandas as pd
import numpy as np
import seaborn as sns
import pickle
import matplotlib.pyplot as plt
from sklearn.preprocessing import LabelEncoder
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler
from imblearn.over_sampling import SMOTE
from sklearn.model_selection import KFold
from sklearn.naive_bayes import GaussianNB
from sklearn.neighbors import KNeighborsClassifier
from sklearn.model_selection import GridSearchCV
from sklearn.tree import DecisionTreeClassifier
from sklearn.linear_model import Perceptron
from sklearn.linear_model import LogisticRegression
from sklearn.ensemble import RandomForestClassifier
from sklearn.ensemble import AdaBoostClassifier
from sklearn.model_selection import RandomizedSearchCV
from xgboost import XGBClassifier
from sklearn.svm import SVC
from sklearn.metrics import roc_curve, auc
# import tensorflow as tf
# from tensorflow import keras
# from tensorflow.keras import Sequential
# from tensorflow.keras.layers import Dense 
# from tensorflow.keras.layers import Dropout
# from tensorflow.keras.losses import BinaryCrossentropy
from sklearn.metrics import accuracy_score,confusion_matrix,classification_report


def load_dataset(path):
    """importing dataset as panda dataframe

    Args:
        path (str): dataset path

    Returns:
        dataframe: panda dataframe
    """
    dataframe = pd.read_csv(path)
    return dataframe

def unique_values(dataframe):
    """we are listing the unique values of each column

    Args:
        dataframe (dataframe): cdc dataset
    """
    # creat a list of features as columns_df
    columns_df = list(dataframe.columns.values)
    # loop over dataset to print all unique values
    for column in columns_df:
        print(column, ':', str(dataframe[column].unique()))

def encode_age_category(dataframe):
    """encoding the AgeCategory column to integer

    Args:
        dataframe (dataframe): cdc dataset
    """
    # create a variable as dictionary and assign all unique values of 'AgeCategory' column
    encode_age = {'55-59':57, '80 or older':80, '65-69':67,
                '75-79':77,'40-44':42,'70-74':72,'60-64':62,
                '50-54':52,'45-49':47,'18-24':21,'35-39':37,
                '30-34':32,'25-29':27}
    # apply the dictionary into the 'AgeCategory' column
    dataframe['AgeCategory'] = dataframe['AgeCategory'].apply(lambda x: encode_age[x])
    # change values of the 'AgeCategory' to integer
    dataframe['AgeCategory'] = dataframe['AgeCategory'].astype(int)

def categorical_features(dataframe):
    """find categorical features

    Args:
        dataframe (datatfram): cdc dataset

    Returns:
        list: list of columns
    """
    # select all categorical features
    encode_age_category(dataframe)
    return dataframe.select_dtypes(include=[object])

def numerical_features(dataframe):
    """find numerical features

    Args:
        dataframe (datatfram): cdc dataset

    Returns:
        list: list of columns
    """
    # select all numerical features
    encode_age_category(dataframe)
    return dataframe.select_dtypes(include=[np.number])

def univariate_categorical_graph(dataframe, cat_features):
    """plot all categorical features

    Args:
        dataframe (dataframe): cdc dataset
        cat_features (list): list of all categorical features
    """
    # call encode_age_category function to encode the values
    # encode_age_category(dataframe)
    i = 1
    # create an empty canvas of size 25x15
    plt.figure(figsize = (25,15))
    # loop over categorical features and plot them
    for feature in cat_features:
        # plot each column in the canvas
        plt.subplot(3,5,i)
        sns.set(palette='Paired')
        sns.set_style("ticks")
        # create x bar under each graph
        plt_ax = sns.countplot(x = feature, data = dataframe)
        # create lable for each graph
        plt_ax.set_xticklabels(plt_ax.get_xticklabels(), rotation=45, ha="center")
        i +=1
    # plot the canvas
    plt.show()

def univariate_numerical_graph(dataframe, num_features):
    """plot all numerical features

    Args:
        dataframe (dataframe): cdc dataset
        num_features (list): list of all numerical features
    """
    i=1
    # creat an empty canvas of size 35*5
    plt.figure(figsize = (35,5))
    # loop over numerical feature and plot them
    for feature in num_features:
        # plot each column in the canvas
        plt.subplot(1,5,i)
        sns.set(palette='dark')
        sns.set_style("ticks")
        # plot hist graph of each feature and use kde(Kernel density estimate) on them
        sns.histplot(dataframe[feature], kde=True)
        # creat x bar for each graph as feature
        plt.xlabel(feature)
        # creat y bar for each graph as count
        plt.ylabel("Count")
        i+=1
    plt.show()

def univariate_numerical_statistic(dataframe, num_features):
    """print numerical feature's statistic

    Args:
        dataframe (dataframe): cdc dataset
    """
    cat_col = list(num_features)
    # creat a list of univariate numreical feature as num_col
    print(dataframe.describe()[1:][cat_col])

def bivariate_categorical_graph(dataframe, cat_features):
    """plot all categorical features, related to target feature

    Args:
        dataframe (dataframe): cdc dataset
        cat_features (list): list of all categorical features
    """
    i = 1
    # creat an empty canvas size of 25*15
    plt.figure(figsize = (25,15))
    # loop over categorical feature and plot them
    for feature in cat_features:
        # plot each column in the canvas
        plt.subplot(3,5,i)
        sns.set(palette='Paired')
        sns.set_style("ticks")
        # hue; plot each feature realted to target column(HeartDisease)
        plt_ax = sns.countplot(x = feature, data = dataframe, hue = 'HeartDisease')
        # creat label for each graph
        plt_ax.set_xticklabels(plt_ax.get_xticklabels(), rotation=45, ha="right")
        i +=1
    plt.show()

def bivariate_numerical_graph(dataframe, num_features, target):
    """plot numerical features, related to col_name

    Args:
        dataframe (dataframe): cdc dataset
        num_features (list): list of numerical features
    """
    i=1
    # creat an empty canvas size of 35*5
    plt.figure(figsize=(35,5))
    sns.set(palette='Paired')
    sns.set_style("ticks")
    # loop over numerical feature and plot them
    for feature in num_features:
        # plot each column in the canvas
        plt.subplot(1,5,i)
        # plot numerical features using boxplot
        sns.boxplot(y=dataframe[feature], x = dataframe[target])
        i+=1
    plt.show()

def preprocessing_encode_columns(dataframe):
    """preprocess data for applying ML models

    Args:
        dataframe (dataframe): cdc dataset

    Returns:
        dataframe: panda dataframe of all encoded features
    """
    encode_age_category(dataframe)
    # Encode columns with exact 2 unique values
    for col in ['HeartDisease', 'Smoking', 'AlcoholDrinking', 'Stroke', 'DiffWalking', 'Sex', 'PhysicalActivity', 'Asthma', 'KidneyDisease', 'SkinCancer']:
        if dataframe[col].dtype == 'O':
            le = LabelEncoder()
            dataframe[col] = le.fit_transform(dataframe[col])
    # One-hot encode for columns with more than 2 unique values
    dataframe = pd.get_dummies(dataframe, columns=['Race', 'Diabetic', 'GenHealth'], prefix = ['Race', 'Diabetic', 'GenHealth'])
    return dataframe

def preprocessing_splitting(dataframe, target, split_size):
    """Preprocessing; splitting dataset: 20/100 and 80/100

    Args:
        dataframe (dataframe): cdc dataset

    Returns:
        dataframe: dataframe
    """
    # split the data into train and test
    train_data, test_data = train_test_split(dataframe, train_size=split_size)
    # split the train data into train and validation
    X_train = train_data.drop(target, axis=1)
    y_train = train_data[target]
    X_test = test_data.drop(target, axis=1)
    y_test = test_data[target]
    # return the splitted data
    return X_train, y_train, X_test, y_test

def preprocessing_scaling(X_train, X_test):
    """Preprocessing; scaling dataset

    Args:
        X_train (dataframe): train dataset
        X_test (dataframe): test dataset

    Returns:
        dataframe: dataframe
    """
    # scale the data
    sc = StandardScaler()
    # fit the data
    X_train=sc.fit_transform(X_train)
    # transform the data
    X_test=sc.transform(X_test)
    # return the scaled data
    return X_train, X_test

def balance_data(method, X_train, y_train):
    """balancing data

    Args:
        method (string): method's name
        X_train (dataframe): dataframe
        y_train (dataframe): dataframe

    Returns:
        dataframe: dataframe
    """
    # if method is SMOTE
    if (method == 'smote'):
        # smote
        smote = SMOTE(sampling_strategy='minority')
        # fit the data
        X_train_blnc, y_train_blnc = smote.fit_resample(X_train,y_train)
        # return the balanced data
        return X_train_blnc, y_train_blnc
    # if method is else
    else:
        return 0

def k_fold_cross_validation(number_of_split):
    """k-fold cross validation

    Args:
        number_of_split (int): number of split

    Returns:
        KFold function: list of n dataframe
    """
    # creat a list of k
    return KFold(n_splits=number_of_split, random_state=None,shuffle=False)

def fit_model(model, X_train, y_train, X_test, y_test, cv, params, v, pkl_file):
    """fit model
    Args:
        model (string): model's name
        X_train (dataframe): dataframe
        y_train (dataframe): dataframe
        cv (list/datframe): list of n dataframe
        params (dictionary): dictionary of parameters
        v (int): verbose: the higher, the more messages
        pkl_file (pkl): pickle file for save the model
    """
    if(model == 'decision_tree'):
        model = DecisionTreeClassifier()
        model_cv = GridSearchCV(model, param_grid=params, cv=cv, verbose=v)
        model_cv.fit(X_train , y_train)
        print(model_cv.best_params_)
        pickle.dump(model_cv, open(pkl_file, 'wb'))
        roc_auc(model_cv, X_test, y_test)
        return model_cv

    elif(model == 'perceptron'):
        model = Perceptron()
        model_cv = GridSearchCV(model , param_grid=params, cv=cv, verbose=v)
        model_cv.fit(X_train, y_train)
        print(model_cv.best_params_)
        pickle.dump(model_cv, open(pkl_file, 'wb'))
        roc_auc(model_cv, X_test, y_test)
        return model_cv

    elif (model == 'naive_bayes'):
        model = GaussianNB()
        model_cv = GridSearchCV(model, param_grid=params ,cv=cv, verbose=v) 
        model_cv.fit(X_train, y_train)
        print(model_cv.best_params_)
        pickle.dump(model_cv, open(pkl_file, 'wb'))
        roc_auc(model_cv, X_test, y_test)
        return model_cv

    elif (model == 'knn'):
        model = KNeighborsClassifier()
        model_cv = GridSearchCV(model, param_grid=params, cv=cv, verbose=v)
        model_cv.fit(X_train, y_train)
        print(model_cv.best_params_)
        # pickle.dump(model_cv, open(pkl_file, 'wb'))
        # roc_auc(model_cv, X_test, y_test)
        return model_cv

    elif(model == 'logistic_regression'):
        model = LogisticRegression()
        model_cv = GridSearchCV(model, param_grid=params, cv=cv, verbose=v)
        model.fit(X_train,y_train)
        print(model_cv.best_params_)
        pickle.dump(model_cv, open(pkl_file, 'wb'))
        roc_auc(model_cv, X_test, y_test)
        return model_cv

    # elif(model == 'nn'):
    #     model=Sequential([
    #         Dense(units=512, activation='relu'),
    #         Dropout(0.2),
    #         Dense(units=512, activation='relu'),
    #         Dropout(0.2),
    #         Dense(units=512, activation='relu'),
    #         Dropout(0.2),
    #         Dense(units=256, activation='relu'),
    #         Dropout(0.2),
    #         Dense(units=128, activation='relu'),
    #         Dropout(0.2),
    #         Dense(units=1 , activation='sigmoid')
    #     ])
    #     model.compile(optimizer='adam', loss='binary_crossentropy',metrics=['accuracy'])
    #     model.fit(X_train, y_train, epochs=30)
    #     pickle.dump(model, open(pkl_file, 'wb'))
    #     return best_params

    elif(model == 'random_forest'):
        model = RandomForestClassifier()
        model_cv = GridSearchCV(model, param_grid=params, cv=cv, verbose=v)
        model_cv.fit(X_train, y_train)
        print(model_cv.best_params_)
        pickle.dump(model_cv, open(pkl_file, 'wb'))
        roc_auc(model_cv, X_test, y_test)
        return model_cv

    elif(model == 'svm'):
        model = SVC()
        model_cv = GridSearchCV(model, param_grid=params, cv=cv, verbose=v)
        model_cv.fit(X_train, y_train)
        print(model_cv.best_params_)
        pickle.dump(model_cv, open(pkl_file, 'wb'))
        roc_auc(model_cv, X_test, y_test)
        return model_cv

    elif(model == 'xg_boost'):
        model = XGBClassifier()
        model_cv = GridSearchCV(model, param_grid=params, cv=cv, verbose=v)
        model_cv.fit(X_train, y_train)
        print(model_cv.best_params_)
        pickle.dump(model_cv, open(pkl_file, 'wb'))
        roc_auc(model_cv, X_test, y_test)
        return model_cv

    elif(model == 'ada_boost'):
        model = AdaBoostClassifier()
        model_cv = GridSearchCV(model, param_grid=params, cv=cv, verbose=v)
        model_cv.fit(X_train, y_train)
        print(model_cv.best_params_)
        pickle.dump(model_cv, open(pkl_file, 'wb'))
        roc_auc(model_cv, X_test, y_test)
        return model_cv

    else:
        print('''model's name is not correct
        please choose one of the following models:
        decision_tree
        perceptron
        naive_bayes
        knn
        logistic_regression
        random_forest
        svm
        xg_boost
        ada_boost''')

def load_model(pkl_file):
    loaded_model = pickle.load(open(pkl_file, 'rb'))
    return loaded_model

def classification_rep(X_test, y_test, loaded_model):
    y_pred = loaded_model.predict(X_test)
    print(classification_report(y_test, y_pred))
    conf_matrix = confusion_matrix(y_test, y_pred)
    plt.figure(figsize=(10,7))
    sns.heatmap(conf_matrix, annot=True, linewidths=0.8, fmt='d', cmap='Blues')
    plt.xlabel('Predicted',fontsize=16)
    plt.ylabel('Truth',fontsize=16)
    plt.show()

def roc_auc(model_cv, X_test, y_test):
    y_score_model = model_cv.predict_proba(X_test)
    yes_probs = y_score_model[:,1]
    plt.figure(figsize=(10,7), dpi=100)
    plt.plot([0,1],[0,1],linestyle='--',label='No Skill')
    fpr , tpr, threshold = roc_curve(y_test,yes_probs)  # false positive, true posistive, threshold
    #AUC
    auc_model = auc(fpr, tpr)
    plt.plot(fpr, tpr, marker='_', label='(auc=%0.3f)' % auc_model, color='Orange')
    plt.xlabel("False Positive Rate")
    plt.ylabel("True Positive Rate")
    plt.legend()
    plt.show()
    

# def main():
#     """Driver
#     """
#     df_cdc = load_dataset('G:\\My Drive\\CDC_ML\\2-Working\\dataset\\heart_2020_cleaned.csv')
#     # unique_values(df_cdc)

#     # print(numerical_features        (df_cdc))
#     # print(categorical_features      (df_cdc)) 
#     # univariate_categorical_graph    (df_cdc, categorical_features(df_cdc))
#     # univariate_numerical_graph      (df_cdc, numerical_features(df_cdc))
#     # univariate_numerical_statistic  (df_cdc, numerical_features(df_cdc))
#     # bivariate_categorical_graph     (df_cdc, categorical_features(df_cdc))
#     # bivariate_numerical_graph       (df_cdc, numerical_features(df_cdc), 'DiffWalking')
    
#     df_encode                           = preprocessing_encode_columns  (df_cdc)
#     X_train, y_train, X_test, y_test    = preprocessing_splitting       (df_encode, 'HeartDisease', 0.8)
#     X_train_scale, X_test_scale         = preprocessing_scaling         (X_train, X_test)    
#     X_train_blnc, y_train_blnc          = balance_data                  ('smote', X_train_scale, y_train)
#     cv                                  = k_fold_cross_validation       (10)

#     params_dt = {
#         "criterion"     :   ["gini", "entropy", "log_loss"], 
#         "max_depth"     :   [100],
#         "random_state"  :   [1024]
#         }

#     params_nb = {
#         'var_smoothing': np.logspace(1,10, num=10)
#         }

#     params_knn = {
#         'algorithm':['ball_tree', 'kd_tree', 'brute'],
#         'n_neighbors':list(range(1,101)),
#         'weights':['uniform', 'distance'],
#         'leaf_size':list(range(1,101)),
#         'p':[1,2],
#         'metric':['minkowski', 'euclidean', 'manhattan'],
#         }

#     y_pred = fit_model('decision_tree', # model name
#                         X_train_blnc,   # X_train
#                         y_train_blnc,   # y_train
#                         X_test_scale,   # X_test
#                         y_test,         # y_test
#                         cv,             # cv
#                         params_dt,      # parameters 
#                         3,              # verbose
#                         'D:\\CDC_ML_pkl\\test.pkl' # pkl file location
#                         )

#     loaded_model = load_model('D:\\CDC_ML_pkl\\test.pkl')
#     classification_rep(X_test_scale, y_test, loaded_model)


# if __name__ == "__main__" :
#     main()