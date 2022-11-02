import streamlit as st
import pandas as pd
from driver import *
from sklearn.model_selection import train_test_split
from sklearn.ensemble import RandomForestRegressor
from sklearn.metrics import mean_squared_error, r2_score
from sklearn.datasets import load_diabetes, load_boston
from PIL import Image

#---------------------------------#
# Page layout
## Page expands to full width
st.set_page_config(page_title='Machine Learning Projects App', initial_sidebar_state='expanded', layout='wide')

#---------------------------------#
st.write("# The Machine Learning App")

#---------------------------------#
with st.sidebar:
    with st.container():
        col1, col2 = st.columns([1,4])

        with col1:
            st.image('https://raw.githubusercontent.com/Machine-Learning-Projects1/CDC_ML/test-ui/assets/logo.png', width=60)
        with col2:
            st.markdown("## Machine Learning Projects [*GitHub*](https://github.com/Machine-Learning-Projects1)")
        st.write("---")

with st.container():
    with st.sidebar.header('1. Upload your CSV data'):
        uploaded_file = st.sidebar.file_uploader("Upload your input CSV file", type=["csv"])
        st.sidebar.write('---')

with st.container():
    with st.sidebar.header('2. Data preprocessing'):
        target = st.sidebar.text_input('Target', 'HeartDisease')
        split_size = st.sidebar.slider('Data split ratio (% for Training Set)', 10, 90, 80, 5)
        split_size = split_size/100
        pp = st.sidebar.button('Preprocess data')
        st.sidebar.write('---')

with st.container():
    with st.sidebar.header('3. Select model'):
        # Add a selectbox to the sidebar:
        model_selection = st.sidebar.selectbox(
            'Choose a model',
            ('k-Nearest Neighbors', 'Decision Tree', 'Perceptron', 'Logistic Regression', 'SVM', 'Neural Network', 'Random Forest', 'XGBoost', 'AdaBoost', 'Naive Bayes', 'GMM', 'Deep Learning', 'Large language model')
        )
        
    st.sidebar.write('\n')
    params_mode = st.sidebar.radio(
    "Choose the hyperparameters mode, either manual or automatic. If automatic, the hyperparameters will be tuned using GridSearchCV. If manual, you will have to enter the hyperparameters manually.",
    ('Automatic', 'Manual'))
    st.sidebar.write('\n')

# Sidebar - Specify parameter settings
if model_selection == 'k-Nearest Neighbors' and params_mode == 'Automatic':
    with st.sidebar.subheader('3.1 Set Parameters'):
        n_neighbors = st.sidebar.slider('Number of neighbors (n_neighbors)', 1, 20, 5)
        weights = st.sidebar.multiselect('Weight function used in prediction', ['uniform', 'distance'])
        algorithm = st.sidebar.multiselect(
            'Algorithm used to compute the nearest neighbors (algorithm)',
            ['auto', 'ball_tree', 'kd_tree', 'brute']
        )
        leaf_size = st.sidebar.slider('Leaf size passed to BallTree or KDTree (leaf_size)', 1, 50, 30)
        p = st.sidebar.multiselect(
            'Power parameter for the Minkowski metric (p)',
            [1, 2]
        )
        metric = st.sidebar.multiselect(
            'The distance metric to use for the tree (metric)',
            ['minkowski', 'euclidean', 'manhattan']
        )
        metric_params = st.sidebar.multiselect(
            'Additional keyword arguments for the metric function (metric_params)',
            ['None', 'distance']
        )
        n_jobs = st.sidebar.multiselect(
            'Number of parallel jobs to run (n_jobs). None means 1 unless in a joblib.parallel_backend context. -1 means using all processors.', [1, -1]
        )
        st.sidebar.write('\n')


# Sidebar - Specify parameter settings
if model_selection == 'k-Nearest Neighbors' and params_mode == 'Manual':
    with st.sidebar.header('3. Set Parameters'):
        n_neighbors = st.sidebar.slider('Number of neighbors (n_neighbors)', 1, 20, 5)
        weights = st.sidebar.radio(
            'Weight function used in prediction (weights)',
            ('uniform', 'distance')
        )
        algorithm = st.sidebar.radio(
            'Algorithm used to compute the nearest neighbors (algorithm)',
            ('auto', 'ball_tree', 'kd_tree', 'brute')
        )
        leaf_size = st.sidebar.slider('Leaf size passed to BallTree or KDTree (leaf_size)', 1, 50, 30)
        p = st.sidebar.radio(
            'Power parameter for the Minkowski metric (p)',
            (1, 2)
        )
        metric = st.sidebar.radio(
            'The distance metric to use for the tree (metric)',
            ('minkowski', 'euclidean', 'manhattan')
        )
        metric_params = st.sidebar.radio(
            'Additional keyword arguments for the metric function (metric_params)',
            ('None', 'distance')
        )
        n_jobs = st.sidebar.radio(
            'Number of parallel jobs to run (n_jobs). None means 1 unless in a joblib.parallel_backend context. -1 means using all processors.', (1, -1)
        )
        st.sidebar.write('---')

if model_selection == 'Decision Tree' and params_mode == 'Manual':
    with st.sidebar.header('3. Set Parameters'):
        criterion = st.sidebar.radio(
            'The function to measure the quality of a split (criterion)',
            ('gini', 'entropy')
        )
        max_depth = st.sidebar.slider('The maximum depth of the tree (max_depth)', 1, 20, 5)
        min_samples_split = st.sidebar.slider('The minimum number of samples required to split an internal node (min_samples_split)', 2, 20, 2)
        min_samples_leaf = st.sidebar.slider('The minimum number of samples required to be at a leaf node (min_samples_leaf)', 1, 20, 1)
        st.sidebar.write('---')

if model_selection == 'Perceptron' and params_mode == 'Manual':
    with st.sidebar.header('3. Set Parameters'):
        penalty = st.sidebar.radio(
            'Used to specify the norm used in the penalization (penalty)',
            ('l2', 'l1', 'elasticnet')
        )
        alpha = st.sidebar.slider('Constant that multiplies the regularization term if regularization is used (alpha)', 0.0, 1.0, 0.0001)
        max_iter = st.sidebar.slider('The maximum number of passes over the training data (aka epochs) (max_iter)', 1, 1000, 100)
        st.sidebar.write('---')

if model_selection == 'Logistic Regression' and params_mode == 'Manual':
    with st.sidebar.header('3. Set Parameters'):
        penalty = st.sidebar.radio(
            'Used to specify the norm used in the penalization (penalty)',
            ('l2', 'l1', 'elasticnet')
        )
        C = st.sidebar.slider('Inverse of regularization strength; must be a positive float (C)', 0.0, 1.0, 0.0001)
        max_iter = st.sidebar.slider('The maximum number of iterations (max_iter)', 1, 1000, 100)
        st.sidebar.write('---')

if model_selection == 'SVM' and params_mode == 'Manual':
    with st.sidebar.header('3. Set Parameters'):
        C = st.sidebar.slider('Penalty parameter C of the error term (C)', 0.0, 1.0, 0.0001)
        kernel = st.sidebar.radio(
            'Specifies the kernel type to be used in the algorithm (kernel)',
            ('linear', 'poly', 'rbf', 'sigmoid')
        )
        degree = st.sidebar.slider('Degree of the polynomial kernel function ("poly") (degree)', 1, 10, 3)
        gamma = st.sidebar.radio(
            'Kernel coefficient for "rbf", "poly" and "sigmoid" (gamma)',
            ('scale', 'auto')
        )
        st.sidebar.write('---')

if model_selection == 'Neural Network' and params_mode == 'Manual':
    with st.sidebar.header('3. Set Parameters'):
        hidden_layer_sizes = st.sidebar.slider('The ith element represents the number of neurons in the ith hidden layer (hidden_layer_sizes)', 1, 100, 10)
        activation = st.sidebar.radio(
            'Activation function for the hidden layer (activation)',
            ('identity', 'logistic', 'tanh', 'relu')
        )
        solver = st.sidebar.radio(
            'The solver for weight optimization (solver)',
            ('lbfgs', 'sgd', 'adam')
        )
        alpha = st.sidebar.slider('L2 penalty (regularization term) parameter (alpha)', 0.0, 1.0, 0.0001)
        learning_rate = st.sidebar.radio(
            'Learning rate schedule for weight updates (learning_rate)',
            ('constant', 'invscaling', 'adaptive')
        )
        st.sidebar.write('---')

if model_selection == 'Random Forest' and params_mode == 'Manual':
    with st.sidebar.header('3. Set Parameters'):
        n_estimators = st.sidebar.slider('The number of trees in the forest (n_estimators)', 1, 100, 10)
        criterion = st.sidebar.radio(
            'The function to measure the quality of a split (criterion)',
            ('gini', 'entropy')
        )
        max_depth = st.sidebar.slider('The maximum depth of the tree (max_depth)', 1, 20, 5)
        min_samples_split = st.sidebar.slider('The minimum number of samples required to split an internal node (min_samples_split)', 2, 20, 2)
        min_samples_leaf = st.sidebar.slider('The minimum number of samples required to be at a leaf node (min_samples_leaf)', 1, 20, 1)
        st.sidebar.write('---')

if model_selection == 'XGBoost' and params_mode == 'Manual':
    with st.sidebar.header('3. Set Parameters'):
        learning_rate = st.sidebar.slider('Boosting learning rate (xgb_learning_rate)', 0.0, 1.0, 0.0001)
        max_depth = st.sidebar.slider('Maximum depth of a tree. Increasing this value will make the model more complex and more likely to overfit (xgb_max_depth)', 1, 20, 5)
        n_estimators = st.sidebar.slider('Number of trees to fit (xgb_n_estimators)', 1, 100, 10)
        objective = st.sidebar.radio(
            'Specify the learning task and the corresponding learning objective or a custom objective function to be used (objective)',
            ('reg:linear', 'reg:logistic', 'binary:logistic', 'binary:logitraw', 'count:poisson', 'multi:softmax', 'multi:softprob', 'rank:pairwise', 'reg:gamma', 'reg:tweedie')
        )
        booster = st.sidebar.radio(
            'Specify which booster to use: gbtree, gblinear or dart (booster)',
            ('gbtree', 'gblinear', 'dart')
        )
        st.sidebar.write('---')

if model_selection == 'AdaBoost' and params_mode == 'Manual':
    with st.sidebar.header('3. Set Parameters'):
        n_estimators = st.sidebar.slider('The maximum number of estimators at which boosting is terminated (n_estimators)', 1, 100, 10)
        learning_rate = st.sidebar.slider('Learning rate shrinks the contribution of each classifier (learning_rate)', 0.0, 1.0, 0.0001)
        algorithm = st.sidebar.radio(
            "If 'SAMME.R' then use the SAMME.R real boosting algorithm. base_estimator must support calculation of class probabilities. If 'SAMME' then use the SAMME discrete boosting algorithm. The SAMME.R algorithm typically converges faster than SAMME, achieving a lower test error with fewer boosting iterations (algorithm)",
            ('SAMME.R', 'SAMME')
        )
        st.sidebar.write('---')

if model_selection == 'Naive Bayes' and params_mode == 'Manual':
    with st.sidebar.header('3. Set Parameters'):
        priors = st.sidebar.radio(
            'Prior probabilities of the classes (priors)',
            ('None', 'Manual')
        )
        if priors == 'Manual':
            priors = st.sidebar.text_input('Prior probabilities of the classes (priors)', value='None')
        st.sidebar.write('\n')


# Building the model
buildb = st.sidebar.button('Build Model')
st.sidebar.write('---')

#---------------------------------#
# Main panel

# Displays the dataset
st.subheader('1. Dataset')

if uploaded_file is not None:
    df = pd.read_csv(uploaded_file)
    
    st.write(df)
    st.write('Dataframe shape', df.shape)
    st.success('Dataset successfully loaded.')
    st.write('---')
    
    st.subheader('2. Data Preprocessing')
    st.info('Awaiting for preprocessing parameters...')

    if pp:
        df_encode = preprocessing_encode_columns(df)
        X_train, y_train, X_test, y_test = preprocessing_splitting(df_encode, target, split_size)
        X_train_scale, X_test_scale = preprocessing_scaling(X_train, X_test)
        X_train_blnc, y_train_blnc = balance_data('smote', X_train_scale, y_train)
        cv = k_fold_cross_validation(10)

        col1, col2 = st.columns(2)

        with col1:
            st.write('X_train', X_train.shape, X_train.head())
            st.write('y_train', y_train.shape, y_train.head())

        with col2:
            st.write('X_test', X_test.shape, X_test.head())
            st.write('y_test', y_test.shape, y_test.head())

        st.success('Data preprocessing successfully completed.')
        st.write('---')

        st.subheader('3. Build Model')
        st.info('Awaiting for select model and build model button.')
        # if buildb:
            

else:
    st.info('Awaiting for CSV file to be uploaded.')
    if st.button('Press to use CDC_2020 Dataset'):
        df = load_dataset('https://raw.githubusercontent.com/Machine-Learning-Projects1/CDC_ML/main/dataset/heart_2020_cleaned.csv')
        st.write(df)
        st.success('Dataset successfully loaded.')

        st.write('---')
        st.subheader('2. Build Model')
        st.info('Awaiting for select model and build model button.')

