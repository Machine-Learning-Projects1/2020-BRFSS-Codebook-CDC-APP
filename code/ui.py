import streamlit as st
import pandas as pd
from driver import *
from sklearn.model_selection import train_test_split
from sklearn.ensemble import RandomForestRegressor
from sklearn.metrics import mean_squared_error, r2_score
from sklearn.datasets import load_diabetes, load_boston
from PIL import Image
from sklearn.neighbors import KNeighborsClassifier
from contextlib import contextmanager
from io import StringIO
from threading import current_thread
import streamlit as st
import sys
import streamlit as st
from streamlit.report_thread import REPORT_CONTEXT_ATTR_NAME
from threading import current_thread
from contextlib import contextmanager
from io import StringIO
import sys
import logging
import time
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
        # show_file = st.sidebar.checkbox('Show file')
        if uploaded_file is not None:
            st.sidebar.success('File uploaded successfully')
        st.sidebar.write('---')

with st.container():
    st.sidebar.header('2. Build your model')
    with st.sidebar.subheader('2.1 Data preprocessing'):
        target = st.sidebar.text_input('Target', 'HeartDisease')
        split_size = st.sidebar.slider('Data split ratio (% for Training Set)', 10, 90, 80, 5)
        split_size = split_size/100
        # processing_data = st.sidebar.button('Preprocess data')
        # if processing_data:
        #     st.success("2. Data preprocessing completed")
        st.sidebar.write('\n')

    with st.sidebar.subheader('2.2 Select model'):
        # Add a selectbox to the sidebar:
        model_selection = st.sidebar.selectbox(
            'Choose a model',
            ('k-Nearest Neighbors', 'Decision Tree', 'Perceptron', 'Logistic Regression', 'SVM', 'Neural Network', 'Random Forest', 'XGBoost', 'AdaBoost', 'Naive Bayes', 'GMM', 'Deep Learning', 'Large language model')
        )
        
    st.sidebar.write('\n')
    params_mode = st.sidebar.radio(
    "Choose the hyperparameters mode, either manual or automatic. If automatic, the hyperparameters will be tuned using GridSearchCV. If manual, you will have to enter the hyperparameters manually.",
    ('Disable' ,'Automatic', 'Manual'), index=0)
    st.sidebar.write('\n')

    if params_mode == 'Disable':
        st.sidebar.warning('Hyperparameters are disabled, Please select Automatic or Manual')

    # Sidebar - Specify parameter settings
    if model_selection == 'k-Nearest Neighbors' and (params_mode == 'Automatic'):
        with st.sidebar.subheader('2.2.1 Set Parameters'):
            
            n_neighbors = st.sidebar.slider('Number of neighbors (n_neighbors). default = 5', 1, 20, 5)
            
            weights = st.sidebar.multiselect(
                'Weight function used in prediction (weights). default = uniform',
                ['uniform', 'distance']
            )
            
            algorithm = st.sidebar.multiselect(
                'Algorithm used to compute the nearest neighbors (algorithm). default = auto',
                ['auto', 'ball_tree', 'kd_tree', 'brute']
            )

            leaf_size = st.sidebar.slider('Leaf size passed to BallTree or KDTree (leaf_size). default = 30', 1, 50, 30)
            
            p = st.sidebar.multiselect(
                'Power parameter for the Minkowski metric (p). default = 2',
                [1, 2]
            )
            
            metric = st.sidebar.multiselect(
                'The distance metric to use for the tree (metric). default = minkowski',
                ['minkowski', 'euclidean', 'manhattan']
            )
            
            n_jobs = st.sidebar.multiselect(
                'Number of parallel jobs to run (n_jobs). None means 1 unless in a joblib.parallel_backend context. -1 means using all processors. default = 1', [1, -1]
            )
            
    # Sidebar - Specify parameter settings
    if model_selection == 'k-Nearest Neighbors' and (params_mode == 'Manual'):
        with st.sidebar.subheader('2.2.1 Set Parameters'):
            n_neighbors = st.sidebar.slider('Number of neighbors (n_neighbors). default = 5', 1, 20, 5)
            
            weights = st.sidebar.radio(
                'Weight function used in prediction (weights). default = uniform',
                ('uniform', 'distance'), index=0
            )
            
            algorithm = st.sidebar.radio(
                'Algorithm used to compute the nearest neighbors (algorithm). default = auto',
                ('auto', 'ball_tree', 'kd_tree', 'brute'), index=0
            )

            leaf_size = st.sidebar.slider('Leaf size passed to BallTree or KDTree (leaf_size). default = 30', 1, 50, 30)
            
            p = st.sidebar.radio(
                'Power parameter for the Minkowski metric (p). default = 2',
                (1, 2), index=1
            )
            
            metric = st.sidebar.radio(
                'The distance metric to use for the tree (metric). default = minkowski',
                ('minkowski', 'euclidean', 'manhattan'), index=0
            )
            
            n_jobs = st.sidebar.radio(
                'Number of parallel jobs to run (n_jobs). None means 1 unless in a joblib.parallel_backend context. -1 means using all processors. default = 1', (1, -1), index=0
            )
           
    kfold = st.sidebar.slider('Number of folds in KFold (kfold). default = 10', 2, 20, 10)
    verbos = st.sidebar.slider('Verbosity mode (verbose). default = 10', 0, 10, 10)

    


    # Building the model
    build_button = st.sidebar.button('Build Model')
    st.sidebar.write('---')

#---------------------------------#
# Main panel

def st_capture(output_func):
    with StringIO() as stdout, redirect_stdout(stdout):
        old_write = stdout.write

        def new_write(string):
            ret = old_write(string)
            output_func(stdout.getvalue())
            return ret
        
        stdout.write = new_write
        yield

def main():
    # Displays the dataset
    if uploaded_file is not None:
        with st.spinner('Uploading dataset...'):
            df_org = pd.read_csv(uploaded_file)
            df = df_org.head(1000)
            # if show_file == True:
            print('-> Dataset is successfully uploaded!')
            st.dataframe(df.head())
            st.write('dataset shape', df.shape)

            
        if build_button:
            with st.spinner('Building model...'):
                print('-> Building model...')
                df_encode = preprocessing_encode_columns(df)
                X_train, y_train, X_test, y_test = preprocessing_splitting(df_encode, target, split_size)
                # X_train_scale, X_test_scale = preprocessing_scaling(X_train, X_test)
                # X_train_blnc, y_train_blnc = balance_data('smote', X_train_scale, y_train)
                cv = k_fold_cross_validation(kfold)
                if params_mode == 'Automatic':
                    params = {
                        'algorithm': algorithm,
                        'n_neighbors':list(range(1, n_neighbors+1)),
                        'weights':weights,
                        'leaf_size':list(range(1, leaf_size+1)),
                        'p':p,
                        'metric':metric,
                        'n_jobs': n_jobs
                    }
                if params_mode == 'Manual':
                    params = {
                        'algorithm': [algorithm],
                        'n_neighbors':[n_neighbors],
                        'weights':[weights],
                        'leaf_size':[leaf_size],
                        'p':[p],
                        'metric':[metric],
                        'n_jobs': [n_jobs]
                }
                    # 'metric_params': [metric_params],

            
                model = KNeighborsClassifier()
                model_cv = GridSearchCV(model, param_grid=params, cv=cv, verbose=10)
                model_cv.fit(X_train, y_train)
                print('\nbest parameters:', model_cv.best_params_)
                print('\n-> Model built successfully!')
                # st.code(model_cv.best_params_)
                # st.success('Model built successfully')
                print('-> Predicting...\n')
                y_pred = model_cv.predict(X_test)
                print(classification_report(y_test, y_pred))
            st.write('---')


    else:
        st.info('Awaiting for CSV file to be uploaded.')
        if st.button('Press to use CDC_2020 Dataset'):
            df = load_dataset('https://raw.githubusercontent.com/Machine-Learning-Projects1/CDC_ML/main/dataset/heart_2020_cleaned.csv')
            st.write(df)

            st.write('---')
            st.subheader('2. Build Model')
            st.info('Awaiting for select model and build model button.')


@contextmanager
def st_redirect(src, dst):
    placeholder = st.empty()
    output_func = getattr(placeholder, dst)

    with StringIO() as buffer:
        old_write = src.write

        def new_write(b):
            if getattr(current_thread(), REPORT_CONTEXT_ATTR_NAME, None):
                buffer.write(b + '')
                output_func(buffer.getvalue() + '')
            else:
                old_write(b)

        try:
            src.write = new_write
            yield
        finally:
            src.write = old_write


@contextmanager
def st_stdout(dst):
    "this will show the prints"
    with st_redirect(sys.stdout, dst):
        yield


@contextmanager
def st_stderr(dst):
    "This will show the logging"
    with st_redirect(sys.stderr, dst):
        yield


if __name__ == '__main__':
    with st_stdout("code"):#, st_stderr("code"):
        main()
