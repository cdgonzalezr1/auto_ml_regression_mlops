from operator import index
import streamlit as st
import plotly.express as px
import pandas_profiling
import pandas as pd
import os 

from sklearn.metrics import mean_absolute_error, mean_squared_error, r2_score, mean_squared_log_error
from pycaret.regression import *
from streamlit_pandas_profiling import st_profile_report
from pyspark.sql import SparkSession
# from pycaret.parallel import FugueBackend

spark = SparkSession.builder.getOrCreate()

st.set_page_config(page_title="AutoRegression MLOps", page_icon="📈", layout="wide", initial_sidebar_state="expanded")

if os.path.exists('./dataset.csv'): 
    df = pd.read_csv('dataset.csv', index_col=None)
    val_df = pd.read_csv('validation.csv', index_col=None)

with st.sidebar: 
    st.image("https://www.onepointltd.com/wp-content/uploads/2019/12/ONE-POINT-01-1.png")
    st.title("Auto regression ML App to MLOps")
    choice = st.radio("Navigation", ["Upload","Profiling","Modelling", "Download"])
    st.info("This project was developed by Christian Gonzalez: cdgonzalezr@unal.edu.co c.gonzalezr@unal.edu.co")

if choice == "Upload":
    st.title("Upload Your Dataset")
    file = st.file_uploader("Upload Your Dataset")
    if file: 
        df = pd.read_csv(file, index_col=None)
        df.to_csv('dataset.csv', index=None)
        st.dataframe(df)
    file_val = st.file_uploader("Upload Your Validation Dataset")
    if file_val: 
        df_val = pd.read_csv(file_val, index_col=None)
        df_val.to_csv('validation.csv', index=None)
        st.dataframe(df_val)

if choice == "Profiling": 
    st.title("Exploratory Data Analysis")
    profile_df = df.profile_report()
    st_profile_report(profile_df)

if choice == "Modelling": 
    st.title("Modelling")
    st.subheader("Antes de iniciar, abre una terminal y abre el diseño de experimentos de mflow usando el comando 'mlflow ui'")
    st.dataframe(df.head())
    train = df.sample(frac=0.8, random_state=786).reset_index(drop=True)
    test = df.drop(train.index).reset_index(drop=True)

    col1, col2, col3 = st.columns(3)
    with col1: 
        chosen_target = st.selectbox('Choose the Target Column', df.columns)
        chosen_numerical_features = st.multiselect('Choose the Numeric Feature Columns', df.columns)
        chosen_categorical_features = st.multiselect('Choose the Categorical Feature Columns', df.columns)
        chosen_exclude_features = st.multiselect('Choose the Exclude Feature Columns', df.columns)
        high_cardinality_features = st.multiselect('Choose the High Cardinality Feature Columns', df.columns)
    with col2: 
        models = st.multiselect('Choose the Models', ['Extra Trees Regressor', 'Lasso Least Angle Regression', 'Decision Tree Regressor', 'Huber Regressor', 'Orthogonal Matching Pursuit', 'Lasso Regression', 'Gradient Boosting Regressor', 'AdaBoost Regressor', 'Random Forest Regressor', 'K Neighbors Regressor', 'Passive Aggressive Regressor', 'Dummy Regressor', 'Bayesian Ridge', 'Elastic Net', 'Ridge Regression', 'Linear Regression', 'Least Angle Regression'])
        mapped_model = {'Extra Trees Regressor': 'et', 'Lasso Least Angle Regression': 'llar', 'Decision Tree Regressor': 'dt', 'Huber Regressor': 'huber', 'Orthogonal Matching Pursuit': 'omp', 'Lasso Regression': 'lasso', 'Gradient Boosting Regressor': 'gbr', 'AdaBoost Regressor': 'ada', 'Random Forest Regressor': 'rf', 'K Neighbors Regressor': 'knn', 'Passive Aggressive Regressor': 'par', 'Dummy Regressor': 'dummy', 'Bayesian Ridge': 'br', 'Elastic Net': 'en', 'Ridge Regression': 'ridge', 'Linear Regression': 'lr', 'Least Angle Regression': 'lar'}
        select_models = [mapped_model[i] for i in models]
    with col3: 
         metric = st.selectbox('Choose the Metric', ['MAE', 'MSE', 'RMSE', 'R2', 'RMSLE', 'MAPE'])
    
    st.markdown("<hr/>", unsafe_allow_html=True)
    
    if st.button('Run Modelling'): 
        setup(train, target=chosen_target, silent=True, numeric_features=chosen_numerical_features, categorical_features=chosen_categorical_features, ignore_features=chosen_exclude_features,  fold=10, log_experiment=True, experiment_name='AutoChrisML', remove_outliers=True, normalize=True, transformation=True, normalize_method='robust', transformation_method='yeo-johnson', imputation_type='iterative', feature_selection=True, feature_selection_threshold=0.8, feature_interaction=True, feature_ratio=True, polynomial_features=True, polynomial_degree=2, combine_rare_levels=True, rare_level_threshold=0.1, remove_multicollinearity=True, multicollinearity_threshold=0.9, create_clusters=True, cluster_iter=20, ignore_low_variance=True, remove_perfect_collinearity=True, use_gpu=True, session_id=123, high_cardinality_features=high_cardinality_features)
        setup_df = pull()
        st.dataframe(setup_df)
        top_models = compare_models(sort= metric , n_select=3, include=select_models)#, parallel = FugueBackend(spark))
        compare_df = pull()
        st.dataframe(compare_df)

        st.text("🔹Tuning the top models")
        try:
            tune_top = [tune_model(i) for i in top_models]
            st.dataframe(tune_top)
        except:
            st.text(" Alguno de los modelos no se puede optimizar")
            tune_top = top_models

        st.text("🔹Bagging the top model")
        try:
            bagged_top = ensemble_model(tune_top[0], method='Bagging', choose_better=True)
        except:
            st.text("El modelo no se puede ensamblar")
            bagged_top = tune_top[0]
        st.text("🔹Boosting the top model")
        try:
            boost_top = ensemble_model(tune_top[0], method='Boosting', choose_better=True)
        except:
            st.text("El modelo no se puede ensamblar")
            boost_top = tune_top[0]

        st.text("🔹Blending the top models")
        try:
            blend_top = blend_models(estimator_list=top_models, fold=10, round=4, choose_better=True)  
        except:
            st.text("Los modelos no se pueden combinar")
            blend_top = tune_top[0]

        st.text("🔹Stacking the top models")
        try:
            stack_top = stack_models(estimator_list=top_models, fold = 10)
            best_model = automl(optimize=metric)
        except:
            st.text("Los modelos no se pueden apilar")
            stack_top = tune_top[0]
            best_model = tune_top[0]

        st.text("🔹Predicting on the test set")
        test['predictions_best_model'] = predict_model(best_model, data=test)['Label']
        test['predictions_blend_top'] = predict_model(blend_top, data=test)['Label']
        test['predictions_stack_top'] = predict_model(stack_top, data=test)['Label']
        test['predictions_bagged_top'] = predict_model(bagged_top, data=test)['Label']
        test['predictions_boost_top'] = predict_model(boost_top, data=test)['Label']
        st.dataframe(test)

        st.text("🔹Comparing the metrics")
        try:
            mae_best_model = mean_absolute_error(test[chosen_target], test['predictions_best_model'])
        except:
            mae_best_model = None
        try:
            mae_blend_top = mean_absolute_error(test[chosen_target], test['predictions_blend_top'])
        except:
            mae_blend_top = None
        try:
            mae_stack_top = mean_absolute_error(test[chosen_target], test['predictions_stack_top'])
        except:
            mae_stack_top = None
        try:
            mae_bagged_top = mean_absolute_error(test[chosen_target], test['predictions_bagged_top'])
        except:
            mae_bagged_top = None
        try:
            mae_boost_top = mean_absolute_error(test[chosen_target], test['predictions_boost_top'])
        except:
            mae_boost_top = None

        try:
            mse_best_model = mean_squared_error(test[chosen_target], test['predictions_best_model'])
        except:
            mse_best_model = None
        try:
            mse_blend_top = mean_squared_error(test[chosen_target], test['predictions_blend_top'])
        except:
            mse_blend_top = None
        try:
            mse_stack_top = mean_squared_error(test[chosen_target], test['predictions_stack_top'])
        except:
            mse_stack_top = None
        try:
            mse_bagged_top = mean_squared_error(test[chosen_target], test['predictions_bagged_top'])
        except:
            mse_bagged_top = None
        try:
            mse_boost_top = mean_squared_error(test[chosen_target], test['predictions_boost_top'])
        except:
            mse_boost_top = None

        try:
            rmse_best_model = np.sqrt(mean_squared_error(test[chosen_target], test['predictions_best_model']))
        except:
            rmse_best_model = None
        try:
            rmse_blend_top = np.sqrt(mean_squared_error(test[chosen_target], test['predictions_blend_top']))
        except:
            rmse_blend_top = None
        try:
            rmse_stack_top = np.sqrt(mean_squared_error(test[chosen_target], test['predictions_stack_top']))
        except:
            rmse_stack_top = None
        try:
            rmse_bagged_top = np.sqrt(mean_squared_error(test[chosen_target], test['predictions_bagged_top']))
        except:
            rmse_bagged_top = None
        try:
            rmse_boost_top = np.sqrt(mean_squared_error(test[chosen_target], test['predictions_boost_top']))
        except:
            rmse_boost_top = None

        try:
            r2_best_model = r2_score(test[chosen_target], test['predictions_best_model'])
        except:
            r2_best_model = None
        try:
            r2_blend_top = r2_score(test[chosen_target], test['predictions_blend_top'])
        except:
            r2_blend_top = None
        try:
            r2_stack_top = r2_score(test[chosen_target], test['predictions_stack_top'])
        except:
            r2_stack_top = None
        try:
            r2_bagged_top = r2_score(test[chosen_target], test['predictions_bagged_top'])
        except:
            r2_bagged_top = None
        try:
            r2_boost_top = r2_score(test[chosen_target], test['predictions_boost_top'])
        except:
            r2_boost_top = None

        try:
            rmsle_best_model = np.sqrt(mean_squared_log_error(test[chosen_target], test['predictions_best_model']))
        except:
            rmsle_best_model = None
        try:
            rmsle_blend_top = np.sqrt(mean_squared_log_error(test[chosen_target], test['predictions_blend_top']))
        except:
            rmsle_blend_top = None
        try:
            rmsle_stack_top = np.sqrt(mean_squared_log_error(test[chosen_target], test['predictions_stack_top']))
        except:
            rmsle_stack_top = None
        try:
            rmsle_bagged_top = np.sqrt(mean_squared_log_error(test[chosen_target], test['predictions_bagged_top']))
        except:
            rmsle_bagged_top = None
        try:
            rmsle_boost_top = np.sqrt(mean_squared_log_error(test[chosen_target], test['predictions_boost_top']))
        except:
            rmsle_boost_top = None

        try:
            mape_best_model = np.mean(np.abs((test[chosen_target] - test['predictions_best_model']) / test[chosen_target])) * 100
        except:
            mape_best_model = None
        try:
            mape_blend_top = np.mean(np.abs((test[chosen_target] - test['predictions_blend_top']) / test[chosen_target])) * 100
        except:
            mape_blend_top = None
        try:
            mape_stack_top = np.mean(np.abs((test[chosen_target] - test['predictions_stack_top']) / test[chosen_target])) * 100
        except:
            mape_stack_top = None
        try:
            mape_bagged_top = np.mean(np.abs((test[chosen_target] - test['predictions_bagged_top']) / test[chosen_target])) * 100
        except:
            mape_bagged_top = None
        try:
            mape_boost_top = np.mean(np.abs((test[chosen_target] - test['predictions_boost_top']) / test[chosen_target])) * 100
        except:
            mape_boost_top = None
        
        st.dataframe(pd.DataFrame({
            'MAE': [mae_best_model, mae_blend_top, mae_stack_top, mae_bagged_top, mae_boost_top],
            'MSE': [mse_best_model, mse_blend_top, mse_stack_top, mse_bagged_top, mse_boost_top],
            'RMSE': [rmse_best_model, rmse_blend_top, rmse_stack_top, rmse_bagged_top, rmse_boost_top],
            'R2': [r2_best_model, r2_blend_top, r2_stack_top, r2_bagged_top, r2_boost_top],
            'RMSLE': [rmsle_best_model, rmsle_blend_top, rmsle_stack_top, rmsle_bagged_top, rmsle_boost_top],
            'MAPE': [mape_best_model, mape_blend_top, mape_stack_top, mape_bagged_top, mape_boost_top]
        }, index=['Best Model', 'Blend Top', 'Stack Top', 'Bagged Top', 'Boost Top']).sort_values(by=metric, ascending=True))

        st.text("🔹 Selecting the best model")
        if metric == 'MAE':
            if mae_best_model == min(mae_best_model, mae_blend_top, mae_stack_top, mae_bagged_top, mae_boost_top):
                selected_model = best_model
            elif mae_blend_top == min(mae_best_model, mae_blend_top, mae_stack_top, mae_bagged_top, mae_boost_top):
                selected_model = blend_top
            elif mae_stack_top == min(mae_best_model, mae_blend_top, mae_stack_top, mae_bagged_top, mae_boost_top):
                selected_model = stack_top
            elif mae_bagged_top == min(mae_best_model, mae_blend_top, mae_stack_top, mae_bagged_top, mae_boost_top):
                selected_model = bagged_top
            elif mae_boost_top == min(mae_best_model, mae_blend_top, mae_stack_top, mae_bagged_top, mae_boost_top):
                selected_model = boost_top
        elif metric == 'MSE':
            if mse_best_model == min(mse_best_model, mse_blend_top, mse_stack_top, mse_bagged_top, mse_boost_top):
                selected_model = best_model
            elif mse_blend_top == min(mse_best_model, mse_blend_top, mse_stack_top, mse_bagged_top, mse_boost_top):
                selected_model = blend_top
            elif mse_stack_top == min(mse_best_model, mse_blend_top, mse_stack_top, mse_bagged_top, mse_boost_top):
                selected_model = stack_top
            elif mse_bagged_top == min(mse_best_model, mse_blend_top, mse_stack_top, mse_bagged_top, mse_boost_top):
                selected_model = bagged_top
            elif mse_boost_top == min(mse_best_model, mse_blend_top, mse_stack_top, mse_bagged_top, mse_boost_top):
                selected_model = boost_top
        elif metric == 'RMSE':
            if rmse_best_model == min(rmse_best_model, rmse_blend_top, rmse_stack_top, rmse_bagged_top, rmse_boost_top):
                selected_model = best_model
            elif rmse_blend_top == min(rmse_best_model, rmse_blend_top, rmse_stack_top, rmse_bagged_top, rmse_boost_top):
                selected_model = blend_top
            elif rmse_stack_top == min(rmse_best_model, rmse_blend_top, rmse_stack_top, rmse_bagged_top, rmse_boost_top):
                selected_model = stack_top
            elif rmse_bagged_top == min(rmse_best_model, rmse_blend_top, rmse_stack_top, rmse_bagged_top, rmse_boost_top):
                selected_model = bagged_top
            elif rmse_boost_top == min(rmse_best_model, rmse_blend_top, rmse_stack_top, rmse_bagged_top, rmse_boost_top):
                selected_model = boost_top
        elif metric == 'R2':
            if r2_best_model == max(r2_best_model, r2_blend_top, r2_stack_top, r2_bagged_top, r2_boost_top):
                selected_model = best_model
            elif r2_blend_top == max(r2_best_model, r2_blend_top, r2_stack_top, r2_bagged_top, r2_boost_top):
                selected_model = blend_top
            elif r2_stack_top == max(r2_best_model, r2_blend_top, r2_stack_top, r2_bagged_top, r2_boost_top):
                selected_model = stack_top
            elif r2_bagged_top == max(r2_best_model, r2_blend_top, r2_stack_top, r2_bagged_top, r2_boost_top):
                selected_model = bagged_top
            elif r2_boost_top == max(r2_best_model, r2_blend_top, r2_stack_top, r2_bagged_top, r2_boost_top):
                selected_model = boost_top
        elif metric == 'RMSLE':
            if rmsle_best_model == min(rmsle_best_model, rmsle_blend_top, rmsle_stack_top, rmsle_bagged_top, rmsle_boost_top):
                selected_model = best_model
            elif rmsle_blend_top == min(rmsle_best_model, rmsle_blend_top, rmsle_stack_top, rmsle_bagged_top, rmsle_boost_top):
                selected_model = blend_top
            elif rmsle_stack_top == min(rmsle_best_model, rmsle_blend_top, rmsle_stack_top, rmsle_bagged_top, rmsle_boost_top):
                selected_model = stack_top
            elif rmsle_bagged_top == min(rmsle_best_model, rmsle_blend_top, rmsle_stack_top, rmsle_bagged_top, rmsle_boost_top):
                selected_model = bagged_top
            elif rmsle_boost_top == min(rmsle_best_model, rmsle_blend_top, rmsle_stack_top, rmsle_bagged_top, rmsle_boost_top):
                selected_model = boost_top
        elif metric == 'MAPE':
            if mape_best_model == min(mape_best_model, mape_blend_top, mape_stack_top, mape_bagged_top, mape_boost_top):
                selected_model = best_model
            elif mape_blend_top == min(mape_best_model, mape_blend_top, mape_stack_top, mape_bagged_top, mape_boost_top):
                selected_model = blend_top
            elif mape_stack_top == min(mape_best_model, mape_blend_top, mape_stack_top, mape_bagged_top, mape_boost_top):
                selected_model = stack_top
            elif mape_bagged_top == min(mape_best_model, mape_blend_top, mape_stack_top, mape_bagged_top, mape_boost_top):
                selected_model = bagged_top
            elif mape_boost_top == min(mape_best_model, mape_blend_top, mape_stack_top, mape_bagged_top, mape_boost_top):
                selected_model = boost_top

        tab1, tab2, tab3, tab4, tab5, tab6 = st.tabs(["Residuals", "Error", "Cooks", "Feature", "Mainfold", "Interpret"])
        with tab1:
            st.write('🔹Plot of the best model Residuals')
            try:
                plot_model(selected_model, plot = 'residuals', save = True)
                st.image('Residuals.png')
            except:
                st.write('No Residuals plot available for this model')
        with tab2:
            st.write('🔹Plot of the best model Error')
            try:
                plot_model(selected_model, plot = 'error', save = True)
                st.image('Prediction Error.png')
            except:
                st.write('No Error plot available for this model')
        with tab3:
            st.write('🔹Plot of the best model Cooks Distance')
            try:
                plot_model(selected_model, plot = 'cooks', save = True)
                st.image('Cooks Distance.png')
            except:
                st.write('No Cooks Distance plot for this model')
        with tab4:
            st.write('🔹Plot of the best model Feature Importance')
            try:
                plot_model(selected_model, plot = 'feature', save = True)
                st.image('Feature Importance.png')
            except:
                st.write('No Feature Importance plot for this model')
        with tab5:
            st.write('🔹Plot of the best model Manifold')
            try:
                plot_model(selected_model, plot = 'manifold', save = True)
                st.image('Manifold Learning.png')
            except:
                st.write('No Manifold Learning plot for this model')
        with tab6:
            st.write('🔹Plot of the best model Interpretation')
            try:
                interpret_model(selected_model, save = True)
                st.image('SHAP summary.png')
            except:
                st.write('No Interpretation plot for this model')
        
        st.text("🔹Predicting on the validation data")
        val_df['Predictions'] = predict_model(selected_model, data = val_df)['Label']

        st.text("🔹Downloading the predictions")
        val_df.to_csv('predictions.csv', index = False)
        # with open('predictions.csv', 'rb') as f:
        #     st.download_button('Download Predictions', f, file_name="predictions.csv")
        
        st.text("🔹Saving the one with the best metric")
        save_model(selected_model, 'best_model')
        st.text("Deploying the model to aws")
        deploy_model(selected_model, 'deployment', platform = 'aws', authentication = { 'bucket'  : 'pycaret-test' })
        # st.text("Loading the model")
        # saved_model = load_model('deployment')
        # st.text("Predicting on the test data")
        # predictions = predict_model(saved_model, data = test_data)
        

if choice == "Download": 
    with open('best_model.pkl', 'rb') as f: 
        st.download_button('Download Model', f, file_name="best_model.pkl")