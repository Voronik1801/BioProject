import numpy as np
import pandas as pd
import model
from sklearn.metrics import mean_squared_error

def final_model():
    df = pd.DataFrame()
    df = pd.read_csv('final_ols_515.csv', sep='\t')

    with open('data_for_train/result_y.txt') as f:
        y = np.array([list(map(float, row.split())) for row in f.readlines()])

    n_cv = 9

    y_predicted = model.ols_prediction(df, y)
    cv = model.cross_validation_ols_n(df.values, y, n=n_cv)
    print('----OLS---')
    rms = mean_squared_error(y, y_predicted, squared=False)
    rms_cv = mean_squared_error(y, cv, squared=False)
    err = model.error(y, cv)
    print(f'rmse y_predict: {rms}')
    print(f'rmse cv: {rms_cv}')
    print(f'err: {err}')


    y_predicted = model.rlm_prediction(df, y)
    cv = model.cross_validation_rlm_n(df.values, y, n=n_cv)
    print('----RLM---')
    rms = mean_squared_error(y, y_predicted, squared=False)
    rms_cv = mean_squared_error(y, cv, squared=False)
    err = model.error(y, cv)
    print(f'rmse y_predict: {rms}')
    print(f'rmse cv: {rms_cv}')
    print(f'cv: {err}')


    y_predicted = model.forest_prediction(df.values, y)
    cv = model.cross_validation_forest_n(df.values, y, n_cv)
    print('----RF---')
    rms = mean_squared_error(y, y_predicted, squared=False)
    rms_cv = mean_squared_error(y, cv, squared=False)
    err = model.error(y, cv)
    print(f'rmse y_predict: {rms}')
    print(f'rmse cv: {rms_cv}')
    print(f'err: {err}')

final_model()