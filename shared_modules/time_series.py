## @package time_series
#
#  Time series module provides class for univariate time series forecasting.
#  Version: 0.12
#  Class time_series contains data as an artibute and methods which operates on that data.
#
# Changes form last version: (add to commit message)
# ### -0.13:
# -sth


import numpy as np
import pandas as pd
import pmdarima as pm # for pm.auto_arima, and arima
from pmdarima.arima import StepwiseContext

from statsmodels.tsa.holtwinters import ExponentialSmoothing
from itertools import product
import scipy.stats as st # for Z calcuation

import logging
import warnings
import math
from numbers import Number
from predictionError import PredictionError

## debuging messages
debug_time_series  = False
## rounding decimal
def_rounding_digit = 1



from log_format import LOG_FORMAT
logging.basicConfig(format=LOG_FORMAT)
LOGGER = logging.getLogger(__name__)

if debug_time_series:
    LOGGER.setLevel(logging.DEBUG)
else:
    LOGGER.setLevel(logging.INFO)

## Automatic forecasting.
#
# Takes data loaded to object, chooses best model and parameters for it, and make forecasting.
#
# @param nr_fore number of predistios forward
# @param per_season if data is seasonal (if not 0), how many periods in season (e.g. if mounthly = 12)
# @param alpha confidance interval
# @param return_model whether return model
#
# @returns dictionary {"forecast": [x,x,x,...],...}
# @returns forecast - point forecast
# @returns conf_int - confidance interval for forecast
# @returns mae - model mean absolute error
# @returns aic - model aic
# @returns model - fitted model (if set)
def _auto_forecast(self, nr_fore, per_season, alpha=0.05, return_model=False):

    try:
        est_dict = self.auto_forecast_est(nr_fore, per_season, alpha, return_model=return_model)    
    except:
        est_dict = dict()
        LOGGER.warning('EST model not created.')
    try:
        arima_dict = self.auto_forecast_arima(nr_fore, per_season, alpha, return_model=return_model)
    except:
        arima_dict = dict()
        LOGGER.warning('ARIMA model not created.')

    if not(bool(est_dict)) and not(bool(arima_dict)):
        LOGGER.error('Nie stowrzono żadnego modelu')
        raise PredictionError('Nie stowrzono żadnego modelu')

    if not(bool(est_dict)):
        return arima_dict
    elif not(bool(arima_dict)):
        return est_dict
    else:
        LOGGER.debug('est_aic: %f', est_dict['aic'])
        LOGGER.debug('airma_aic: %f', arima_dict['aic'])
        # comparison based on AIC:
        if est_dict['aic'] < arima_dict['aic']:
            LOGGER.debug('est_dict')
            return est_dict
        else:
            LOGGER.debug('arima_dict')
            return arima_dict

    




## Automatic SARIMA forecasting.
#
# Takes data loaded to object, choose best parameters for SARIMA model, and make forecasting.
#
# @param nr_fore number of predistios forward
# @param per_season if data is seasonal (if not 0), how many periods in season (e.g. if mounthly = 12)
# @param alpha confidance interval
# @param max_p maximum order of p and P parameter during evaluation
# @param max_d maximum order of d and D parameter during evaluation
# @param max_q maximum order of q and Q parameter during evaluation
# @param return_model whether return model
#
# @returns dictionary {"forecast": [x,x,x],...}
# @returns forecast - point forecast
# @returns conf_int - confidance interval for forecast
# @returns mae - model mean absolute error
# @returns aic - model aic
# @returns model - fitted model (if set)
def _auto_forecast_arima(self, pred_nr, per_season, alpha = 0.05, max_p=4, max_d=2, max_q=4, return_model=False):
    try:
        if per_season:
            with StepwiseContext(max_steps=500): # how many repetition
                model_auto = pm.auto_arima(self.history,
                                        start_p=0, start_q=0,
                                        max_p=max_p, max_q=max_q,  # maximum p and q
                                        m=per_season,              # frequency of series
                                        d=None,           # let model determine 'd'
                                        seasonal=True,   # Seasonality
                                        start_P=0, start_Q=0,
                                        max_P=max_p, max_Q=max_q,
                                        D=0,
                                        trace=False,    # print created models during fitting
                                        error_action='warn',
                                        suppress_warnings=True,
                                        stepwise=True)
        else:
            with StepwiseContext(max_steps=500):
                model_auto = pm.auto_arima(self.history, 
                                        start_p=0, start_q=0,
                                        max_p=max_p, max_q=max_q,  # maximum p and q
                                        d=None,           # let model determine 'd'
                                        seasonal=False,   # Seasonality
                                        trace=False,    # print created models during fitting
                                        error_action='warn',
                                        suppress_warnings=True,
                                        stepwise=True)

        LOGGER.debug('Model created')
        # jesli model nie ma parametrow
        order = model_auto.get_params()['order'] 
        seasonal_order = model_auto.get_params()['seasonal_order']
        if order == (0,0,0) and seasonal_order == (0,0,0,0):
            LOGGER.error('Model ARIMA z paramaetrami (0,0,0)(0,0,0,0)')
            raise PredictionError('Żaden model nie został dopasowany.')
        model_auto_fit = model_auto.fit(self.history)
        [auto_forecast, conf_int] = model_auto_fit.predict( pred_nr , return_conf_int=True, alpha=alpha ) # time series with predictions
        LOGGER.debug('Fitted and predictid')
        LOGGER.debug(f'Predictions: {auto_forecast}')
        auto_forecast = list(auto_forecast)
        # MAE
        residuals = model_auto_fit.resid()
        LOGGER.debug('Residuals to mae: {value}'.format(value=residuals))
        model_mae = round(np.absolute(residuals).mean(), def_rounding_digit)
        # AIC
        model_aic = round(model_auto_fit.aic(),def_rounding_digit)
    
        if isinstance(auto_forecast,list):
            auto_forecast = [round(num,def_rounding_digit) for num in auto_forecast]
        else: # napisac obsługę błedu
            auto_forecast = round(auto_forecast,def_rounding_digit)

        conf_int = [ [round(num_nested,def_rounding_digit) for num_nested in num] for num in conf_int]

        if return_model:
            return dict({ "forecast": auto_forecast, "conf_int": conf_int, "mae": model_mae, "aic": model_aic, "model": model_auto_fit})
        else:
            return dict({ "forecast": auto_forecast, "conf_int": conf_int, "mae": model_mae, "aic": model_aic})
    
    except:
        LOGGER.error('Żaden model nie zotał dopasowany')
        raise PredictionError('Żaden model nie został dopasowany.')

## Manual SARIMA forecasting.
#
# Takes data loaded to object, creates SARIMA model based on parameters, and make forecasting.
#
# @param pred_nr number of predistios forward
# @param parameters in form of: [p, d, q, P, D, Q, m], m-per season
# @param alpha confidance interval
# @param return_model whether return model
#
# @returns dictionary {"forecast": [x,x,x],...}
# @returns forecast - point forecast
# @returns conf_int - confidance interval for forecast
# @returns mae - model mean absolute error
# @returns aic - model aic
# @returns model - fitted model (if set)
def _man_forecast_arima(self, pred_nr, parameters, alpha=0.05, return_model=False):
    
    (p, d, q, P, D, Q, m) = parameters
    n_order = (p,d,q) 
    s_order = (P,D,Q,m)
    
    try:
        warnings.filterwarnings("ignore")
        man_model = pm.ARIMA(order=n_order, seasonal_order=s_order) # creating model
        man_model_fit = man_model.fit(self.history) # tarin model on data
    except:
        LOGGER.error('Nie można utowrzyć modelu z danymi parametrami:%s',str(parameters))
        return dict()
    
    [man_forecast, conf_int] = man_model_fit.predict( pred_nr , return_conf_int=True, alpha=alpha )
    man_forecast = list(man_forecast)
    
    # MAE
    residuals = man_model_fit.resid()
    model_mae = round(np.absolute(residuals).mean(), def_rounding_digit)
    # AIC
    model_aic = round(man_model_fit.aic(),def_rounding_digit)
    
    
    if isinstance(man_forecast,list):
        man_forecast = [round(num,def_rounding_digit) for num in man_forecast]
    else: # napisac obsługę błedu
        man_forecast = round(man_forecast,def_rounding_digit)
    
    conf_int = [ [round(num_nested,def_rounding_digit) for num_nested in num] for num in conf_int]
    
    
    if return_model:
        return dict({ "forecast": man_forecast, "conf_int": conf_int, "mae": model_mae, "aic": model_aic, "model": man_model_fit})
    else:
        return dict({ "forecast": man_forecast, "conf_int": conf_int, "mae": model_mae, "aic": model_aic})
    
    


## Automatic EST (exponential smoothing) forecasting.
#
# Takes data loaded to object, choose best parameters for EST model, and make forecasting.
#
# @param pred_nr number of predistios forward
# @param per_season if data is seasonal (if not 0), how many periods in season (e.g. if mounthly = 12)
# @param alpha confidance interval
# @param add if add new prediction to the history and make next with new model (new history)
# @param return_model whether return model
#
# @returns dictionary {"forecast": [x,x,x],...}
# @returns forecast - point forecast
# @returns conf_int - confidance interval for forecast
# @returns mae - model mean absolute error
# @returns aic - model aic
# @returns model - fitted model (if set)
def _auto_forecast_est(self, pred_nr, per_season, alpha = 0.05, add = False, return_model=False):
    
    results = self.find_best_est_model(self.history, per_season)
    LOGGER.debug(f"results: {results}")
    if not results:
        LOGGER.error('Żaden model nie zotał dopasowany')
        raise PredictionError('Żaden model nie został dopasowany.')
    results.sort()
    best_config = results[0][1]
    LOGGER.debug('Best config: %s', str(best_config))
    if return_model:
        est_dict = self.man_forecast_est(pred_nr, per_season, alpha ,best_config, add=add, return_model=return_model)
    else:
        est_dict = self.man_forecast_est(pred_nr, per_season, alpha ,best_config, add=add)
    return est_dict
    


## Manual EST forecasting.
#
# Takes data loaded to object, creates EST model based on parameters, and make forecasting.
#
# @note config [t,d,s,b,r]: \n
# t-trend: {'add', 'mul', None}, \n
# d-damped trend: {True, False}, \n
# s-seasonal: {'add', 'mul', None}, \n
# b-use_boxcox: {True, False}, \n
# r-remove_bias: {True, False}, \n
# more: https://www.statsmodels.org/dev/generated/statsmodels.tsa.holtwinters.ExponentialSmoothing.html
#
# @param pred_nr number of predistios forward
# @param per_season if data is seasonal (if not 0), how many periods in season (e.g. if mounthly = 12)
# @param alpha confidance interval
# @param config [t,d,s,b,r]
# @param add if add new prediction to the history and make next with new model (new history)
# @param return_model whether return model
#
# @returns dictionary {"forecast": [x,x,x],...}
# @returns forecast - point forecast
# @returns conf_int - confidance interval for foreca
# @returns mae - model mean absolute error
# @returns aic - model aic
# @returns model - fitted model (if set)
def _man_forecast_est(self, pred_nr, per_season, alpha=0.05, config = [None,False,None,False,False], add = False, return_model=False):
    
    man_forecast = list()
    conf_int = list()
    
    if add:     # tutaj powinno sie od nowa wybierac parametry dla modelu po mamy po każdej iteracji nowy ts!!!!!!!!!!!!!
                # inaczej czasem sie wysypuje   
        tmp_history = list(self.history)

        p = per_season
        t,d,s,b,r = config
        
        for i in range(pred_nr):
            LOGGER.debug(tmp_history)

            # define model
            model = ExponentialSmoothing(tmp_history, trend=t, damped=d, seasonal=s, seasonal_periods=p)
            # fit model
            model_fit = model.fit(optimized=True, use_boxcox=b, remove_bias=r)
            # make one step forecast
            yhat = model_fit.predict()[0]
            
            # calculating confidence interval for more than one prediction
            c_int = self.confidence_interval(yhat, model_fit, pred_ordinal=(i+1), alpha=alpha)
            
            man_forecast.append(yhat)
            conf_int.append(c_int)
            tmp_history.append(yhat)
    
    else: # if not add
        p = per_season
        t,d,s,b,r = config
        # define model
        try:
            model = ExponentialSmoothing(self.history, trend=t, damped=d, seasonal=s, seasonal_periods=p)
            # fit model
            model_fit = model.fit(optimized=True, use_boxcox=b, remove_bias=r)
            # make few steps forecst
            man_forecast = list(model_fit.forecast(pred_nr))
        except:
            LOGGER.error('Nie można stworzyć modelu')
            return dict()

        # confidence interval calculation
        for i in range(pred_nr):
            c_int = self.confidence_interval(man_forecast[i], model_fit, pred_ordinal=(i+1), alpha=alpha)
            conf_int.append(c_int)
            
    LOGGER.debug('Man EST predicted')
    # MAE
    residuals = model_fit.resid
    LOGGER.debug(f'Residuals: {residuals}')
    model_mae = round(np.absolute(residuals).mean(), def_rounding_digit)
    # AIC 
    model_aic = round(model_fit.aic,def_rounding_digit)
    if isinstance(man_forecast,list):
        man_forecast = [round(num,def_rounding_digit) for num in man_forecast]
    else: # napisac obsługę błedu
        man_forecast = round(man_forecast,def_rounding_digit)

    conf_int = [ [round(num_nested,def_rounding_digit) for num_nested in num] for num in conf_int]
    if return_model: # if add then last model
        return dict({ "forecast": man_forecast, "conf_int": conf_int, "mae": model_mae, "aic": model_aic, "model": model_fit})
    else:
        return dict({ "forecast": man_forecast, "conf_int": conf_int, "mae": model_mae, "aic": model_aic})


## Calculating condifence inteval for prediction
#
# @param y point prediction (value)
# @param model required for .resid, .see
# @param pred_ordinal which in order is current prodiction (which in row)
# @param alpha percentage of interval
#
# @return confidence (prediction) interval for given point y in form of [y_low, y_high]
def _confidence_interval(self, y, model, pred_ordinal=1, alpha=0.05):
    
    alpha_proc = 1 - alpha
    norm_dist = st.norm.ppf(1-(1-alpha_proc)/2) # normal distribution
    z = round(norm_dist,2)
    
    LOGGER.debug('Z= {value}'.format(value=z))
    
    sse = model.sse # sum of squared error
    data_size = len(model.resid)
    std_dev = np.sqrt(sse/data_size)
    
    std_dev = std_dev*np.sqrt(pred_ordinal) # estimation of standard deviation for few forecast
    
    y_low = y - z*std_dev 
    y_high  = y + z*std_dev
    
    return [y_low, y_high]



## Finding best exponential smoothing model
#
# Creates model for each configuration form _exp_smoothing_configs() and calculating AIC
#
# @param data fitting model to this data 
# @param per_season if data is seasonal (if not 0), how many periods in season (e.g. if mounthly = 12)
# 
# @return list of AICs of model and configurations 
def _find_best_est_model(self, data, per_season):
    confs = self.exp_smoothing_configs()
    results = list()
    for con in confs:
        try:
            warnings.filterwarnings("ignore")
            results.append(self.model_score(data, per_season, con))
        except:
            pass
    return results




## Calculates AIC for the model
#
#
# @param data fitting model to this data
# @param per_season if data is seasonal (if not 0), how many periods in season (e.g. if mounthly = 12)
# @param config [t,d,s,b,r]
#
# @return AIC for model and configuration
def _model_score(self, data, per_season, config):
    p = per_season
    t,d,s,b,r = config
    
    model = ExponentialSmoothing(data, trend=t, damped=d, seasonal=s, seasonal_periods=p)
    model_fit = model.fit(optimized=True, use_boxcox=b, remove_bias=r)
    model_aic = model_fit.aic
    LOGGER.debug(f'model_aic: {model_aic}')
    LOGGER.debug(f'model_aic type: {type(model_aic)}')

    # check if we can predict with this model
    fst_forecast = model_fit.forecast(1)[0]
    LOGGER.debug(f'1st forecast: {fst_forecast}')
    if (not isinstance(fst_forecast, Number)) or (not np.isfinite(fst_forecast)) : 
        LOGGER.warning('Can not predict with this model')
        raise ValueError('Man we can not predict with this model ')

    if (not isinstance(model_aic, Number)) or (not np.isfinite(model_aic)) :
        LOGGER.warning('Coś nie tak z model AIC')
        raise ValueError('Model AIC nie jest liczbą')
    
    LOGGER.debug('Returning: model_aic: {v_aic}, config: {conf}'.format(v_aic= model_aic, conf=config))
    return [model_aic, config]


## Create a set of exponential smoothing configs
#
# @return List of all possible configurations
def _exp_smoothing_configs(self):
    models = list()
    # define config lists
    t_params = ['add', 'mul', None]
    d_params = [True, False]
    s_params = ['add', 'mul', None]
    b_params = [True, False]
    r_params = [True, False]
    # create config instances
    models = list(product(t_params,d_params,s_params,b_params,r_params))

    return models


## Change different types to flat list
#
# @param data possibly: list, ndarray, nested ndarray, nested list
#
# @returns flat list e.g. [1,2,3,4,5]
def _change_to_flat_list(self, data):
    final_list = 0
    LOGGER.debug('raw data= {value}'.format(value=data))
    if isinstance(data, list):
        if isinstance(data[0], list): # nested list
            final_list = [item for sublist in data for item in sublist]
        else:
            #final_list = data 
            final_list = [item for item in data]
            
    if isinstance(data, np.ndarray):
        if isinstance(data[0], list) or isinstance(data[0], np.ndarray): # nested list or array in array
            # it has to be one level nested list to works properly (otherwise will work but not good)
            # [[1],[2],[3]...]
            final_list = [item for sublist in data.tolist() for item in sublist]
        else: # value
            final_list = data.tolist()
    LOGGER.debug('final_list= %s', str(final_list))        
    if final_list == 0:
        LOGGER.error('Wrong initial value of series') 
        # rise error
    return final_list
        

## Main class of module
#
# This class contains time series and methods to forecasting
# history - parameter which is essential and assigned to object
class time_series:
    
    ## The constructor
    # 
    # @param data univarient time series, which contains data to forecast.
    def __init__(self, data = [0]):
        self.history = self.change_to_flat_list(data)
        # replaceing negative numbers with 0
        self.history = [0 if i<0 else i for i in self.history]
        LOGGER.debug('self.history= %s', str(self.history))

        
    ## @var history
    #  Data to forecastnumeric_level = getattr(logging, loglevel
    
    ## Replace history atribute.
    # 
    # @param data time series to replace old one.
    def set_hisotry(self, data):
        self.history = data
    
    # MAJORS METHODS
    ## Link to _auto_forecast()
    auto_forecast = _auto_forecast
    ## Link to _auto_forecast_arima()
    auto_forecast_arima = _auto_forecast_arima
    ## Link to _man_forecast_arima()
    man_forecast_arima = _man_forecast_arima
    ## Link to _auto_forecast_est()
    auto_forecast_est = _auto_forecast_est
    ## Link to _auto_forecast_est()
    man_forecast_est = _man_forecast_est
    
    # PRIVATE METHODS
    ## Link to _confidence_interval()
    confidence_interval = _confidence_interval
    ## Link to _exp_smoothing_configs()
    exp_smoothing_configs = _exp_smoothing_configs
    ## Link to _find_best_model()
    find_best_est_model = _find_best_est_model
    ## Link to _model_score()
    model_score = _model_score
    ## Link to _change_to_flat_list()
    change_to_flat_list = _change_to_flat_list
