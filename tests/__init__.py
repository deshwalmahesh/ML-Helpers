import pandas as pd
import numpy as np
from scipy.stats import skew, kurtosis, shapiro,norm, anderson, normaltest, pearsonr, jarque_bera, kstest
import statsmodels.stats.api as sms
import statsmodels.api as sm
from statsmodels.stats.outliers_influence import variance_inflation_factor


def normality_test(data:np.ndarray)->dict:
    '''
    Test whether the given array of data is Normal Gaussian distribution or not 
    If Data Is Gaussian: Use Parametric Statistical Methods | Else: Use Nonparametric Statistical Methods
    '''
    names = ['Shapiro-Wilk', "Ralph D'Agostino", "Jarque-Bera", "Kolmogorov-Smirnov", "Anderson-Darling"]
    tests = [shapiro, normaltest, jarque_bera, kstest, anderson]
    
    result = ''
    for i, test in enumerate(tests): 
        alpha = 0.05 # because we are replacing value of  alpha in Anderson-Darling test
        
        if i == 3: # Kolmogorov test takes 1 extra arg
            stat, p = test(data,'norm') 
            
        elif i == 4: # It gives Different values for each Confidence
            test_result = test(data)
            alpha, p = test_result.statistic, test_result.critical_values[2]
            
        else:
            stat, p = test(data)
            
        if p <= alpha: # Write only failed tests
            result += f"{names[i]} , "
 
    if len(result):
        return "Not normal according to following tests with 95% significance: "+ result


def multi_colinearity_test(df:pd.DataFrame, label_col:str, corr_thresh:float=0.9, vif_thresh:int=5)->str:
    '''
    Find Multi Co-linearity among variables. Drop the variables which are co linear as it'll create a bias
    args:
        df: Whole Pandas DataFrame
        label_col: Column name which contains the values we are trying to predict
        corr_thresh: Threshold for correlation. Values exceeding this value are cause colinearity with others
        vif_thresh: Variance Inflation Factor threshold
    '''
    X = df.drop([label_col],axis=1)
    col_names = X.columns
    
    cor_matrix = df.drop([label_col],axis=1).corr()
    upper_tri = cor_matrix.where(np.triu(np.ones(cor_matrix.shape),k=1).astype(np.bool)).abs()
    corr_res = [column for column in upper_tri.columns if any(upper_tri[column] > corr_thresh)]
    
    X_constant = sm.add_constant(X)
    vif = [variance_inflation_factor(X_constant.values, i) for i in range(X_constant.shape[1])][1:]
    v = pd.DataFrame({'vif': vif},index=col_names)
    vif_res = v[v['vif'] > vif_thresh].index.tolist()
    
    if len(vif_res) or len(corr_res):
        return f"These features are found to be causing Multi-Colinearity: Pearson Correlation with threshold {corr_thresh}: {corr_res} | Variance Inflation Factor with threshold {vif_thresh}: {vif_res}"


