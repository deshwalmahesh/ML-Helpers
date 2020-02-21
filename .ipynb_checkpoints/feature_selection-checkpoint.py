import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns


class ClassFeatSel():
    '''
    class to find the feature importance and selection feature selections
    '''
    def feature_importances(self,X_train,y_train,column_names,use_type='all'): 
        '''
        Method to find the feature importances of different features based on different models
        args:
            X_train: numpy array of attributes
            y_train: 'target' or classes column
            column_names: list of names of all the columns that are present in X_train
            use_type: {string} 'base' for Logistic Regression, 'tree' for ExtraTree and RandomForest, 'support_vector' 
            for LinearSVM, 'linear' for SGDClassifier and 'all' for all the above
        out:
            dataframe displaying importances of all the columns
        '''

        assert use_type in ['base','linear','support_vector', 'tree','all'], "provide suitable value for 'use_type'. See Docs"

        feature_imp = pd.DataFrame(index=column_names) 

        def base_model(X_train,y_train,column_names,feature_imp):
            from sklearn.linear_model import LogisticRegression
            clf = LogisticRegression().fit(X_train,y_train)
            feature_imp['Logistic Reg'] = clf.coef_.ravel()
            return feature_imp


        def svc_model(X_train,y_train,column_names,feature_imp):
            from sklearn.svm import LinearSVC
            from sklearn.linear_model import SGDClassifier
            clf = LinearSVC().fit(X_train,y_train)
            feature_imp['Support Vector'] = clf.coef_.ravel()
            return feature_imp


        def linear_model(X_train,y_train,column_names,feature_imp):
            from sklearn.linear_model import SGDClassifier
            clf = SGDClassifier().fit(X_train,y_train)
            feature_imp['Linear SGD'] = clf.coef_.ravel()
            return feature_imp


        def trees_model(X_train,y_train,column_names,feature_imp):
            from sklearn.ensemble import ExtraTreesClassifier, RandomForestClassifier
            tree = [('Random Forest',RandomForestClassifier()),('Extra Tree',ExtraTreesClassifier())]
            for tup in tree:
                clf = tup[1].fit(X_train,y_train)
                feature_imp[tup[0]] = clf.feature_importances_
            
            return feature_imp


        def run_all(X_train,y_train,column_names,feature_imp):
            '''
            method to return the dataframe of importance for all the attributes of data given using all of the
            possible classification models such as Logistic Reg, SVM, Trees etc
            '''
            features = base_model(X_train,y_train,column_names,feature_imp,True)
            features.join(linear_model(X_train,y_train,column_names,feature_imp,True))
            features.join(svc_model(X_train,y_train,column_names,feature_imp,True))
            features.join(trees_model(X_train,y_train,column_names,feature_imp,True))
            return features

        if use_type == 'base':
            return base_model(X_train,y_train,column_names,feature_imp)

        elif use_type == 'support_vector':
            return svc_model(X_train,y_train,column_names,feature_imp)

        elif use_type == 'linear':
            return linear_model(X_train,y_train,column_names,feature_imp)

        elif use_type == 'tree':
            return trees_model(X_train,y_train,column_names,feature_imp)

        else:
            return run_all(X_train,y_train,column_names,feature_imp)
      
    
    
    def univariate_test(self,X_train,y_train,column_names,method='all',transform=False,num_feat=1):
    
        '''
        method to get the contribution of attributes towards the final 'target' for classification problem
        args:
            X_train: numpy array of training attributes
            y_train: 'target' column
            column_names: names of columns from the original array so that the relative names and their contribution 
            can be displayed
            method: {string} 'mutual_info','f_score', 'chi' or 'all'. Check sklearn.feature_selection.SelectKBest for more
            transform: Whether to return the transformed X_train or not
            num_feat: if transform=True, how many features finally wanted after transformation
        out:
            scores: importances of all the column for classification
            X_train: transformed X_train with top 'num_feat' columns (optional)
        '''

        from sklearn.feature_selection import SelectKBest
        from sklearn.feature_selection import f_classif,mutual_info_classif

        assert method in ['mutual_info','f_score','chi','all'], "choose suitable 'method' type. Check documentation"
        if method=='both': # both provides just the scores not transformed data
            assert transform!=True, "'transform should be False with 'method=all'"


        def f_score(X_train,y_train,column_names,transform=transform,num_feat=num_feat):
            '''
            method to implement 'f_classif' inside sklearn.feature_selection.SelectKBest
            '''
            from sklearn.feature_selection import f_classif
            x = SelectKBest(f_classif,k=num_feat).fit(X_train,y_train)
            scores = pd.DataFrame(x.scores_,index=column_names,columns=['F Score'])
            if transform:
                return (scores,x.transform(X_train))
            else:
                return scores

        def mutual(X_train,y_train,column_names,transform=transform,num_feat=num_feat):
            '''
            method to implement 'mutual_info_classif' inside sklearn.feature_selection.SelectKBest
            '''
            from sklearn.feature_selection import mutual_info_classif
            x = SelectKBest(mutual_info_classif,k=num_feat).fit(X_train,y_train)
            scores = pd.DataFrame(x.scores_,index=column_names,columns=['Mutual Info'])
            if transform:
                return (scores,x.transform(X_train))
            else:
                return scores
        
        
        def chi(X_train,y_train,column_names,transform=transform,num_feat=num_feat):
            '''
            method to implement 'chi-square' inside sklearn.feature_selection.SelectKBest
            '''
            from sklearn.feature_selection import chi2
            x = SelectKBest(chi2,k=num_feat).fit(X_train,y_train)
            scores = pd.DataFrame(x.scores_,index=column_names,columns=['Mutual Info'])
            if transform:
                return (scores,x.transform(X_train))
            else:
                return scores

            
        if method=='mutual_info':
            return mutual(X_train,y_train,column_names,transform,num_feat)

        elif method =='f_score':
            return f_score(X_train,y_train,column_names,transform,num_feat)
        
        elif method =='chi_square':
            return chi(X_train,y_train,column_names,transform,num_feat)
        

        else:
            scores = mutual(X_train,y_train,column_names,transform=False)
            scores['F Score'] = f_score(X_train,y_train,column_names,transform=False).values
            scores['Chi Square'] = chi(X_train,y_train,column_names,transform=False).values
            return scores
     
    

class RegFeatSel():
    '''
    class to find the feature importance and selection feature selections
    '''
    def feature_importances(self,X_train,y_train,column_names,use_type='all'): 
        '''
        Method to find the feature importances of different features based on different models
        args:
            X_train: numpy array of attributes
            y_train: 'target' or classes column
            column_names: list of names of all the columns that are present in X_train
            use_type: {string} 'base' for OLS,'linear' for Lasso, 'tree' for ExtraTree and RandomForest, 'all' for 
            all the above
        out:
            dataframe displaying importances of all the columns
        '''

        assert use_type in ['base','linear','tree','all'], "provide suitable value for 'use_type'. See Docs"

        feature_imp = pd.DataFrame(index=column_names) 

        def base_model(X_train,y_train,column_names,feature_imp):
            import statsmodels.api as sm
            X_train = sm.add_constant(X_train)
            model = sm.OLS(y_train,X_train).fit()
            feature_imp['OLS p-values '] = model.pvalues[1:]
            return feature_imp
        
        
        def linear_model(X_train,y_train,column_names,feature_imp):
            from sklearn.linear_model import Lasso
            model = Lasso().fit(X_train,y_train)
            feature_imp['Lasso Regression'] = model.coef_.ravel()
            return feature_imp


        def trees_model(X_train,y_train,column_names,feature_imp):
            from sklearn.ensemble import ExtraTreesRegressor, RandomForestRegressor
            tree = [('Random Forest',RandomForestRegressor()),('Extra Tree',ExtraTreesRegressor())]
            for tup in tree:
                model = tup[1].fit(X_train,y_train)
                feature_imp[tup[0]] = model.feature_importances_
            return feature_imp


        def run_all(X_train,y_train,column_names,feature_imp):
            '''
            method to return the dataframe of importance for all the attributes of data given using all of the
            possible regression models such as Lasso Reg, Trees etc
            '''
            features = base_model(X_train,y_train,column_names,feature_imp,True)
            features.join(linear_model(X_train,y_train,column_names,feature_imp,True))
            features.join(trees_model(X_train,y_train,column_names,feature_imp,True))
            return features

        if use_type=='base':
            return base_model(X_train,y_train,column_names,feature_imp)
        
        if use_type == 'linear':
            return linear_model(X_train,y_train,column_names,feature_imp)

        elif use_type == 'tree':
            return trees_model(X_train,y_train,column_names,feature_imp)

        else:
            return run_all(X_train,y_train,column_names,feature_imp)
      
    
    
    def univariate_test(self,X_train,y_train,column_names,method='both',transform=False,num_feat=1):
    
        '''
        method to get the contribution of attributes towards the final 'target' for Regression problem
        args:
            X_train: numpy array of training attributes
            y_train: 'target' column
            column_names: names of columns from the original array so that the relative names and their contribution 
            can be displayed
            method: {string} 'mutual_info','f_score' or 'all'. Check sklearn.feature_selection.SelectKBest for more
            transform: Whether to return the transformed X_train or not
            num_feat: if transform=True, how many features finally wanted after transformation
        out:
            scores: importances of all the column for regression
            X_train: transformed X_train with top 'num_feat' columns (optional)
        '''

        from sklearn.feature_selection import SelectKBest
        from sklearn.feature_selection import f_regression,mutual_info_regression

        assert method in ['mutual_info','f_score','all'], "choose suitable 'method' type. Check documentation"
        if method=='both': # both provides just the scores not transformed data
            assert transform!=True, "'transform should be False with 'method=all'"


        def f_score(X_train,y_train,column_names,transform=transform,num_feat=num_feat):
            '''
            method to implement 'f_regression' inside sklearn.feature_selection.SelectKBest
            '''
            from sklearn.feature_selection import f_regression
            x = SelectKBest(f_regression,k=num_feat).fit(X_train,y_train)
            scores = pd.DataFrame(x.scores_,index=column_names,columns=['F Score'])
            if transform:
                return (scores,x.transform(X_train))
            else:
                return scores

        def mutual(X_train,y_train,column_names,transform=transform,num_feat=num_feat):
            '''
            method to implement 'mutual_info_regression' inside sklearn.feature_selection.SelectKBest
            '''
            from sklearn.feature_selection import mutual_info_regression
            x = SelectKBest(mutual_info_regression,k=num_feat).fit(X_train,y_train)
            scores = pd.DataFrame(x.scores_,index=column_names,columns=['Mutual Info'])
            if transform:
                return (scores,x.transform(X_train))
            else:
                return scores


        if method=='mutual_info':
            return mutual(X_train,y_train,column_names,transform,num_feat)

        elif method =='f_score':
            return f_score(X_train,y_train,column_names,transform,num_feat)

        else:
            scores = mutual(X_train,y_train,column_names,transform=False)
            scores['F Score'] = f_score(X_train,y_train,column_names,transform=False).values
            return scores
        
        

class BasicTest():
    '''
    class to implement basic tests for feature selections like Variance threshold, Variance Inflation Factor etc
    '''
    def var_thresh_based(self,df_x,label_col,transform=False,threshold=0):
        '''
        method to find and get the features which have variance more than a threshold
        args:
            df_x: pandas dataframe. df should be raw. No standardisation or scaling should be in the df
            label_col: target y column name {optional}
            thresh: float. return the columns which have variance greater than this after transformation  
        out:
            pandas series with variances of individual columns
            transformed numpy array {if transform=True} WITHOUT 'label_col'
        '''
        from sklearn.feature_selection import VarianceThreshold
        
        y=df_x[label_col].values
        df = df_x.drop(label_col,axis=1)
        assert type(df)==pd.core.frame.DataFrame, "'df' should be a Pandas DataFrame"
        
        scores = pd.DataFrame(df.var(),columns=['Variance'])
        model = VarianceThreshold(threshold=threshold).fit(df,y)
        if transform:
            return (scores,model.transform(df))
        else:
            return scores
        
        
    def var_inflation(self,df_raw,label_col,thresh=5,transform=False):
        '''
        method to perform the variance threshold analysis on a given Pandas Dataframe. For more information 
        google 'variance_inflation_factor'
        args:
            df_raw: pandas dataframe {raw df without scaling or standardization}
            thresh: threshold to drop the columns
                    thresh <= 1 : not correlated
                    1 <thresh< 5 : moderately correlated
                    thresh> 5 :    highly correlated
            transform: {optional} whether to drop the columns based on the threshold
            
        out:
            score: pandas dataframe showing Variance Inflation Factors of different columns
            transformed Dataframe with dropped columns
        '''
        
        from statsmodels.stats.outliers_influence import variance_inflation_factor as VIF
        
        df = df_raw.drop(label_col,axis=1)
        assert type(df)==pd.core.frame.DataFrame, "'df' should be a Pandas DataFrame"
        
        vif = pd.DataFrame(index=df.columns)
        vif["VIF Factor"] = [VIF(df.values, i) for i in range(df.shape[1])]
        if transform:
            return (vif,df.loc[:,(vif['VIF Factor']>thresh).values])
        else:
            return vif
        
    
    def correlation(self,df,label_col,method='pearson',transform=False,thresh=0.10):
        '''
        method to find the columns which columns have most correlation with the target label
        args:
            df: pandas datafrme WITHOUT normalized,scaled or Standardized
            label_col: name of target column
            method: any one from 'pearson', 'kendall', 'spearman'
            transform: whether to return transformed dataframe with less columns
            thresh: drop the columns which have a correlation value less than threshold
        out:
            corr: pandas series with correlation value of each column to target_col
            dataframe: transformed dataframe with less columns
        '''
        if transform:
            assert thresh>0, "provide a value of 'thresh' > 0 with 'transform=True'"
            
        corr = df.corr(method=method)[label_col].drop(label_col).rename(label_col+' Corr')
        if transform:
            return (corr,df.drop(label_col,axis=1).loc[:,abs(corr)>thresh])
        else:
            return corr
        
        
    def get_all(self,df,label_col):
        '''
        apply all of the 'var_thresh_based', 'var_inflation' and 'correlation' without transformation
        '''
        result = self.var_thresh_based(df,label_col)
        result = result.join(self.var_inflation(df,label_col))
        result = result.join(self.correlation(df,label_col))
        return result
    
    
    
class F_B_R_Selection():
    '''
    Implement Forward, Backward or Recursive Feature Selection
    NOTE: Data should be scaled for some processes
    '''
    def __init__(self,problem,model_type='base'):
        '''
        constructor of the class
        args:
            problem: 'regression'/ 'classification'
            model: 'base', 'support_vector','tree','ensemble'
        '''
        
        self.problem = problem
        assert self.problem in ['regression','classification'], "'problem' should be either 'regression' or 'classification'"
        try:
            from mlxtend.feature_selection import SequentialFeatureSelector as SFS
        except ModuleNotFoundError:
            !pip install mlxtend
            from mlxtend.feature_selection import SequentialFeatureSelector as SFS
        
        
        if self.problem =='classification': # if classification
            if model_type=='base':
                from sklearn.linear_model import LogisticRegression
                self.model = LogisticRegression()
            elif model_type=='linear':
                from sklearn.svm import LinearSVC
                self.model = LinearSVC()
            elif model_type =='tree':
                from sklearn.ensemble import RandomForestClassifier
                self.model = RandomForestClassifier()
            elif model_type == 'ensemble':
                from sklearn.ensemble import ExtraTreesClassifier
                self.model = ExtraTreesClassifier()
        
        else: # if problem is regression
            if model_type =='base':
                from sklearn.linear_model import Lasso
                self.model = Lasso()
            elif model_type =='linear':
                from sklearn.svm import LinearSVR
                self.model = LinearSVR()
            elif model_type =='tree':
                from sklearn.ensemble import RandomForestRegressor
                self.model = RandomForestRegressor()
            elif model_type == 'ensemble':
                from sklearn.ensemble import ExtraTreesRegressor
                self.model = ExtraTreesRegressor()
         
        
        
        
    def execute(self,X,y,n_features,selection_type='forward',scoring=None,transform=False):
        '''
        perform Forward feature selection for the data given
        NOTE: X should be Scaled for some models as it might fail to converge
        args:
            X: Features dataframe
            y: label column
            n_features: final number of features
            selection_type: 'forward','backward','step','recursive'
            model: 'base', 'support_vector','tree','ensemble'
            scoring: {string,callable,method} optional
            transform: whether to return a transformed dataframe
        out: 
            names of n selected features is 'transform' is False
            transformed array with n selected features if transform is True
        '''
        assert type(X)==pd.core.frame.DataFrame, "'X' should be a Pandas DataFrame"
        from mlxtend.feature_selection import SequentialFeatureSelector as SFS
        
        
        if selection_type == 'recursive':
            from sklearn.feature_selection import RFE
            sfs = RFE(self.model,n_features).fit(X,y)
            selected = sfs.support_
            selected = X.loc[:,selected.tolist()].columns.tolist()
        
        
        else:
        
            if selection_type == 'forward':
                sfs = SFS(self.model,n_features,forward=True,scoring=scoring,floating=False).fit(X,y)
            elif selection_type =='backward':
                sfs = SFS(self.model,n_features,forward=False,scoring=scoring,floating=False).fit(X,y)
            elif selection_type == 'step':
                sfs = SFS(self.model,n_features,forward=False,scoring=scoring,floating=True).fit(X,y)   
            
            selected = sfs.k_feature_names_
        
        if transform:
            return sfs.transform(X)
        else:
            return selected          