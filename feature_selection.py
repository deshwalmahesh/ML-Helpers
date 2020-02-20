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

        feature_imp = pd.DataFrame(np.zeros(len(column_names)),index=column_names) 

        def base_model(X_train,y_train,column_names,feature_imp,with_0=False):
            from sklearn.linear_model import LogisticRegression
            clf = LogisticRegression().fit(X_train,y_train)
            feature_imp['Logistic Reg'] = clf.coef_.ravel()
            if with_0:
                return feature_imp
            else:
                return feature_imp.drop(0,axis=1)


        def svc_model(X_train,y_train,column_names,feature_imp,with_0=False):
            from sklearn.svm import LinearSVC
            from sklearn.linear_model import SGDClassifier
            clf = LinearSVC().fit(X_train,y_train)
            feature_imp['Support Vector'] = clf.coef_.ravel()
            if with_0:
                return feature_imp
            else:
                return feature_imp.drop(0,axis=1) 


        def linear_model(X_train,y_train,column_names,feature_imp,with_0=False):
            from sklearn.linear_model import SGDClassifier
            clf = SGDClassifier().fit(X_train,y_train)
            feature_imp['Linear SGD'] = clf.coef_.ravel()
            if with_0:
                return feature_imp
            else:
                return feature_imp.drop(0,axis=1)


        def trees_model(X_train,y_train,column_names,feature_imp,with_0=
                        False):
            from sklearn.ensemble import ExtraTreesClassifier, RandomForestClassifier
            tree = [('Random Forest',RandomForestClassifier()),('Extra Tree',ExtraTreesClassifier())]
            for tup in tree:
                clf = tup[1].fit(X_train,y_train)
                feature_imp[tup[0]] = clf.feature_importances_
            if with_0:
                return feature_imp
            else:
                return feature_imp.drop(0,axis=1)


        def run_all(X_train,y_train,column_names,feature_imp):
            '''
            method to return the dataframe of importance for all the attributes of data given using all of the
            possible classification models such as Logistic Reg, SVM, Trees etc
            '''
            features = base_model(X_train,y_train,column_names,feature_imp,True)
            features.merge(linear_model(X_train,y_train,column_names,feature_imp,True),on=0)
            features.merge(svc_model(X_train,y_train,column_names,feature_imp,True),on=0)
            features.merge(trees_model(X_train,y_train,column_names,feature_imp,True),on=0)
            return features.drop(0,axis=1)

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
      
    
    
    def stat_test(self,X_train,y_train,column_names,method='all',transform=False,num_feat=1):
    
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
            use_type: {string} 'base' for Lasso, 'tree' for ExtraTree and RandomForest, 'all' for all the above
        out:
            dataframe displaying importances of all the columns
        '''

        assert use_type in ['base','tree','all'], "provide suitable value for 'use_type'. See Docs"

        feature_imp = pd.DataFrame(np.zeros(len(column_names)),index=column_names) 

        def base_model(X_train,y_train,column_names,feature_imp,with_0=False):
            from sklearn.linear_model import Lasso
            clf = Lasso().fit(X_train,y_train)
            feature_imp['Lasso Regression'] = clf.coef_.ravel()
            if with_0:
                return feature_imp
            else:
                return feature_imp.drop(0,axis=1)



        def trees_model(X_train,y_train,column_names,feature_imp,with_0=False):
            from sklearn.ensemble import ExtraTreesRegressor, RandomForestRegressor
            tree = [('Random Forest',RandomForestRegressor()),('Extra Tree',ExtraTreesRegressor())]
            for tup in tree:
                clf = tup[1].fit(X_train,y_train)
                feature_imp[tup[0]] = clf.feature_importances_
            if with_0:
                return feature_imp
            else:
                return feature_imp.drop(0,axis=1)


        def run_all(X_train,y_train,column_names,feature_imp):
            '''
            method to return the dataframe of importance for all the attributes of data given using all of the
            possible regression models such as Lasso Reg, Trees etc
            '''
            features = base_model(X_train,y_train,column_names,feature_imp,True)
            features.merge(trees_model(X_train,y_train,column_names,feature_imp,True),on=0)
            return features.drop(0,axis=1)

        if use_type == 'base':
            return base_model(X_train,y_train,column_names,feature_imp)

        elif use_type == 'tree':
            return trees_model(X_train,y_train,column_names,feature_imp)

        else:
            return run_all(X_train,y_train,column_names,feature_imp)
      
    
    
    def stat_test(self,X_train,y_train,column_names,method='both',transform=False,num_feat=1):
    
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