import numpy as np
import pandas as pd



def transform_data(data,t_type=0,m='min'):
    '''
    A t_type to transform data in Gaussian which is not Gaussian in nature using different techniques
    param:
        data: input data in the form of numpy array or pandas series
        t_type: transformation type 
                options:   int {
                                0: Square Root
                                1: Normalization
                                2: Sigmoid
                                3: Cube Root
                                4: Normalized Cube Root
                                5: Log
                                6: Log Max Root
                                7: Normalized Log
                                8: Normalized Log Max Root
                                9: Hyperbolic Tangent
                                10: 
                            }
    out:
        transformed data 
    '''

    def normalize_(data):
        upper = data.max()
        lower = data.min()
        return (column - lower)/(upper-lower)
    
    def sigmoid_(data):
        e = np.exp(1)
        return 1/(1+e**(-data))

    def log_(data):
        if data.min()>0:
            return np.log(data)
        else:
            return np.log(data+1)
    
    
    if t_type==0:
        return np.sqrt(data)

    if t_type==1:
        return normalize_(data) # normalize
    
    elif t_type==2:
        return sigmoid_(data) # sigmoid
    
    elif t_type==3:
        return data**(1/3) # cube root

    elif t_type==4:
        return normalize_(data**(1/3)) # normalized cube root

    elif t_type==5:
        return log_(data) # log

    elif t_type==6:
        return data**(np.log(data.max())) # log-max-root 

    elif t_type==7:
        return normalize_(log_(data)) # normalized log

    elif t_type==8:
        return normalize_(data**(np.log(data.max()))) # normalized log-max-root

    elif t_type==9:
        return np.tanh(data) # hyperbolic tangent
    
    elif t_type==10:
        return data.rank(method=m).apply(lambda x: (x-1)/len(data)-1)

    else:
        print('No Suitable t_type Specified. Returning Data')
        return(data) 
    

print(transform_data([1,2,3],t_type=2))



class PreProcessing():
    '''
    class to perform data transformations like correlation removal, scaling and more...
    '''
    
    
    def change_distribution(self,data,t_type=0,m='min'):
        '''
        A t_type to transform data in Gaussian which is not Gaussian in nature using different techniques
        param:
            data: input data in the form of numpy array or pandas series
            t_type: transformation type 
                    options:   int {
                                    0: Square Root
                                    1: Normalization
                                    2: Sigmoid
                                    3: Cube Root
                                    4: Normalized Cube Root
                                    5: Log
                                    6: Log Max Root
                                    7: Normalized Log
                                    8: Normalized Log Max Root
                                    9: Hyperbolic Tangent
                                    10: 
                                }
            m: method for ranking procedure. Default 'min'. Check numpy docs for more

        out:
            transformed data 
        '''

        def normalize_(data):
            upper = data.max()
            lower = data.min()
            return (column - lower)/(upper-lower)

        def sigmoid_(data):
            e = np.exp(1)
            return 1/(1+e**(-data))

        def log_(data):
            if data.min()>0:
                return np.log(data)
            else:
                return np.log(data+1)


        if t_type==0:
            return np.sqrt(data)

        if t_type==1:
            return normalize_(data) # normalize

        elif t_type==2:
            return sigmoid_(data) # sigmoid

        elif t_type==3:
            return data**(1/3) # cube root

        elif t_type==4:
            return normalize_(data**(1/3)) # normalized cube root

        elif t_type==5:
            return log_(data) # log

        elif t_type==6:
            return data**(np.log(data.max())) # log-max-root 

        elif t_type==7:
            return normalize_(log_(data)) # normalized log

        elif t_type==8:
            return normalize_(data**(np.log(data.max()))) # normalized log-max-root

        elif t_type==9:
            return np.tanh(data) # hyperbolic tangent

        elif t_type==10: # ranking 
            return data.rank(method=m).apply(lambda x: (x-1)/len(data)-1)

        else:
            print('No Suitable t_type Specified. Returning Data')
            return(data) 

    
    def transform(self,df,method='standard',columns=None):
        '''
        Method to transform the data using different data transformation techniques such as 
        Standardization, MinMax, Robust and Normalization
        
        args:
            df: pandas dataframe or numpy array. Requires only numerical values
            method: method to do the scaling
                    {
                    'standard' :  default for Standardization
                    'minmax': Minmax Scaling
                    'norm': Normalized Scaling
                    'robust': Robust Scaling
                    }
            columns: list of name of columns in case of pandas frame else the whole df is used
        out:
            arr = numpy transformed array
        '''
        if columns:
            df = df[columns]
            
        if method == 'standard':
            from sklearn.preprocessing import StandardScaler
            return StandardScaler().fit_transform(df)
        
        elif method == 'minmax':
            from skimage.preprocessing import MinmaxScaler
            return MinmaxScaler().fit_transform(df)
        
        elif method =='norm':
            from sklearn.preprocessing import Normalizer
            return Normalizer().fit_transform(df)
        
        elif method == 'robust':
            from sklearn.preprocessing import RobustScaler
            return RobustScaler.fit_transform(df)
        
        else:
            print('No suitable method found. Plese check the documentation')
            
    
    def remove_corr(self,df,thresh=0.90,drop=True,inplace=False,show_correlated=False):
        '''
        Remove the highly correlated features from a given dataset given on the threshold
        params:
            df: pandas Dataframe
            thresh: threshold value to remove the correlated features
            drop: whether to drop the correlated columns or not
            inplace: Bool. drops the dataframe value in place. default false 
            show_correlated: Bool. show the names of correlated features 
        out:
            dataframe with dropped columns
        '''
        
        corr_matrix = df.corr().abs()

        upper = corr_matrix.where(np.triu(np.ones(corr_matrix.shape), k=1).astype(np.bool))
        # Select upper triangle of correlation matrix

        to_drop = [column for column in upper.columns if any(upper[column] > thresh)]
        # Find index of feature columns with correlation greater than thresh
        
        if show_correlated:
            print(to_drop)
        
        if drop:
            return df.drop(df[to_drop], axis=1,inplace=inplace)



class PostProcessing():
    '''
    Class to attain Data Post Processing tasks like class balancing and more.
    A balanced distribution of the classes using either under or over sampling techniques on training
    examples is done for ML and AI
    '''
    def __init__(self,method='over_sample', for_NN=False):
        '''
        Constructor to store the under or over sample argument
        args:
            method: method to use for the sampling. over_sampling' by default {'over_sample','under_sample'}
        '''
        self.method = method
        self.for_NN = for_NN
        
    
    def over_sample(self,X_train=None,y_train=None, show_dist=False):
        '''
        Function to perform over sampling of minority classes by using SMOTE
        args:
            X_train: Training numerical data in form of numpy array
            y_train: Training labels which are not perfectly seperated
        out:
            X_train,y_train: over sampled minority data with increase number of training examples
        '''
        
        from imblearn.over_sampling import SMOTE
        X_train, y_train = SMOTE().smt.fit_sample(X_train, y_train)
        return X_train, y_train
    
    
    def under_sample(self,X_train=None,y_train=None,show_dist=False):
         '''
        Function to perform over sampling of minority classes by using SMOTE
        args:
            X_train: Training numerical data in form of numpy array
            y_train: Training labels which are not perfectly seperated
        out:
            X_train,y_train: over sampled minority data with increase number of training examples
        '''
            
        from imblearn.under_sampling import NearMiss
        X_train, y_train = Nearmiss().fit_sample(X_train, y_train)   
        return X_train, y_train
    
    
    def class_weight_NN(self,y_train):
        '''
        used in Neural Networks to provide bias to the under sampled classes. Use output in the class_weight
        args:
            y_train: training data labels
        out: 
            weights of corresponding classes where under sampled classes have a bias given
        '''
        
        from sklearn.utils import class_weight
        return class_weight.compute_class_weight('balanced',np.unique(y_train),y_train)
        
    
    
    def balance(self,y_train,X_train=None,for_neural=False):
        '''
        perform balancing of classes using the above methods for users who don't know what to use
        args:
            X_train :  training set in form numpy array
            y_train : labels in form numpy array
        out:
            sampled X_train, y_train
        '''
        if self.for_NN:
            for_neural = self.for_NN
        
        if for_neural:
            return self.class_weight_NN(y_train)
        
        else:
            if self.method == 'over_sample':
                return self.over_sample(X_train,y_train)
            else:
                return self.under_sample(X_train,y_train)
                
    