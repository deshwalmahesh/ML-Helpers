import pandas as pd
import numpy as np
import seaborn as sns
import matplotlib.pyplot as plt
from scipy import stats
from scipy.stats import norm


class EDA():
    '''
    Very basic EDA class to show the very basic nature of data. 
    dataframe is not required for the __init__() constructor to make it very generic.
    A pandas dataframe is required to be in every method.
    '''    


    def find_missing(self,df):
        '''
        Find the missing values by either looking at the np.nan() sum or by looking at minimum values which are not 
        supposed to be there such as height can not be 0 or negative
        '''

        print('Missing Values Count in each Column\n\n',df.isna().sum(),'\n')
        print('Check if missing value is replaced by 0\n\n',df.min(),'\n')
        print('Check unique values in each column\n\n',df.nunique(),'\n')


    def check_distribution(self,df,column_name):
        '''
        Checks the distribution of column in data given the column is categorical in nature.
        input:
            class_name: name of the column in dataframe
        out:
            display a pie chart with respective percentages of the classes 
        '''

        df[column_name].value_counts().plot(kind='pie',autopct='%1.2f%%', rotatelabels=True)


    def basic_plots(self,df,force=False,cols=3,PAD=3,width=15,return_cols=False,tight=False):
        '''
        if there are less than 15 features, it plots the basic plots to get the visualization of data. 
        If data is categorical, it plots the bar else the distribution for numerical. Last plot is always violin
        args:
            df: pandas dataframe
            force: whether to plot the who dataFrame
            cols: number of columns for the subplots
            PAD: padding for spacing of subplots
            width: change whole width of the graph
            return_cols: return a list of categirical and numerical columns
            tight: plt.tight_layout()
        out: 
            2 lists of names of categorical and numerical columns
        '''
        cat_cols = []
        num_cols = []
        rows=df.shape[1]//cols+1
        if (not force and df.shape[1]<=15) or force:
            f,ax = plt.subplots(nrows=rows,ncols=cols, figsize=(width,df.shape[1]+3))
            ax = ax.ravel()

            for i , column in enumerate(df.columns): # if purely categorical
                ax[i].set_xlabel(column)
                if df[column].dtypes == 'O': # class names a,b,c,d
                    df[column].value_counts().plot(kind='bar',ax=ax[i])
                    cat_cols.append(column)

                else:
                    if df[column].nunique() < df.shape[0]//10: # classes 1,2,3,4
                        df[column].value_counts().plot(kind='bar',ax=ax[i])
                        cat_cols.append(column)
                    else:
                        sns.distplot(df[column],ax=ax[i])
                        num_cols.append(column)

        
            for axis_num in range(i+1,(cols*rows)): # delete the remaining empty plots
                f.delaxes(ax[axis_num])
            
            if tight:
                plt.tight_layout()
            plt.show()
            
            if return_cols:
                return cat_cols,num_cols

        else: 
            print("There are more columns than allowed. Please pass 'force=True'")
       
    
        
    def skew_plot(self,df,num_cols,cols=4,width=15,height=12,tight=True):
        '''
        Find skewness of the numerical data
        args:
            df: pandas dataframe
            num_cols: list of names of numerical cols of the dataframe
            cols: number of columns for the subplots
            PAD: padding for spacing of subplots
            width: width of figure size
            height: figure height
            tight: plt.tight_layout()
        '''
        rows = len(num_cols)//2+1
        f,ax = plt.subplots(nrows=rows,ncols=cols, figsize=(width,height))
        ax = ax.ravel()
        
        i=0
        col_num=0
        while col_num<=len(num_cols)-1:
            column = num_cols[col_num]
            sns.distplot(df[column], fit=norm,ax=ax[i])
            stats.probplot(df[column], plot=ax[i+1])
            i+=2
            col_num+=1
        
        for axis_num in range(i,(cols*rows)): # delete the remaining empty plots
                f.delaxes(ax[axis_num])
        
        if tight:
            plt.tight_layout()
        plt.show()


    
    def box_vio(self,df,num_cols,cols=4,width=15,height=12,tight=True):
        '''
        plot box and violin plots of numerical plots
        args:
            df: pandas dataframe
            num_cols: list of names of numerical cols of the dataframe
            cols: number of columns for the subplots
            PAD: padding for spacing of subplots
            width: width of figure size
            height: figure height
            tight: plt.tight_layout()
        '''
        rows = len(num_cols)//2+1
        f,ax = plt.subplots(nrows=rows,ncols=cols, figsize=(width,height))
        ax = ax.ravel()
        
        i=0
        col_num=0
        while col_num<=len(num_cols)-1:
            column = num_cols[col_num]
            sns.boxplot(y=df[column],ax=ax[i],color='m')
            ax[i].set_xlabel(column+ ' Box')
            sns.violinplot(y=df[column],ax=ax[i+1],color='teal')
            ax[i+1].set_xlabel(column+ ' Violin')
            i+=2
            col_num+=1
        
        for axis_num in range(i,(cols*rows)): # delete the remaining empty plots
                f.delaxes(ax[axis_num])
        
        if tight:
            plt.tight_layout()
        plt.show()


    def fix_heatmap(self):
        '''
        fix the half cut upper and lower boxes of a heatmap in newer versions of Seaborn
        '''
        b, t = plt.ylim() # discover the values for bottom and top
        b += 0.5 # Add 0.5 to the bottom
        t -= 0.5 # Subtract 0.5 from the top
        plt.ylim(b, t) # update the ylim(bottom, top) values
        plt.show()


    def show_corr(self,df,plot=True,return_corr_df=False,size=(10,10)):
        '''
        Show the correlation within a dataframe
        args:
            df: pandas datadframe
            plot: {bool} default False. Whether to show the heatmap or not
            return_corr_df: return the correlated df or not
            size: tuple (w,h) for the size of plot
        out:
            returns correlated squared DataFrame 
        '''

        if plot:
            fig, ax = plt.subplots(figsize=size)
            sns.heatmap(df.corr(),annot=True, cmap='coolwarm',facecolor='b',lw=2, ax=ax)
            self.fix_heatmap()

        if return_corr_df:
            print(f'Correlation of each column:\n\n')
            return df.corr()    


    
    def classification_prob_plot(self,df,col_name,cols=3,force=False,PAD=3,width=15,tight=False):
        '''
        method to show the relations of all the columns with the categorical column.
        args:
            df: pandas dataframe
            col_name: any categorical column whose relation/distribution we want to see with all the columns in that df
            rows: number of rows for subplots
            force: whether to show each and every column
            PAD: Padding for subplots spacing
            width: used for width of graph
        '''

        rows=df.shape[1]//cols+1
        if (not force and df.shape[1]<=15) or force:
            f,ax = plt.subplots(nrows=rows,ncols=cols, figsize=(width,df.shape[1]+PAD))
            ax = ax.ravel()

            for i , column in enumerate(df.columns): # if purely categorical
                ax[i].set_xlabel(column)
                if df[column].dtypes == 'O': # class names a,b,c,d
                    sns.countplot(x=column, hue=col_name, data=df,ax=ax[i])

                else:
                    if df[column].nunique() < df.shape[0]//10: # classes 1,2,3,4
                        sns.countplot(hue=col_name, x=column, data=df,ax=ax[i])

                    else:
                        sns.stripplot(y=column, x=col_name,data=df,ax=ax[i],alpha=0.8)
                        sns.violinplot(y=column,x=col_name,data=df,ax=ax[i],inner=None,color=".88")

            for axis_num in range(i+1,(cols*rows)): # delete the remaining empty plots
                f.delaxes(ax[axis_num])
            if tight:
                plt.tight_layout()
            plt.show()

        else: 
            print("There are more columns than allowed. Please allow 'force=True'")
      
    
    
    def regression_prob_plot(self,df,col_name,cols=3,force=False,PAD=3,width=15,tight=False):
        '''
        Find the relation between numerical column and all the other columns
        args:
            df: pandas dataframe
            col_name: any categorical column whose relation/distribution we want to see with all the columns in that df
            rows: number of rows for subplots
            force: whether to show each and every column
            PAD: Padding for subplots spacing
            width: used for width of graph
        '''

        rows=df.shape[1]//cols+1
        if (not force and df.shape[1]<=15) or force:
            f,ax = plt.subplots(nrows=rows,ncols=cols, figsize=(width,df.shape[1]+PAD))
            ax = ax.ravel()

            for i , column in enumerate(df.columns): # if purely categorical
                ax[i].set_xlabel(column)
                sns.regplot(x=col_name, y=column, data=df,ax=ax[i], y_jitter=0.05)
                
            for axis_num in range(i+1,(cols*rows)): # delete the remaining empty plots
                f.delaxes(ax[axis_num])
            if tight:
                plt.tight_layout()
            plt.show()

        else: 
            print("There are more columns than allowed. Please allow 'force=True'")
            
    
    def cat_vs_nums(self,df,cat_col_name,numeric_cols,cols=3,width=12,height=7,tight=False,**kwargs):
        '''
        method to plot the distribution categorical within all  of the numerical columns
        args:
            df: pandas dataframe
            cat_col_name: name of the column whose distribution we want to check among all the numreical columns
            numeric_cols: {list} of all  numerical the column names 
            cols: number of graphs to plot per row. default 3
            width: width of plot
            height: height of plot
            tight: plt.tight_layout()
            **kwargs: sns.kdeplot() arguments
        '''
        
        class_values = df[cat_col_name].value_counts().index.tolist() # unique target classes
        rows = len(num_cols)//cols+1
        f,ax = plt.subplots(rows,cols,figsize=(width,height))
        ax = ax.ravel()
        for i,column in enumerate(numeric_cols):
            for j in class_values:
                sns.kdeplot(df.loc[df[cat_col_name] == j, column], label = f'{cat_col_name} {j}',ax=ax[i],**kwargs)
            
        for axis_num in range(i+1,(cols*rows)): # delete the remaining empty plots
            f.delaxes(ax[axis_num])
        
        if tight:
                plt.tight_layout()
        plt.show()  
        
    
    def cat_vs_cats(self,df,col_name,cat_cols,cols=4,width=15,height=9,tight=False,**kwargs):
        '''
        method to distribution of classes of a categorical among all other categorical columns
        args:
            df: pandas dataframe
            col_name: name of categorical column whose distribution we want to check
            cat_cols: {list/tuple} of all the other categorical columns
            cols: number of figures to plot at each row
            width: width of plot
            height: height of plot
            tight: plt.tight_layout()
            **kwargs: arguments for seaborn.countplot()
        '''
        rows = len(cat_cols)//cols+1
        f,ax = plt.subplots(rows,cols,figsize=(width,height))
        ax = ax.ravel()
        for i,column in enumerate(cat_cols):
            sns.countplot(x=column, hue=col_name, data=df,ax=ax[i],**kwargs)
            
        for axis_num in range(i+1,(cols*rows)): # delete the remaining empty plots
            f.delaxes(ax[axis_num])
        
        if tight:
                plt.tight_layout()
        plt.show()      
        
    
    
    def num_vs_nums(self,df,col_name,num_cols,cols=3,width=15,height=15,tight=False,**kwargs):
        '''
        method to plot the relation of one numerical column with all the other columns in a dataframe
        args:
            df: pandas dataframe
            col_name: name of numeric column whose distribution we want to check
            num_cols: {list/tuple} of all the other numerical columns
            cols: number of figures to plot at each row
            width: width of plot
            height: height of plot
            tight: plt.tight_layout()
            **kwargs: arguments for seaborn.regplot()
        '''
        rows = len(cat_cols)//cols+1
        f,ax = plt.subplots(rows,cols,figsize=(width,height))
        ax = ax.ravel()
        for i,column in enumerate(num_cols):
            sns.regplot(x=col_name, y=column, data=df,ax=ax[i], y_jitter=0.05,**kwargs)
            
        for axis_num in range(i+1,(cols*rows)): # delete the remaining empty plots
            f.delaxes(ax[axis_num])
        
        if tight:
                plt.tight_layout()
        plt.show() 
        
        
    
    def num_vs_cats(self,df,col_name,cat_cols,cols=3,width=15,height=15,tight=False,plot_type='reg',**kwargs):
        '''
        method to plot the relation distribution of one categorical column with all the other columns in a dataframe
        args:
            df: pandas dataframe
            col_name: name of numeric column whose distribution we want to check
            num_cols: {list/tuple} of all the other numerical columns
            cols: number of figures to plot at each row
            width: width of plot
            height: height of plot
            tight: plt.tight_layout()
            plot_type = {str} "strip", "swarm", "reg"
            **kwargs: arguments for seaborn.regplot() / strip / swarm /
        '''
        rows = len(cat_cols)//cols+1
        f,ax = plt.subplots(rows,cols,figsize=(width,height))
        ax = ax.ravel()
        for i,column in enumerate(cat_cols):
            if plot_type == 'reg':
                sns.regplot(x=col_name, y=column, data=df,ax=ax[i], y_jitter=0.05,**kwargs)
                
            elif plot_type == 'strip':
                sns.stripplot(x=column, y=col_name,data=df,ax=ax[i],alpha=0.8)
                sns.violinplot(x=column,y=col_name,data=df,ax=ax[i],inner=None,color="0.88")
                
            elif plot_type == 'swarm':
                sns.swarmplot(x=column, y=col_name, data=df,ax=ax[i],alpha=0.8,**kwargs)
                sns.violinplot(x=column,y=col_name,data=df,ax=ax[i],inner=None,color="0.88")
                
            
        for axis_num in range(i+1,(cols*rows)): # delete the remaining empty plots
            f.delaxes(ax[axis_num])
        
        if tight:
                plt.tight_layout()
        plt.show()
    