from sklearn.ensemble import RandomForestClassifier
from sklearn.model_selection import StratifiedKFold
from sklearn.model_selection import train_test_split
from sklearn.inspection import permutation_importance
import etl
import pandas as pd
import numpy as np

class Model:
    
    def _init_(self):
       
       # italizes a blank model
        
        #Let's store some placeholder meta-data that we will change in our fit.
        """
        I created these 4 new self variables which allow me to use the train and test x and y's through          out the class. 
        I don't think I need other variable and maybe the 4 new variable i added are redundent but it was the only way i think I could make this code work.
        """
        self._Xtrain = None
        self._Xtest = None
        self._ytrain = None
        self._ytest = None
        self._target = None
        self._model = None
        self._shape = None
        self._columns = []



    def learn(self, df, target = None, kind = 'RF', *args, **kwargs):
        """
        1.Make sure there is a value for target, if there isnt then throw an error 
        2. Then check if the classifier is a Random Forest and if it isnt then check if the 
        classifier is a Linear classifier.
        3.Split the data in to training and testing(i saw the key-word args in the parameters of the funcation so maybe i could apply them to both the RandomForestClassifier and train_test_split
        4.fit the model
        """
       
        
        if target is None:
            raise ValueError("The target cannot be None.")
            ...
        if kind == "RF":
            model = RandomForestClassifier(*args, **kwargs)
        elif kind == "Linear":
            raise ValueError("The target should be RF, for now")
        transformer = etl.PandasTransformer()
        transformer.fit(df)
        X,y = transformer.transform(df)
      
      
        X_train, X_test, y_train, y_test = train_test_split(X,y, **kwargs)
        

        # now fit the model
        model.fit(X,y)
        self._Xtrain = X_train
        self._Xtest = X_test
        self._ytrain = y_train
        self._ytest = y_train
        self._model = model
        self._target = target
        self._shape = X.shape
        self._columns = df.columns

    def predict(self, df):
        """
        first we check if there is a trained model
        if there is a trained model I return a a prediction using the self._model.predict.
        need to figure out a way to transform my numpy back to pandas
        i make it a pandas dataframe and make the column the name of the target
        """
        
        
        if self._model is None:
            raise ValueError("You must train a model first with None first.")
        else:
           p_value =pd.DataFrame({self._target:self._model.predict(df.drop(self._target,axis = 1))})

        return p_value

    def accuracy(self, df):
        """
        so I score both the training and testing sets of data and then print out what the values would be
        """
        train_score = self._model.score(self._Xtrain,self._ytrain)
        test_score = self._model.score(self._Xtest,self._ytest)
        return "Train Score: ", train_score, "\nTest Score: ", test_score
    
    def auc(self,df):
        """
        using the sci kit learn shown below i found the roc_auc_score
        https://scikit-learn.org/stable/modules/generated/sklearn.metrics.roc_auc_score.html
        """
        return roc_auc_score(self._ytest, RFC.predict_proba(self.Xtest)[:,1])

    def crossValidate(self,df,splits):
        """
        first import libraries 
        define my X and Y
        Then I 
        """
        
        transformer = PandasTransformer()
        transformer.fit(df)
        X,y = transformer.transform(df)
       

       #folds = StratifiedKFold(n_splits = splits)

        scores = []

        for train_index, test_index in kf.split(X,y):
            X_train, X_test, y_train, y_test = X.iloc[train_index],X.iloc[test_index], y.iloc[train_index], y.iloc[test_index]
            self._model.fit(X_train,y_train)
            scores.append(get_scores(self._model, X_train, X_test, y_train, y_test))

        cv_scores = pd.DataFrame({"Cross-Validation Scores":scores_rf})

        return cv_scores
            
        

    def feature_importances(self,df, num_trials):
        X = df.drop(self._target, axis = 1)
        y = df.iloc[:,self._target]
        r_multi = permutation_importance(self._model, X, y,
                                         n_repeats = num_trials,
                                         random_state = 0)
        importance_score =pd.DataFrame({"Feature Importances":r_multi["importances_mean"]})
                                      # self._model.feature_names)

        return importance_score
        
   # def grid_search(df,target, *args, **kwargs):
            
    def readData(self, loc):
        #function that has a dataset url/file location/ etc as a parameter
        import pandas as pd #import pandas
        df = pd.read_csv(loc) #read the url given
        return df #return dataframe

"""
    def split(X, y, size, rand, shuf):
       
        splits a dataset in a training and testing, needs the X and Y variables, size of testing sample, what the random state should be and if shuffle is True or false
       
        from sklearn.model_selection import train_test_split
    import numpy as np
    # import numpy for .shape()

    X_train , X_test, y_train, y_test = train_test_split(X, y, test_size = size, random_state = rand, shuffle = shuf)
    #spliting the data in testing and training

      #return print("Training Set: ",np.shape(X_train), np.shape(y_train),"Testing set: ",np.shape(X_test), np.shape(y_test))#printing the training and testing shape
    return X_train, y_train, X_test, y_test


def get_score(model, X_train, X_test, y_train, y_test):
    model.fit(X_train, y_train)
    return model.score(X_test, y_test)
"""

    
    
    
