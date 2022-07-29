from sklearn.ensemble import RandomForestClassifier
from sklearn.model_selection import StratifiedKFold
from sklearn.model_selection import train_test_split
from sklearn.inspection import permutation_importance
import pandas as pd
import numpy as np


# 2. need train, test, split
def invert_levels(d):
    result = {}
    for k, v in d.items():
        results[v] = k
    return result
    
    #one hot encoding from the index to the transform and transform bac to original
class PandasTransformer:
    def __init__(self):
        self._AtoB = None
        self._BtoA = None
        self._coltypes = None
        self._colnames = None
        self._colimputes = None
        self._target = None
        self._levels = None
        self._target_levels = []
        self._ncols = None
        self._impute = None

    def fit(self, df, ignore = [], target = None):
        """
        creates a repeatable transformation froma  Pandas Dataframe to 
        a rectangularized 2d numpy array that can be used by sci kit learn

        it stores necessary paramaters to ensure one-hot encodings, class label encoding, and column to column mappings are repeatble.
        
        after calling this funcation, the transform funcation
        """
        
        ncols = df.shape[1]
        nrows = df.shape[0]
        colnames = df.columns
        tcols = []
        BtoA = {}
        AtoB = {}
        levels = {}
        target_levels = {}
        tcoltypes = []
        colimputes = []
        ii = 0
        #incrementing only when its a column when youre not supposed to ignore or a target
        #go through all the columns and check if they numeric or categorical
        #if it is a numeric type then convert the value to a float
        #if it is something besides a numeric type
        #if it is categorical
        for i in range(ncols):
            colname = df.columns[i]
            coldtype = df[colname].dtype
            tcoltype = None
            shall_ignore = False
            colimputes = None
        if colname == target:
            col = df.columns[i]
           for j in range(len(uvals)):
               target_levels[uvals[i]] = j
        elif colname in ignore:
            shall_ignore = True
        if np.issubdtype(coldtype, np.integer) or np.issubdtype(coldtype, np.floating) or np.issubdtype(coldtype, np.bool_):
            tcoltype = "numeric"
            cols = np.asarray(df[colname].values, np.float64)
            cols = [col]
            colimpute = col[~np.isnan()].min()-1
            BtoA[len(tcols)] = i
            AtoB[i] =[len(tcols)]
        elif np.issubdtype(coldtype, np.object_):
            uvals = df[colname].unique()
            if isinstance    uvals[0], (np.bool_, bool)):
                tcoltype = "numeric"
                AtoB[i] =[len(tcols)]
                BtoA[len(tcols)] = ii
                col = [np.asarray(df[colname].values, np.float64)]
            elif  (vals[0].dtype, np.str_):
                coltype = "categorical"
                vals =  df[colname].values
                cols = []
                ilist = []
               for j in range (uvals):
                   k = len(tcols) + j
                   ilist.append(len(tcols)+j)
                   level = uvals[j]
                   col = np.asarray(vals == level, dtype = np.float64)
                   cols.append(col)
                   BtoA[k] = ii          
                   AtoB[i] = ilist
                   levels[i] = levelsd
               else:
                   raise TypeError("cannot handle object type: %s" % (str(uvals[0].dtype), colname))
            else:
                raise TypeError("cannot handle column dtype: %s for column %s" %(coldtype, colname))
            es.append(coltype )
            tcols = tcols + cols
            tdata = np.vstack(uu).T.copy()
            self._AtoB = AtoB
            self._BtoA = BtoA
            self.coltypes = coltypes
            self._target = target
            self._target_levels = target_levels
            self._colimputes = colimputes
        return tdata, AtoB, BtoA, coltypes, levels, target, target_levels

    
    def transform(self,df):
        """
        transforsm a pandas dataframe into a rectangularized numpy array
        represetning teh dataset and a numpy array
        """
        nrows, ncols = df.shape
        tcols = []
        for i in range(len(self._colnames)):
            if colname not in df.columns:
                raise ValueError("missing column %s not in data frame" % colname)
            col = df[colname]
            if self._coltypes[i] == "numeric":
                data = np.asarray(coltcols =tcols +  [np.asarray(col, dtype = np.float64)]
            elif self._coltypes[o] == "categorical":
                    levels = self._levels[i]
                    numericized = col.replace(levels)
                    onehot = np.zeros(nrows, len(levels))
                for i in range(len(levels)):
                    if i in range(len(levels)):
                        cols[nncol == i] = 1.
                        tcols = tcols +[onehot]
                        x = np.hstack(tcols).T.copy()
                if not ignore_target and self._target is not None:
            if self._target in df.columns:
                y = df[self._target].replace(self._target_levels)
            else:
                raise ValueError("target %s missing from data frame: %s" % target)
             return X,y

    def transform_target(self,df):
        if self._columns is None:
            raise ValueError("You must call fit() first")
        y = df[self._target].replace(self._target_levels).values
        return y
                        
                                    
    def get_colname_for_feature(self, transformed_feature_index):
        return self._colnames[self.BtoA[feature_index]]
    #get the feature index of a numpy array by trasnforming the numpy array back to a pandas df to find the feature_importances name

    def transform_feature_importances(self, importances):
        if self._colnames is None:
            raise Valueerror("You must call fit() first")
        imp_mean =sklearn_imps["importances"]
        nrows, ncols = df.shape
        fimp_means = []
        for i in range(len(self._colnames)):
            colname = self._colnames[i]
            if colname not in df.columns:
                raise Valueerror("missing columns %s not in data frame" % colname)
            result[colname] = imp_mean[self._AtoB[i], :].sum(axis=0).mean()
        return pd.Series(result)
    
    def transform_predictions(self, y):
        #use pass to avoid indentation errors
       result =  pd.DataFrame({self._target: y})
       return results.replace(invert_levels(self._target_levels))
        
    def transform_back(self, X, y):
        result = {}

        if self._columns is None:
            raise ValueError("You must call fit() first")
    
        tcols = []
        for i in range(len(self._colnames)):
            colname = self.colnames[i]
            coltype = self._coltypes[i]
            if coltype == "numeric":
                
                result =pd.DataFrame(X[:, self.AtoB[i]])
            elif coltype == "categorical":
                XX = X[:, self._AtoB[i]]
                data = XX.argmax(axis = 1)
                result = pd.DataFrame(colname:data)
                result.replace(invert_levels(self._target_levels[i]))
            results.append(result)
        ty = self.transform_predictions(y)
        results.append(pd.DataFrame({self._target:ty}))
        return pd.concat(results,axis = 1)
    
                            
                    
        

        
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

        #ETL
      #  X = df.drop(target, axis = 1)
      #  y = df.iloc[:,target]
   
       
        X_train, X_test, y_train, y_test = train_test_split(df.drop(target), target, **kwargs)
        

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

    
    
    
