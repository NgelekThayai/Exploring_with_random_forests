# ETL Code
#
# Code largely provided by my mentor, Dr. Damian Eads during our one-on-one
# meeting. Improvements made by myself.

import numpy as np
import pandas as pd

def invert_levels(d):
    result = {}
    for k, v in d.items():
        result[v] = k
    return result

class PandasTransformer:

    def __init__(self):
        """
        Creates a new uninitialized PandasTransformer. Call fit() first.
        """
        self._AtoB = None
        self._BtoA = None
        self._coltypes = None
        self._colnames = None
        self._colimputes = None
        self._target = None
        self._target_levels = []
        self._levels = None
        self._impute = None

    def fit(self, df, ignore=[], target=None):
        """
        Creates a repeatable transformation from a Pandas DataFrame to
        a rectangularized 2D numpy array that can be used by scikit learn.
        
        It stores necessary parameters to ensure one-hot encodings, class
        label encodings, and column to column mappings are repeatable.

        After calling this function, the transform* functions may be used.
        """
        ncols = df.shape[1]
        nrows = df.shape[0]
        tcols = []
        BtoA = {}
        AtoB = {}
        levels = {}
        target_levels = {}
        coltypes = []
        colnames = []
        colimputes = []
        columns = []
        ii = 0
        for i in range(ncols):
            colname = df.columns[i]#error is here, tried making a columns[] but that didnt help.
            coldtype = df[colname].dtype
            tcoltype = None
            shall_ignore = False
            colimpute = None
            if colname == target:
                uvals = df[colname].unique()
                for j in range(len(uvals)):
                    target_levels[uvals[j]] = j
                shall_ignore = True
            elif colname in ignore:
                shall_ignore = True
            elif np.issubdtype(coldtype, np.integer) or np.issubdtype(coldtype, np.floating) or np.issubdtype(coldtype, np.bool_):
                coltype = "numeric"
                col = np.asarray(df[colname].values, np.float64)
                cols = [col]
                colimpute = col[~np.isnan(col)].min()-1
                BtoA[len(tcols)] = ii
                AtoB[ii] = [len(tcols)]
            elif np.issubdtype(coldtype, np.object_):
                uvals = df[colname].unique()
                if isinstance(uvals[0], (np.bool_, bool)):
                    coltype = "numeric"
                    colimpute = col[~np.isnan(col)].min()-1
                    AtoB[ii] = [len(tcols)]
                    BtoA[len(tcols)] = ii
                    cols = [np.asarray(df[colname].values, np.float64)]
                elif isinstance(uvals[0], (str,)):
                    coltype = "categorical"
                    vals = df[colname].values
                    cols = []
                    ilist = []
                    levelsd = {}
                    for j in range(len(uvals)):
                        k = len(tcols) + j
                        ilist.append(k)
                        level = uvals[j]
                        levelsd[level] = j
                        col = np.asarray(vals == level, dtype=np.float64)
                        cols.append(col)
                        BtoA[k] = ii
                    AtoB[ii] = ilist
                    levels[ii] = levelsd
                else:
                    raise TypeError("cannot handle object type %s for column %s" % (str(type(uvals[0])), colname))
            else:
                raise TypeError("cannot handle column dtype %s for column %s" % (coldtype, colname))
            if not shall_ignore:
                coltypes.append(coltype)
                colnames.append(colname)
                colimputes.append(colimpute)
                tcols = tcols + cols
                ii = ii + 1
        tdata = np.vstack(tcols).T.copy()
        self._AtoB = AtoB
        self._BtoA = BtoA
        self._coltypes = coltypes
        self._colnames = colnames
        self._colimputes = colimputes
        self._levels = levels
        self._target = target
        self._target_levels = target_levels
        
        return tdata, AtoB, BtoA, coltypes, colnames, target, target_levels, levels

    def transform(self, df, ignore_target=False):
        """
        Transforms a Pandas Data frame into a rectangularized NumPy array
        representing the data set and a NumPy array representing the label
        vector for input to scikit learn.
        """
        if self._colnames is None:
            raise ValueError("You must call fit() first")
        nrows, ncols = df.shape
        tcols = []
        for i in range(len(self._colnames)):
            colname = self._colnames[i] 
            if colname not in df.columns:
                raise ValueError("missing column %s not in data frame" % colname)
            col = df[colname]
            if self._coltypes[i] == "numeric":
                data = np.asarray(col, dtype=np.float64).reshape(nrows)
                data[np.isnan(data)] = self._colimputes[i]
                if np.isnan(data.min()):
                    pp
                data = data.reshape(nrows, 1)
                tcols = tcols + [data]
            elif self._coltypes[i] == "categorical":
                levels = self._levels[i]
                numericized = col.replace(levels).values
                onehot = np.zeros((nrows, len(levels)))
                for i in range(len(levels)):
                    onehot[numericized == i, i] = 1.
                tcols = tcols + [onehot]
        X = np.hstack(tcols)
        y = None
        if not ignore_target and self._target is not None:
            if self._target in df.columns:
                y = df[self._target].replace(self._target_levels).values
            else:
                raise ValueError("target %s missing from data frame: %s" % target)
        
        return X, y

    def transform_target(self, df):
        if self._colnames is None:
            raise ValueError("You must call fit() first")
        y = df[self._target].replace(self._target_levels).values
        return y

    def get_colname_for_feature(self, transformed_feature_index):
        if self._colnames is None:
            raise ValueError("You must call fit() first")
        return self._colnames[self._BtoA[transformed_feature_index]]

    def transform_feature_importances(self, sklearn_imps,df):
        if self._colnames is None:
            raise ValueError("You must call fit() first")
        imp = sklearn_imps["importances"]
        nrows, ncols = df.shape
        result = {}
        for i in range(len(self._colnames)):
            colname = self._colnames[i]
            if colname not in df.columns:
                raise ValueError("missing column %s not in data frame" % colname)
            result[colname] = imp[self._AtoB[i], :].sum(axis=0).mean()
        return pd.Series(result)

    def transform_predictions(self, y):
        if self._colnames is None:
            raise ValueError("You must call fit() first")
        result = pd.DataFrame({self._target: y})
        return result.replace(invert_levels(self._target_levels))

    def transform_back(self, X, y):
        results = []
        if self._colnames is None:
            raise ValueError("You must call fit() first")
        tcols = []
        for i in range(len(self._colnames)):
            colname = self._colnames[i]
            coltype = self._coltypes[i]
            if coltype == "numeric":
                data = X[:, self._AtoB[i]].ravel()
                result = pd.DataFrame({colname: data})
            elif coltype == "categorical":
                XX = X[:, self._AtoB[i]]
                data = XX.argmax(axis=1).ravel()
                result = pd.DataFrame({colname: data})
                result = result.replace(invert_levels(self._levels[i]))
            results.append(result)
        ty = self.transform_predictions(y)
        results.append(ty)
        return pd.concat(results, axis=1)
    
    def make_train_test_split(self, df, provide_index=False, n_splits=None, test_size=0.2, random_state=None):
        """
        Performs ``n_splits`` trials of two-way train, test partitioning of
        a data frame using stratified sampling. If ``n_splits`` is not provided,
        just 
        """
        from sklearn.model_selection import StratifiedShuffleSplit
        nrows, ncols = df.shape
        y = self.transform_target(df)
        # we don't need any features for label-based stratified sampling. So create dummy data here. The number of features is not relevant to generating indices, so 2 columns are generated.
        X = np.random.rand(nrows, 2)

        # If ``n_splits`` is not provided, assume that there is only one trial.
        if n_splits is None:
            sss = StratifiedShuffleSplit(n_splits=1, test_size=test_size, random_state=random_state)
            train_index, test_index = next(sss.split(X, y))
            if provide_index:
                return train_index, test_index
            else:
                return df.iloc[train_index], df.iloc[test_index]
        else:
            sss = StratifiedShuffleSplit(n_splits=n_splits, test_size=test_size, random_state=random_state)
            if provide_index:
                return [(train_index, test_index) for train_index, test_index in sss.split(X, y)]
            else:
                return ((df.iloc[train_index], df.iloc[test_index]) for train_index, test_index in sss.split(X, y))
