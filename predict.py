def forwardselection(X, y):
    """Forward variable selection based on the Scikit learn API
    
    
    Output:
    ----------------------------------------------------------------------------------
    Scikit learn OLS regression object for the best model
    """

    # Functions
    from sklearn.linear_model import LinearRegression
    from sklearn.model_selection import cross_val_score

    # Initialisation
    base = []
    p = X.shape[1]
    candidates = list(np.arange(p))

    # Forward recursion
    i=1
    bestcvscore=-np.inf    
    while i<=p:
        bestscore = 0
        for variable in candidates:
            ols = LinearRegression()
            ols.fit(X.iloc[:, base + [variable]], y)
            score = ols.score(X.iloc[:, base + [variable]], y)
            if score > bestscore:
                bestscore = score 
                best = ols
                newvariable=variable
        base.append(newvariable)
        candidates.remove(newvariable)
        
        cvscore = cross_val_score(best, X.iloc[:, base], y, scoring='neg_mean_squared_error').mean() 
        
        if cvscore > bestcvscore:
            bestcvscore=cvscore
            bestcv = best
            subset = base[:]
        i+=1
    
    #Finalise
    return bestcv, subset


class forward:
    def __init__(self):
        pass

    def fit(self, X, y):
        self.ols, self.subset = forwardselection(X, y)

    def predict(self, X):
        return self.ols.predict(X.iloc[:, self.subset])

    def cv_score(self, X, y, cv=10):
        from sklearn.model_selection import cross_val_score
        scores = cross_val_score(self.ols, X.iloc[:, self.subset], np.ravel(y), cv=cv, scoring='neg_mean_squared_error')
        return np.sqrt(-1*np.mean(scores))

#Split data intro training and test datasets
from sklearn.model_selection import train_test_split
import numpy as np
import pandas as pd
import warnings
warnings.filterwarnings('ignore')
import matplotlib.pyplot as plt

#change train_df to train_df_out to apply removing outliers feature

def makeModel(df):
    X_train, X_test = df[0:-80], df[-80:]

    Y_train = X_train['price']
    Y_test = X_test['price']
    X_train.drop('price',axis=1,inplace=True)
    X_train.drop('datetime',axis=1,inplace=True)
    X_train.drop('date',axis=1,inplace=True)

    X_test.drop('price',axis=1,inplace=True)
    X_test.drop('datetime',axis=1,inplace=True)
    X_test.drop('date',axis=1,inplace=True)
    np.random.seed(0)

    fwd = forward()
    fwd.fit(X_train, Y_train)
    y_pred = fwd.predict(X_test)
    pred = pd.Series(y_pred)
    pred.index +=274
    return pred

def predPlot(df,pred):
    fig = plt.figure(figsize=(10, 4))
    df = df.set_index('date')
    ax = df['price'].plot(label='truth')
    pred.plot(linestyle='--', color='#ff7823', ax=ax, label='forecast', alpha=.7, figsize=(14, 7))

    ax.set_xlabel('date')
    ax.set_ylabel('harga')
    plt.legend()
    return fig
