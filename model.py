from sklearn import model_selection,metrics
import pandas as pd


#Change Directory
os.chdir('/home/kishlay/Documents/Data Analytics Practice Problems/Titanic')


#Setting up data and target,predictors and ID columns
target = 'Survived'
IDcols = ['PassengerId']
train = pd.read_csv('train_processed.csv')
test = pd.read_csv('test_processed.csv')
predictors = [x for x in train.columns if x not in [target]+IDcols]

#Generic modeling function
def model_fun(alg,train,test,predictors,y_col,IDcols,fname):
    
    #Fit the model
    alg.fit(train[predictors],train[y_col])
    
    
    #Perform cross validation
    ypred = model_selection.cross_val_predict(alg,train[predictors],train[y_col],cv=20)
    print(metrics.accuracy_score(train[y_col],y_pred=ypred))
    print(metrics.classification_report(train[y_col],y_pred=ypred))
    
    #Predict the values
    test[y_col] = alg.predict(test[predictors])
    test[y_col] = test[y_col].astype(int)
    
    #Store the results
    IDcols.append(y_col)
    submission = pd.DataFrame({ x: test[x] for x in IDcols})
    submission.to_csv(fname, index=False)
    
    
#Model 1
from sklearn.linear_model import LogisticRegression
IDcols = ['PassengerId']
model_fun(LogisticRegression(),train,test,predictors,target,IDcols,'logreg.csv')


#Model 2
from sklearn.ensemble import RandomForestClassifier
IDcols = ['PassengerId']
model_fun(RandomForestClassifier(),train=train,test=test,predictors=predictors,y_col=target,IDcols=IDcols,fname='randomforest.csv')


#Model 3
from sklearn.tree import DecisionTreeClassifier
IDcols = ['PassengerId']
model_fun(DecisionTreeClassifier(),train=train,test=test,predictors=predictors,y_col=target,IDcols=IDcols,fname='decisiontree.csv')


#Model 4
from sklearn.neural_network import MLPClassifier
IDcols = ['PassengerId']
nn_model = MLPClassifier(solver='lbfgs',hidden_layer_sizes=(6, 3) ,alpha=1e-3,random_state=1)
model_fun(nn_model,train=train,test=test,predictors=predictors,y_col=target,IDcols=IDcols,fname='nnmodel.csv')


# Model 5
from xgboost import XGBClassifier
IDcols = ['PassengerId']
X_train, X_test, y_train, y_test = model_selection.train_test_split(train[predictors], train[target], test_size=0.25, random_state=7)
xboost = XGBClassifier()
xboost.fit(X_train,y_train)
ypred = xboost.predict(X_test)
ypred = ypred.astype(int)
y_test = y_test.astype(int)
print(metrics.accuracy_score(y_test,ypred))
test[target] = xboost.predict(test[predictors])
test[target] = test[target].astype(int)
IDcols.append(target)
results = pd.DataFrame({x:test[x] for x in IDcols})