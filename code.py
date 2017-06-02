import pandas as pd
import numpy as np
import matplotlib as mplb

os.chdir('/home/kishlay/Documents/Data Analytics Practice Problems/Titanic')

#Load the data and combine
df1 = pd.read_csv('train.csv')
df2 = pd.read_csv('test.csv')
df1['source'] = 'train'
df2['source'] = 'test'
df3 = pd.concat([df1,df2])


############################## Exploration of the data #####################################

#count the null values in each column
print("\nNo of empty values in columns is:")
print(df3.apply(lambda x:sum(x.isnull())))


#count the unique values in columns
print("\nNo of unique values in columns is:")
print(df3.apply(lambda x: len(x.unique())))


#filter categorical columns
categorical_columns = [x for x in df3.dtypes.index if df3.dtypes[x] == 'object']
categorical_columns = [x for x in categorical_columns if x not in ['Name','source','Ticket','Cabin']]
for col in categorical_columns:
    print('\nCount of values in Categorical column %s'%col)
    print(df3[col].value_counts())

################################### Data Cleaning ###########################################

#delete cabin as it is almost empty
del df3['Cabin']


#Fill empty fare data with mean price for its 'Pclass'
miss_data = df3['Fare'].isnull()
mean_fare = df3.pivot_table(index='Pclass',values='Fare')
print('\nMissing Fare data:%d'%sum(miss_data))
df3.loc[miss_data,'Fare'] = mean_fare.loc[df3.loc[miss_data,'Pclass']].iloc[0].iloc[0]
print('Missing Fare data:%d'%sum(df3['Fare'].isnull()))


#Fill Embarked with the mode of embarked
miss_data = df3['Embarked'].isnull()
print('\nMissing Embarked data before:%d'%sum(miss_data))
df3.loc[miss_data,'Embarked'] = df3['Embarked'].mode().iloc[0]
print('Missing Embarked data after:%d'%sum(df3['Embarked'].isnull()))


#Fill age with mean of same group [Sex,SibSp,Parch]
miss_data = df3['Age'].isnull()
print('\nMissing Age data before:%d'%sum(miss_data))
mean_group_age = df3.pivot_table(index=['Sex','SibSp','Parch'],values='Age')
df3.loc[miss_data,'Age'] = df3[miss_data].T.apply(lambda x:mean_group_age.loc[x['Sex'],x['SibSp'],x['Parch']]).T['Age']
print('Missing Age data after 1st attempt:%d'%sum(df3['Age'].isnull()))
miss_data = df3['Age'].isnull()
df3.loc[miss_data,'Age'] = np.mean(df3.Age)
print('Missing Age data after 2nd attempt:%d'%sum(df3['Age'].isnull()))


################################### Data Transformation ########################################


#Extracting titles from Names
A = df3['Name'].str.split('(.*, )|(\\..*)').tolist()
df3['Name'] = pd.Series([x[3] for x in A])
print("\nUnique Titles Before:\n",df3['Name'].value_counts())

# miss women
miss_women = df3['Name'].isin(['Ms','Mlle'])
df3.loc[miss_women,'Name'] = 'Miss'
# mrs women
df3.loc[df3['Name'] == 'Mme','Name'] = 'Mrs'

# officer
officers = df3['Name'].isin(['Capt', 'Col', 'Don', 'Dr', 'Major', 'Rev'])
df3.loc[officers,'Name'] = 'officer'

# royalty
royalty = df3['Name'].isin(['Dona', 'Lady', 'the Countess','Sir', 'Jonkheer'])
df3.loc[royalty,'Name'] = 'royalty'

print("\nUnique Titles After:\n",df3['Name'].value_counts())



#Transforming Categorical Variables
from sklearn.preprocessing import LabelEncoder
var_mod = ['Sex','Name']
le = LabelEncoder()
for i in var_mod:
    df3[i] = le.fit_transform(df3[i])


#One Hot Coding:
df3 = pd.get_dummies(df3, columns=['Sex','Embarked','Name'])
df3.drop(['Ticket'],axis=1,inplace=True)

#Store results to csv files
train_processed = df3[df3['source'] == 'train']
train_processed.to_csv('train_processed.csv',index=False)
test_processed = df3[df3['source'] == 'test']
test_processed.to_csv('test_processed.csv',index=False)
