#create your individual project here!
from sklearn.preprocessing import LabelEncoder
from sklearn.neighbors import KNeighborsClassifier
import matplotlib.pyplot as plt
import seaborn as sns
import pandas as pd
df = pd.read_csv('train.csv')
print(df.info())
print(df.head())
print(df.isna().sum())
df.drop(['bdate','' 'education_form','langs', 'city','id', 'occupation_type', 'occupation_name', 'last_seen'], axis = 1, inplace = True)
print(df.info())
print(df)
def transform(career):
    if career == 'False':
        return '-'
    return career

df['career_start'] = df['career_start'].apply(transform)
df['career_end'] = df['career_end'].apply(transform)
categorical_cols = df.select_dtypes(include=['object']).columns
label_encoders ={}
for col in categorical_cols:
    label_encoders[col] = LabelEncoder()
    df[col] = label_encoders[col].fit_transform(df[col])
dorr = df.corr()
df_test = pd.read_csv("test.csv")
df_test.drop(['bdate','' 'education_form','langs', 'city','id', 'occupation_type', 'occupation_name', 'last_seen'], axis = 1, inplace = True)

categorical_cols = df_test.select_dtypes(include=['object']).columns
label_encoders ={}
for col in categorical_cols:
    label_encoders[col] = LabelEncoder()
    df_test[col] = label_encoders[col].fit_transform(df_test[col])

print(df_test.head())
plt.figure(figsize=(50,7))
sns.heatmap(dorr, annot = True, fmt='.2g')
x_train = df.drop('result', axis = 1)
y_train = df['result']
model_classifier = KNeighborsClassifier(n_neighbors = 5, metric = 'minkowski', p = 2)
model_classifier.fit(x_train, y_train)
y_pred = model_classifier.predict(df_test)
#print(df.head())
#print(df.info())
#plt.show()
print(y_pred)
df_test = df['result']
data = {'Prediction': y_pred[:1989], 'Real data':df_test[:1989]}
data_result = pd.DataFrame(data)
print(data_result)
data_result.to_csv('result_digitalEdu.csv',index = False)