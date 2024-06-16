
import pandas as pd
import numpy as np
from sklearn.preprocessing import StandardScaler
from sklearn.linear_model import LinearRegression
from sklearn.metrics import  mean_squared_error, r2_score 
from sklearn.neighbors import KNeighborsRegressor
from sklearn.preprocessing import RobustScaler
from sklearn.ensemble import RandomForestRegressor
from sklearn.pipeline import make_pipeline, Pipeline
from sklearn.preprocessing import MinMaxScaler

def wmae_test(test, pred,weights): # WMAE for test 
   
    error = np.sum(weights * np.abs(test - pred), axis=0) / np.sum(weights)
    return error


df = pd.read_csv('./walmart_cleaned.csv')

#144mod45 + 1 = 10
df_10 = df[df.loc[:,'Store']==10]

#copy of the original df
df_10_copy = df_10.copy()


df_10_copy['Date']= pd.to_datetime(df_10_copy['Date'],dayfirst=True)
df_10_copy['year']=df_10_copy.Date.dt.year
df_10_copy['weekday'] = df_10_copy.Date.dt.weekday
df_10_copy['month'] = df_10_copy.Date.dt.month

#drop unnecessary columns

columns = ["Unnamed: 0","Store","MarkDown1","MarkDown2","MarkDown3","MarkDown4","MarkDown5","Type","Size","Next week"]
df_10_copy.drop(columns=columns,inplace=True)


#3 Departments picked randomly:  10, 20, 30
#Department 10 from store 10
dept_10=df_10_copy[df_10_copy['Dept']==10]
dept_10=dept_10.drop(columns=["Dept"])
dept_10=dept_10.reset_index(drop=True)

train_set_10=dept_10[(dept_10['year']==2010) | (dept_10['year']==2012) ]
x_train_10 = train_set_10.drop(columns=['year','Weekly_Sales','Date'])
y_train_10 = train_set_10['Weekly_Sales']

test_set_10=dept_10[dept_10['year']==2011]
x_test_10 = test_set_10.drop(columns=['year','Weekly_Sales','Date'])
y_test_10 = test_set_10['Weekly_Sales']


#feature scaling 
std = StandardScaler()
x_train_10_std = std.fit_transform(x_train_10)
x_train_10_std = pd.DataFrame(x_train_10_std , columns = x_train_10.columns)

x_test_10_std = std.transform(x_test_10)
x_test_10_std = pd.DataFrame(x_test_10_std,columns=x_test_10.columns)

MLR = LinearRegression().fit(x_train_10_std,y_train_10)
y_predicted_test = MLR.predict(x_test_10_std)

print('Department 10 model performance:')
print('Test set')
print("Mean Squared Error:%.2f"%mean_squared_error(y_test_10, y_predicted_test))
print("R2:%.2f"% r2_score(y_test_10, y_predicted_test))

#calculating next week
dept_10_copy  = dept_10.copy()
dept_10_copy.drop(columns=['year','Weekly_Sales','Date'],inplace=True)

dept_10_copy_x_std = std.transform(dept_10_copy)
dept_10_copy_x_std = pd.DataFrame(dept_10_copy_x_std,columns=dept_10_copy.columns)

dept_10_copy_y = MLR.predict(dept_10_copy_x_std)
dept_10_copy_y = pd.DataFrame(dept_10_copy_y)

dept_10_copy_y.drop(index=df.index[0],axis=0,inplace=True)
dept_10_copy_y= dept_10_copy_y.reset_index(drop=True)
dept_10_copy_y.loc[len(dept_10_copy_y.index)]=0
dept_10['Next Week'] = dept_10_copy_y
dept_10.to_csv(path_or_buf='./results/store_10_dept_10.csv')


#Department 20 from store 10
dept_20=df_10_copy[df_10_copy.loc[:,'Dept']==20]
dept_20=dept_20.drop(columns=["Dept"])
dept_20=dept_20.reset_index(drop=True)

train_set_20=dept_20[(dept_20['year']==2010) | (dept_20['year']==2012) ]
x_train_20 = train_set_20.drop(columns=['year','Weekly_Sales','Date'])
y_train_20 = train_set_20['Weekly_Sales']

test_set_20=dept_20[dept_20['year']==2011]
x_test_20 = test_set_20.drop(columns=['year','Weekly_Sales','Date'])
y_test_20 = test_set_20['Weekly_Sales']

RFR = RandomForestRegressor(n_estimators=50, random_state=42, n_jobs=-1, max_depth=35,max_features = 'sqrt',min_samples_split = 10)
scaler = RobustScaler()
pipe = make_pipeline(scaler,RFR)

pipe.fit(x_train_20, y_train_20 )
y_predicted_test = pipe.predict(x_test_20)

weights = x_test_20['IsHoliday'].apply(lambda is_holiday:5 if is_holiday else 1)


print('Department 20 model performance:')
print('Test set')
print('WMAE: %.2f'% wmae_test(y_test_20, y_predicted_test,weights))

#calculating next week
dept_20_copy  = dept_20.copy()
dept_20_copy.drop(columns=['year','Weekly_Sales','Date'],inplace=True)

dept_20_copy_y = pipe.predict(dept_20_copy)
dept_20_copy_y = pd.DataFrame(dept_20_copy_y)

dept_20_copy_y.drop(index=df.index[0],axis=0,inplace=True)
dept_20_copy_y= dept_20_copy_y.reset_index(drop=True)
dept_20_copy_y.loc[len(dept_20_copy_y.index)]=0
dept_20['Next Week'] = dept_20_copy_y
dept_20.to_csv(path_or_buf='./results/store_10_dept_20.csv')

#Department 30 from store 10
dept_30=df_10_copy[df_10_copy.loc[:,'Dept']==30]
dept_30=dept_30.drop(columns=["Dept"])
dept_30=dept_30.reset_index(drop=True)


train_set_30=dept_30[(dept_30['year']==2010) | (dept_30['year']==2012) ]
x_train_30 = train_set_30.drop(columns=['year','Weekly_Sales','Date'])
y_train_30 = train_set_30['Weekly_Sales']

test_set_30=dept_30[dept_30['year']==2011]
x_test_30 = test_set_30.drop(columns=['year','Weekly_Sales','Date'])
y_test_30 = test_set_30['Weekly_Sales']

KNN = KNeighborsRegressor()
scaler = MinMaxScaler()
pipe = make_pipeline(scaler,KNN)
pipe.fit(x_train_30,y_train_30)
y_predicted_test = pipe.predict(x_test_30)


weights = x_test_30['IsHoliday'].apply(lambda is_holiday:5 if is_holiday else 1)

print('Department 30 model performance:')
print('Test set')
print('WMAE: %.2f'% wmae_test(y_test_30, y_predicted_test,weights))

#calculating next week
dept_30_copy  = dept_30.copy()
dept_30_copy.drop(columns=['year','Weekly_Sales','Date'],inplace=True)

dept_30_copy_y = pipe.predict(dept_30_copy)
dept_30_copy_y = pd.DataFrame(dept_30_copy_y)

dept_30_copy_y.drop(index=df.index[0],axis=0,inplace=True)
dept_30_copy_y= dept_30_copy_y.reset_index(drop=True)
dept_30_copy_y.loc[len(dept_30_copy_y.index)]=0
dept_30['Next Week'] = dept_30_copy_y
dept_30.to_csv(path_or_buf='./results/store_10_dept_30.csv')