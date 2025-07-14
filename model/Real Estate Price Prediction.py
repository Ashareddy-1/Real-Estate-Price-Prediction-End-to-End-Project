import pandas as pd
import numpy as np
import matplotlib
from matplotlib import pyplot as plt

# Load the dataset
df1 = pd.read_csv(r'C:\Users\SESA610197\Documents\AIMLLNG\ML\Real Estate Price Prediction E2E Project\bengaluru_house_prices.csv')
print(df1)

# Group by 'area_type' and count
area_type = df1.groupby('area_type')['area_type'].agg('count')
print(area_type)

# Drop unnecessary columns
df2 = df1.drop(['area_type', 'society', 'balcony', 'availability'], axis='columns')
print(df2)

# Check for null values
df3 = df2.isnull().sum()
print(df3)

# Drop rows with NaN values
df4 = df2.dropna()
print(df4.isnull().sum())

# Convert 'size' column to string type
df4['size'] = df4['size'].astype(str)

# Print unique values in 'size' column
unique_values = df4['size'].unique()
for value in unique_values:
    print(value)

# Apply the lambda function to extract the number of bedrooms
df4['bhk'] = df4['size'].apply(lambda x: int(x.split(' ')[0]))
print(df4)

print(df4['bhk'].unique())

print(df4[df4.bhk>20])

print(df4.total_sqft.unique())

def is_float(x):
    try:
        float(x)
    except:
        return False
    return True

print(df4[~df4['total_sqft'].apply(is_float)].head(10))

def convert_sqft_to_num(x):
    tokens = x.split('-')
    if len(tokens) == 2:
        return (float(tokens[0])+float(tokens[1]))/2
    try:
        return float(x)
    except:
        return None

print(convert_sqft_to_num('2166'))
print(convert_sqft_to_num('2100 - 2850'))

df5 = df4.copy()
df5['total_sqft'] = df5['total_sqft'].apply(convert_sqft_to_num)
print(df5)

print(df5.loc[30])

df6 = df5.copy()
df6['price_per_sqft'] = df6['price']*100000/df6['total_sqft']
print(df6)

print(df6.location.unique())
print(len(df6.location.unique()))
df6.location = df6.location.apply(lambda x : x.strip())
location_stats = df6.groupby('location')['location'].agg('count').sort_values(ascending = False)
print(location_stats)

print(len(location_stats[location_stats<=10]))

location_stats_less_than_10 = location_stats[location_stats<=10]
print(location_stats_less_than_10)

print(len(df6.location.unique()))

df6.location = df6.location.apply(lambda x : 'other' if x in location_stats_less_than_10 else x)
print(len(df6.location.unique()))

print(df6[df6.total_sqft/df6.bhk < 300])
print(df6.shape)

df7 = df6[~(df6.total_sqft/df6.bhk<300)]
print(df7.shape)

print(df7.price_per_sqft.describe())

def remove_pps_outliers(df):
    df_out = pd.DataFrame()
    for key, subdf in df.groupby('location'):
        m = np.mean(subdf.price_per_sqft)
        st = np.std(subdf.price_per_sqft)
        reduced_df = subdf[(subdf.price_per_sqft>(m-st))&(subdf.price_per_sqft<=(m+st))]
        df_out = pd.concat([df_out, reduced_df], ignore_index = True)
    return df_out
df8 = remove_pps_outliers(df7)
print(df8.shape)

def plot_scatter_chart(df,location):
    bhk2 = df[(df.location==location) & (df.bhk==2)]
    bhk3 = df[(df.location==location) & (df.bhk==3)]
    matplotlib.rcParams['figure.figsize'] = (15,10)
    plt.scatter(bhk2.total_sqft,bhk2.price,color='blue',label='2 BHK', s=50)
    plt.scatter(bhk3.total_sqft,bhk3.price,marker='+', color='green',label='3 BHK', s=50)
    plt.xlabel("Total Square Feet Area")
    plt.ylabel("Price (Lakh Indian Rupees)")
    plt.title(location)
    plt.legend()
    plt.show()
    
plot_scatter_chart(df8,"Rajaji Nagar")    
    
def remove_bhk_outliers(df):
    exclude_indices = np.array([])
    for location, location_df in df.groupby('location'):
        bhk_stats = {}
        for bhk, bhk_df in location_df.groupby('bhk'):
            bhk_stats[bhk] = {
                'mean': np.mean(bhk_df.price_per_sqft),
                'std': np.std(bhk_df.price_per_sqft),
                'count': bhk_df.shape[0]
            }
        for bhk, bhk_df in location_df.groupby('bhk'):
            stats = bhk_stats.get(bhk-1)
            if stats and stats['count']>5:
                exclude_indices = np.append(exclude_indices, bhk_df[bhk_df.price_per_sqft<(stats['mean'])].index.values)
    return df.drop(exclude_indices,axis='index')
df9 = remove_bhk_outliers(df8)
# df9 = df8.copy()
print(df9.shape)

plot_scatter_chart(df9,"Rajaji Nagar")

import matplotlib
matplotlib.rcParams["figure.figsize"] = (20,10)
plt.hist(df9.price_per_sqft,rwidth=0.8)
plt.xlabel("Price Per Square Feet")
plt.ylabel("Count")
plt.show()

print(df9.bath.unique())
print(df9[df9.bath>10])

plt.hist(df9.bath, rwidth=0.8)
plt.xlabel("Number of bathrooms")
plt.ylabel("Count")
plt.show()
print(df9[df9.bath>df9.bhk+2])

df10 = df9[df9.bath<df9.bhk+2]
print(df10.shape)

df11 = df10.drop(['size','price_per_sqft'],axis='columns')
print(df11)

dummies = pd.get_dummies(df11.location)
print(dummies)

df12 = pd.concat([df11,dummies.drop('other', axis = 'columns')], axis='columns')
print(df12)

df13 = df12.drop('location', axis='columns')
print(df13)

x = df13.drop('price', axis = 'columns')
print(x)

y = df13.price
print(y)

from sklearn.model_selection import train_test_split
x_train, x_test, y_train, y_test = train_test_split(x,y,test_size=0.2, random_state=10)

from sklearn.linear_model import LinearRegression
lr_clf = LinearRegression()
lr_clf.fit(x_train, y_train)
score=lr_clf.score(x_test, y_test)
print(score)

from sklearn.model_selection import ShuffleSplit
from sklearn.model_selection import cross_val_score

cv = ShuffleSplit(n_splits=5, test_size=0.2, random_state=0)
cross_score = cross_val_score(LinearRegression(), x, y, cv=cv)
print(cross_score)

import pandas as pd
from sklearn.model_selection import GridSearchCV, ShuffleSplit
from sklearn.linear_model import LinearRegression, Lasso
from sklearn.tree import DecisionTreeRegressor

def find_best_model_using_gridsearchcv(x, y):
    algos = {
        'linear_regression': {
            'model': LinearRegression(),
            'params': {
                # LinearRegression does not have hyperparameters that benefit from GridSearchCV in basic form
            }
        },
        'lasso': {
            'model': Lasso(),
            'params': {
                'alpha': [1, 2],
                'selection': ['random', 'cyclic']
            }
        },
        'decision_tree': {
            'model': DecisionTreeRegressor(),
            'params': {
                'criterion': ['squared_error', 'friedman_mse'],
                'splitter': ['best', 'random']
            }
        }
    }
    scores = []
    cv = ShuffleSplit(n_splits=5, test_size=0.2, random_state=0)

    for algo_name, config in algos.items():
        if config['params']:  # Only run GridSearchCV if there are parameters to tune
            gs = GridSearchCV(config['model'], config['params'], cv=cv, return_train_score=False)
        else:
            gs = GridSearchCV(config['model'], {}, cv=cv, return_train_score=False)
        gs.fit(x, y)
        scores.append({
            'model': algo_name,
            'best_score': gs.best_score_,
            'best_params': gs.best_params_
        })

    results_df = pd.DataFrame(scores, columns=['model', 'best_score', 'best_params'])

    # Print the best model and its parameters
    best_model = results_df.loc[results_df['best_score'].idxmax()]
    print(f"Best Model: {best_model['model']}")
    print(f"Best Score: {best_model['best_score']}")
    print(f"Best Parameters: {best_model['best_params']}")

    return results_df
find_best_model_using_gridsearchcv(x,y)

def predict_price(location, sqft, bath, bhk):
    # Create a zero vector with the same number of features as x
    x_input = np.zeros(len(x.columns))
    
    # Assign values to the first few features (assuming order: sqft, bath, bhk)
    x_input[0] = sqft
    x_input[1] = bath
    x_input[2] = bhk

    # Set the location index to 1 if it exists in the columns
    if location in x.columns:
        loc_index = np.where(x.columns == location)[0][0]
        x_input[loc_index] = 1

    # Predict using the trained model
    return lr_clf.predict([x_input])[0]


print(predict_price('1st Phase JP Nagar',1000, 2, 2))
print(predict_price('1st Phase JP Nagar',1000, 3, 3))
print(predict_price('Indira Nagar',1000, 2, 2))
print(predict_price('Indira Nagar',1000, 3, 3))

import pickle
with open('banglore_home_prices_model.pickle', 'wb') as f:
    pickle.dump(lr_clf, f)

import json
columns = {
    'data_columns' : [col.lower() for col in x.columns]
}
with open("columns.json","w") as f:
    f.write(json.dumps(columns))






            







                  
