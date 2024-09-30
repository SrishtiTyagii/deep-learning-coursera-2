# Cox Proportional Hazards and Random Survival Forests

Welcome to the final assignment in Course 2! In this assignment you'll develop risk models using survival data and a combination of linear and non-linear techniques. We'll be using a dataset with survival data of patients with Primary Biliary Cirrhosis (pbc). PBC is a progressive disease of the liver caused by a buildup of bile within the liver (cholestasis) that results in damage to the small bile ducts that drain bile from the liver. Our goal will be to understand the effects of different factors on the survival times of the patients. Along the way you'll learn about the following topics: 

- Cox Proportional Hazards
  - Data Preprocessing for Cox Models.
- Random Survival Forests
  - Permutation Methods for Interpretation.

## Table of Contents

- [1. Import Packages](#1)
- [2. Load the Dataset](#2)
- [3. Explore the Dataset](#3)
- [4. Cox Proportional Hazards](#4)
    - [Exercise 1 - to_one_hot](#ex-1)
- [5. Fitting and Interpreting a Cox Model](#5)
- [6. Hazard ratio](#3)
    - [Exercise 2 - hazard_ratio](#ex-2)
- [7. Harrell's C-Index](#7)
    - [Exercise 3 - harrell_c](#ex-3)
- [8. Random Survival Forests](#8)
- [9. Permutation Method for Interpretation](#9)

<a name='1'></a>
## 1. Import Packages

We'll first import all the packages that we need for this assignment. 

- `sklearn` is one of the most popular machine learning libraries.
- `numpy` is the fundamental package for scientific computing in python.
- `pandas` is what we'll use to manipulate our data.
- `matplotlib` is a plotting library.
- `lifelines` is an open-source survival analysis library.


```python
import sklearn
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt

from lifelines import CoxPHFitter
from lifelines.utils import concordance_index as cindex
from sklearn.model_selection import train_test_split

from util import load_data

from public_tests import *
```

<a name='2'></a>
## 2. Load the Dataset

Run the next cell to load the data.


```python
df = load_data('data/pbc.csv')
```

<a name='3'></a>
## 3. Explore the Dataset

In the lecture videos `time` was in months, however in this assignment, `time` will be converted into years. Also notice that we have assigned a numeric value to `sex`, where `female = 0` and `male = 1`.

Next, familiarize yourself with the data and the shape of it. 


```python
print(df.shape)

# df.head() only outputs the top few rows
df.head()
```

    (258, 19)





<div>
<style scoped>
    .dataframe tbody tr th:only-of-type {
        vertical-align: middle;
    }

    .dataframe tbody tr th {
        vertical-align: top;
    }

    .dataframe thead th {
        text-align: right;
    }
</style>
<table border="1" class="dataframe">
  <thead>
    <tr style="text-align: right;">
      <th></th>
      <th>time</th>
      <th>status</th>
      <th>trt</th>
      <th>age</th>
      <th>sex</th>
      <th>ascites</th>
      <th>hepato</th>
      <th>spiders</th>
      <th>edema</th>
      <th>bili</th>
      <th>chol</th>
      <th>albumin</th>
      <th>copper</th>
      <th>alk.phos</th>
      <th>ast</th>
      <th>trig</th>
      <th>platelet</th>
      <th>protime</th>
      <th>stage</th>
    </tr>
  </thead>
  <tbody>
    <tr>
      <th>0</th>
      <td>1.095890</td>
      <td>1.0</td>
      <td>0.0</td>
      <td>58.765229</td>
      <td>0.0</td>
      <td>1.0</td>
      <td>1.0</td>
      <td>1.0</td>
      <td>1.0</td>
      <td>14.5</td>
      <td>261.0</td>
      <td>2.60</td>
      <td>156.0</td>
      <td>1718.0</td>
      <td>137.95</td>
      <td>172.0</td>
      <td>190.0</td>
      <td>12.2</td>
      <td>4.0</td>
    </tr>
    <tr>
      <th>1</th>
      <td>12.328767</td>
      <td>0.0</td>
      <td>0.0</td>
      <td>56.446270</td>
      <td>0.0</td>
      <td>0.0</td>
      <td>1.0</td>
      <td>1.0</td>
      <td>0.0</td>
      <td>1.1</td>
      <td>302.0</td>
      <td>4.14</td>
      <td>54.0</td>
      <td>7394.8</td>
      <td>113.52</td>
      <td>88.0</td>
      <td>221.0</td>
      <td>10.6</td>
      <td>3.0</td>
    </tr>
    <tr>
      <th>2</th>
      <td>2.772603</td>
      <td>1.0</td>
      <td>0.0</td>
      <td>70.072553</td>
      <td>1.0</td>
      <td>0.0</td>
      <td>0.0</td>
      <td>0.0</td>
      <td>0.5</td>
      <td>1.4</td>
      <td>176.0</td>
      <td>3.48</td>
      <td>210.0</td>
      <td>516.0</td>
      <td>96.10</td>
      <td>55.0</td>
      <td>151.0</td>
      <td>12.0</td>
      <td>4.0</td>
    </tr>
    <tr>
      <th>3</th>
      <td>5.273973</td>
      <td>1.0</td>
      <td>0.0</td>
      <td>54.740589</td>
      <td>0.0</td>
      <td>0.0</td>
      <td>1.0</td>
      <td>1.0</td>
      <td>0.5</td>
      <td>1.8</td>
      <td>244.0</td>
      <td>2.54</td>
      <td>64.0</td>
      <td>6121.8</td>
      <td>60.63</td>
      <td>92.0</td>
      <td>183.0</td>
      <td>10.3</td>
      <td>4.0</td>
    </tr>
    <tr>
      <th>6</th>
      <td>5.019178</td>
      <td>0.0</td>
      <td>1.0</td>
      <td>55.534565</td>
      <td>0.0</td>
      <td>0.0</td>
      <td>1.0</td>
      <td>0.0</td>
      <td>0.0</td>
      <td>1.0</td>
      <td>322.0</td>
      <td>4.09</td>
      <td>52.0</td>
      <td>824.0</td>
      <td>60.45</td>
      <td>213.0</td>
      <td>204.0</td>
      <td>9.7</td>
      <td>3.0</td>
    </tr>
  </tbody>
</table>
</div>



Take a minute to examine particular cases.


```python
i = 20
df.iloc[i, :]
```




    time          11.175342
    status         1.000000
    trt            0.000000
    age           44.520192
    sex            1.000000
    ascites        0.000000
    hepato         1.000000
    spiders        0.000000
    edema          0.000000
    bili           2.100000
    chol         456.000000
    albumin        4.000000
    copper       124.000000
    alk.phos    5719.000000
    ast          221.880000
    trig         230.000000
    platelet      70.000000
    protime        9.900000
    stage          2.000000
    Name: 23, dtype: float64



Now, split your dataset into train, validation and test set using 60/20/20 split. 


```python
np.random.seed(0)
df_dev, df_test = train_test_split(df, test_size = 0.2)
df_train, df_val = train_test_split(df_dev, test_size = 0.25)

print("Total number of patients:", df.shape[0])
print("Total number of patients in training set:", df_train.shape[0])
print("Total number of patients in validation set:", df_val.shape[0])
print("Total number of patients in test set:", df_test.shape[0])
```

    Total number of patients: 258
    Total number of patients in training set: 154
    Total number of patients in validation set: 52
    Total number of patients in test set: 52


Before proceeding to modeling, let's normalize the continuous covariates to make sure they're on the same scale. Again, we should normalize the test data using statistics from the train data.


```python
continuous_columns = ['age', 'bili', 'chol', 'albumin', 'copper', 'alk.phos', 'ast', 'trig', 'platelet', 'protime']
mean = df_train.loc[:, continuous_columns].mean()
std = df_train.loc[:, continuous_columns].std()
df_train.loc[:, continuous_columns] = (df_train.loc[:, continuous_columns] - mean) / std
df_val.loc[:, continuous_columns] = (df_val.loc[:, continuous_columns] - mean) / std
df_test.loc[:, continuous_columns] = (df_test.loc[:, continuous_columns] - mean) / std
```

Let's check the summary statistics on our training dataset to make sure it's standardized.


```python
df_train.loc[:, continuous_columns].describe()
```




<div>
<style scoped>
    .dataframe tbody tr th:only-of-type {
        vertical-align: middle;
    }

    .dataframe tbody tr th {
        vertical-align: top;
    }

    .dataframe thead th {
        text-align: right;
    }
</style>
<table border="1" class="dataframe">
  <thead>
    <tr style="text-align: right;">
      <th></th>
      <th>age</th>
      <th>bili</th>
      <th>chol</th>
      <th>albumin</th>
      <th>copper</th>
      <th>alk.phos</th>
      <th>ast</th>
      <th>trig</th>
      <th>platelet</th>
      <th>protime</th>
    </tr>
  </thead>
  <tbody>
    <tr>
      <th>count</th>
      <td>1.540000e+02</td>
      <td>1.540000e+02</td>
      <td>1.540000e+02</td>
      <td>1.540000e+02</td>
      <td>1.540000e+02</td>
      <td>1.540000e+02</td>
      <td>1.540000e+02</td>
      <td>1.540000e+02</td>
      <td>1.540000e+02</td>
      <td>1.540000e+02</td>
    </tr>
    <tr>
      <th>mean</th>
      <td>9.833404e-16</td>
      <td>-3.258577e-16</td>
      <td>1.153478e-16</td>
      <td>1.153478e-16</td>
      <td>5.767392e-18</td>
      <td>1.326500e-16</td>
      <td>-1.263059e-15</td>
      <td>8.074349e-17</td>
      <td>2.018587e-17</td>
      <td>1.291896e-14</td>
    </tr>
    <tr>
      <th>std</th>
      <td>1.000000e+00</td>
      <td>1.000000e+00</td>
      <td>1.000000e+00</td>
      <td>1.000000e+00</td>
      <td>1.000000e+00</td>
      <td>1.000000e+00</td>
      <td>1.000000e+00</td>
      <td>1.000000e+00</td>
      <td>1.000000e+00</td>
      <td>1.000000e+00</td>
    </tr>
    <tr>
      <th>min</th>
      <td>-2.304107e+00</td>
      <td>-5.735172e-01</td>
      <td>-1.115330e+00</td>
      <td>-3.738104e+00</td>
      <td>-9.856552e-01</td>
      <td>-7.882167e-01</td>
      <td>-1.489281e+00</td>
      <td>-1.226674e+00</td>
      <td>-2.058899e+00</td>
      <td>-1.735556e+00</td>
    </tr>
    <tr>
      <th>25%</th>
      <td>-6.535035e-01</td>
      <td>-4.895812e-01</td>
      <td>-5.186963e-01</td>
      <td>-5.697976e-01</td>
      <td>-6.470611e-01</td>
      <td>-5.186471e-01</td>
      <td>-8.353982e-01</td>
      <td>-6.884514e-01</td>
      <td>-6.399831e-01</td>
      <td>-7.382590e-01</td>
    </tr>
    <tr>
      <th>50%</th>
      <td>-6.443852e-03</td>
      <td>-3.846612e-01</td>
      <td>-2.576693e-01</td>
      <td>5.663556e-02</td>
      <td>-3.140636e-01</td>
      <td>-3.416086e-01</td>
      <td>-2.260984e-01</td>
      <td>-2.495932e-01</td>
      <td>-4.100373e-02</td>
      <td>-1.398807e-01</td>
    </tr>
    <tr>
      <th>75%</th>
      <td>5.724289e-01</td>
      <td>2.977275e-02</td>
      <td>1.798617e-01</td>
      <td>6.890921e-01</td>
      <td>3.435366e-01</td>
      <td>-4.620597e-03</td>
      <td>6.061159e-01</td>
      <td>3.755727e-01</td>
      <td>6.617988e-01</td>
      <td>3.587680e-01</td>
    </tr>
    <tr>
      <th>max</th>
      <td>2.654276e+00</td>
      <td>5.239050e+00</td>
      <td>6.243146e+00</td>
      <td>2.140730e+00</td>
      <td>5.495204e+00</td>
      <td>4.869263e+00</td>
      <td>3.058176e+00</td>
      <td>5.165751e+00</td>
      <td>3.190823e+00</td>
      <td>4.447687e+00</td>
    </tr>
  </tbody>
</table>
</div>



<a name='4'></a>
## 4. Cox Proportional Hazards

Our goal is to build a risk score using the survival data that we have. We'll begin by fitting a Cox Proportional Hazards model to your data.

Recall that the Cox Proportional Hazards model describes the hazard for an individual $i$ at time $t$ as 

$$
\lambda(t, x) = \lambda_0(t)e^{\theta^T X_i}
$$

The $\lambda_0$ term is a baseline hazard and incorporates the risk over time, and the other term incorporates the risk due to the individual's covariates. After fitting the model, we can rank individuals using the person-dependent risk term $e^{\theta^T X_i}$. 

Categorical variables cannot be used in a regression model as they are. In order to use them, conversion to a series of variables is required.

Since our data has a mix of categorical (`stage`) and continuous (`wblc`) variables, before we proceed further we need to do some data engineering. To tackle the issue at hand we'll be using the `Dummy Coding` technique. In order to use Cox Proportional Hazards, we will have to turn the categorical data into one hot features so that we can fit our Cox model. Luckily, Pandas has a built-in function called `get_dummies` that will make it easier for us to implement our function. It turns categorical features into multiple binary features.

<img src="images/1-hot-encode.png" style="padding-top: 5px;width: 60%;left: 0px;margin-left: 150px;margin-right: 0px;">




<a name='ex-1'></a>
### Exercise 1 - to_one_hot
In the cell below, implement the `to_one_hot(...)` function.

<details>    
<summary>
    <font size="3" color="darkgreen"><b>Hints</b></font>
</summary>
<p>
<ul>
    <li>Remember to drop the first dummy for each each category to avoid convergence issues when fitting the proportional hazards model.</li>
    <li> Check out the <a href="https://pandas.pydata.org/pandas-docs/stable/reference/api/pandas.get_dummies.html" > get_dummies() </a>  documentation. </li>
    <li>Use <code>dtype=np.float64</code>.</li>
</ul>
</p>


```python
# UNQ_C1 (UNIQUE CELL IDENTIFIER, DO NOT EDIT)
def to_one_hot(dataframe, columns):
    '''
    Convert columns in dataframe to one-hot encoding.
    Args:
        dataframe (dataframe): pandas dataframe containing covariates
        columns (list of strings): list categorical column names to one hot encode
    Returns:
        one_hot_df (dataframe): dataframe with categorical columns encoded
                            as binary variables
    '''
    
    ### START CODE HERE (REPLACE INSTANCES OF 'None' with your code) ###
    
    one_hot_df = pd.get_dummies(dataframe,columns=columns, drop_first = True, dtype=np.float64)
    
    ### END CODE HERE ###
    
    return one_hot_df
```


```python
# test cell ex1 - do not modify this test cell
to_one_hot_test(to_one_hot, df_train, df_val, df_test)
```

    One hot val columns:
    
     ['time', 'status', 'trt', 'age', 'sex', 'ascites', 'hepato', 'edema', 'bili', 'chol', 'albumin', 'copper', 'alk.phos', 'ast', 'trig', 'platelet', 'protime', 'spiders_1.0', 'stage_2.0', 'stage_3.0', 'stage_4.0'] 
    
    There are 21 columns
    
    [92m All tests passed.


#### Expected Output:
```
One hot val columns:

 ['time', 'status', 'trt', 'age', 'sex', 'ascites', 'hepato', 'edema', 'bili', 'chol', 'albumin', 'copper', 'alk.phos', 'ast', 'trig', 'platelet', 'protime', 'spiders_1.0', 'stage_2.0', 'stage_3.0', 'stage_4.0'] 

There are 21 columns

 All tests passed.
```

Now you'll use the function you coded to transform the training, validation, and test sets.


```python
to_encode = ['edema', 'stage']

one_hot_train = to_one_hot(df_train, to_encode)
one_hot_val = to_one_hot(df_val, to_encode)
one_hot_test = to_one_hot(df_test, to_encode)
```

### Look for new features
Now, let's take a peek at one of the transformed data sets. Do you notice any new features?


```python
print(one_hot_train.shape)
one_hot_train.head()
```

    (154, 22)





<div>
<style scoped>
    .dataframe tbody tr th:only-of-type {
        vertical-align: middle;
    }

    .dataframe tbody tr th {
        vertical-align: top;
    }

    .dataframe thead th {
        text-align: right;
    }
</style>
<table border="1" class="dataframe">
  <thead>
    <tr style="text-align: right;">
      <th></th>
      <th>time</th>
      <th>status</th>
      <th>trt</th>
      <th>age</th>
      <th>sex</th>
      <th>ascites</th>
      <th>hepato</th>
      <th>spiders</th>
      <th>bili</th>
      <th>chol</th>
      <th>...</th>
      <th>alk.phos</th>
      <th>ast</th>
      <th>trig</th>
      <th>platelet</th>
      <th>protime</th>
      <th>edema_0.5</th>
      <th>edema_1.0</th>
      <th>stage_2.0</th>
      <th>stage_3.0</th>
      <th>stage_4.0</th>
    </tr>
  </thead>
  <tbody>
    <tr>
      <th>279</th>
      <td>3.868493</td>
      <td>0.0</td>
      <td>0.0</td>
      <td>-0.414654</td>
      <td>0.0</td>
      <td>0.0</td>
      <td>0.0</td>
      <td>0.0</td>
      <td>-0.300725</td>
      <td>-0.096081</td>
      <td>...</td>
      <td>0.167937</td>
      <td>0.401418</td>
      <td>0.330031</td>
      <td>0.219885</td>
      <td>-1.137178</td>
      <td>0.0</td>
      <td>0.0</td>
      <td>0.0</td>
      <td>1.0</td>
      <td>0.0</td>
    </tr>
    <tr>
      <th>137</th>
      <td>3.553425</td>
      <td>1.0</td>
      <td>0.0</td>
      <td>0.069681</td>
      <td>1.0</td>
      <td>0.0</td>
      <td>1.0</td>
      <td>0.0</td>
      <td>0.895363</td>
      <td>0.406085</td>
      <td>...</td>
      <td>0.101665</td>
      <td>0.472367</td>
      <td>1.621764</td>
      <td>-0.120868</td>
      <td>-0.239610</td>
      <td>0.0</td>
      <td>0.0</td>
      <td>0.0</td>
      <td>1.0</td>
      <td>0.0</td>
    </tr>
    <tr>
      <th>249</th>
      <td>4.846575</td>
      <td>0.0</td>
      <td>1.0</td>
      <td>-0.924494</td>
      <td>0.0</td>
      <td>0.0</td>
      <td>1.0</td>
      <td>0.0</td>
      <td>-0.510565</td>
      <td>-0.225352</td>
      <td>...</td>
      <td>0.245463</td>
      <td>1.899020</td>
      <td>-0.580807</td>
      <td>0.422207</td>
      <td>0.159309</td>
      <td>0.0</td>
      <td>0.0</td>
      <td>0.0</td>
      <td>0.0</td>
      <td>1.0</td>
    </tr>
    <tr>
      <th>266</th>
      <td>0.490411</td>
      <td>1.0</td>
      <td>0.0</td>
      <td>1.938314</td>
      <td>0.0</td>
      <td>1.0</td>
      <td>1.0</td>
      <td>1.0</td>
      <td>0.748475</td>
      <td>-0.608191</td>
      <td>...</td>
      <td>-0.650254</td>
      <td>-0.288898</td>
      <td>-0.481443</td>
      <td>-0.727833</td>
      <td>1.356065</td>
      <td>0.0</td>
      <td>1.0</td>
      <td>0.0</td>
      <td>0.0</td>
      <td>1.0</td>
    </tr>
    <tr>
      <th>1</th>
      <td>12.328767</td>
      <td>0.0</td>
      <td>0.0</td>
      <td>0.563645</td>
      <td>0.0</td>
      <td>0.0</td>
      <td>1.0</td>
      <td>1.0</td>
      <td>-0.405645</td>
      <td>-0.210436</td>
      <td>...</td>
      <td>2.173526</td>
      <td>-0.144699</td>
      <td>-0.531125</td>
      <td>-0.450972</td>
      <td>-0.139881</td>
      <td>0.0</td>
      <td>0.0</td>
      <td>0.0</td>
      <td>1.0</td>
      <td>0.0</td>
    </tr>
  </tbody>
</table>
<p>5 rows × 22 columns</p>
</div>



<a name='5'></a>
## 5. Fitting and Interpreting a Cox Model

Run the following cell to fit your Cox Proportional Hazards model using the `lifelines` package.


```python
cph = CoxPHFitter()
cph.fit(one_hot_train, duration_col = 'time', event_col = 'status', step_size=0.1)
```




    <lifelines.CoxPHFitter: fitted with 154 total observations, 90 right-censored observations>



You can use `cph.print_summary()` to view the coefficients associated with each covariate as well as confidence intervals. 


```python
cph.print_summary()
```


<div>
<style scoped>
    .dataframe tbody tr th:only-of-type {
        vertical-align: middle;
    }

    .dataframe tbody tr th {
        vertical-align: top;
    }

    .dataframe thead th {
        text-align: right;
    }
</style>
<table border="1" class="dataframe">
  <tbody>
    <tr>
      <th>model</th>
      <td>lifelines.CoxPHFitter</td>
    </tr>
    <tr>
      <th>duration col</th>
      <td>'time'</td>
    </tr>
    <tr>
      <th>event col</th>
      <td>'status'</td>
    </tr>
    <tr>
      <th>number of observations</th>
      <td>154</td>
    </tr>
    <tr>
      <th>number of events observed</th>
      <td>64</td>
    </tr>
    <tr>
      <th>partial log-likelihood</th>
      <td>-230.82</td>
    </tr>
    <tr>
      <th>time fit was run</th>
      <td>2024-09-30 15:57:36 UTC</td>
    </tr>
  </tbody>
</table>
</div><table border="1" class="dataframe">
  <thead>
    <tr style="text-align: right;">
      <th></th>
      <th>coef</th>
      <th>exp(coef)</th>
      <th>se(coef)</th>
      <th>coef lower 95%</th>
      <th>coef upper 95%</th>
      <th>exp(coef) lower 95%</th>
      <th>exp(coef) upper 95%</th>
      <th>z</th>
      <th>p</th>
      <th>-log2(p)</th>
    </tr>
  </thead>
  <tbody>
    <tr>
      <th>trt</th>
      <td>-0.22</td>
      <td>0.80</td>
      <td>0.30</td>
      <td>-0.82</td>
      <td>0.37</td>
      <td>0.44</td>
      <td>1.45</td>
      <td>-0.73</td>
      <td>0.46</td>
      <td>1.11</td>
    </tr>
    <tr>
      <th>age</th>
      <td>0.23</td>
      <td>1.26</td>
      <td>0.19</td>
      <td>-0.13</td>
      <td>0.60</td>
      <td>0.88</td>
      <td>1.82</td>
      <td>1.26</td>
      <td>0.21</td>
      <td>2.27</td>
    </tr>
    <tr>
      <th>sex</th>
      <td>0.34</td>
      <td>1.41</td>
      <td>0.40</td>
      <td>-0.45</td>
      <td>1.14</td>
      <td>0.64</td>
      <td>3.11</td>
      <td>0.84</td>
      <td>0.40</td>
      <td>1.33</td>
    </tr>
    <tr>
      <th>ascites</th>
      <td>-0.10</td>
      <td>0.91</td>
      <td>0.56</td>
      <td>-1.20</td>
      <td>1.01</td>
      <td>0.30</td>
      <td>2.75</td>
      <td>-0.17</td>
      <td>0.86</td>
      <td>0.21</td>
    </tr>
    <tr>
      <th>hepato</th>
      <td>0.31</td>
      <td>1.36</td>
      <td>0.38</td>
      <td>-0.44</td>
      <td>1.06</td>
      <td>0.64</td>
      <td>2.89</td>
      <td>0.81</td>
      <td>0.42</td>
      <td>1.26</td>
    </tr>
    <tr>
      <th>spiders</th>
      <td>-0.18</td>
      <td>0.83</td>
      <td>0.38</td>
      <td>-0.94</td>
      <td>0.57</td>
      <td>0.39</td>
      <td>1.77</td>
      <td>-0.47</td>
      <td>0.64</td>
      <td>0.66</td>
    </tr>
    <tr>
      <th>bili</th>
      <td>0.05</td>
      <td>1.05</td>
      <td>0.18</td>
      <td>-0.29</td>
      <td>0.39</td>
      <td>0.75</td>
      <td>1.48</td>
      <td>0.29</td>
      <td>0.77</td>
      <td>0.37</td>
    </tr>
    <tr>
      <th>chol</th>
      <td>0.19</td>
      <td>1.20</td>
      <td>0.15</td>
      <td>-0.10</td>
      <td>0.47</td>
      <td>0.91</td>
      <td>1.60</td>
      <td>1.28</td>
      <td>0.20</td>
      <td>2.33</td>
    </tr>
    <tr>
      <th>albumin</th>
      <td>-0.40</td>
      <td>0.67</td>
      <td>0.18</td>
      <td>-0.75</td>
      <td>-0.06</td>
      <td>0.47</td>
      <td>0.94</td>
      <td>-2.28</td>
      <td>0.02</td>
      <td>5.46</td>
    </tr>
    <tr>
      <th>copper</th>
      <td>0.30</td>
      <td>1.35</td>
      <td>0.16</td>
      <td>-0.01</td>
      <td>0.61</td>
      <td>0.99</td>
      <td>1.84</td>
      <td>1.91</td>
      <td>0.06</td>
      <td>4.14</td>
    </tr>
    <tr>
      <th>alk.phos</th>
      <td>-0.22</td>
      <td>0.80</td>
      <td>0.14</td>
      <td>-0.49</td>
      <td>0.05</td>
      <td>0.61</td>
      <td>1.05</td>
      <td>-1.62</td>
      <td>0.11</td>
      <td>3.24</td>
    </tr>
    <tr>
      <th>ast</th>
      <td>0.21</td>
      <td>1.24</td>
      <td>0.16</td>
      <td>-0.10</td>
      <td>0.53</td>
      <td>0.91</td>
      <td>1.69</td>
      <td>1.34</td>
      <td>0.18</td>
      <td>2.48</td>
    </tr>
    <tr>
      <th>trig</th>
      <td>0.20</td>
      <td>1.23</td>
      <td>0.16</td>
      <td>-0.11</td>
      <td>0.52</td>
      <td>0.89</td>
      <td>1.68</td>
      <td>1.27</td>
      <td>0.21</td>
      <td>2.28</td>
    </tr>
    <tr>
      <th>platelet</th>
      <td>0.14</td>
      <td>1.15</td>
      <td>0.15</td>
      <td>-0.16</td>
      <td>0.43</td>
      <td>0.86</td>
      <td>1.54</td>
      <td>0.92</td>
      <td>0.36</td>
      <td>1.48</td>
    </tr>
    <tr>
      <th>protime</th>
      <td>0.36</td>
      <td>1.43</td>
      <td>0.17</td>
      <td>0.03</td>
      <td>0.69</td>
      <td>1.03</td>
      <td>1.99</td>
      <td>2.15</td>
      <td>0.03</td>
      <td>4.97</td>
    </tr>
    <tr>
      <th>edema_0.5</th>
      <td>1.24</td>
      <td>3.47</td>
      <td>0.46</td>
      <td>0.35</td>
      <td>2.14</td>
      <td>1.42</td>
      <td>8.50</td>
      <td>2.72</td>
      <td>0.01</td>
      <td>7.28</td>
    </tr>
    <tr>
      <th>edema_1.0</th>
      <td>2.02</td>
      <td>7.51</td>
      <td>0.60</td>
      <td>0.84</td>
      <td>3.20</td>
      <td>2.31</td>
      <td>24.43</td>
      <td>3.35</td>
      <td>&lt;0.005</td>
      <td>10.28</td>
    </tr>
    <tr>
      <th>stage_2.0</th>
      <td>1.21</td>
      <td>3.35</td>
      <td>1.08</td>
      <td>-0.92</td>
      <td>3.33</td>
      <td>0.40</td>
      <td>28.06</td>
      <td>1.11</td>
      <td>0.27</td>
      <td>1.91</td>
    </tr>
    <tr>
      <th>stage_3.0</th>
      <td>1.18</td>
      <td>3.27</td>
      <td>1.09</td>
      <td>-0.96</td>
      <td>3.33</td>
      <td>0.38</td>
      <td>27.86</td>
      <td>1.08</td>
      <td>0.28</td>
      <td>1.84</td>
    </tr>
    <tr>
      <th>stage_4.0</th>
      <td>1.41</td>
      <td>4.10</td>
      <td>1.15</td>
      <td>-0.85</td>
      <td>3.67</td>
      <td>0.43</td>
      <td>39.43</td>
      <td>1.22</td>
      <td>0.22</td>
      <td>2.18</td>
    </tr>
  </tbody>
</table><div>
<style scoped>
    .dataframe tbody tr th:only-of-type {
        vertical-align: middle;
    }

    .dataframe tbody tr th {
        vertical-align: top;
    }

    .dataframe thead th {
        text-align: right;
    }
</style>
<table border="1" class="dataframe">
  <tbody>
    <tr>
      <th>Concordance</th>
      <td>0.83</td>
    </tr>
    <tr>
      <th>Log-likelihood ratio test</th>
      <td>97.63 on 20 df, -log2(p)=38.13</td>
    </tr>
  </tbody>
</table>
</div>


**Question:**

- According to the model, was treatment `trt` beneficial? 
- What was its associated hazard ratio? 
    - Note that the hazard ratio is how much an incremental increase in the feature variable changes the hazard.

<details>    
<summary>
    <font size="3" color="darkgreen"><b>Check your answer!</b></font>
</summary>
<p>
<ul>
<ul>
    <li>You should see that the treatment (trt) was beneficial because it has a negative impact on the hazard (the coefficient is negative, and exp(coef) is less than 1).</li>
    <li>The associated hazard ratio is ~0.8, because this is the exp(coef) of treatment.</li>
</ul>
</p>

We can compare the predicted survival curves for treatment variables. Run the next cell to plot survival curves using the `plot_covariate_groups()` function. 
- The y-axis is th survival rate
- The x-axis is time


```python
cph.plot_covariate_groups('trt', values=[0, 1]);
```


![png](output_34_0.png)


Notice how the group without treatment has a lower survival rate at all times (the x-axis is time) compared to the treatment group.

<a name='6'></a>
## 6. Hazard Ratio

Recall from the lecture videos that the Hazard Ratio between two patients was the likelihood of one patient (e.g smoker) being more at risk than the other (e.g non-smoker).
$$
\frac{\lambda_{smoker}(t)}{\lambda_{nonsmoker}(t)} = e^{\theta (X_{smoker} - X_{nonsmoker})^T}
$$

Where

$$
\lambda_{smoker}(t) = \lambda_0(t)e^{\theta X_{smoker}^T}
$$
and
$$
\lambda_{nonsmoker}(t) = \lambda_0(t)e^{\theta X_{nonsmoker}^T} \\
$$

<a name='ex-2'></a>
### Exercise 2 - hazard_ratio
In the cell below, write a function to compute the hazard ratio between two individuals given the cox model's coefficients.

<details>    
<summary>
    <font size="3" color="darkgreen"><b>Hints</b></font>
</summary>
<p>
<ul>
    <li>use numpy.dot</li>
    <li>use nump.exp</li>
</ul>
</p>



```python
# UNQ_C2 (UNIQUE CELL IDENTIFIER, DO NOT EDIT)
def hazard_ratio(case_1, case_2, cox_params):
    '''
    Return the hazard ratio of case_1 : case_2 using
    the coefficients of the cox model.
    
    Args:
        case_1 (np.array): (1 x d) array of covariates
        case_2 (np.array): (1 x d) array of covariates
        model (np.array): (1 x d) array of cox model coefficients
    Returns:
        hazard_ratio (float): hazard ratio of case_1 : case_2
    '''
    
    ### START CODE HERE (REPLACE INSTANCES OF 'None' with your code) ###
    
    hr = np.exp(np.dot(cox_params,(case_1 - case_2).T))
    
    ### END CODE HERE ###
    
    return hr
```

Now, evaluate it on the following pair of indivduals: `i = 1` and `j = 5`. Given your implementation of `hazard_ratio` is correct, you will **only** pass the test when `i = 1` and `j = 5`.


```python
# test cell ex2

# Set i = 1 and j = 5 to pass the test
i = 1
j = 5

# DO NOT modify anything below
hazard_ratio_test(hazard_ratio, i, j, one_hot_train, cph)
```

    15.029017732492221 
    
    [92m All tests passed.


#### Expected Output:
```
15.029017732492221 

 All tests passed.
```

**Question:** 

Is `case_1` or `case_2` at greater risk? 

<details>    
<summary>
    <font size="3" color="darkgreen"><b>Check your answer!</b></font>
</summary>
<p>
<ul>
<ul>
    Important! The following answer only applies if you picked i = 1 and j = 5
    <li>You should see that `case_1` is at higher risk.</li>
    <li>The hazard ratio of case 1 / case 2 is greater than 1, so case 1 had a higher hazard relative to case 2</li>
</ul>
</p>

Inspect different pairs, and see if you can figure out which patient is more at risk.


```python
i = 4
case_1 = one_hot_train.iloc[i, :].drop(['time', 'status'])

j = 7
case_2 = one_hot_train.iloc[j, :].drop(['time', 'status'])

print("Case 1\n\n", case_1, "\n")
print("Case 2\n\n", case_2, "\n")
print("Hazard Ratio:", hazard_ratio(case_1.values, case_2.values, cph.params_.values))
```

    Case 1
    
     trt          0.000000
    age          0.563645
    sex          0.000000
    ascites      0.000000
    hepato       1.000000
    spiders      1.000000
    bili        -0.405645
    chol        -0.210436
    albumin      1.514297
    copper      -0.481961
    alk.phos     2.173526
    ast         -0.144699
    trig        -0.531125
    platelet    -0.450972
    protime     -0.139881
    edema_0.5    0.000000
    edema_1.0    0.000000
    stage_2.0    0.000000
    stage_3.0    1.000000
    stage_4.0    0.000000
    Name: 1, dtype: float64 
    
    Case 2
    
     trt          0.000000
    age          0.463447
    sex          0.000000
    ascites      0.000000
    hepato       1.000000
    spiders      0.000000
    bili        -0.489581
    chol        -0.309875
    albumin     -1.232371
    copper      -0.504348
    alk.phos     2.870427
    ast         -0.936261
    trig        -0.150229
    platelet     3.190823
    protime     -0.139881
    edema_0.5    0.000000
    edema_1.0    0.000000
    stage_2.0    0.000000
    stage_3.0    0.000000
    stage_4.0    1.000000
    Name: 38, dtype: float64 
    
    Hazard Ratio: 0.1780450006997129


<details>    
<summary>
    <font size="3" color="darkgreen"><b>Check your answer!</b></font>
</summary>
<p>
<ul>
<ul>
    Important! The following answer only applies if you picked i = 4 and j = 7
    <li>You should see that `case_2` is at higher risk.</li>
    <li>The hazard ratio of case 1 / case 2 is less than 1, so case 2 had a higher hazard relative to case 1</li>
</ul>
</p>

<a name='7'></a>
## 7. Harrell's C-index

To evaluate how good our model is performing, we will write our own version of the C-index. Similar to the week 1 case, C-index in the survival context is the probability that, given a randomly selected pair of individuals, the one who died sooner has a higher risk score. 

However, we need to take into account censoring. Imagine a pair of patients, $A$ and $B$. 

#### Scenario 1
- A was censored at time $t_A$ 
- B died at $t_B$
- $t_A < t_B$. 

Because of censoring, we can't say whether $A$ or $B$ should have a higher risk score. 

#### Scenario 2
Now imagine that $t_A > t_B$.

- A was censored at time $t_A$ 
- B died at $t_B$
- $t_A > t_B$

Now we can definitively say that $B$ should have a higher risk score than $A$, since we know for a fact that $A$ lived longer. 

Therefore, when we compute our C-index
- We should only consider pairs where at most one person is censored
- If they are censored, then their censored time should occur *after* the other person's time of death. 

The metric we get if we use this rule is called **Harrel's C-index**.

Note that in this case, being censored at time $t$ means that the true death time was some time AFTER time $t$ and not at $t$. 
- Therefore if $t_A = t_B$ and A was censored:
    - Then $A$ actually lived longer than $B$. 
    - This will effect how you deal with ties in the exercise below!



<a name='ex-3'></a>
### Exercise 3 - harrell_c
Fill in the function below to compute Harrel's C-index.

<details>    
<summary>
    <font size="3" color="darkgreen"><b>Hints</b></font>
</summary>
<p>
<ul>
    <li>If you get a division by zero error, consider checking how you count when a pair is permissible (in the case where one patient is censored and the other is not censored).</li>
</ul>
</p>


```python
# UNQ_C3 (UNIQUE CELL IDENTIFIER, DO NOT EDIT)
def harrell_c(y_true, scores, event):
    '''
    Compute Harrel C-index given true event/censoring times,
    model output, and event indicators.
    
    Args:
        y_true (array): array of true event times
        scores (array): model risk scores
        event (array): indicator, 1 if event occurred at that index, 0 for censorship
    Returns:
        result (float): C-index metric
    '''
    
    n = len(y_true)
    assert (len(scores) == n and len(event) == n)
    
    concordant = 0.0
    permissible = 0.0
    ties = 0.0
    
    result = 0.0
    
    ### START CODE HERE (REPLACE INSTANCES OF 'None' and 'pass' with your code) ###
    
   # use double for loop to go through cases
    for i in range(n):
        # set lower bound on j to avoid double counting
        for j in range(i+1, n):
            
            # check if at most one is censored
            if not (event[i] == 0 and event[j] == 0):

                # check if neither are censored
                if event[i] == 1 and event[j] == 1:
                    permissible += 1
                    
                    # check if scores are tied
                    if y_true[i] == y_true[j]:
                        ties += 1
                    
                    # check for concordant
                    elif y_true[i] > y_true[j] and scores[i] < scores[j]:
                        concordant += 1
                    elif y_true[i] < y_true[j] and scores[i] > scores[j]:
                        concordant += 1
                
                # check if one is censored
                elif event[i] == 0 or event[j] == 0:
                    
                    # get censored index
                    censored = j
                    uncensored = i
                    
                    if event[i] == 0:
                        censored = i
                        uncensored = j
                        
                    # check if permissible
                    # Note: in this case, we are assuming that censored at a time
                    # means that you did NOT die at that time. That is, if you
                    # live until time 30 and have event = 0, then you lived THROUGH
                    # time 30.
                    if y_true[censored] >= y_true[uncensored]:
                        permissible += 1
                        
                        # check if scores are tied
                        if scores[i] == scores[j]:
                            # update ties 
                            ties += 1
                            
                        # check if scores are concordant 
                        if y_true[censored] >= y_true[uncensored] and scores[censored] < scores[uncensored]:
                            concordant += 1
    
    # set result to c-index computed from number of concordant pairs,
    # number of ties, and number of permissible pairs (REPLACE 0 with your code) 
    result = (concordant + 0.5 * ties) / permissible
    
    ### END CODE HERE ###
    
    return result   
```


```python
# test cell ex1 - do not modify this test cell
harrell_c_test(harrell_c)    
```

    Test Case 1
    
    y_true:  [30, 12, 84, 9]
    scores:  [0.5, 0.9, 0.1, 1.0]
    event:   [1, 1, 1, 1]
    Output:  1.0
    
    Test Case 2
    
    y_true:  [30, 12, 84, 9]
    scores:  [0.9, 0.5, 1.0, 0.1]
    event:   [1, 1, 1, 1]
    Output:  0.0
    
    Test Case 3
    
    y_true:  [30, 12, 84, 9]
    scores:  [0.5, 0.9, 0.1, 1.0]
    event:   [1, 0, 1, 1]
    Output:  1.0
    
    Test Case 4
    
    y_true:  [30, 30, 20, 20]
    scores:  [10, 5, 15, 20]
    event:   [1, 0, 1, 0]
    Output:  0.75
    
    Test Case 5
    
    y_true:  [20, 20, 30, 30, 30]
    scores:  [20, 15, 5, 10, 15]
    event:   [0, 1, 0, 1, 0]
    Output:  0.5833333333333334
    
    Tes Case 6
    
    y_true:  [10, 10]
    scores:  [4, 5]
    event:   [0, 1]
    Output:  1.0 
    
    [92m All tests passed.


#### Expected Output:
```
Test Case 1
Output:  1.0

Test Case 2
Output:  0.0

Test Case 3
Output:  1.0

Test Case 4
Output:  0.75

Test Case 5
Output:  0.5833333333333334

Tes Case 6
Output:  1.0 

 All tests passed.
```

Now use the Harrell's C-index function to evaluate the cox model on our data sets.


```python
# Train
scores = cph.predict_partial_hazard(one_hot_train)
cox_train_scores = harrell_c(one_hot_train['time'].values, scores.values, one_hot_train['status'].values)
# Validation
scores = cph.predict_partial_hazard(one_hot_val)
cox_val_scores = harrell_c(one_hot_val['time'].values, scores.values, one_hot_val['status'].values)
# Test
scores = cph.predict_partial_hazard(one_hot_test)
cox_test_scores = harrell_c(one_hot_test['time'].values, scores.values, one_hot_test['status'].values)

print("Train:", cox_train_scores)
print("Val:", cox_val_scores)
print("Test:", cox_test_scores)
```

    Train: 0.8265139116202946
    Val: 0.8544776119402985
    Test: 0.8478543563068921


What do these values tell us ?

<a name='8'></a>
## 8. Random Survival Forests

This performed well, but you have a hunch you can squeeze out better performance by using a machine learning approach. You decide to use a Random Survival Forest. To do this, you can use the `RandomForestSRC` package in R. To call R function from Python, we'll use the `r2py` package. Run the following cell to import the necessary requirements. 



```python
%load_ext rpy2.ipython
%R require(ggplot2)

from rpy2.robjects.packages import importr
# import R's "base" package
base = importr('base')

# import R's "utils" package
utils = importr('utils')

# import rpy2's package module
import rpy2.robjects.packages as rpackages

forest = rpackages.importr('randomForestSRC', lib_loc='R')

from rpy2 import robjects as ro
R = ro.r

from rpy2.robjects import pandas2ri
pandas2ri.activate()
```

    R[write to console]: Loading required package: ggplot2
    


Instead of encoding our categories as binary features, we can use the original dataframe since trees deal well with raw categorical data (can you think why this might be?).

Run the code cell below to build your forest.


```python
model = forest.rfsrc(ro.Formula('Surv(time, status) ~ .'), data=df_train, ntree=300, nodedepth=5, seed=-1)
```


```python
print(model)
```

                             Sample size: 154
                        Number of deaths: 64
                         Number of trees: 300
               Forest terminal node size: 15
           Average no. of terminal nodes: 6.54
    No. of variables tried at each split: 5
                  Total no. of variables: 17
           Resampling used to grow trees: swor
        Resample size used to grow trees: 97
                                Analysis: RSF
                                  Family: surv
                          Splitting rule: logrank *random*
           Number of random split points: 10
                              Error rate: 19.07%
    
    


Finally, let's evaluate on our validation and test sets, and compare it with our Cox model.


```python
result = R.predict(model, newdata=df_val)
scores = np.array(result.rx('predicted')[0])

print("Cox Model Validation Score:", cox_val_scores)
print("Survival Forest Validation Score:", harrell_c(df_val['time'].values, scores, df_val['status'].values))
```

    Cox Model Validation Score: 0.8544776119402985
    Survival Forest Validation Score: 0.8296019900497512



```python
result = R.predict(model, newdata=df_test)
scores = np.array(result.rx('predicted')[0])

print("Cox Model Test Score:", cox_test_scores)
print("Survival Forest Validation Score:", harrell_c(df_test['time'].values, scores, df_test['status'].values))
```

    Cox Model Test Score: 0.8478543563068921
    Survival Forest Validation Score: 0.8621586475942783


Your random forest model should be outperforming the Cox model slightly. Let's dig deeper to see how they differ.

<a name='9'></a>
## 9. Permutation Method for Interpretation

We'll dig a bit deeper into interpretation methods for forests a bit later, but for now just know that random surival forests come with their own built in variable importance feature. The method is referred to as VIMP, and for the purpose of this section you should just know that higher absolute value of the VIMP means that the variable generally has a larger effect on the model outcome.

Run the next cell to compute and plot VIMP for the random survival forest.


```python
vimps = np.array(forest.vimp(model).rx('importance')[0])

y = np.arange(len(vimps))
plt.barh(y, np.abs(vimps))
plt.yticks(y, df_train.drop(['time', 'status'], axis=1).columns)
plt.title("VIMP (absolute value)")
plt.show()
```


![png](output_68_0.png)


### Question:

How does the variable importance compare to that of the Cox model? Which variable is important in both models? Which variable is important in the random survival forest but not in the Cox model? You should see that `edema` is important in both the random survival forest and the Cox model. You should also see that `bili` is important in the random survival forest but not the Cox model .

## Congratulations!

You've finished the last assignment in course 2! Take a minute to look back at the analysis you've done over the last four assignments. You've done a great job!
