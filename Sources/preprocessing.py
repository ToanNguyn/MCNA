import numpy as np
from sklearn.preprocessing import StandardScaler
from imblearn.over_sampling import SMOTE

def handle_numeric_outliers(df, method='iqr'):
    numeric_cols = df.select_dtypes(include=np.number).columns
    binary_cols = [col for col in numeric_cols if df[col].nunique() == 2]
    numeric_cols = list(set(numeric_cols) - set(binary_cols))

    for col in numeric_cols:
        Q1 = df[col].quantile(0.25)
        Q3 = df[col].quantile(0.75)
        IQR = Q3 - Q1
        lower = Q1 - 1.5 * IQR
        upper = Q3 + 1.5 * IQR

        df[col] = df[col].clip(lower, upper)

    return df

def frequency_encode(X_train, X_test, col='State'):
    freq = X_train[col].value_counts(normalize=True) 
    X_train[col + '_freq'] = X_train[col].map(freq)
    X_test[col + '_freq'] = X_test[col].map(freq)
    X_train.drop(col, axis=1, inplace=True)
    X_test.drop(col, axis=1, inplace=True)
    return X_train, X_test

def convert_plan_features(X):
    X['International plan'] = X['International plan'].map({'No': 0, 'Yes': 1})
    X['Voice mail plan'] = X['Voice mail plan'].map({'No': 0, 'Yes': 1})
    return X

def feature_engineering(X):
    X['avg_day_call_duration'] = X['Total day minutes'] / (X['Total day calls'] + 1e-5)
    X['avg_eve_call_duration'] = X['Total eve minutes'] / (X['Total eve calls'] + 1e-5)
    X['avg_night_call_duration'] = X['Total night minutes'] / (X['Total night calls'] + 1e-5)
    X['avg_intl_call_duration'] = X['Total intl minutes'] / (X['Total intl calls'] + 1e-5)

    X['total_minutes'] = X['Total day minutes'] + X['Total eve minutes'] + X['Total night minutes'] + X['Total intl minutes']
    X['day_ratio'] = X['Total day minutes'] / (X['total_minutes'] + 1e-5)
    X['eve_ratio'] = X['Total eve minutes'] / (X['total_minutes'] + 1e-5)
    X['night_ratio'] = X['Total night minutes'] / (X['total_minutes'] + 1e-5)
    X['intl_ratio'] = X['Total intl minutes'] / (X['total_minutes'] + 1e-5)

    X['total_calls'] = X['Total day calls'] + X['Total eve calls'] + X['Total night calls'] + X['Total intl calls']
    X['intl_calls_ratio'] = X['Total intl calls'] / (X['total_calls'] + 1e-5)

    X['Total charge'] = X['Total day charge'] + X['Total eve charge'] + X['Total night charge'] + X['Total intl charge']
    return X

def scale_features(X_train, X_test): # boosting ko cần scaling 
    scaler = StandardScaler()
    X_train_scaled = scaler.fit_transform(X_train)
    X_test_scaled = scaler.transform(X_test)
    return X_train_scaled, X_test_scaled

def balance_data(X_train, y_train):
    smote = SMOTE(random_state=42)
    X_resampled, y_resampled = smote.fit_resample(X_train, y_train)
    return X_resampled, y_resampled

def preprocess_pipeline(train, test, model='logreg'):
    X_train = train.drop('Churn', axis=1).copy()
    y_train = train['Churn'].astype(int)
    X_test = test.drop('Churn', axis=1).copy()
    y_test = test['Churn'].astype(int)

    X_train = convert_plan_features(X_train)
    X_test = convert_plan_features(X_test)

    if model == 'logreg':
        X_train = handle_numeric_outliers(X_train)
        X_test = handle_numeric_outliers(X_test)

    X_train = feature_engineering(X_train)
    X_test = feature_engineering(X_test)

    X_train, X_test = frequency_encode(X_train, X_test)

    if model == 'logreg':
        feature_names = X_train.columns.tolist()
        X_train, X_test = scale_features(X_train, X_test)

    X_train, y_train = balance_data(X_train, y_train)

    if model == 'logreg':
        return X_train, y_train, X_test, y_test, feature_names
    else:
        return X_train, y_train, X_test, y_test

