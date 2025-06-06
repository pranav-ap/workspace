from utils.logger_setup import logger

import numpy as np
import pandas as pd

from pprint import pprint


def extract_or_replace(pattern, value):
    import re
    match = re.search(pattern, value)
    if match is not None:
        return match.group(1)

    return None


#
# def get_datasets():
#     logger.debug("Getting Dataset!")
#
#     # Load the dataset
#     train_data = pd.read_csv("../data/used_car_prices/train.csv")
#     test_data = pd.read_csv("../data/used_car_prices/test.csv")
#
#     ids = test_data['id']
#
#     # Determine lengths of train and test sets
#     train_length = len(train_data)
#     test_length = len(test_data)
#
#     # Add a source column
#     train_data['source'] = 'train'
#     test_data['source'] = 'test'
#
#     combined = pd.concat([train_data, test_data])
#     combined = combined.drop(['id'], axis=1)
#
#     # Extract features from the 'engine' column
#     combined['hp'] = combined['engine'].map(lambda value: extract_or_replace(r'(\d+\.\d+)HP', value))
#     combined['litres'] = combined['engine'].map(lambda value: extract_or_replace(r'(\d+\.\d+)L', value))
#     combined['cylinders'] = combined['engine'].map(lambda value: extract_or_replace(r'(\d+) Cylinder', value))
#
#     combined['hp'] = combined['hp'].astype('float64')
#     combined['litres'] = combined['litres'].astype('float64')
#     combined['cylinders'] = combined['cylinders'].astype('float64')
#
#     combined['accident'] = combined['accident'].map({
#         'None reported': 0,
#         'At least 1 accident or damage reported': 1
#     })
#
#     combined = combined.drop(['engine'], axis=1)
#
#     # Define categorical features
#     categorical_features = ['brand', 'model', 'fuel_type', 'ext_col', 'int_col', 'clean_title']
#
#     # Define transformers
#     categorical_transformer = Pipeline(steps=[
#         ('imputer', SimpleImputer(strategy='most_frequent')),  # Handle missing values
#         ('onehot', OneHotEncoder(drop='first'))  # One-hot encode categorical features
#     ])
#
#     # Create a preprocessor
#     preprocessor = ColumnTransformer(
#         transformers=[
#             ('cat', categorical_transformer, categorical_features)
#         ]
#     )
#
#     # Apply transformations
#     X = preprocessor.fit_transform(combined)
#     combined_transformed = pd.DataFrame(X)
#
#     # Split transformed data back into train and test sets
#     train_data_transformed = combined_transformed.iloc[:train_length]
#     test_data_transformed = combined_transformed.iloc[train_length:]
#
#     # Features and target variable
#     X_train, y_train = train_data_transformed.drop("price", axis=1), combined.query('source == "train"')["price"]
#     X_test, y_test = test_data_transformed.drop("price", axis=1), None
#
#     return X_train, X_test, y_train, y_test, ids
#

def get_datasets():
    logger.debug("Getting Dataset!")

    # Load the dataset
    train_data = pd.read_csv("../data/used_car_prices/train.csv")
    test_data = pd.read_csv("../data/used_car_prices/test.csv")

    ids = test_data['id']

    # Determine lengths of train and test sets
    train_length = len(train_data)
    test_length = len(test_data)

    # Add a source column
    train_data['source'] = 'train'
    test_data['source'] = 'test'

    combined = pd.concat([train_data, test_data])
    combined = combined.drop(['id', 'ext_col', 'int_col', 'transmission', 'fuel_type', 'clean_title', 'accident'], axis=1)

    for col in ['brand', 'model']:
        combined[col] = combined[col].astype('category')

    combined['hp'] = combined['engine'].map(
        lambda value: extract_or_replace(r'(\\d+\\.\\d+)HP', value))

    combined['litres'] = combined['engine'].map(
        lambda value: extract_or_replace('(\\d+\\.\\d+)L', value))

    combined['cylinders'] = combined['engine'].map(
        lambda value: extract_or_replace('(\\d+) Cylinder', value))

    combined['hp'] = combined['hp'].astype('float64')
    combined['litres'] = combined['litres'].astype('float64')
    combined['cylinders'] = combined['cylinders'].astype('float64')

    # combined['accident'] = combined['accident'].map({
    #     'None reported': 0,
    #     'At least 1 accident or damage reported': 1
    # })

    train_data = combined.query('source == "train"')
    test_data = combined.query('source == "test"')

    # for col in ['accident']:
    #     mode_val = train_data[col].mode()
    #     train_data.fillna({col: mode_val}, inplace=True)
    #
    #     mode_val = test_data[col].mode()
    #     test_data.fillna({col: mode_val}, inplace=True)

    for col in ['litres', 'hp', 'cylinders', 'price']:
        mean_val = train_data[col].mean()
        train_data.fillna({col: mean_val}, inplace=True)

        mean_val = test_data[col].mean()
        test_data.fillna({col: mean_val}, inplace=True)

    combined = pd.concat([train_data, test_data])
    combined = combined.drop(['engine'], axis=1)

    # Convert categorical columns to numerical using one-hot encoding
    combined = pd.get_dummies(combined.drop(['source'], axis=1), drop_first=True, sparse=True)

    # from sklearn.preprocessing import OneHotEncoder
    # encoder = OneHotEncoder(drop='first')
    # combined = encoder.fit_transform(combined)
    # combined = pd.DataFrame(combined)
    # print(combined.columns)

    # train_data = combined_encoded[combined['source'] == 'train']
    # test_data = combined_encoded[combined['source'] == 'test']

    # train_data = (combined[combined['source'] == 'train']).drop(['source'], axis=1)
    # test_data = (combined[combined['source'] == 'test']).drop(['source'], axis=1)

    # Split transformed data back into train and test sets
    train_data, test_data = combined.iloc[:train_length], combined.iloc[train_length:]
    del combined

    logger.info(f"{train_data.shape}, {test_data.shape}")

    # Features and target variable
    X_train, y_train = train_data.drop("price", axis=1), train_data["price"]
    X_test, y_test = test_data.drop("price", axis=1), None

    return X_train, X_test, y_train, y_test, ids


def prepare_RandomForestRegressor():
    from sklearn.ensemble import RandomForestRegressor

    model = RandomForestRegressor()

    param_grid = {
        "n_estimators": [300],
        "min_samples_split": [50, 60, 80],
        "max_depth": [15],
        'min_samples_leaf': [5]
    }

    return model, param_grid


def train(X, y, model, param_grid):
    from sklearn.metrics import make_scorer, mean_squared_error

    # Define RMSE as a scoring function
    def rmse(y_true, y_pred):
        return np.sqrt(mean_squared_error(y_true, y_pred))

    # Create a custom RMSE scorer
    rmse_scorer = make_scorer(rmse, greater_is_better=False)

    from sklearn.preprocessing import StandardScaler
    scaler = StandardScaler()
    X_transformed = scaler.fit_transform(X)

    from sklearn.model_selection import GridSearchCV
    grid = GridSearchCV(estimator=model,
                        param_grid=param_grid,
                        cv=5,
                        scoring=rmse_scorer,
                        verbose=1,
                        n_jobs=-1)

    logger.info("Go Forth and Grid Search!")
    grid.fit(X_transformed, y)

    logger.info(f"Best score        : {grid.best_score_ * 100:.2f}%")
    logger.info(f"Best parameters   : ")
    pprint(grid.best_params_)

    best_model = grid.best_estimator_

    logger.info("Retraining best model on full training data")
    best_model.fit(X_transformed, y)

    best_model.scaler = scaler

    return best_model


def main():
    logger.success("Start!")

    # Prepare model
    model, param_grid = prepare_RandomForestRegressor()

    # Get datasets
    X_train, X_test, y_train, y_test, ids = get_datasets()
    best_model = train(X_train, y_train, model, param_grid)

    logger.info("Predicting on Test Set")
    X_test_transformed = best_model.scaler.transform(X_test)

    from utils.common import plot_validation_curve, plot_learning_curve
    plot_validation_curve(X_train, y_train, best_model, param_name='n_estimators', param_range=[50, 100])
    plot_learning_curve(X_train, y_train, best_model)

    y_pred = best_model.predict(X_test_transformed)
    submission = pd.DataFrame({
        "id": ids,
        "price": y_pred
    })

    submission.to_csv('../data/used_car_prices/submission.csv', index=False)

    logger.success("Done!")

    return best_model


if __name__ == '__main__':
    main()


