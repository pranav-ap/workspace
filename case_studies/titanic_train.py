from utils.logger_setup import logger

import numpy as np
import pandas as pd

from pprint import pprint


def prepare_dataset(data, train=True):
    # Drop unnecessary columns
    columns_to_drop = ['PassengerId', 'Name', 'Ticket', 'Cabin', 'Embarked']
    data = data.drop(columns_to_drop, axis=1)

    data['Pclass'] = data['Pclass'].astype('category')
    data['Sex'] = data['Sex'].astype('category')

    # Handle missing values
    mean_age = data['Age'].mean()
    data['Age'] = data['Age'].fillna(mean_age)

    # Create new features
    data["Family_Size"] = data['SibSp'] + data['Parch'] + 1
    data['Family_Type'] = pd.cut(
        data['Family_Size'],
        5,
        labels=np.arange(5)
    )

    no_of_bins = 5

    data['Age_Group'] = pd.cut(
        data['Age'],
        no_of_bins,
        labels=np.arange(no_of_bins)
    )

    # data = data.drop(['SibSp', 'Parch', 'Family_Size', 'Age'], axis=1)
    data = data.drop(['SibSp', 'Parch'], axis=1)

    data['Family_Type'] = data['Family_Type'].astype('category')
    data['Age_Group'] = data['Age_Group'].astype('category')

    if train:
        from sklearn.utils import resample

        # separate majority and minority classes
        majority = data[data['Survived'] == 0]
        minority = data[data['Survived'] == 1]

        # oversample the minority class
        minority_oversampled = resample(minority,
                                        replace=True,
                                        n_samples=len(majority))

        # combine majority class with oversampled minority class
        data = pd.concat([majority, minority_oversampled])

    # Convert categorical columns to numerical using one-hot encoding
    data = pd.get_dummies(data, drop_first=True)

    # Features and target variable
    X = data.drop("Survived", axis=1) if train else data
    y = data["Survived"] if train else None

    return X, y


def get_datasets():
    logger.debug("Getting Dataset!")

    # Load the Titanic dataset
    train_data = pd.read_csv("../data/titanic/train.csv")
    test_data = pd.read_csv("../data/titanic/test.csv")

    testPassengerId = test_data['PassengerId']

    X_train, y_train = prepare_dataset(train_data)
    X_test, y_test = prepare_dataset(test_data, train=False)

    return X_train, X_test, y_train, y_test, testPassengerId


def prepare_RandomForestClassifier():
    from sklearn.ensemble import RandomForestClassifier

    model = RandomForestClassifier()

    param_grid = {
        "n_estimators": [300],
        "min_samples_split": [50, 60, 80],
        "max_depth": [15],
        'min_samples_leaf': [5]
    }

    return model, param_grid


def prepare_xgboostClassifier():
    import xgboost as xgb

    model = xgb.XGBClassifier()

    param_grid = {
      'n_estimators': [10, 20, 25],
      'learning_rate': [0.01, 0.1],
      'max_depth': [4, 7, 10],
      'gamma': [0.1],
      'subsample': [0.8],
      'colsample_bytree': [0.8]
    }

    return model, param_grid


def prepare_LogisticRegression():
    from sklearn.linear_model import LogisticRegression

    model = LogisticRegression()  # Increased max_iter to ensure convergence

    param_grid = {
        "penalty": ['l1', 'l2', 'elasticnet', 'none'],
        "C": [0.01, 0.1, 1, 10, 100],
        "solver": ['liblinear', 'saga'],
        "l1_ratio": [0.1, 0.5, 0.9]  # Only used if penalty is 'elasticnet'
    }

    return model, param_grid


def train(X, y, model, param_grid):
    from sklearn.preprocessing import StandardScaler
    scaler = StandardScaler()
    X_transformed = scaler.fit_transform(X)

    from sklearn.model_selection import GridSearchCV
    grid = GridSearchCV(estimator=model,
                        param_grid=param_grid,
                        cv=5,
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
    model, param_grid = prepare_RandomForestClassifier()
    # model, param_grid = prepare_LogisticRegression()
    # model, param_grid = prepare_xgboostClassifier()

    # Get datasets
    X_train, X_test, y_train, y_test, testPassengerId = get_datasets()
    best_model = train(X_train, y_train, model, param_grid)

    logger.info("Predicting on Test Set")
    X_test_transformed = best_model.scaler.transform(X_test)

    from utils.common import plot_validation_curve, plot_learning_curve
    plot_validation_curve(X_train, y_train, best_model, param_name='n_estimators', param_range=[50, 100])
    # plot_validation_curve(X_train, y_train, best_model, param_name='C', param_range=[50, 100])
    plot_learning_curve(X_train, y_train, best_model)

    y_pred = best_model.predict(X_test_transformed)
    submission = pd.DataFrame({
        "PassengerId": testPassengerId,
        "Survived": y_pred
    })

    submission.to_csv('../data/titanic/submission.csv', index=False)

    logger.success("Done!")

    return best_model


if __name__ == '__main__':
    main()
