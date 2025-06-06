from utils.logger_setup import logger

import seaborn as sns
from pprint import pprint


def get_datasets():
    logger.debug("Getting Dataset!")

    # Load
    data = sns.load_dataset('iris')

    # X & y
    X = data.drop("species", axis=1)
    y = data["species"]

    # Dont use dummy variables. Does not work with random forest classifier

    # Splitting
    from sklearn.model_selection import train_test_split
    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, stratify=y)

    return X_train, X_test, y_train, y_test


def prepare_RandomForestClassifier():
    from sklearn.ensemble import RandomForestClassifier

    model = RandomForestClassifier()

    param_grid = {
        "n_estimators": [50, 100],
        "min_samples_split": [2, 10, 25, 50],
        "max_depth": [None, 2, 5, 10, 20]
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
    best_model.scaler = scaler

    return best_model


def eval_model(y_test, y_pred, y_pred_proba=None):
    logger.info("Evaluation on Test Set")

    from sklearn.metrics import accuracy_score
    logger.info(f"Accuracy Score : {accuracy_score(y_test, y_pred):.4f}")

    from sklearn.metrics import classification_report
    print(classification_report(y_test, y_pred, zero_division=0))

    from utils.common import plot_confusion_matrix
    plot_confusion_matrix(y_test, y_pred)

    if y_pred_proba is not None:
        from utils.common import plot_roc_curve, plot_precision_recall_curve
        plot_roc_curve(y_test, y_pred_proba, is_binary=False)
        plot_precision_recall_curve(y_test, y_pred_proba, is_binary=False)


def main():
    logger.success("Start!")

    # model, param_grid = prepare_LogisticRegressionClassifier()
    model, param_grid = prepare_RandomForestClassifier()

    X_train, X_test, y_train, y_test = get_datasets()
    best_model = train(X_train, y_train, model, param_grid)

    logger.info("Predicting on Test Set")
    X_test_transformed = best_model.scaler.transform(X_test)

    if hasattr(best_model, 'predict_proba'):
        y_pred_proba = best_model.predict_proba(X_test_transformed)
    else:
        y_pred_proba = None

    y_pred = best_model.predict(X_test_transformed)
    eval_model(y_test, y_pred, y_pred_proba)

    from utils.common import plot_validation_curve, plot_learning_curve
    plot_validation_curve(X_train, y_train, best_model, param_name='n_estimators', param_range=[50, 100])
    plot_learning_curve(X_train, y_train, best_model)

    logger.success("Done!")

    return best_model


if __name__ == '__main__':
    main()

