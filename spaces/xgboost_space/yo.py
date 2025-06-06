import seaborn as sns
import xgboost as xgb
from sklearn.model_selection import train_test_split


def train():
    diamonds = sns.load_dataset("diamonds")

    X, y = diamonds.drop('price', axis=1), diamonds[['price']]

    X_train, X_test, y_train, y_test = train_test_split(X, y)

    dmatrix_train = xgb.DMatrix(X_train, y_train, enable_categorical=True)
    dmatrix_test = xgb.DMatrix(X_test, y_test, enable_categorical=True)

    evals = [
        (dmatrix_train, "train"),
        (dmatrix_test, "validation")
    ]

    params = {"objective": "reg:squarederror", "tree_method": "gpu_hist"}
    n = 1000
    early_stopping_rounds = 20
    nfold = 5

    results = xgb.cv(
        params=params,
        dtrain=dmatrix_train,
        num_boost_round=n,
        nfold=nfold,
        early_stopping_rounds=early_stopping_rounds
    )

    best_rmse = results['test-rmse-mean'].min()
    print(f'Best rmse : {best_rmse}')

    best_num_boost_round = len(results)
    print(f'Best num_boost_round : {len(results)}')

    final_model = xgb.train(
        params,
        dmatrix_train,
        num_boost_round=best_num_boost_round,
        evals=evals,
        verbose_eval=10,
        early_stopping_rounds=50,
    )

    from sklearn.metrics import mean_squared_error

    preds = final_model.predict(dmatrix_test)
    rmse = mean_squared_error(y_test, preds, squared=False)

    print(f"RMSE of the base model: {rmse:.3f}")

    final_model.save_model('final_model.model')
    print("XGBoost model saved successfully.")

    # loaded_model = xgb.Booster()
    # loaded_model.load_model('model_new_hyper.model')


def main():
    train()


if __name__ == '__main__':
    main()
