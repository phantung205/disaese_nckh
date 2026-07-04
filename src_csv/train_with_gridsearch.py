from sklearn.model_selection import GridSearchCV
import os
from src_csv import preprocessing
from configs import config_csv
import argparse
from sklearn.linear_model import LogisticRegression
from sklearn.ensemble import RandomForestClassifier
from sklearn.svm import SVC
from xgboost import XGBClassifier
from sklearn.pipeline import Pipeline
from sklearn.impute import SimpleImputer
from sklearn.preprocessing import StandardScaler


def get_args():
    parser = argparse.ArgumentParser(description="best parameter")
    parser.add_argument("--random_state","-r",type=int,default=config_csv.random_state, help="random state")
    parser.add_argument("--model_name","-m",type=str,default="xgboost", help="choice model")
    args = parser.parse_args()
    return args

def parameter(args):
    if args.model_name == "random_forest":
        clf = Pipeline([
            ("imputer", SimpleImputer(strategy="mean")),
            ("classifier", RandomForestClassifier(
                random_state=args.random_state
            ))
        ])

        param_grid = {
            "imputer__strategy": ["mean", "median"],
            "classifier__n_estimators": [100, 200, 300],
            "classifier__criterion": ["gini", "entropy"],
            "classifier__max_depth": [None, 5, 10],
        }

    elif args.model_name == "logistic":
        clf = Pipeline([
            ("imputer", SimpleImputer(strategy="mean")),
            ("scaler", StandardScaler()),
            ("classifier", LogisticRegression(random_state=args.random_state,solver="lbfgs"))
        ])

        param_grid = {
            "imputer__strategy": ["mean", "median"],
            "classifier__C": [0.01, 0.1, 1, 10],
            "classifier__max_iter": [500, 1000,2000],
        }

    elif args.model_name == "svm":
        clf = Pipeline([
            ("imputer", SimpleImputer(strategy="mean")),
            ("scaler", StandardScaler()),
            ("classifier", SVC(random_state=args.random_state))
        ])
        param_grid = {
            "imputer__strategy": ["mean", "median"],
            "classifier__C": [0.1, 1, 10],
            "classifier__kernel": ["rbf", "linear"],
            "classifier__gamma": ["scale", "auto"]
        }

    elif args.model_name == "xgboost":
        clf = Pipeline([
            ("imputer", SimpleImputer(strategy="mean")),
            ("classifier", XGBClassifier(
                random_state=args.random_state,
                eval_metric="logloss"
            ))
        ])
        param_grid = {
            "imputer__strategy": ["mean", "median"],
            "classifier__n_estimators": [200, 300, 500],
            "classifier__learning_rate": [0.03, 0.05, 0.1],
            "classifier__max_depth": [3, 4, 5],
            "classifier__subsample": [0.8, 1.0],
            "classifier__colsample_bytree": [0.8, 1.0],
        }

    else:
        raise ValueError(
            f"Model '{args.model_name}' is not supported. "
            "Choose from: random_forest, logistic, svm"
        )
    return clf,param_grid

def main():
    args = get_args()

    # load data
    x_train, x_test, y_train, y_test = preprocessing.preprocess_and_split()

    model,param_grid = parameter(args)

    grid_search = GridSearchCV(estimator=model,param_grid=param_grid,cv=4,scoring="recall",verbose=2)
    grid_search.fit(x_train, y_train)

    best_params = grid_search.best_params_
    best_score = grid_search.best_score_

    print("Best Params:")
    print(best_params)
    print(f"Best Recall: {best_score:.4f}")

    os.makedirs(config_csv.dir_parameter_optimal, exist_ok=True)
    save_path = os.path.join(config_csv.dir_parameter_optimal, f"best_params_{args.model_name}.txt")

    # ghi file
    with open(save_path, "w", encoding="utf-8") as f:
        f.write("Best Parameters:\n")
        for key, value in best_params.items():
            f.write(f"{key}: {value}\n")
        f.write(f"\nBest Recall: {best_score:.4f}\n")

    print(f"Saved best parameters to: {save_path}")


if __name__ == '__main__':
    main()

