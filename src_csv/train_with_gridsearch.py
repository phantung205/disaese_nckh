from sklearn.model_selection import GridSearchCV
import os
from src_csv import preprocessing
from configs import config_csv
import argparse
from sklearn.linear_model import LogisticRegression
from sklearn.ensemble import RandomForestClassifier
from xgboost import XGBClassifier
from sklearn.pipeline import Pipeline
from sklearn.impute import SimpleImputer
from sklearn.preprocessing import StandardScaler
from sklearn.model_selection import StratifiedKFold
from sklearn.preprocessing import OneHotEncoder
from sklearn.compose import ColumnTransformer
from lightgbm import LGBMClassifier


def get_args():
    parser = argparse.ArgumentParser(description="best parameter")
    parser.add_argument("--random_state","-r",type=int,default=config_csv.random_state, help="random state")
    parser.add_argument("--model_name","-m",type=str,default="lightgbm", help="choice model")
    args = parser.parse_args()
    return args

def parameter(args):
    if args.model_name == "random_forest":
        clf = RandomForestClassifier(random_state=args.random_state,class_weight ="balanced")

        param_grid = {
            "classifier__n_estimators": [200, 500],
            "classifier__criterion": ["gini"],
            "classifier__max_depth": [10, 20, None],
            "classifier__min_samples_split": [2, 5],
            "classifier__min_samples_leaf": [1, 2],
            "classifier__max_features": ["sqrt"],
            "classifier__bootstrap": [True]
        }

    elif args.model_name == "logistic":
        clf = LogisticRegression(random_state=args.random_state,solver="lbfgs",class_weight ="balanced")

        param_grid = {
            "classifier__C": [0.001,0.01,0.1,1,10,100],
            "classifier__penalty": ["l2"],
            "classifier__max_iter": [500,1000,2000],
        }

    elif args.model_name == "lightgbm":
        clf = LGBMClassifier(random_state=42,class_weight = "balanced")

        param_grid = {
            "classifier__n_estimators": [200, 500],
            "classifier__learning_rate": [0.03, 0.05],
            "classifier__max_depth": [-1, 10],
            "classifier__num_leaves": [31, 63],
        }

    elif args.model_name == "xgboost":
        clf = XGBClassifier(random_state=args.random_state,eval_metric="logloss")

        param_grid = {
            "classifier__n_estimators": [300, 500],
            "classifier__learning_rate": [0.03, 0.05],
            "classifier__max_depth": [3, 5],
            "classifier__subsample": [0.8, 1.0],
            "classifier__colsample_bytree": [0.8, 1.0],
            "classifier__min_child_weight": [1, 3],
            "classifier__gamma": [0, 0.1],
            "classifier__scale_pos_weight": [1, 10.34],
            "classifier__reg_alpha": [0, 0.1],
            "classifier__reg_lambda": [1, 3]
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

    num_transformer = Pipeline([
        ("imputer", SimpleImputer(strategy="mean")),
        ("scaler", StandardScaler())
    ])

    cat_transformer = Pipeline([
        ("imputer", SimpleImputer(strategy="most_frequent")),
        ("onehot", OneHotEncoder(handle_unknown="ignore"))
    ])

    preprocessor = ColumnTransformer([
        ("num_feature", num_transformer, config_csv.numeric_features),
        ("cat_feature", cat_transformer, config_csv.categorical_features),
        ("binary", "passthrough", config_csv.binary_features)
    ])

    model,param_grid = parameter(args)

    pipeline = Pipeline([
        ("preprocessor", preprocessor),
        ("classifier", model)
    ])

    cv = StratifiedKFold(
        n_splits=5,
        shuffle=True,
        random_state=args.random_state
    )

    grid_search = GridSearchCV(
        estimator=pipeline,
        param_grid=param_grid,
        cv=cv,
        scoring="recall",
        n_jobs=-1,
        verbose=2
    )
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

