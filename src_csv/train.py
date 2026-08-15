import os
import joblib
from sklearn.compose import  ColumnTransformer
from src_csv import preprocessing
from configs import config_csv
import argparse
from sklearn.impute import SimpleImputer
from sklearn.preprocessing import StandardScaler,OrdinalEncoder,OneHotEncoder
from sklearn.linear_model import LogisticRegression
from sklearn.ensemble import RandomForestClassifier
from sklearn.svm import SVC
from sklearn.pipeline import Pipeline
from sklearn.metrics import classification_report,roc_auc_score,recall_score
from xgboost import XGBClassifier
from lightgbm import LGBMClassifier
# from imblearn.over_sampling import SMOTE


def parse_args():
    p = argparse.ArgumentParser(description="Train a classification model for diabetes")
    # common
    p.add_argument("--random_state","-r",type=int,default=config_csv.random_state)
    p.add_argument("--test_size","-t",type=float,default=config_csv.test_size)
    p.add_argument("--model_name","-m",type=str,default="logistic")

    # RandomForest
    p.add_argument("--n_estimators","-n",type=int,default=200)
    p.add_argument("--criterion","-rf_c",type=str,default="gini")
    p.add_argument("--max_depth","-md",type=int,default=10)
    p.add_argument("--min_samples_split", type=int, default=2)
    p.add_argument("--min_samples_leaf", type=int, default=1)
    p.add_argument("--max_features", type=str, default="sqrt")
    p.add_argument("--bootstrap", type=bool, default=True)

    #  Logistic
    p.add_argument("--c_logis","-c",type=float,default=0.001)
    p.add_argument("--max_iter","-i",type=int,default=500)

    # LightGBM
    p.add_argument("--lgbm_n_estimators",type=int,default=200)
    p.add_argument("--lgbm_learning_rate",type=float,default=0.03)
    p.add_argument("--lgbm_max_depth",type=int,default=10)
    p.add_argument("--lgbm_num_leaves",type=int,default=31)


    #  XGBoost
    p.add_argument("--xgb_n_estimators","-xn",type=int,default=300)
    p.add_argument("--xgb_learning_rate","-lr",type=float,default=0.03)
    p.add_argument("--xgb_max_depth","-xmd",type=int,default=5)
    p.add_argument("--xgb_subsample",type=float,default=1.0)
    p.add_argument("--xgb_colsample_bytree",type=float,default=1.0)
    p.add_argument("--xgb_min_child_weight",type=int,default=1)
    p.add_argument("--xgb_gamma",type=float,default=0.1)
    p.add_argument("--xgb_reg_alpha",type=float,default=0.1)
    p.add_argument("--xgb_reg_lambda",type=float,default=1)
    p.add_argument("--xgb_scale_pos_weight",type=float,default=5)

    return p.parse_args()

def build_model(args):


    if args.model_name == "random_forest":
        clf = RandomForestClassifier(
            n_estimators=args.n_estimators,
            criterion=args.criterion,
            max_depth=args.max_depth,
            min_samples_split=args.min_samples_split,
            min_samples_leaf=args.min_samples_leaf,
            max_features=args.max_features,
            bootstrap=args.bootstrap,
            class_weight={0:1, 1: 4.5},
            random_state=args.random_state,
            n_jobs=-1
        )

    elif args.model_name == "logistic":
        clf = LogisticRegression(
            C=args.c_logis,
            penalty="l2",
            solver="lbfgs",
            max_iter=args.max_iter,
            class_weight={0:1, 1: 5},
            random_state=args.random_state
        )



    elif args.model_name == "lightgbm":
        clf = LGBMClassifier(
            n_estimators=args.lgbm_n_estimators,
            learning_rate=args.lgbm_learning_rate,
            max_depth=args.lgbm_max_depth,
            num_leaves=args.lgbm_num_leaves,
            class_weight=  {0:1, 1: 4.5},
            random_state=args.random_state,
            n_jobs=-1,
            verbosity=-1
        )


    elif args.model_name == "xgboost":
        clf = XGBClassifier(
            n_estimators=args.xgb_n_estimators,
            learning_rate=args.xgb_learning_rate,
            max_depth=args.xgb_max_depth,
            subsample=args.xgb_subsample,
            colsample_bytree=args.xgb_colsample_bytree,
            min_child_weight=args.xgb_min_child_weight,
            gamma=args.xgb_gamma,
            reg_alpha=args.xgb_reg_alpha,
            reg_lambda=args.xgb_reg_lambda,
            scale_pos_weight=args.xgb_scale_pos_weight,
            random_state=args.random_state,
            eval_metric="logloss",
            n_jobs=-1
        )
    else:
        raise ValueError("Model not supported")

    return clf

def main(args):
    # lấy ra dữ liệu đã chia
    x_train, x_test, y_train, y_test = preprocessing.preprocess_and_split(args.test_size,args.random_state)

    #  dùng smote để cân bằng data
    # smote = SMOTE(
    #     sampling_strategy="auto",
    #     k_neighbors=5,
    #     random_state=42
    # )
    # x_train,y_train = smote.fit_resample(x_train,y_train)

    # tạo pipeline chuẩn hóa
    num_transformer = Pipeline([
        ("imputer", SimpleImputer(strategy="mean")),
        ("scaler", StandardScaler())
    ])
    cat_transformer = Pipeline([
        ("imputer", SimpleImputer(strategy="most_frequent")),
        ("onehot", OneHotEncoder(handle_unknown="ignore"))
    ])

    # thực hiện chuẩn hóa
    preprocessor = ColumnTransformer([
        ("num_feature", num_transformer, config_csv.numeric_features),
        ("cat_feature", cat_transformer, config_csv.categorical_features),
        ("binary", "passthrough", config_csv.binary_features)
    ])

    # tạo model pipline
    clf = build_model(args)
    pipeline = Pipeline([
        ("preprocessor", preprocessor),
        ("clf", clf),
    ])

    # fit dữ liệu vào
    pipeline.fit(x_train, y_train)

    # test model
    y_proba = pipeline.predict_proba(x_test)[:, 1]
    threshold = 0.45  # ưu tiên recall
    y_predict = (y_proba >= threshold).astype(int)

    roc_auc = roc_auc_score(y_test, y_proba)
    recall = recall_score(y_test, y_predict)

    # print result
    print(classification_report(y_test, y_predict))
    print(f"ROC-AUC: {roc_auc:.4f}")
    print(f"Recall: {recall:.4f}")

    # lưu model kết quả đánh giá lại
    if not os.path.isdir(config_csv.dir_evaluate):
        os.makedirs(config_csv.dir_evaluate)
    path_result = os.path.join(config_csv.dir_evaluate,f"train_report_{args.model_name}.txt")
    with open(path_result,"w") as f:
        f.write(f"Model: {args.model_name.replace('_', ' ').title()}\n\n")
        f.write(classification_report(y_test, y_predict))
        f.write(f"\nROC-AUC: {roc_auc:.4f}\n")
        f.write(f"\nRecall: {recall:.4f}\n")

    # lưu model
    if not os.path.isdir(config_csv.dir_csv_model):
        os.makedirs(config_csv.dir_csv_model)
    model_file = f"{args.model_name}.pkl"
    model_path = os.path.join(config_csv.dir_csv_model,model_file)
    joblib.dump(pipeline, model_path)
    print("save model successfull")


if __name__ == '__main__':
    args = parse_args()
    main(args)


