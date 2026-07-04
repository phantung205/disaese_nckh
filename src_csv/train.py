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
# from imblearn.over_sampling import SMOTE


def parse_args():
    p = argparse.ArgumentParser(description="Train a classification model for diabetes")
    # common
    p.add_argument("--random_state","-r",type=int,default=config_csv.random_state)
    p.add_argument("--test_size","-t",type=float,default=config_csv.test_size)
    p.add_argument("--model_name","-m",type=str,default="xgboost")

    # RandomForest
    p.add_argument("--n_estimators","-n",type=int,default=300)
    p.add_argument("--criterion","-rf_c",type=str,default="gini")
    p.add_argument("--max_depth","-md",type=int,default=None)

    #  Logistic
    p.add_argument("--c_logis","-c",type=float,default=1)
    p.add_argument("--max_iter","-i",type=int,default=500)

    #  SVM
    p.add_argument("--kernel","-k",type=str,default="linear")
    p.add_argument("--svm_c","-sc",type=float,default=0.1)
    p.add_argument("--gamma","-g",type=str,default="scale")

    #  XGBoost
    p.add_argument("--xgb_n_estimators","-xn",type=int,default=200)
    p.add_argument("--learning_rate","-lr",type=float,default=0.03)
    p.add_argument("--xgb_max_depth","-xmd",type=int,default=3)
    p.add_argument("--scale_pos_weight","-spw",type=float,default=1.5)

    #  class weight (cho imbalance)
    p.add_argument("--class_weight_0", type=float, default=1.0)
    p.add_argument("--class_weight_1", type=float, default=1.5)

    return p.parse_args()

def build_model(args):

    class_weight = {
        0: args.class_weight_0,
        1: args.class_weight_1
    }

    if args.model_name == "random_forest":
        clf = RandomForestClassifier(
            n_estimators=args.n_estimators,
            criterion=args.criterion,
            max_depth=args.max_depth,
            random_state=args.random_state,
            class_weight=class_weight
        )

    elif args.model_name == "logistic":
        clf = LogisticRegression(
            C=args.c_logis,
            penalty="l2",
            solver="lbfgs",
            max_iter=args.max_iter,
            class_weight=class_weight
        )

    elif args.model_name == "svm":
        clf = SVC(
            kernel=args.kernel,
            C=args.svm_c,
            gamma=args.gamma,
            class_weight=class_weight,
            probability=True
        )

    elif args.model_name == "xgboost":
        clf = XGBClassifier(
            n_estimators=args.xgb_n_estimators,
            learning_rate=args.learning_rate,
            max_depth=args.xgb_max_depth,
            scale_pos_weight=args.scale_pos_weight,
            random_state=args.random_state
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
    num_transformer = Pipeline(steps=[
        ("imputer",SimpleImputer(strategy="median")),
        ("scaler",StandardScaler())
    ])

    # thực hiện chuẩn hóa
    preprocessor = ColumnTransformer(transformers=[
        ("num_feature",num_transformer,config_csv.numerical_col)
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
    threshold = 0.4  # ưu tiên recall
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


