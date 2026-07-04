import argparse
import os
import joblib
from src_csv import preprocessing
from sklearn.metrics import classification_report, roc_auc_score
from configs import config_csv

def parse_args():
    p = argparse.ArgumentParser(description="test  model ")
    p.add_argument("--model_name","-m",type=str,default="svm",help="Model name to test")

    return p.parse_args()

def main(args):
    # lấy ra dữ liệu file test
    _, x_test, _, y_test = preprocessing.preprocess_and_split()

    # laod model
    model_path = os.path.join(config_csv.dir_csv_model,f"{args.model_name}.pkl")
    if not os.path.isfile(model_path):
        print("You need to train the model to have checkpoints before testing.")
        exit(0)
    model = joblib.load(model_path)

    y_pred = model.predict(x_test)
    y_proba = model.predict_proba(x_test)[:,1]
    roc_auc = roc_auc_score(y_test,y_proba)

    print("\n Classification Report:")
    print(classification_report(y_test, y_pred))

    print(f"ROC-AUC: {roc_auc:.4f}")

if __name__ == '__main__':
    args = parse_args()
    main(args)