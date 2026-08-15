from sklearn.model_selection import train_test_split
import pandas as pd
from configs import config_csv
import os
import numpy as np


def load_data():
    return pd.read_csv(config_csv.path_csv_raw)

def  clean_raw_data(df, is_train=True):
    df = df.copy()

    #  xóa các cột có dữ liệu trung lặp
    if is_train:
        df = df.drop_duplicates()

    # sử lý các giá trị cột ko thể nhỏ hơn 0
    # if "HbA1c_level" in df.columns:
    #     df = df[df["HbA1c_level"] > 0]
    # if "blood_glucose_level" in df.columns:
    #     df = df[df["blood_glucose_level"] > 0]

    # xóa bỏ dữ liệu ko hợp lý
    if "smoking_history" in df.columns:
        df = df[df['smoking_history'] != 'No Info']

    # xóa các cột ko cẩn thiết
    # if "..." in df.columns:
    #     df = df.drop("...", axis=1)

    # Lọc age không hợp lệ
    if "age" in df.columns:
        df = df[(df["age"] >= 0) & (df["age"] <= 120)]

    # xóa các giá trị ko hợp lệ
    if "hypertension" in df.columns:
        df = df[df["hypertension"].isin([0, 1])]
    if "heart_disease" in df.columns:
        df = df[df["heart_disease"].isin([0, 1])]


    # dữ lại các cột cần
    if is_train:
        required_cols = (
            config_csv.numeric_features +
            config_csv.categorical_features +
            config_csv.binary_features +
            config_csv.target_col
        )
        df = df[required_cols]
    else :
        required_cols = (
                config_csv.numeric_features +
                config_csv.categorical_features +
                config_csv.binary_features
        )
        df = df[required_cols]

    return df

def preprocess_and_split(test_size=None,random_state=None):
    processed_dir = config_csv.dir_csv_processed

    if test_size is None:
        test_size = config_csv.test_size
    if random_state is None:
        random_state = config_csv.random_state

    # load data
    df = load_data()

    # clear data
    df = clean_raw_data(df,True)

    # split targit , sample
    x = df.drop(config_csv.target_col,axis=1)
    y = df[config_csv.target_col]

    # train , test split
    x_train, x_test, y_train, y_test = train_test_split(x,y,test_size=test_size,random_state=random_state,stratify=y)

    os.makedirs(processed_dir, exist_ok=True)
    x_train.to_csv(os.path.join(processed_dir, "x_train.csv"), index=False)
    x_test.to_csv(os.path.join(processed_dir, "x_test.csv"), index=False)
    y_train.to_csv(os.path.join(processed_dir, "y_train.csv"), index=False)
    y_test.to_csv(os.path.join(processed_dir, "y_test.csv"), index=False)

    return x_train, x_test, y_train, y_test



if __name__ == '__main__':
    x_train, x_test, y_train, y_test = preprocess_and_split()
    print(x_train.head(2))
    print(x_test.head(2))
    print(y_train.head(2))
    print(y_test.head(2))