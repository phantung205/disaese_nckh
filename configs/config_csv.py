import os
from configs import paths_common

"""data"""
# path data raw csv
path_csv_raw = os.path.join(paths_common.dir_data_raw,"csv","diabetes_prediction_dataset.csv")

# directory data processed csv
dir_csv_processed = os.path.join(paths_common.dir_data_processed,"csv")


"""reports"""
# directory reports csv
dir_csv_report = os.path.join(paths_common.dir_reports,"csv")

# directory statistic data
dir_csv_eda = os.path.join(dir_csv_report,"eda")
file_name_report = "report_diabetes.html"

# directory Evaluate model
dir_evaluate = os.path.join(dir_csv_report,"evaluate")

# directory parameter optimal
dir_parameter_optimal = os.path.join(dir_csv_report,"parameter_optimal")


"""columns,parameter,model"""
# required columns
target_col = ["diabetes"]
numeric_features = ["age",
    "bmi",
    "HbA1c_level",
    "blood_glucose_level"
]

binary_features = [
    "hypertension",
    "heart_disease"
]

categorical_features = [
    "gender",
    "smoking_history"
]

# test size và random-state
test_size = 0.2
random_state = 42


# directory save model
dir_csv_model = os.path.join(paths_common.dir_model,"csv")


""" config neron"""
batch_size = 256
epochs = 100
learning_rate = 0.001
weight_decay = 1e-4
# dir tensorboader
path_tensorboard = os.path.join(dir_csv_report,"tensorboard")
path_best_model = os.path.join(dir_csv_model, "best_nn.pt")

