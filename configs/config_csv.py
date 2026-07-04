import os
from configs import paths_common

"""data"""
# path data raw csv
path_csv_raw = os.path.join(paths_common.dir_data_raw,"csv","diabetes.csv")

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
target_col = [
    "Outcome"]
numerical_col = ["Pregnancies","Glucose","BloodPressure","SkinThickness","Insulin","BMI","DiabetesPedigreeFunction","Age"]

# test size và random-state
test_size = 0.10
random_state = 42

# directory save model
dir_csv_model = os.path.join(paths_common.dir_model,"csv")