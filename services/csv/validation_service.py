import os

def validate_input(data):
    # Numerical features
    if data["age"] <= 0:
        raise ValueError("Tuổi phải lớn hơn 0")

    if data["bmi"] <= 0:
        raise ValueError("BMI phải lớn hơn 0")

    if data["HbA1c_level"] <= 0:
        raise ValueError("HbA1c_level phải lớn hơn 0")

    if data["blood_glucose_level"] <= 0:
        raise ValueError("Blood glucose level phải lớn hơn 0")

    # Binary features
    if data["hypertension"] not in [0, 1]:
        raise ValueError("Hypertension chỉ được phép là 0 hoặc 1")

    if data["heart_disease"] not in [0, 1]:
        raise ValueError("Heart disease chỉ được phép là 0 hoặc 1")

    # Categorical features
    if data["gender"] not in ["Male","Female","Other"]:
        raise ValueError("Gender phải là Male, Female hoặc Other")

    if data["smoking_history"] not in ["never","former","current", "not current","ever"]:
        raise ValueError("Smoking history không hợp lệ")


def validate_file(file):
    filename = file.filename

    # check xem đã chọn file chưa
    if filename == "":
        raise ValueError("Chưa chọn file")

    # check đuôi đúng định dạng chưa
    ext = os.path.splitext(filename)[1].lower()
    allowed_ext = {".csv", ".xlsx", ".xls"}
    if ext not in allowed_ext:
        raise ValueError("Chỉ hỗ trợ file CSV hoặc Excel")