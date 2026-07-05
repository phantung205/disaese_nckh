import os

def validate_input(data):
    if data["Age"] <= 0:
        raise ValueError("Tuổi phải lớn hơn 0")
    if data["BMI"] <= 0:
        raise ValueError("BMI phải lớn hơn 0")
    if data["Glucose"] <= 0:
        raise ValueError("Glucose phải lớn hơn 0")
    if data["Pregnancies"] < 0:
        raise ValueError("Pregnancies phải lớn hơn 0")
    if data["BloodPressure"] <= 0:
        raise ValueError("BloodPressure phải lớn hơn 0")
    if data["SkinThickness"] <= 0:
        raise ValueError("SkinThickness phải lớn hơn 0")
    if data["DiabetesPedigreeFunction"] < 0:
        raise ValueError("DiabetesPedigreeFunction phải lớn hơn 0")
    if data["Insulin"] <= 0:
        raise ValueError("Insulin phải lớn hơn 0")


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