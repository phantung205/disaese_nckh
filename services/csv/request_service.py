def get_form_data(form):
    data = {
        "Pregnancies": int(form["Pregnancies"]),
        "Glucose": float(form["Glucose"]),
        "BloodPressure": float(form["BloodPressure"]),
        "SkinThickness": float(form["SkinThickness"]),
        "Insulin": float(form["Insulin"]),
        "BMI": float(form["BMI"]),
        "DiabetesPedigreeFunction": float(form["DiabetesPedigreeFunction"]),
        "Age": int(form["Age"])
    }

    return data

def get_form_model_name(form):
    model_name = form.get("model_name","logistic")
    return model_name


def get_form_file(files):
    file = files["file"]
    return file

