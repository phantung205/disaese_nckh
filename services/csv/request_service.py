def get_form_data(form):
    data = {
        "age": float(form["age"]),
        "bmi": float(form["bmi"]),
        "HbA1c_level": float(form["HbA1c_level"]),
        "blood_glucose_level": float(form["blood_glucose_level"]),
        "hypertension": int(form["hypertension"]),
        "heart_disease": int(form["heart_disease"]),
        "gender":form["gender"],
        "smoking_history": form["smoking_history"]
    }

    return data

def get_form_model_name(form):
    model_name = form.get("model_name","logistic")
    return model_name


def get_form_file(files):
    file = files["file"]
    return file

