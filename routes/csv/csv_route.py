from flask import Blueprint,render_template,request,send_file
from services.csv import validation_service, inference_service,request_service,file_service


csv_bp = Blueprint("predict_csv",__name__)


@csv_bp.route("/")
def home():
    return render_template(
        "index.html",
        data=None,
        error=None,
        prediction=None,
        proba_dict=None,
        selected_model="logistic",
        output_save=None
    )

@csv_bp.route("/predict_csv_input", methods=["POST"])
def predict_form():
    try:
        model_name = request_service.get_form_model_name(request.form)

        # lấy dữ liệu người dùng nhập vào và tên model
        data = request_service.get_form_data(request.form)

        # validation dữ liệu
        validation_service.validate_input(data)

        # dự đoán
        prediction, proba_dict = inference_service.predict_dict(data,model_name)

        return render_template("index.html",data=data,prediction=prediction,proba_dict=proba_dict,selected_model=model_name,error=None,output_save=None)

    except Exception as e:
        model_name = request_service.get_form_model_name(request.form)
        return render_template("index.html",data=request.form,error=str(e),prediction=None,proba_dict=None,selected_model=model_name,output_save=None)


@csv_bp.route("/predict_csv_file", methods=["POST"])
def predict_file():
    try:
        # lấy tên model
        model_name = request_service.get_form_model_name(request.form)

        # lấy ra tên file upload
        file = request_service.get_form_file(request.files)

        # validation
        validation_service.validate_file(file)

        #lưu file người dùng upload
        input_path, name, timestamp = file_service.save_upload_file(file)

        # dự đoán
        df_result = inference_service.predict_file(input_path,model_name)

        # lưu file kết quả
        output_filename = file_service.save_prediction_result(df_result, name, timestamp)

        return render_template("index.html",data=None,error=None,prediction=None,proba_dict=None,selected_model=model_name,output_save=output_filename)
    except Exception  as e:
        model_name = request_service.get_form_model_name(request.form)
        return render_template("index.html", data=None, error=str(e), prediction=None, proba_dict=None,selected_model=model_name,output_save=None)

@csv_bp.route("/download/<filename>")
def download_file(filename):
    file_path = file_service.get_download_path(filename)
    return send_file(
        file_path,
        as_attachment=True
    )