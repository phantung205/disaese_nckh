from flask import Blueprint,request ,render_template,send_from_directory
import os
from configs import paths_common
from services.image import inference_service,validation_service,image_service

image_bp = Blueprint("predict_image",__name__)

# tạo thư mục lưu file người dùng upload
upload = paths_common.dir_uploads



@image_bp.route("/image",methods=["GET","POST"])
def home():
    try:
        if request.method == "GET":
            return render_template("predict_image.html", result=None, image_name=None, error=None)

        elif request.method == "POST":
            # lấy dữ liệu từ form
            image = request.files["image"]

            #validation
            validation_service.validation_image(image)

            # save upload
            path_upload,new_filename = image_service.save_upload_image(image)

            #predict_image
            result = inference_service.predict_image(path_upload)

            return render_template(
                "predict_image.html",
                result=result,
                image_name=new_filename,
                error=None
            )
    except Exception as e:
        return render_template(
            "predict_image.html",
            result=None,
            image_name=None,
            error=str(e)
        )

@image_bp.route("/uploads/<filename>")
def uploaded_image(filename):
    return send_from_directory(upload,filename)