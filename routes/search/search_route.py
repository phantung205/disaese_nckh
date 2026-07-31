from flask import Blueprint,render_template,request,send_from_directory
from services.search import search_service,load_model_search,validation_service
from configs import paths_common,config_image

model = load_model_search.load_model_search()
upload = paths_common.dir_uploads


search_bp = Blueprint("search_image",__name__)


@search_bp.route("/search",methods=["GET","POST"])
def search():
    try:
        if request.method == "GET":
            return render_template("search_image.html", error=None, image_results=None, top_k=None,query_image=None)

        elif request.method == "POST":
            image = request.files.get("image_search")
            top_k = int(request.form.get("top_k", 5))

            # validation
            validation_service.validate_search(image,top_k)


            results,query_image = search_service.search_image(file=image, model=model, k_top=top_k)

            return render_template("search_image.html", error=None, image_results=results, top_k=top_k,query_image=query_image)

    except Exception as e:
        return render_template("search_image.html", error=str(e), image_results=None, top_k=None, query_image=None)


@search_bp.route("/dataset/<path:filename>")
def dataset_image(filename):
    return send_from_directory(
        config_image.dir_image_search,filename)

@search_bp.route("/uploads/<filename>")
def uploaded_image(filename):
    return send_from_directory(upload,filename)
