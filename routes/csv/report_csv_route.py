from flask import Blueprint
from flask import render_template
from services.csv import report_service

report_csv_bp = Blueprint("report_csv",__name__)

@report_csv_bp.route("/report_csv_model")
def show_reports():

    result = report_service.get_report_names()

    return render_template("report_csv.html",reports=result)