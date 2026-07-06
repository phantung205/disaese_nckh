from flask import Flask
from routes.csv.csv_route import csv_bp
from routes.csv.report_csv_route import report_csv_bp
from routes.image.image_route import image_bp
from routes.chat.chat_route import chat_bp
from configs import paths_common
import os
from services.image.inference_service import load_model


load_model()   # preload model

os.makedirs(paths_common.dir_uploads, exist_ok=True)
os.makedirs(paths_common.dir_results, exist_ok=True)

app = Flask(__name__)

app.register_blueprint(csv_bp)

app.register_blueprint(report_csv_bp)

app.register_blueprint(image_bp)

app.register_blueprint(chat_bp)

if __name__ == '__main__':
    app.run(
        host="0.0.0.0",
        port=5000,
        debug=True
    )