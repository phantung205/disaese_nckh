from flask import Blueprint
from flask import request
from flask import jsonify

from services.chat.chat_service import ask

chat_bp = Blueprint("chat", __name__)


@chat_bp.route("/api/chat", methods=["POST"])
def api_chat():

    question = request.form.get("question", "")

    image = request.files.get("image")

    answer = ask(
        question=question,
        image=image
    )

    return jsonify({
        "answer": answer
    })