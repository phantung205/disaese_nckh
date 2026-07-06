from flask import Blueprint
from flask import request
from flask import jsonify

from services.chat.chat_service import ask

chat_bp = Blueprint("chat", __name__)


@chat_bp.route("/api/chat", methods=["POST"])
def api_chat():

    data = request.get_json()

    question = data.get("question", "")

    answer = ask(question)

    return jsonify({
        "answer": answer
    })