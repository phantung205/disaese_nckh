const toggle = document.getElementById("chat-toggle");
const container = document.getElementById("chat-container");
const close = document.getElementById("chat-close");

const sendBtn = document.getElementById("send-btn");
const input = document.getElementById("chat-input");
const messages = document.getElementById("chat-messages");
const clearBtn = document.getElementById("clear-chat");


// =========================
// Mở chat
// =========================
toggle.onclick = function () {
    container.style.display = "flex";
}

// =========================
// Đóng chat
// =========================
close.onclick = function () {
    container.style.display = "none";
}


// =========================
// xóa cuộc trò chuyện cũ
// =========================
clearBtn.onclick = function(){

    if(confirm("Bạn có muốn xóa toàn bộ cuộc trò chuyện?")){

        showWelcomeMessage();
    }

}


// =========================
// Gửi câu hỏi
// =========================
sendBtn.onclick = async function () {

    const question = input.value.trim();

    if (question === "") return;

    messages.innerHTML += `
        <div style="text-align:right;margin:10px">
            <b>Bạn:</b><br>${question}
        </div>
    `;

    input.value = "";

    const response = await fetch("/api/chat", {

        method: "POST",

        headers: {
            "Content-Type": "application/json"
        },

        body: JSON.stringify({
            question: question
        })

    });

    const data = await response.json();

    messages.innerHTML += `
        <div style="margin:10px">
            <b>Bot:</b><br>${data.answer}
        </div>
    `;
    messages.scrollTop = messages.scrollHeight;

}


// hàm hiển thị lời chào
function showWelcomeMessage() {
    messages.innerHTML = `
        <div style="margin:10px">
            <b>Bot:</b><br>
            Xin chào! Tôi có thể giúp gì cho bạn?
        </div>
    `;
}

showWelcomeMessage();