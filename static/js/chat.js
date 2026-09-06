const toggle = document.getElementById("chat-toggle");
const container = document.getElementById("chat-container");
const close = document.getElementById("chat-close");
const sendBtn = document.getElementById("send-btn");
const input = document.getElementById("chat-input");
const messages = document.getElementById("chat-messages");
const clearBtn = document.getElementById("clear-chat");
const imageBtn = document.getElementById("image-btn");
const imageInput = document.getElementById("chat-image");
const selectedImagePreview = document.getElementById("selected-image-preview");

// MỞ CHAT
toggle.onclick = function () {
    container.style.display = "flex";
    input.focus();
};

toggle.addEventListener("keydown", function (event) {
    if (event.key === "Enter" || event.key === " ") {
        event.preventDefault();
        toggle.click();
    }
});

// ĐÓNG CHAT
close.onclick = function () {
    container.style.display = "none";
};

// NÚT CHỌN ẢNH
imageBtn.onclick = function () {
    imageInput.click();
};

// KIỂM TRA FILE ẢNH
imageInput.onchange = function () {
    const image = imageInput.files[0];
    if (!image) return;

    const allowedTypes = ["image/png", "image/jpeg", "image/webp"];
    if (!allowedTypes.includes(image.type)) {
        alert("Chỉ hỗ trợ ảnh PNG, JPG hoặc WEBP.");
        removeSelectedImage();
        return;
    }

    const maxSize = 10 * 1024 * 1024;
    if (image.size > maxSize) {
        alert("Ảnh không được lớn hơn 10MB.");
        removeSelectedImage();
        return;
    }
    showSelectedImage(image);
};

// XÓA CUỘC TRÒ CHUYỆN
clearBtn.onclick = function () {
    if (confirm("Bạn có muốn xóa toàn bộ cuộc trò chuyện?")) {
        showWelcomeMessage();
    }
};

// GỬI MESSAGE
async function sendMessage() {
    const question = input.value.trim();
    const image = imageInput.files[0];

    // Không có text và cũng không có ảnh
    if (question === "" && !image) return;

    // HIỂN THỊ MESSAGE CỦA USER
    let userMessage = `<div class="user-message"><div class="message-bubble user-bubble"><b>Bạn:</b><br>`;
    if (question !== "") userMessage += `${escapeHtml(question)}`;
    if (image) userMessage += `<br>📷 ${escapeHtml(image.name)}`;
    userMessage += `</div></div>`;

    messages.innerHTML += userMessage;
    messages.scrollTop = messages.scrollHeight; // Cuộn xuống cuối

    // RESET INPUT VÀ LƯU ẢNH
    input.value = "";
    const selectedImage = image;

    // TẠO FORM DATA
    const formData = new FormData();
    formData.append("question", question);
    if (selectedImage) formData.append("image", selectedImage);

    // Xóa file đã chọn
    imageInput.value = "";

    // HIỂN THỊ "ĐANG TRẢ LỜI"
    const loadingId = "loading-" + Date.now();
    messages.innerHTML += `<div class="bot-message" id="${loadingId}"><div class="message-bubble bot-bubble"><b>Bot:</b><br><span>Đang phân tích...</span></div></div>`;
    messages.scrollTop = messages.scrollHeight;

    // KHÓA NÚT GỬI
    sendBtn.disabled = true;
    sendBtn.innerText = "Đang gửi...";

    try {
        // GỌI FLASK API
        const response = await fetch("/api/chat", {
            method: "POST",
            body: formData
        });

        if (!response.ok) {
            throw new Error("Server trả về lỗi HTTP " + response.status);
        }

        const data = await response.json();

        // XÓA LOADING
        const loadingElement = document.getElementById(loadingId);
        if (loadingElement) loadingElement.remove();

        // HIỂN THỊ CÂU TRẢ LỜI
        messages.innerHTML += `<div class="bot-message"><div class="message-bubble bot-bubble"><b>Bot:</b><br>${formatBotAnswer(data.answer)}</div></div>`;
        messages.scrollTop = messages.scrollHeight;

    } catch (error) {
        console.error("Chat error:", error);
        const loadingElement = document.getElementById(loadingId);
        if (loadingElement) loadingElement.remove();

        messages.innerHTML += `<div class="bot-message"><div class="message-bubble bot-bubble"><b>Bot:</b><br>Xin lỗi, hiện tại không thể kết nối với máy chủ.<br><br>Vui lòng thử lại sau.</div></div>`;
        messages.scrollTop = messages.scrollHeight;
    }

    // MỞ LẠI NÚT GỬI
    sendBtn.disabled = false;
    sendBtn.innerText = "Gửi";
}

// CLICK NÚT GỬI
sendBtn.onclick = function () {
    sendMessage();
};

// NHẤN ENTER ĐỂ GỬI
input.addEventListener("keydown", function (event) {
    if (event.key === "Enter" && !event.shiftKey) {
        event.preventDefault();
        sendMessage();
    }
});

// HIỂN THỊ LỜI CHÀO
function showWelcomeMessage() {
    messages.innerHTML = `<div class="bot-message"><div class="message-bubble bot-bubble"><b>Bot:</b><br>Xin chào! Tôi có thể giúp gì cho bạn?</div></div>`;
}

// ESCAPE HTML
function escapeHtml(text) {
    if (!text) return "";
    return text.replace(/&/g, "&amp;").replace(/</g, "&lt;").replace(/>/g, "&gt;").replace(/"/g, "&quot;").replace(/'/g, "&#039;");
}

// FORMAT CÂU TRẢ LỜI BOT
function formatBotAnswer(answer) {
    if (!answer) return "Không nhận được câu trả lời.";
    return escapeHtml(answer).replace(/\n/g, "<br>");
}

// XỬ LÝ GIAO DIỆN PREVIEW ẢNH
function showSelectedImage(image) {
    const imageUrl = URL.createObjectURL(image);
    const fileSize = formatFileSize(image.size); // Lưu ý: Hàm này chưa có trong đoạn code bạn gửi

    selectedImagePreview.innerHTML = `<div class="selected-image"><img src="${imageUrl}" alt="Ảnh đã chọn"><div class="selected-image-info"><div class="selected-image-name">${escapeHtml(image.name)}</div><div class="selected-image-size">${fileSize}</div></div><button id="remove-image-btn" type="button" title="Hủy ảnh">✕</button></div>`;

    selectedImagePreview.style.display = "block";

    // NÚT HỦY ẢNH
    document.getElementById("remove-image-btn").onclick = function () {
        removeSelectedImage();
    };
}

function removeSelectedImage() {
    imageInput.value = "";
    selectedImagePreview.innerHTML = "";
    selectedImagePreview.style.display = "none";
}

// HÀM TÍNH TOÁN HIỂN THỊ DUNG LƯỢNG ẢNH
function formatFileSize(bytes) {
    if (bytes === 0) return '0 Bytes';
    const k = 1024;
    const sizes = ['Bytes', 'KB', 'MB'];
    const i = Math.floor(Math.log(bytes) / Math.log(k));
    return parseFloat((bytes / Math.pow(k, i)).toFixed(2)) + ' ' + sizes[i];
}

// KHỞI TẠO CHƯƠNG TRÌNH
showWelcomeMessage();