import os
from datetime import datetime
from PIL import Image,ImageDraw,ImageFont
from configs import paths_common

def create_prediction_report(data,prediction,proba_dict,model_name):
    # tạo tên file ảnh
    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    filename = f"prediction_{timestamp}.png"
    file_path = os.path.join(
        paths_common.dir_results,
        filename
    )

    # tạo ảnh
    width = 1000
    height = 1000
    image = Image.new("RGB",(width, height),"white")
    draw = ImageDraw.Draw(image)

    #font
    try:
        font_title = ImageFont.truetype("arial.ttf",36)
        font_header = ImageFont.truetype("arial.ttf",26)
        font_normal = ImageFont.truetype("arial.ttf",22)
        font_result = ImageFont.truetype("arialbd.ttf",30)

    except OSError:
        font_title = ImageFont.load_default()
        font_header = ImageFont.load_default()
        font_normal = ImageFont.load_default()
        font_result = ImageFont.load_default()

    # tiêu đề
    title =  "PHIEU KET QUA DU DOAN"
    draw.text((width // 2, 40),title,fill="black",font=font_title,anchor="ma")
    draw.text((width // 2, 90),"DIABETES PREDICTION",fill="black",font=font_header,anchor="ma")

    # đường kẻ
    draw.line((50, 135, 950, 135),fill="black", width=2)

    # thông tin người dùng
    draw.text((70, 170),"THONG TIN DU DOAN",fill="black",font=font_header)

    y = 220
    fields = [
        ("Gioi tinh", data.get("gender", "")),
        ("Tuoi", data.get("age", "")),
        ("BMI", data.get("bmi", "")),
        ("HbA1c", data.get("HbA1c_level", "")),
        ("Duong huyet", data.get("blood_glucose_level", "")),
        ("Tang huyet ap","Co" if str(data.get("hypertension")) == "1" else "Khong"),
        ("Benh tim","Co" if str(data.get("heart_disease")) == "1" else "Khong"),
        ("Tien su hut thuoc",data.get("smoking_history", ""))
    ]

    for label, value in fields:
        text = f"{label}: {value}"
        draw.text((80, y),text,fill="black",font=font_normal)
        y += 45

    # kết quả dự đoán
    y += 20
    draw.line((50, y, 950, y),fill="black",width=2)

    y += 35
    draw.text((70, y),"KET QUA DU DOAN",fill="black",font=font_header)

    y += 55
    if prediction == 1:
        result_text = "CO NGUY CO MAC TIEU DUONG"
    else:
        result_text = "KHONG CO NGUY CO TIEU DUONG"

    draw.text((width // 2, y),result_text,fill="black",font=font_result,anchor="ma")

    # sác xuất
    y += 70
    for cls, prob in proba_dict.items():
        if cls == 0 or str(cls) == "0":
            label = "Khong mac tieu duong"
        else:
            label = "Co nguy co tieu duong"

        text = f"{label}: {prob}%"
        draw.text((100, y),text,fill="black",font=font_normal)
        y += 45

    # model
    y += 20
    draw.line((50, y, 950, y),fill="black",width=2)

    y += 35
    draw.text((70, y),f"Model su dung: {model_name}",fill="black",font=font_normal)

    # thời gian dự đoán
    y += 45
    current_time = datetime.now().strftime("%d/%m/%Y %H:%M:%S")
    draw.text((70, y),f"Thoi gian: {current_time}",fill="black",font=font_normal)

    image.save(file_path,format="PNG")

    return filename

def get_report_path(filename):
    return os.path.join(paths_common.dir_results,filename)