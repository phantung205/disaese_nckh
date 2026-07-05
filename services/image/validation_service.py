import os.path


def validation_image(image):
    # kiểm tra người dùng đã chọn ảnh hay chưa
    if image.filename == "":
        raise ValueError("bạn chưa chọn ảnh")

    # lấy ra tên đuôi file người dùng up
    name,ext = os.path.splitext(image.filename)

    # kiểm tra đuôi file có đúng ko
    ext = ext.lower()
    allowed_extensions = {".jpg",".png",".jpeg"}
    if ext not in allowed_extensions:
        raise ValueError("Chỉ chấp nhận các định dạng ảnh: .jpg, .jpeg, .png")
