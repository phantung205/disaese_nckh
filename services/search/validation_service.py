import os

def validate_search(image, top_k):
    # validation ảnh
    image_name = image.filename

    if image_name == "":
        raise ValueError("bạn chưa chọn ảnh")

    _,ext = os.path.splitext(image_name)
    ext = ext.lower()

    allowed_extensions = {".jpg", ".png", ".jpeg"}
    if ext not in allowed_extensions:
        raise ValueError("Chỉ chấp nhận các định dạng ảnh: .jpg, .jpeg, .png")

    # validation top_k
    if not isinstance(top_k,int) or top_k <= 0:
        raise  ValueError(" phải là số nguyên và lớn hơn 0")
