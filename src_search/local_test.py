from src_search.search_engine import ImageSearchEngine
import matplotlib.pyplot as plt
from PIL import Image
import os

def visualize(query, results):

    plt.figure(figsize=(12, 6))

    # Hiển thị ảnh query
    plt.subplot(1, len(results) + 1, 1)
    plt.title("Query")

    if isinstance(query, str) and os.path.exists(query):
        plt.imshow(Image.open(query))
        plt.xlabel(os.path.basename(query), fontsize=8)
    else:
        plt.text(0.5, 0.5, query, ha="center", va="center")

    plt.axis("off")

    # Hiển thị kết quả
    for i, img_path in enumerate(results):
        plt.subplot(1, len(results) + 1, i + 2)
        plt.imshow(Image.open(img_path))

        image_name = os.path.splitext(os.path.basename(img_path))[0]

        plt.title(image_name, fontsize=8)
        plt.axis("off")

    plt.tight_layout()
    plt.show()


if __name__ == "__main__":

    engine = ImageSearchEngine()


    query = "no.png"

    results = engine.search(query)

    visualize(query, results)