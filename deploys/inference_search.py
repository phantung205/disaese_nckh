from src_search.search_engine import ImageSearchEngine


def load_model():
    search_engine = ImageSearchEngine()
    return search_engine

def search_image(image_path,model,top_k):
    results = model.search(image_path,top_k=top_k)
    return results