from deploys import inference_search


search_engine = None

def load_model_search():
    global search_engine

    if search_engine is None:
        search_engine = inference_search.load_model()

    return search_engine