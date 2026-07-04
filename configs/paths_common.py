import os

""" root """
# directory root project
root_project = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))


""" data """
# directory data
dir_data = os.path.join(root_project,"data")

# directory data raw
dir_data_raw = os.path.join(dir_data,"raw")

# directory data processed
dir_data_processed = os.path.join(dir_data,"processed")


""" reports """
# directory report
dir_reports = os.path.join(root_project,"reports")


""" models """
# directory models
dir_model = os.path.join(root_project,"models")

"""uploads and results"""
#directory uploads
dir_uploads = os.path.join(root_project,"uploads")

# directory results predict
dir_results = os.path.join(root_project,"results")