import os

from configs import config_csv


def get_report_names():

    reports = {}

    for file in os.listdir(config_csv.dir_evaluate):
        if file.endswith(".txt"):
            path = os.path.join(config_csv.dir_evaluate,file)

            with open(path,"r",encoding="utf-8") as f:
                reports[file] = f.read()

    return reports
