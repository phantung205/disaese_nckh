import os
from configs import paths_common


"""data"""
# directory raw image
dir_image_raw = os.path.join(paths_common.dir_data_raw,"image")

# directory processed image
dir_image_processed = os.path.join(paths_common.dir_data_processed,"image")


"""models , report"""
dir_image_model = os.path.join(paths_common.dir_model,"image")
dir_image_report = os.path.join(paths_common.dir_reports,"image")


"""split, category"""
splits = ["train","valid"]
categorys = ["No_DR","DR"]


"""train"""
#parameter
batch_size = 16
image_size = 224
train_ratio = 0.8
learning_rate = 1e-4
weight_decay = 1e-4
momentum = 0.9
epochs = 100

# path save models train and tensorboard
path_tensorboard = os.path.join(dir_image_report,"tensorboard")
path_model = os.path.join(dir_image_model,"trained_models")
# path model best good
path_best_model = os.path.join(path_model,"best_cnn.pt")



