import os
import sys
import time
import logging

import matplotlib.pyplot as plt
import numpy as np
import paddle
from paddle.metric import Accuracy, Auc
from paddle.nn import CrossEntropyLoss, NLLLoss
from visualdl import LogWriter
from utils.utils_single import load_yaml, create_data_loader, save_model, load_model
from dataset import DinDataset, DienDataset
import net.din
import net.dien

from train import test, model_dict

if len(sys.argv) > 1:
    config = load_yaml(str(sys.argv[1]))
else:
    config = load_yaml("config_din.yaml")

place = paddle.set_device('gpu')

model_name = config.get("runner.model_name", "din")
# train_dataset = model_dict[model_name]["dataset"](config.get("runner.train_data_dir", "./train"), config)
# test_dataset = model_dict[model_name]["dataset"](config.get("runner.test_data_dir", "./test"), config)
RecDataset = model_dict[model_name]["dataset"]
# train_dataloader = create_data_loader(config=config, RecDataset=RecDataset, place=place)
test_dataloader = create_data_loader(config=config, RecDataset=RecDataset, place=place, mode="test")

model_method = model_dict[model_name]["function"]
callback = paddle.callbacks.VisualDL(log_dir='log/')

test(config, model_method, test_dataloader)