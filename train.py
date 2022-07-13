import os
import sys
import matplotlib.pyplot as plt
import numpy as np
import paddle
from paddle.metric import Accuracy
from paddle.nn import CrossEntropyLoss
from utils.utils_single import load_yaml, create_data_loader
from dataset import DinDataset, DienDataset
from net.din import DINLayer
from net.dien import DIENLayer

model_dict = {
    "din": {
        "function": DINLayer,
        "dataset": DinDataset
    },
    "dien": {
        "function": DIENLayer,
        "dataset": DienDataset
    }
}


def train(config, model_method, train_loader, valid_loader, resume_train=False, EPOCHS=100, callback=None):
    item_emb_size = config.get("hyper_parameters.item_emb_size", 64)
    cat_emb_size = config.get("hyper_parameters.cat_emb_size", 64)
    act = config.get("hyper_parameters.act", "sigmoid")
    is_sparse = False
    use_DataLoader = True
    item_count = config.get("hyper_parameters.item_count", 63001)
    cat_count = config.get("hyper_parameters.cat_count", 801)
    model = paddle.Model(model_method(item_emb_size, cat_emb_size, act, is_sparse,
                 use_DataLoader, item_count, cat_count))

    save_dir = config.get("runner.model_save_path", "checkpoint")

    if resume_train:
        model.load(os.path.join(save_dir, "final"), skip_mismatch=True)

    lr = config.get("hyper_parameters.optimizer.learning_rate_base_lr", 0.01)
    optimizer = paddle.optimizer.SGD(learning_rate=lr, parameters=model.parameters())

    dataiter = iter(train_loader)
    batch, label = dataiter.next()
    model.summary(batch)

    return model

def main():
    if len(sys.argv) > 1:
        config = load_yaml(str(sys.argv[1]))
    else:
        config = load_yaml("config_din.yaml")

    place = paddle.set_device('gpu')

    model_name = config.get("runner.model_name", "din")
    # train_dataset = model_dict[model_name]["dataset"](config.get("runner.train_data_dir", "./train"), config)
    # test_dataset = model_dict[model_name]["dataset"](config.get("runner.test_data_dir", "./test"), config)
    RecDataset = model_dict[model_name]["dataset"]
    train_dataloader = create_data_loader(config=config, RecDataset=RecDataset, place=place)
    test_dataloader = create_data_loader(config=config, RecDataset=RecDataset, place=place, mode="test")

    model_method = model_dict[model_name]["function"]
    callback = paddle.callbacks.VisualDL(log_dir='log/')

    Epoch = config.get("runner.epochs", 8)

    model = train(config, model_method, train_dataloader, test_dataloader, EPOCHS=Epoch, callback=callback)


if __name__ == '__main__':
    main()
