import logging
import os
import sys
import time

import paddle
from visualdl import LogWriter

import net.dien
import net.din
from dataset import DinDataset, DienDataset
from utils.utils_single import load_yaml, create_data_loader, save_model, load_model

model_dict = {
    "din": {
        "function": net.din.DINLayer,
        "dataset": DinDataset
    },
    "dien": {
        "function": net.dien.DIENLayer,
        "dataset": DienDataset
    }
}

logging.basicConfig(
    format='%(asctime)s - %(levelname)s - %(message)s', level=logging.INFO)
logger = logging.getLogger(__name__)


# define metrics such as auc/acc
# multi-task need to define multi metric
def create_metrics():
    metrics_list_name = ["auc"]
    # auc_metric = paddle.metric.Auc(num_thresholds=self.bucket)
    auc_metric = paddle.metric.Auc("ROC")
    metrics_list = [auc_metric]
    return metrics_list, metrics_list_name


# define optimizer
def create_optimizer(dy_model, config):
    boundaries = [410000]
    base_lr = config.get(
        "hyper_parameters.optimizer.learning_rate_base_lr")
    values = [base_lr, 0.2]
    sgd_optimizer = paddle.optimizer.SGD(
        learning_rate=paddle.optimizer.lr.PiecewiseDecay(
            boundaries=boundaries, values=values),
        parameters=dy_model.parameters())
    return sgd_optimizer


def train(config, model_method, train_dataloader, valid_loader, resume_train=False, EPOCHS=100, print_interval=100,
          tensor_print_dict=None, callback=None):
    item_emb_size = config.get("hyper_parameters.item_emb_size", 64)
    cat_emb_size = config.get("hyper_parameters.cat_emb_size", 64)
    act = config.get("hyper_parameters.act", "sigmoid")
    print_interval = config.get("runner.print_interval", None)
    is_sparse = False
    use_DataLoader = True
    item_count = config.get("hyper_parameters.item_count", 63001)
    cat_count = config.get("hyper_parameters.cat_count", 801)
    model_init_path = config.get("runner.model_init_path", None)
    # model = paddle.Model(model_method(item_emb_size, cat_emb_size, act, is_sparse,
    #                                   use_DataLoader, item_count, cat_count))
    model = model_method(item_emb_size, cat_emb_size, act, is_sparse,
                         use_DataLoader, item_count, cat_count)

    save_dir = config.get("runner.model_save_path", "checkpoint")

    if resume_train and model_init_path is not None:
        load_model(model_init_path, model)
        # model.load(os.path.join(save_dir, "final"), skip_mismatch=True)

    lr = config.get("hyper_parameters.optimizer.learning_rate_base_lr", 0.01)
    # optimizer = paddle.optimizer.SGD(learning_rate=lr, parameters=model.parameters())
    optimizer = paddle.optimizer.Adam(learning_rate=0.001, parameters=model.parameters())
    # 可视化观察网络结构
    # dataiter = iter(train_loader)
    # batch = dataiter.next()
    # model.summary(batch)
    #
    # model.prepare(optimizer, NLLLoss(), Accuracy())
    #
    # model.fit(train_loader,
    #           valid_loader,
    #           epochs=EPOCHS,
    #           # batch_size=BATCH_SIZE,
    #           eval_freq=5,  # 多少epoch 进行验证
    #           save_freq=5,  # 多少epoch 进行模型保存
    #           log_freq=100,  # 多少steps 打印训练信息
    #           save_dir=save_dir,
    #           callbacks=callback)

    # Create a log_visual object and store the data in the path
    log_visual = LogWriter("log/")

    # use fleet run collective
    # if use_fleet:
    #     from paddle.distributed import fleet
    #     strategy = fleet.DistributedStrategy()
    #     fleet.init(is_collective=True, strategy=strategy)
    #     optimizer = fleet.distributed_optimizer(optimizer)
    #     dy_model = fleet.distributed_model(dy_model)

    logger.info("read data")

    last_epoch_id = config.get("last_epoch", -1)
    step_num = 0

    model.train()
    for epoch_id in range(last_epoch_id + 1, EPOCHS):
        # set train mode
        metric_list, metric_list_name = create_metrics()
        # auc_metric = paddle.metric.Auc("ROC")
        epoch_begin = time.time()
        interval_begin = time.time()
        train_reader_cost = 0.0
        train_run_cost = 0.0
        total_samples = 0
        reader_start = time.time()

        # we will drop the last incomplete batch when dataset size is not divisible by the batch size
        assert any(train_dataloader(
        )), "train_dataloader is null, please ensure batch size < dataset size!"

        for batch_id, batch in enumerate(train_dataloader()):
            train_reader_cost += time.time() - reader_start
            optimizer.clear_grad()
            train_start = time.time()
            batch_size = len(batch[0])

            loss, metric_list, tensor_print_dict = model.train_forward(
                metric_list, batch, config)

            loss.backward()
            optimizer.step()
            train_run_cost += time.time() - train_start
            total_samples += batch_size

            if batch_id % print_interval == 0:
                metric_str = ""
                for metric_id in range(len(metric_list_name)):
                    metric_str += (
                            metric_list_name[metric_id] +
                            ":{:.6f}, ".format(metric_list[metric_id].accumulate())
                    )
                    log_visual.add_scalar(
                        tag="train/" + metric_list_name[metric_id],
                        step=step_num,
                        value=metric_list[metric_id].accumulate())
                tensor_print_str = ""
                if tensor_print_dict is not None:
                    for var_name, var in tensor_print_dict.items():
                        tensor_print_str += (
                                "{}:".format(var_name) +
                                str(var.numpy()).strip("[]") + ",")
                        log_visual.add_scalar(
                            tag="train/" + var_name,
                            step=step_num,
                            value=var.numpy())
                logger.info(
                    "epoch: {}, batch_id: {}, ".format(
                        epoch_id, batch_id) + metric_str + tensor_print_str +
                    " avg_reader_cost: {:.5f} sec, avg_batch_cost: {:.5f} sec, avg_samples: {:.5f}, ips: {:.5f} ins/s".
                    format(train_reader_cost / print_interval, (
                            train_reader_cost + train_run_cost) / print_interval,
                           total_samples / print_interval, total_samples / (
                                   train_reader_cost + train_run_cost)))
                train_reader_cost = 0.0
                train_run_cost = 0.0
                total_samples = 0
            reader_start = time.time()
            step_num = step_num + 1

        metric_str = ""
        for metric_id in range(len(metric_list_name)):
            metric_str += (
                    metric_list_name[metric_id] +
                    ": {:.6f},".format(metric_list[metric_id].accumulate()))
            metric_list[metric_id].reset()

        tensor_print_str = ""
        if tensor_print_dict is not None:
            for var_name, var in tensor_print_dict.items():
                tensor_print_str += (
                        "{}:".format(var_name) + str(var.numpy()).strip("[]") + ","
                )

        logger.info("epoch: {} done, ".format(epoch_id) + metric_str +
                    tensor_print_str + " epoch time: {:.2f} s".format(
            time.time() - epoch_begin))

        save_model(
            model, optimizer, save_dir, epoch_id, prefix='rec')
        # if use_fleet:
        #     trainer_id = paddle.distributed.get_rank()
        #     if trainer_id == 0:
        #         save_model(
        #             model,
        #             optimizer,
        #             model_save_path,
        #             epoch_id,
        #             prefix='rec')
        # else:
        #     save_model(
        #         model, optimizer, model_save_path, epoch_id, prefix='rec')

    # return model


def test(config, model_method, test_dataloader, model=None, tensor_print_dict=None):
    if model is None:
        item_emb_size = config.get("hyper_parameters.item_emb_size", 64)
        cat_emb_size = config.get("hyper_parameters.cat_emb_size", 64)
        act = config.get("hyper_parameters.act", "sigmoid")
        is_sparse = False
        use_DataLoader = True
        item_count = config.get("hyper_parameters.item_count", 63001)
        cat_count = config.get("hyper_parameters.cat_count", 801)
        model = model_method(item_emb_size, cat_emb_size, act, is_sparse,
                             use_DataLoader, item_count, cat_count)

    print_interval = config.get("runner.print_interval", None)
    infer_batch_size = config.get("runner.infer_batch_size", None)
    model_load_path = config.get("runner.infer_load_path", "model_output")
    start_epoch = config.get("runner.infer_start_epoch", 0)
    end_epoch = config.get("runner.infer_end_epoch", 10)

    logger.info("read data")
    epoch_begin = time.time()
    interval_begin = time.time()

    metric_list, metric_list_name = create_metrics()
    step_num = 0

    for epoch_id in range(start_epoch, end_epoch):
        logger.info("load model epoch {}".format(epoch_id))
        model_path = os.path.join(model_load_path, str(epoch_id))
        load_model(model_path, model)
        model.eval()
        infer_reader_cost = 0.0
        infer_run_cost = 0.0
        reader_start = time.time()

        # we will drop the last incomplete batch when dataset size is not divisible by the batch size
        assert any(test_dataloader(
        )), "test_dataloader is null, please ensure batch size < dataset size!"

        for batch_id, batch in enumerate(test_dataloader()):
            infer_reader_cost += time.time() - reader_start
            infer_start = time.time()
            batch_size = len(batch[0])

            metric_list, tensor_print_dict = model.infer_forward(
                metric_list, batch, config)

            infer_run_cost += time.time() - infer_start

            if batch_id % print_interval == 0:
                tensor_print_str = ""
                if tensor_print_dict is not None:
                    for var_name, var in tensor_print_dict.items():
                        tensor_print_str += (
                                "{}:".format(var_name) +
                                str(var.numpy()).strip("[]") + ",")
                metric_str = ""
                for metric_id in range(len(metric_list_name)):
                    metric_str += (
                            metric_list_name[metric_id] +
                            ": {:.6f},".format(metric_list[metric_id].accumulate())
                    )
                logger.info(
                    "epoch: {}, batch_id: {}, ".format(
                        epoch_id, batch_id) + metric_str + tensor_print_str +
                    " avg_reader_cost: {:.5f} sec, avg_batch_cost: {:.5f} sec, avg_samples: {:.5f}, ips: {:.2f} ins/s".
                    format(infer_reader_cost / print_interval, (
                            infer_reader_cost + infer_run_cost) / print_interval,
                           infer_batch_size, print_interval * batch_size / (
                                   time.time() - interval_begin)))
                interval_begin = time.time()
                infer_reader_cost = 0.0
                infer_run_cost = 0.0
            step_num = step_num + 1
            reader_start = time.time()

        metric_str = ""
        for metric_id in range(len(metric_list_name)):
            metric_str += (
                    metric_list_name[metric_id] +
                    ": {:.6f},".format(metric_list[metric_id].accumulate()))
            metric_list[metric_id].reset()

        tensor_print_str = ""
        if tensor_print_dict is not None:
            for var_name, var in tensor_print_dict.items():
                tensor_print_str += (
                        "{}:".format(var_name) + str(var.numpy()).strip("[]") + ","
                )

        logger.info("epoch: {} done, ".format(epoch_id) + metric_str +
                    tensor_print_str + " epoch time: {:.2f} s".format(
            time.time() - epoch_begin))
        epoch_begin = time.time()


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
