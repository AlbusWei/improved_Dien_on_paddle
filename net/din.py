import paddle
import paddle.nn as nn
import math
import numpy as np
from paddle.nn.functional import nll_loss, binary_cross_entropy_with_logits


def create_model(config):
    item_emb_size = config.get("hyper_parameters.item_emb_size", 64)
    cat_emb_size = config.get("hyper_parameters.cat_emb_size", 64)
    act = config.get("hyper_parameters.act", "sigmoid")
    is_sparse = config.get("hyper_parameters.is_sparse", False)
    use_DataLoader = config.get("hyper_parameters.use_DataLoader", False)
    item_count = config.get("hyper_parameters.item_count", 63001)
    cat_count = config.get("hyper_parameters.cat_count", 801)
    din_model = DINLayer(item_emb_size, cat_emb_size, act, is_sparse,
                         use_DataLoader, item_count, cat_count)
    return din_model


class DINLayer(nn.Layer):
    def __init__(self, item_emb_size, cat_emb_size, act, is_sparse,
                 use_DataLoader, item_count, cat_count):
        super(DINLayer, self).__init__()

        self.item_emb_size = item_emb_size
        self.cat_emb_size = cat_emb_size
        self.act = act
        self.is_sparse = is_sparse
        self.use_DataLoader = use_DataLoader
        self.item_count = item_count
        self.cat_count = cat_count

        self.hist_item_emb_attr = paddle.nn.Embedding(
            self.item_count,
            self.item_emb_size,
            sparse=self.is_sparse,
            weight_attr=paddle.framework.ParamAttr(
                initializer=paddle.nn.initializer.XavierUniform()),
            name="item_emb")
        self.hist_cat_emb_attr = paddle.nn.Embedding(
            self.cat_count,
            self.cat_emb_size,
            sparse=self.is_sparse,
            weight_attr=paddle.framework.ParamAttr(
                initializer=paddle.nn.initializer.XavierUniform()),
            name="cat_emb")
        self.target_item_emb_attr = paddle.nn.Embedding(
            self.item_count,
            self.item_emb_size,
            sparse=self.is_sparse,
            weight_attr=paddle.framework.ParamAttr(
                initializer=paddle.nn.initializer.XavierUniform()),
            name="item_emb")
        self.target_cat_emb_attr = paddle.nn.Embedding(
            self.cat_count,
            self.cat_emb_size,
            sparse=self.is_sparse,
            weight_attr=paddle.framework.ParamAttr(
                initializer=paddle.nn.initializer.XavierUniform()),
            name="cat_emb")
        self.target_item_seq_emb_attr = paddle.nn.Embedding(
            self.item_count,
            self.item_emb_size,
            sparse=self.is_sparse,
            weight_attr=paddle.framework.ParamAttr(
                initializer=paddle.nn.initializer.XavierUniform()),
            name="item_emb")

        self.target_cat_seq_emb_attr = paddle.nn.Embedding(
            self.cat_count,
            self.cat_emb_size,
            sparse=self.is_sparse,
            weight_attr=paddle.framework.ParamAttr(
                initializer=paddle.nn.initializer.XavierUniform()),
            name="cat_emb")

        self.item_b_attr = paddle.nn.Embedding(
            self.item_count,
            1,
            sparse=self.is_sparse,
            weight_attr=paddle.ParamAttr(
                initializer=paddle.nn.initializer.Constant(value=0.0)))

        self.attention_layer = []
        sizes = [(self.item_emb_size + self.cat_emb_size) * 4
                 ] + [80] + [40] + [1]
        acts = [self.act for _ in range(len(sizes) - 2)] + [None]

        for i in range(len(sizes) - 1):
            linear = paddle.nn.Linear(
                in_features=sizes[i],
                out_features=sizes[i + 1],
                weight_attr=paddle.framework.ParamAttr(
                    initializer=paddle.nn.initializer.XavierUniform()),
                bias_attr=paddle.ParamAttr(
                    initializer=paddle.nn.initializer.Constant(value=0.0)))
            self.add_sublayer('linear_%d' % i, linear)
            self.attention_layer.append(linear)
            if acts[i] == 'sigmoid':
                act = paddle.nn.Sigmoid()
                self.add_sublayer('act_%d' % i, act)
                self.attention_layer.append(act)

        self.con_layer = []

        self.firInDim = self.item_emb_size + self.cat_emb_size
        self.firOutDim = self.item_emb_size + self.cat_emb_size

        linearCon = paddle.nn.Linear(
            in_features=self.firInDim,
            out_features=self.firOutDim,
            weight_attr=paddle.framework.ParamAttr(
                initializer=paddle.nn.initializer.XavierUniform()),
            bias_attr=paddle.ParamAttr(
                initializer=paddle.nn.initializer.Constant(value=0.0)))
        self.add_sublayer('linearCon', linearCon)
        self.con_layer.append(linearCon)

        conDim = self.item_emb_size + self.cat_emb_size + self.item_emb_size + self.cat_emb_size

        conSizes = [conDim] + [80] + [40] + [1]
        conActs = ["sigmoid" for _ in range(len(conSizes) - 2)] + [None]

        for i in range(len(conSizes) - 1):
            linear = paddle.nn.Linear(
                in_features=conSizes[i],
                out_features=conSizes[i + 1],
                weight_attr=paddle.framework.ParamAttr(
                    initializer=paddle.nn.initializer.XavierUniform()),
                bias_attr=paddle.ParamAttr(
                    initializer=paddle.nn.initializer.Constant(value=0.0)))
            self.add_sublayer('linear_%d' % i, linear)
            self.con_layer.append(linear)
            if conActs[i] == 'sigmoid':
                act = paddle.nn.Sigmoid()
                self.add_sublayer('act_%d' % i, act)
                self.con_layer.append(act)

    # define feeds which convert numpy of batch data to paddle.tensor
    def create_feeds(self, batch):
        hist_item_seq = batch[0]
        hist_cat_seq = batch[1]
        target_item = batch[2]
        target_cat = batch[3]
        label = paddle.reshape(batch[4], [-1, 1])
        mask = batch[5]
        target_item_seq = batch[6]
        target_cat_seq = batch[7]
        return hist_item_seq, hist_cat_seq, target_item, target_cat, label, mask, target_item_seq, target_cat_seq

    # define metrics such as auc/acc
    # multi-task need to define multi metric
    def create_metrics(self):
        metrics_list_name = ["auc"]
        # auc_metric = paddle.metric.Auc(num_thresholds=self.bucket)
        auc_metric = paddle.metric.Auc("ROC")
        metrics_list = [auc_metric]
        return metrics_list, metrics_list_name

    def train_forward(self, metrics_list, batch_data, config, loss_function=binary_cross_entropy_with_logits):
        label = paddle.reshape(batch_data[4], [-1, 1])
        raw_pred = self.forward(batch_data)
        loss = loss_function(raw_pred, label)
        predict = paddle.nn.functional.sigmoid(raw_pred)
        predict_2d = paddle.concat([1 - predict, predict], 1)
        label_int = paddle.cast(label, 'int64')
        metrics_list[0].update(
            preds=predict_2d.numpy(), labels=label_int.numpy())

        print_dict = {'loss': loss}
        return loss, metrics_list, print_dict

    def infer_forward(self, metrics_list, batch_data, config):
        raw_pred = self.forward(batch_data)
        label = paddle.reshape(batch_data[4], [-1, 1])
        predict = paddle.nn.functional.sigmoid(raw_pred)
        predict_2d = paddle.concat([1 - predict, predict], 1)
        label_int = paddle.cast(label, 'int64')
        metrics_list[0].update(
            preds=predict_2d.numpy(), labels=label_int.numpy())

        return metrics_list, None

    def forward(self, batch):
        hist_item_seq, hist_cat_seq, target_item, target_cat, label, mask, target_item_seq, target_cat_seq \
            = self.create_feeds(batch)
        hist_item_emb = self.hist_item_emb_attr(hist_item_seq)
        hist_cat_emb = self.hist_cat_emb_attr(hist_cat_seq)
        target_item_emb = self.target_item_emb_attr(target_item)
        target_cat_emb = self.target_cat_emb_attr(target_cat)
        target_item_seq_emb = self.target_item_seq_emb_attr(target_item_seq)
        target_cat_seq_emb = self.target_cat_seq_emb_attr(target_cat_seq)
        item_b = self.item_b_attr(target_item)

        hist_seq_concat = paddle.concat([hist_item_emb, hist_cat_emb], axis=2)
        target_seq_concat = paddle.concat(
            [target_item_seq_emb, target_cat_seq_emb], axis=2)
        target_concat = paddle.concat(
            [target_item_emb, target_cat_emb], axis=1)

        concat = paddle.concat(
            [
                hist_seq_concat, target_seq_concat,
                hist_seq_concat - target_seq_concat,
                hist_seq_concat * target_seq_concat
            ],
            axis=2)

        for attlayer in self.attention_layer:
            concat = attlayer(concat)

        atten_fc3 = concat + mask
        atten_fc3 = paddle.transpose(atten_fc3, perm=[0, 2, 1])
        atten_fc3 = paddle.scale(atten_fc3, scale=self.firInDim ** -0.5)
        weight = paddle.nn.functional.softmax(atten_fc3)

        output = paddle.matmul(weight, hist_seq_concat)

        output = paddle.reshape(output, shape=[0, self.firInDim])

        for firLayer in self.con_layer[:1]:
            concat = firLayer(output)

        embedding_concat = paddle.concat([concat, target_concat], axis=1)

        for colayer in self.con_layer[1:]:
            embedding_concat = colayer(embedding_concat)

        logit = embedding_concat + item_b
        return logit
