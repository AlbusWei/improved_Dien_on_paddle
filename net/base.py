import paddle
import math
from functools import partial

import numpy as np


def create_model(config):
    item_emb_size = config.get("hyper_parameters.item_emb_size", 64)
    cat_emb_size = config.get("hyper_parameters.cat_emb_size", 64)
    position_emb_size = config.get("hyper_parameters.position_emb_size",
                                   64)
    act = config.get("hyper_parameters.act", "sigmoid")
    is_sparse = config.get("hyper_parameters.is_sparse", False)
    # significant for speeding up the training process
    use_DataLoader = config.get("hyper_parameters.use_DataLoader", False)
    item_count = config.get("hyper_parameters.item_count", 63001)
    user_count = config.get("hyper_parameters.user_count", 192403)

    cat_count = config.get("hyper_parameters.cat_count", 801)
    position_count = config.get("hyper_parameters.position_count", 5001)
    n_encoder_layers = config.get("hyper_parameters.n_encoder_layers", 1)
    d_model = config.get("hyper_parameters.d_model", 96)
    d_key = config.get("hyper_parameters.d_key", None)
    d_value = config.get("hyper_parameters.d_value", None)
    n_head = config.get("hyper_parameters.n_head", None)
    dropout_rate = config.get("hyper_parameters.dropout_rate", 0.0)
    postprocess_cmd = config.get("hyper_parameters.postprocess_cmd", "da")
    preprocess_cmd = config.get("hyper_parameters.postprocess_cmd", "n")
    prepostprocess_dropout = config.get(
        "hyper_parameters.prepostprocess_dropout", 0.0)
    d_inner_hid = config.get("hyper_parameters.d_inner_hid", 512)
    relu_dropout = config.get("hyper_parameters.relu_dropout", 0.0)
    layer_sizes = config.get("hyper_parameters.fc_sizes", None)

    bst_model = BaseModel(
        user_count, item_emb_size, cat_emb_size, position_emb_size, act,
        is_sparse, use_DataLoader, item_count, cat_count, position_count,
        n_encoder_layers, d_model, d_key, d_value, n_head, dropout_rate,
        postprocess_cmd, preprocess_cmd, prepostprocess_dropout,
        d_inner_hid, relu_dropout, layer_sizes)

    return bst_model


# define feeds which convert numpy of batch data to paddle.tensor
def create_feeds(batch_data, config):
    dense_feature_dim = config.get('hyper_parameters.dense_input_dim')
    sparse_tensor = []
    for b in batch_data:
        sparse_tensor.append(
            paddle.to_tensor(b.numpy().astype('int64').reshape(-1,
                                                               len(b[0]))))
    label = sparse_tensor[0]
    return label, sparse_tensor[1], sparse_tensor[2], sparse_tensor[
        3], sparse_tensor[4], sparse_tensor[5], sparse_tensor[
               6], sparse_tensor[7]


class BaseModel(paddle.nn.Layer):
    def __init__(self, user_count, item_emb_size, cat_emb_size,
                 position_emb_size, act, is_sparse, use_DataLoader, item_count,
                 cat_count, position_count, n_encoder_layers, d_model, d_key,
                 d_value, n_head, dropout_rate, postprocess_cmd,
                 preprocess_cmd, prepostprocess_dropout, d_inner_hid,
                 relu_dropout, layer_sizes):
        super(BaseModel, self).__init__()

        self.item_emb_size = item_emb_size
        self.cat_emb_size = cat_emb_size
        self.position_emb_size = position_emb_size
        self.act = act
        self.is_sparse = is_sparse
        # significant for speeding up the training process
        self.use_DataLoader = use_DataLoader
        self.item_count = item_count
        self.cat_count = cat_count
        self.position_count = position_count
        self.user_count = user_count
        self.n_encoder_layers = n_encoder_layers
        self.d_model = d_model
        self.d_key = d_key
        self.d_value = d_value
        self.n_head = n_head
        self.dropout_rate = dropout_rate
        self.postprocess_cmd = postprocess_cmd
        self.preprocess_cmd = preprocess_cmd
        self.prepostprocess_dropout = prepostprocess_dropout
        self.d_inner_hid = d_inner_hid
        self.relu_dropout = relu_dropout
        self.layer_sizes = layer_sizes

        self.base = base(user_count, item_emb_size, cat_emb_size,
                       position_emb_size, act, is_sparse, use_DataLoader,
                       item_count, cat_count, position_count, n_encoder_layers,
                       d_model, d_key, d_value, n_head, dropout_rate,
                       postprocess_cmd, preprocess_cmd, prepostprocess_dropout,
                       d_inner_hid, relu_dropout, layer_sizes)

        self.bias = paddle.create_parameter(
            shape=[1],
            dtype='float32',
            default_initializer=paddle.nn.initializer.Constant(value=0.0))

    def forward(self, userid, hist_item_seq, hist_cat_seq, position_seq,
                target_item, target_cat, target_position):
        y_out = self.base.forward(userid, hist_item_seq, hist_cat_seq,
                                 position_seq, target_item, target_cat,
                                 target_position)

        predict = paddle.nn.functional.sigmoid(y_out + self.bias)
        return predict


class base(paddle.nn.Layer):
    def __init__(self, user_count, item_emb_size, cat_emb_size,
                 position_emb_size, act, is_sparse, use_DataLoader, item_count,
                 cat_count, position_count, n_encoder_layers, d_model, d_key,
                 d_value, n_head, dropout_rate, postprocess_cmd,
                 preprocess_cmd, prepostprocess_dropout, d_inner_hid,
                 relu_dropout, layer_sizes):

        super(base, self).__init__()
        self.item_emb_size = item_emb_size
        self.cat_emb_size = cat_emb_size
        self.position_emb_size = position_emb_size
        self.act = act
        self.is_sparse = is_sparse
        # significant for speeding up the training process
        self.use_DataLoader = use_DataLoader
        self.item_count = item_count
        self.cat_count = cat_count
        self.user_count = user_count
        self.position_count = position_count
        self.n_encoder_layers = n_encoder_layers
        self.d_model = d_model
        self.d_key = d_key
        self.d_value = d_value
        self.n_head = n_head
        self.dropout_rate = dropout_rate
        self.postprocess_cmd = postprocess_cmd
        self.preprocess_cmd = preprocess_cmd
        self.prepostprocess_dropout = prepostprocess_dropout
        self.d_inner_hid = d_inner_hid
        self.relu_dropout = relu_dropout
        self.layer_sizes = layer_sizes

        init_value_ = 0.1
        self.hist_item_emb_attr = paddle.nn.Embedding(
            self.item_count,
            self.item_emb_size,
            sparse=False,
            weight_attr=paddle.ParamAttr(
                initializer=paddle.nn.initializer.TruncatedNormal(
                    mean=0.0,
                    std=init_value_ / math.sqrt(float(self.item_emb_size)))))

        self.hist_cat_emb_attr = paddle.nn.Embedding(
            self.cat_count,
            self.cat_emb_size,
            sparse=False,
            weight_attr=paddle.ParamAttr(
                initializer=paddle.nn.initializer.TruncatedNormal(
                    mean=0.0,
                    std=init_value_ / math.sqrt(float(self.cat_emb_size)))))

        self.hist_position_emb_attr = paddle.nn.Embedding(
            self.position_count,
            self.position_emb_size,
            sparse=False,
            weight_attr=paddle.ParamAttr(
                initializer=paddle.nn.initializer.TruncatedNormal(
                    mean=0.0,
                    std=init_value_ /
                        math.sqrt(float(self.position_emb_size)))))

        self.target_item_emb_attr = paddle.nn.Embedding(
            self.item_count,
            self.item_emb_size,
            sparse=False,
            weight_attr=paddle.ParamAttr(
                initializer=paddle.nn.initializer.TruncatedNormal(
                    mean=0.0,
                    std=init_value_ / math.sqrt(float(self.item_emb_size)))))

        self.target_cat_emb_attr = paddle.nn.Embedding(
            self.cat_count,
            self.cat_emb_size,
            sparse=False,
            weight_attr=paddle.ParamAttr(
                initializer=paddle.nn.initializer.TruncatedNormal(
                    mean=0.0,
                    std=init_value_ / math.sqrt(float(self.cat_emb_size)))))

        self.target_position_emb_attr = paddle.nn.Embedding(
            self.position_count,
            self.position_emb_size,
            sparse=False,
            weight_attr=paddle.ParamAttr(
                initializer=paddle.nn.initializer.TruncatedNormal(
                    mean=0.0,
                    std=init_value_ /
                        math.sqrt(float(self.position_emb_size)))))

        self.userid_attr = paddle.nn.Embedding(
            self.user_count,
            self.d_model,
            sparse=False,
            weight_attr=paddle.ParamAttr(
                initializer=paddle.nn.initializer.TruncatedNormal(
                    mean=0.0,
                    std=init_value_ / math.sqrt(float(self.d_model)))))

        self._dnn_layers = []
        sizes = [d_model] + layer_sizes + [1]
        acts = ["relu" for _ in range(len(layer_sizes))] + [None]
        for i in range(len(layer_sizes) + 1):
            linear = paddle.nn.Linear(
                in_features=sizes[i],
                out_features=sizes[i + 1],
                weight_attr=paddle.ParamAttr(
                    initializer=paddle.nn.initializer.Normal(
                        std=0.1 / math.sqrt(sizes[i]))))
            self.add_sublayer('dnn_linear_%d' % i, linear)
            self._dnn_layers.append(linear)
            if acts[i] == 'relu':
                act = paddle.nn.LeakyReLU()
                self.add_sublayer('dnn_act_%d' % i, act)
                self._dnn_layers.append(act)

        self.drop_out = paddle.nn.Dropout(p=dropout_rate)

    def forward(self, userid, hist_item_seq, hist_cat_seq, position_seq,
                target_item, target_cat, target_position):

        user_emb = self.userid_attr(userid)

        hist_item_emb = self.hist_item_emb_attr(hist_item_seq)

        hist_cat_emb = self.hist_cat_emb_attr(hist_cat_seq)

        hist_position_emb = self.hist_position_emb_attr(position_seq)

        target_item_emb = self.target_item_emb_attr(target_item)

        target_cat_emb = self.target_cat_emb_attr(target_cat)

        target_position_emb = self.target_position_emb_attr(target_position)

        item_sequence = paddle.concat(
            [hist_item_emb, hist_cat_emb, hist_position_emb], axis=2)
        target_sequence = paddle.concat(
            [target_item_emb, target_cat_emb, target_position_emb], axis=2)

        # print(whole_embedding)
        enc_output = item_sequence
        enc_output = paddle.sum(enc_output, axis=1)
        enc_output = paddle.reshape(enc_output, [len(enc_output), 1, -1])

        # enc_output = self.encoder_layer(enc_output)
        # enc_output = self.pre_post_process_layer(
        #     enc_output, self.preprocess_cmd, self.prepostprocess_dropout)
        _concat = paddle.concat([user_emb, enc_output, target_sequence], axis=1)
        dnn_input = _concat
        for n_layer in self._dnn_layers:
            dnn_input = n_layer(dnn_input)
        dnn_input = paddle.sum(x=dnn_input, axis=1)
        return dnn_input
