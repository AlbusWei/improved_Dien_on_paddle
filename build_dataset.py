# Copyright (c) 2020 PaddlePaddle Authors. All Rights Reserved.
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
#     http://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.

from __future__ import print_function
import random
import pickle
import sys
import os


def main():
    if len(sys.argv)>1:
        build_dataset(str(sys.argv[1]))
    else:
        build_dataset()


def build_dataset(class_name="Electronics"):
    random.seed(1234)
    remap_path = os.path.join("rawdata", "remap_" + class_name + ".pkl")

    print("read and process data")

    with open(remap_path, 'rb') as f:
        reviews_df = pickle.load(f)
        cate_list = pickle.load(f)
        user_count, item_count, cate_count, example_count = pickle.load(f)

    train_set = []
    test_set = []
    for reviewerID, hist in reviews_df.groupby('reviewerID'):
        pos_list = hist['asin'].tolist()

        def gen_neg():
            neg = pos_list[0]
            while neg in pos_list:
                neg = random.randint(0, item_count - 1)
            return neg

        neg_list = [gen_neg() for i in range(len(pos_list))]

        for i in range(1, len(pos_list)):
            hist = pos_list[:i]
            if i != len(pos_list) - 1:
                train_set.append((reviewerID, hist, pos_list[i], 1))
                train_set.append((reviewerID, hist, neg_list[i], 0))
            else:
                label = (pos_list[i], neg_list[i])
                test_set.append((reviewerID, hist, label))

    random.shuffle(train_set)
    random.shuffle(test_set)

    assert len(test_set) == user_count

    def print_to_file(data, fout):
        for i in range(len(data)):
            fout.write(str(data[i]))
            if i != len(data) - 1:
                fout.write(' ')
            else:
                fout.write(';')

    train_path = os.path.join("train", class_name+"_train.txt")
    test_path = os.path.join("test", class_name + "_test.txt")

    print("make train data")
    with open(train_path, "w") as fout:
        for line in train_set:
            history = line[1]
            target = line[2]
            label = line[3]
            cate = [cate_list[x] for x in history]
            print_to_file(history, fout)
            print_to_file(cate, fout)
            fout.write(str(target) + ";")
            fout.write(str(cate_list[target]) + ";")
            fout.write(str(label) + "\n")

    print("make test data")
    with open(test_path, "w") as fout:
        for line in test_set:
            history = line[1]
            target = line[2]
            cate = [cate_list[x] for x in history]

            print_to_file(history, fout)
            print_to_file(cate, fout)
            fout.write(str(target[0]) + ";")
            fout.write(str(cate_list[target[0]]) + ";")
            fout.write("1\n")

            print_to_file(history, fout)
            print_to_file(cate, fout)
            fout.write(str(target[1]) + ";")
            fout.write(str(cate_list[target[1]]) + ";")
            fout.write("0\n")

    print("make config data")
    with open(class_name+'_config.txt', 'w') as f:
        f.write(str(user_count) + "\n")
        f.write(str(item_count) + "\n")
        f.write(str(cate_count) + "\n")


if __name__ == '__main__':
    main()
