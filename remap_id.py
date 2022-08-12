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
import numpy as np
import os

random.seed(1234)
# set this as you need
files = ["Clothing_Shoes_and_Jewelry",
         "Electronics",
         "Health_and_Personal_Care",
         "Kindle_Store",
         "Office_Products",
         "Movies_and_TV"]


def build_map(df, col_name):
    key = sorted(df[col_name].unique().tolist())
    m = dict(zip(key, range(len(key))))
    df[col_name] = df[col_name].map(lambda x: m[x])
    return m, key


def main():
    for file in files:
        review_dir = "reviews_" + file + "_5.json"
        meta_dir = "meta_" + file + ".json"
        remap(review_dir, meta_dir, file)


def remap(review_dir, meta_dir, file):
    # review_file = os.path.join("rawdata", review_dir)
    # meta_file = os.path.join("rawdata", meta_dir)
    pkl_path = os.path.join("rawdata", review_dir[:-4] + "pkl")
    meta_pkl = os.path.join("rawdata", meta_dir[:-4] + "pkl")
    remap_path = os.path.join("rawdata", "remap_" + file + ".pkl")

    with open(pkl_path, 'rb') as f:
        reviews_df = pickle.load(f)
        reviews_df = reviews_df[['reviewerID', 'asin', 'unixReviewTime']]
    with open(meta_pkl, 'rb') as f:
        meta_df = pickle.load(f)
        meta_df = meta_df[['asin', 'categories']]
        meta_df['categories'] = meta_df['categories'].map(lambda x: x[-1][-1])

    asin_map, asin_key = build_map(meta_df, 'asin')
    cate_map, cate_key = build_map(meta_df, 'categories')
    revi_map, revi_key = build_map(reviews_df, 'reviewerID')

    print("asin_map, asin_key", asin_map, asin_key)
    print("cate_map, cate_key", cate_map, cate_key)
    print("revi_map, revi_key", revi_map, revi_key)

    user_count, item_count, cate_count, example_count = \
        len(revi_map), len(asin_map), len(cate_map), reviews_df.shape[0]
    print('user_count: %d\titem_count: %d\tcate_count: %d\texample_count: %d' %
          (user_count, item_count, cate_count, example_count))

    meta_df = meta_df.sort_values('asin')
    meta_df = meta_df.reset_index(drop=True)
    reviews_df['asin'] = reviews_df['asin'].map(lambda x: asin_map[x])
    reviews_df = reviews_df.sort_values(['reviewerID', 'unixReviewTime'])
    reviews_df = reviews_df.reset_index(drop=True)
    reviews_df = reviews_df[['reviewerID', 'asin', 'unixReviewTime']]

    cate_list = [meta_df['categories'][i] for i in range(len(asin_map))]
    cate_list = np.array(cate_list, dtype=np.int32)

    with open(remap_path, 'wb') as f:
        pickle.dump(reviews_df, f, pickle.HIGHEST_PROTOCOL)  # uid, iid
        pickle.dump(cate_list, f, pickle.HIGHEST_PROTOCOL)  # cid of iid line
        pickle.dump((user_count, item_count, cate_count, example_count), f,
                    pickle.HIGHEST_PROTOCOL)
        pickle.dump((asin_key, cate_key, revi_key), f, pickle.HIGHEST_PROTOCOL)


if __name__ == '__main__':
    # main()
    remap("Books_5.json", "meta_Books.json", "Books")
