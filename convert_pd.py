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
import pickle
import pandas as pd
import os

# set this as you need
files = ["Clothing_Shoes_and_Jewelry",
         "Electronics",
         "Health_and_Personal_Care",
         "Kindle_Store",
         "Office_Products",
         "Movies_and_TV"]


def to_df(file_path):
    with open(file_path, 'r') as fin:
        df = {}
        i = 0
        for line in fin:
            df[i] = eval(line)
            i += 1
        df = pd.DataFrame.from_dict(df, orient='index')
        return df


def convert(review_dir, meta_dir):
    review_file = os.path.join("rawdata", review_dir)
    meta_file = os.path.join("rawdata", meta_dir)
    pkl_path = os.path.join("rawdata", review_dir[:-4] + "pkl")
    meta_pkl = os.path.join("rawdata", meta_dir[:-4] + "pkl")

    print("start to " + review_file)
    reviews_df = to_df(review_file)
    with open(pkl_path, 'wb') as f:
        pickle.dump(reviews_df, f, pickle.HIGHEST_PROTOCOL)

    print("start to analyse " + meta_file)
    meta_df = to_df(meta_file)
    meta_df = meta_df[meta_df['asin'].isin(reviews_df['asin'].unique())]
    meta_df = meta_df.reset_index(drop=True)
    with open(meta_pkl, 'wb') as f:
        pickle.dump(meta_df, f, pickle.HIGHEST_PROTOCOL)


def main():
    for file in files:
        review_dir = "reviews_" + file + "_5.json"
        meta_dir = "meta_" + file + ".json"
        convert(review_dir, meta_dir)


if __name__ == '__main__':
    main()
