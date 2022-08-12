import os
import random

import numpy as np
import pandas as pd


# TODO:为BST和DIN提供相应规格的movielens数据
# 改造思路：label只取第一个，反正方法一致应该效果差距不大

# remap
def build_map(df, col_name):
    key = sorted(df[col_name].unique().tolist())
    m = dict(zip(key, range(len(key))))
    df[col_name] = df[col_name].map(lambda x: m[x])
    return m, key


def print_to_file(data, fout):
    for i in range(len(data)):
        fout.write(str(data[i]))
        if i != len(data) - 1:
            fout.write(' ')
        else:
            fout.write(';')


class DataProcessingClass:
    def __init__(self):
        self.user2user_encoded = {}
        self.userencoded2user = {}
        self.movie2movie_encoded = {}
        self.movie_encoded2movie = {}
        self.cat2cat_encoded = {}
        self.cat_encoded2cat = {}
        self.num_users = 0
        self.num_movies = 0
        self.cate_list=[]

    # 获得只有user, item, rating, timestamp 和 category的df
    def pre_process(self, rating_dir, movies_dir):
        # 处理ratings文件部分
        # load data
        rating_names = ['user_id', 'movie_id', 'rating', 'timestamp']
        df = pd.read_table(rating_dir, sep='::', header=None, names=rating_names, encoding='ISO-8859-1')

        # user id
        user_ids = df["user_id"].unique().tolist()
        print(len(user_ids))
        # remap
        self.user2user_encoded = {x: i for i, x in enumerate(user_ids)}
        self.userencoded2user = {i: x for i, x in enumerate(user_ids)}

        # movie id
        movie_ids = df["movie_id"].unique().tolist()
        print(len(movie_ids))
        # remap ps：though the max id is 3592, there are no so many movies, so remap is necessary
        self.movie2movie_encoded = {x: i for i, x in enumerate(movie_ids)}
        self.movie_encoded2movie = {i: x for i, x in enumerate(movie_ids)}
        df["user"] = df["user_id"].map(self.user2user_encoded)
        df["movie"] = df["movie_id"].map(self.movie2movie_encoded)

        num_users = len(self.user2user_encoded)
        num_movies = len(self.movie_encoded2movie)
        self.num_users = num_users
        self.num_movies = num_movies
        df["rating"] = df["rating"].values.astype(np.float32)

        min_rating = min(df["rating"])
        max_rating = max(df["rating"])

        # 需要把电影的category加进来
        movies_names = ['movie_id', "title", "genres"]
        mv_df = pd.read_table(movies_dir, sep='::', header=None, names=movies_names, encoding='ISO-8859-1')
        # 处理cate
        for i, row in mv_df.iterrows():
            genres = row["genres"]
            mv_df.loc[i, 'category'] = genres.split('|')[0]
        mv_df["movie"] = mv_df["movie_id"].map(self.movie2movie_encoded)
        mv_df = mv_df[['movie', 'category']]
        cat_ids = mv_df["category"].unique().tolist()
        print(len(cat_ids))
        self.cat2cat_encoded = {x: i for i, x in enumerate(cat_ids)}
        self.cat_encoded2cat = {i: x for i, x in enumerate(cat_ids)}
        mv_df['category'] = mv_df['category'].map(self.cat2cat_encoded)
        # 按新movie序号排序，这样后面才能直接映射成cate
        mv_df = mv_df.sort_values('movie')
        mv_df = mv_df.reset_index(drop=True)

        cate_list = [mv_df['category'][i] for i in range(self.num_movies)]
        self.cate_list = np.array(cate_list, dtype=np.int32)
        cate_count = len(cat_ids)

        print(
            "Number of users: {}, Number of Movies: {}, Number of categories: {}, Min rating: {}, Max rating: {}".format(
                num_users, num_movies, cate_count, min_rating, max_rating
            )
        )

        # 添加一列category标签
        df["category"] = 0
        for i, row in df.iterrows():
            movie = row["movie"]
            df.loc[i, 'category'] = list(mv_df[mv_df["movie"] == movie]["category"])[0]

        return df, mv_df, min_rating, max_rating, num_users, num_movies

    def build_dataset(self, df, mv_df):
        train_set = []
        test_set = []

        reviews_df = df.sort_values(['user', 'timestamp'])
        reviews_df = reviews_df.reset_index(drop=True)
        reviews_df = reviews_df[['user', 'movie', 'timestamp']]

        for reviewerID, hist in reviews_df.groupby('user'):
            pos_list = hist['movie'].tolist()

            def gen_neg():
                neg = pos_list[0]
                while neg in pos_list:
                    neg = random.randint(0, self.num_movies - 1)
                return neg

            neg_list = [gen_neg() for i in range(len(pos_list))]
            # neg_list = [gen_neg()]

            # for i in range(1, len(pos_list)):
            #     hist = pos_list[:i]
            #     if i != len(pos_list) - 1:
            #         train_set.append((reviewerID, hist, pos_list[i], 1))
            #     else:
            #         label = (pos_list[i], neg_list[i])
            #         test_set.append((reviewerID, hist, label))

            # 改成留一法，不然数据量太大了
            i = max(1, len(pos_list)-2)
            if i != len(pos_list) - 1:
                train_set.append((reviewerID, pos_list[:i], pos_list[len(pos_list)-2], 1))
                train_set.append((reviewerID, pos_list[:i], neg_list[len(pos_list)-2], 0))
            label = (pos_list[len(pos_list)-1], neg_list[len(pos_list)-1])
            test_set.append((reviewerID, pos_list[:(len(pos_list) - 1)], label))

        random.shuffle(train_set)
        random.shuffle(test_set)

        assert len(test_set) == self.num_users

        def print_to_file(data, fout):
            for i in range(len(data)):
                fout.write(str(data[i]))
                if i != len(data) - 1:
                    fout.write(' ')
                else:
                    fout.write(';')

        train_path = os.path.join("../train", "ml1M_train.txt")
        test_path = os.path.join("../test", "ml1M_test.txt")

        print("make train data")
        with open(train_path, "w") as fout:
            for line in train_set:
                history = line[1]
                target = line[2]
                label = line[3]
                cate = [self.cate_list[x] for x in history]
                print_to_file(history, fout)
                print_to_file(cate, fout)
                fout.write(str(target) + ";")
                fout.write(str(self.cate_list[target]) + ";")
                fout.write(str(label) + "\n")

        print("make test data")
        with open(test_path, "w") as fout:
            for line in test_set:
                history = line[1]
                target = line[2]
                cate = [self.cate_list[x] for x in history]

                print_to_file(history, fout)
                print_to_file(cate, fout)
                fout.write(str(target[0]) + ";")
                fout.write(str(self.cate_list[target[0]]) + ";")
                fout.write("1\n")

                print_to_file(history, fout)
                print_to_file(cate, fout)
                fout.write(str(target[1]) + ";")
                fout.write(str(self.cate_list[target[1]]) + ";")
                fout.write("0\n")

        print("make config data")
        with open('ml1M_config.txt', 'w') as f:
            f.write(str(self.num_users) + "\n")
            f.write(str(self.num_movies) + "\n")
            f.write(str(len(self.cate_list)) + "\n")


if __name__ == '__main__':
    DC = DataProcessingClass()
    df, mv_df, min_rating, max_rating, num_users, num_movies = DC.pre_process("../rawdata/ratings.dat", "../rawdata/movies.dat")
    DC.build_dataset(df, mv_df)
