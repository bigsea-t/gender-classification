import numpy as np
import tensorflow as tf

import csv
import os
import re
import collections
import itertools
from sklearn import preprocessing


def read_csv(filepath):
    with tf.gfile.GFile(filepath, "r") as f:
        reader = csv.reader(f, delimiter=',')
        return list(reader)

def string_processing(str):
    str = str.replace("\n", " ")
    str = str.strip()
    str = str.replace(".", " <EOS>")
    str = str.replace("(", "( ")
    str = str.replace(")", " )")
    str = str.replace(",", " ,")
    str = re.sub(r"\s+", " ", str)
    return str


def build_vocab(posts, min_appear=5):
    flattened = [word for post in posts for word in post]
    counter = collections.Counter(flattened)

    count_pairs = sorted(counter.items(), key=lambda x: (-x[1], x[0]))

    words, _ = list(zip(*count_pairs))
    word_to_id = {k:v if counter[k] >= min_appear else len(words) for k, v in zip(words, range(len(words)))}

    return word_to_id

def converted_data(data_dir, min_nwords=200, max_len=None):
    filepath = os.path.join(data_dir, "blog-gender-dataset.csv")

    posts = []
    labels = []

    raw_data = read_csv(filepath)

    for row in raw_data:
        words = string_processing(row[0]).split(" ")
        if len(words) < min_nwords:
            continue

        posts.append(words)
        gen2num = {'f':0, 'm': 1}
        labels.append(gen2num[row[1].strip().lower()])

    lengths = [len(post) for post in posts]

    if max_len is None:
        max_len = max(lengths)

    padded_posts = [[post[j] if j < lengths[i] else "<PAD>" for j in range(max_len)]
                                                            for i, post in enumerate(posts)]

    word_to_id = build_vocab(padded_posts)

    id_posts = [[word_to_id[word] for word in post] for post in padded_posts]

    print('vocab size:', len(collections.Counter([w for p in id_posts for w in p])))

    id_posts = np.array(id_posts, dtype=np.int32)

    labels = np.array(labels)

    bin_labels = np.zeros((labels.shape[0], 2))
    bin_labels[labels==0, 0] = 1
    bin_labels[labels==1, 1] = 1

    return id_posts, bin_labels


def split_rawdata(raw_data, ratio=[.7, .1, .2]):
    posts, labels = raw_data
    n_data = len(labels)

    n_train = int(n_data * ratio[0])
    n_valid = int(n_data * ratio[1])

    return (posts[:n_train], labels[:n_train]),\
           (posts[n_train:n_train+n_valid], labels[n_train:n_train+n_valid]),\
           (posts[n_train+n_valid:], labels[n_train+n_valid:])


def print_stats(posts, labels):
    len_posts = [len(p) for p in posts]

    print('num posts:', len(posts))
    print('ave words:', sum(len_posts) / len(posts))
    print('min words:', min(len_posts))
    print('max words:', max(len_posts))

    print('num Male:', sum(labels))
    print('num Female:', len(posts) - sum(labels))


def data_iterator(data, batch_size, num_steps):
    """
    :param data:
    :param batch_size: num of posts in a batch
    :param num_steps: num of words in a single row of batch
    :return:
    """
    posts, labels = data

    num_posts, len_post = posts.shape
    assert num_posts == labels.shape[0]

    len_batch = num_posts // batch_size
    len_step = len_post // num_steps


    if len_batch < 1:
        raise ValueError("batch_size must be <= num posts")

    if len_step < 1:
        raise ValueError("num_step must be <= len_posts")

    for i_batch in range(len_batch):
        for i_step in range(len_step):
            yield posts[i_batch*batch_size:(i_batch+1)*batch_size, i_step*num_steps:(i_step+1)*num_steps],\
                  labels[i_batch*batch_size:(i_batch+1)*batch_size]


if __name__ == '__main__':
    print('reader main')
    posts, labels = converted_data('data', min_nwords=100)
    print_stats(posts, labels)