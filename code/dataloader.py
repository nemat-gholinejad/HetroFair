from torch import Tensor
import pandas as pd
import numpy as np
import torch
import json
import os


# load user and item nodes
def load_node_csv(path: str) -> dict:
    uids = []
    trainItem = []
    with open(path) as f:
        for l in f.readlines():
            if len(l) > 0:
                l = l.strip('\n').split(' ')
                items = [int(i) for i in l[1:]]
                uids.append(int(l[0]))
                trainItem.extend(items)

    user_mapping = {index: i for i, index in enumerate(uids)}
    start_index = len(user_mapping)
    item_mapping = {index: i + start_index for i, index in enumerate(np.unique(trainItem))}

    return user_mapping, item_mapping


# load edges between users and items
def load_edge_csv(path: str, users_mapping, items_mapping) -> Tensor:
    unique_users, item, user = [], [], []
    train_data_size = 0
    with open(path) as f:
        for l in f.readlines():
            if len(l) > 0:
                l = l.strip('\n').split(' ')
                # remove if item just seen in test data
                items = [items_mapping.get(int(i), -1) for i in l[1:]]
                items = list(filter(lambda x: x != -1, items))
                uid = int(l[0])
                unique_users.append(uid)
                user.extend([uid] * len(items))
                item.extend(items)
                train_data_size += len(items)

    edge_index = np.array([user, item])
    undirect_edge_index = np.concatenate((edge_index, np.flip(edge_index, axis=0)), axis=1)
    edge_index[1] = edge_index[1] - len(users_mapping)
    return torch.as_tensor(edge_index).to(torch.long), torch.as_tensor(undirect_edge_index).to(torch.long)


def get_positive_intractions(directed_edge_index):
    pos_items = pd.DataFrame(directed_edge_index.T, columns=['uid', 'iid']).groupby('uid')['iid'].apply(list).tolist()
    sorted_list = [sorted(sub_list) for sub_list in pos_items]
    return sorted_list


def read_category(path):
    if path.split('/')[2].startswith('movielens'):
        with open(f'{os.path.split(path)[0]}/movie_to_id.txt') as f:
            lines = f.readlines()
            arr = np.array(list(map(lambda x: x.strip().split(','), lines))).astype(int)
            old2newid = dict(zip(arr[:, 0], arr[:, 1]))
        cat = []
        with open(path, 'r') as f:
            lines = f.readlines()
            for line in lines:
                line = line.strip().split(',')
                item, category = int(line[0]), int(line[1])
                if item in old2newid:
                    cat.append([old2newid[item], category])

            cat.sort(key=lambda x: x[0])
            dic = dict(zip(np.array(cat)[:, 0], np.array(cat)[:, 1]))
    else:
        cat = []
        with open(path, 'r') as f:
            lines = f.readlines()
            for line in lines:
                line = line.strip().split(',')
                item, category = int(line[0]), int(line[1])
                cat.append([item, category])

            cat.sort(key=lambda x: x[0])
            dic = dict(zip(np.array(cat)[:, 0], np.array(cat)[:, 1]))
    return dic
