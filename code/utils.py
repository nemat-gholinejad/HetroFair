from torch_sparse import SparseTensor
from collections import Counter
from copy import deepcopy
import numpy as np
import torch
import yaml


def set_seed(seed: int) -> None:
    np.random.seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed(seed)
        torch.cuda.manual_seed_all(seed)
    torch.manual_seed(seed)


def minibatch(batch_size, *tensors):
    if len(tensors) == 1:
        tensor = tensors[0]
        for i in range(0, len(tensor), batch_size):
            yield tensor[i:i + batch_size]
    else:
        for i in range(0, len(tensors[0]), batch_size):
            yield tuple(x[i:i + batch_size] for x in tensors)


def negative_sampling(postive_list, di_edge_index, neg_count):
    users = np.random.randint(0, len(postive_list), di_edge_index.size()[1])
    num_item = di_edge_index[1].max().item() + 1
    sample = []
    for i, user in enumerate(users):
        negative_items = []
        user_positives = postive_list[user]
        if len(user_positives) == 0:
            continue

        positem = user_positives[np.random.randint(0, len(user_positives))]
        while len(negative_items) < neg_count:
            negitem = np.random.randint(0, num_item)
            if negitem in user_positives or negitem in negative_items:
                continue
            else:
                negative_items.append(negitem)

        sample.append([user, positem] + negative_items)

    return torch.Tensor(sample).long()


def shuffle(*arrays: object, **kwargs: object) -> object:
    require_indices = kwargs.get('indices', False)

    if len(set(len(x) for x in arrays)) != 1:
        raise ValueError('All inputs to shuffle must have '
                         'the same length.')

    shuffle_indices = np.arange(len(arrays[0]))
    np.random.shuffle(shuffle_indices)

    if len(arrays) == 1:
        result = arrays[0][shuffle_indices]
    else:
        result = tuple(x[shuffle_indices] for x in arrays)

    if require_indices:
        return result, shuffle_indices
    else:
        return result


def read_config(config_path, model_name):
    with open(f"{config_path}/{model_name}/config.yml") as f:
        config = yaml.safe_load(f)

    return config


def tensorboard_writer(wrtr, data, train_loss, val_loss, **kwargs):
    epoch = kwargs.pop('epoch')
    splited_metrics = kwargs.pop('Split')
    wrtr.add_scalar(f"{data}_Loss/Train", train_loss, epoch)
    wrtr.add_scalar(f"{data}_Loss/Test", val_loss, epoch)
    for metric, value in kwargs.items():
        wrtr.add_scalar(f"{data}_Test/{metric}", value, epoch)

    for metric, value in splited_metrics.items():
        if metric.startswith('pop'):
            wrtr.add_scalar(f"{data}_Popular/{metric[4:]}", value, epoch)
        else:
            wrtr.add_scalar(f"{data}_UnPopular/{metric[6:]}", value, epoch)


def create_sparse_tensor(row, col, size):
    return SparseTensor(row=row, col=col, sparse_sizes=size)


def get_items_popularity(edge_index, top_rated=0.2):
    edge_index = edge_index.numpy()
    item_count = np.array(list(Counter(edge_index[1, :]).items()))
    item_count = np.array(sorted(item_count, key=lambda x: x[1], reverse=True))

    popular_items = item_count[:int(item_count.shape[0] * top_rated)][:, 0]
    long_tails = item_count[int(item_count.shape[0] * top_rated):][:, 0]

    return popular_items, long_tails


# Checkpoint for model save and Early Stopping
class Monitor:
    def __init__(self, model, optimizer, metric_direction='lower', patience=0, filename='best_model_checkpoint.ph'):
        self.model = model
        self.optimizer = optimizer
        self.filename = filename
        self.metric_direction = metric_direction
        self.best_metric = None
        self.best_model_state_dict = None
        self.best_optimizer_state_dict = None
        self.best_weight_decay = None
        self.best_results = None
        self.best_epoch = None
        self.patience = patience

    def metric_check(self, current_metric, best_metric):
        if self.metric_direction == 'lower':
            return current_metric < best_metric
        elif self.metric_direction == 'higher':
            return current_metric > best_metric
        else:
            raise ValueError("Invalid metric_direction. It should be 'lower' or 'higher'.")

    def is_better(self, metric):
        return self.best_metric is None or self.metric_check(metric, self.best_metric)

    def coninue_check(self, epoch, metric_value, **kwargs):
        _continue = True
        if self.is_better(metric_value):
            self.best_metric = metric_value
            self.best_model_state_dict = deepcopy(self.model.state_dict())
            self.best_optimizer_state_dict = deepcopy(self.optimizer.state_dict())
            self.best_epoch = epoch
            self.best_results = kwargs
        elif epoch - self.best_epoch > self.patience:
            _continue = False

        return _continue

    def save_best_model(self):
        if self.best_model_state_dict is not None:
            checkpoint = {
                'model_state_dict': self.best_model_state_dict,
                'optimizer_state_dict': self.best_optimizer_state_dict,
                'best_metric': self.best_metric,
                'direction': self.metric_direction,
                'epoch': self.best_epoch,
                'results': self.best_results
            }
            torch.save(checkpoint, self.filename)
            print(f"Best Model:\nEpoch: {checkpoint['epoch']}, result: {checkpoint['results']}")

    def load_best_model(self):
        checkpoint = torch.load(self.filename)
        return checkpoint['model_state_dict']
