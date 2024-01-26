from torch.utils.tensorboard import SummaryWriter
from torch import nn, optim, Tensor
from torch_sparse import matmul
from tqdm.notebook import tqdm
from parse import parse_args
import numpy as np
import dataloader
import evaluator
import datetime
import warnings
import models
import torch
import utils
import os
import sys

warnings.filterwarnings("ignore")


def train(model, optimizer, sparse_edge_index, train_edge_index, train_pos_intactions, config):
    model.train()
    s = utils.negative_sampling(train_pos_intactions, train_edge_index, config.negative)
    s = s.to(config.device)
    users, pos_items, neg_items = s[:, 0], s[:, 1], s[:, 2:]
    neg_items = neg_items.reshape((neg_items.size()[0], config.negative))

    users, pos_items, neg_items = utils.shuffle(users, pos_items, neg_items)

    total_batch = len(users) // config.batch_size + 1
    avg_loss = 0
    for (batch_i, (b_users, b_pos, b_neg)) in enumerate(
            utils.minibatch(config.batch_size, users, pos_items, neg_items)):
        users_emb_final, items_emb_final = model(sparse_edge_index)

        loss = model.cal_loss(users_emb_final, items_emb_final, config, b_users, b_pos, b_neg)
        avg_loss += loss

        optimizer.zero_grad()
        loss.backward()
        optimizer.step()

    avg_loss /= total_batch
    return round(avg_loss.item(), 4)


def evaluate(model, sparse_edge_index, test_edge_index, exclude_edge_indices, test_pos_intactions, config, tail_res):
    model.eval()
    neg_cnt = 1

    s = utils.negative_sampling(test_pos_intactions, test_edge_index, neg_cnt)
    s = s.to(config.device)
    users, pos_items, neg_items = s[:, 0], s[:, 1], s[:, 2]
    users, pos_items, neg_items = utils.shuffle(users, pos_items, neg_items)
    neg_items = neg_items.reshape((neg_items.size()[0], neg_cnt))
    with torch.no_grad():
        user_embedding, item_embedding = model(sparse_edge_index)
        loss = model.cal_loss(user_embedding, item_embedding, config, users, pos_items, neg_items)
        f = torch.nn.Sigmoid()
        rating = f(torch.matmul(user_embedding, item_embedding.T))
        for exclude_edge_index in exclude_edge_indices:
            exclude_edge_index = exclude_edge_index.cpu().numpy()
            user_pos_items = dataloader.get_positive_intractions(exclude_edge_index)

            exclude_users = []
            exclude_items = []
            for user, items in enumerate(user_pos_items):
                exclude_users.extend([user] * len(items))
                exclude_items.extend(items)

            rating[exclude_users, exclude_items] = -(1 << 10)

        _, top_K_items = torch.topk(rating, k=config.topk)

        users = test_edge_index[0].unique().cpu().numpy().tolist()

        top_K_items = top_K_items.cpu().numpy()
        r = []
        for user in users:
            label = list(map(lambda x: x in test_pos_intactions[user], top_K_items[user]))
            r.append(label)

        r = torch.Tensor(np.array(r).astype('float'))
        ndcg = evaluator.NDCGatK_r(test_pos_intactions, r, config.topk)
        pru = evaluator.PRU_AtK(users, rating, test_pos_intactions, exclude_items)
        pri = evaluator.PRI_AtK(rating, test_pos_intactions, exclude_items)
        mrr = evaluator.MRR_AtK(r)
        ma_precision = evaluator.MAP_AtK(test_pos_intactions, r)

        splited_result = {}
        if tail_res:
            diff = [popular_items, long_tails]
            mark = ['unpop', 'pop']
            for i in range(2):
                temp_test_pos_intractions = {}
                for user in users:
                    items = list(set(test_pos_intactions[user]) - set(diff[i]))
                    if len(items) > 0:
                        temp_test_pos_intractions[user] = items
                r = []
                for user in list(temp_test_pos_intractions.keys()):
                    label = list(map(lambda x: x in temp_test_pos_intractions[user], top_K_items[user]))
                    r.append(label)
                r = torch.Tensor(np.array(r).astype('float'))
                splited_result[f"{mark[i]}_NDCG"] = evaluator.NDCGatK_r(list(temp_test_pos_intractions.values()), r,
                                                                        config.topk)
                splited_result[f"{mark[i]}_MAP"] = evaluator.MAP_AtK(list(temp_test_pos_intractions.values()), r)
                splited_result[f"{mark[i]}_MRR"] = evaluator.MRR_AtK(r)

    return round(loss.item(), 4), ndcg, mrr, ma_precision, pru, pri, splited_result


if __name__ == "__main__":
    config = parse_args()
    data_path = '../Datasets'
    dataset = config.dataset
    config.device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    utils.set_seed(config.seed)
    path = f'{data_path}/{dataset}'

    train_path = f'{path}/train.txt'
    test_path = f'{path}/test.txt'

    user_mapping, item_mapping = dataloader.load_node_csv(train_path)
    num_users, num_items = len(user_mapping), len(item_mapping)

    di_train_edge_index, undi_train_edge_index = dataloader.load_edge_csv(train_path, user_mapping, item_mapping)
    di_test_edge_index, undi_test_edge_index = dataloader.load_edge_csv(test_path, user_mapping, item_mapping)

    popular_items, long_tails = utils.get_items_popularity(di_train_edge_index)

    train_pos_items = dataloader.get_positive_intractions(di_train_edge_index)
    test_pos_items = dataloader.get_positive_intractions(di_test_edge_index)

    sprs_size = (num_users + num_items, num_users + num_items)
    train_sparse_edge_index = utils.create_sparse_tensor(undi_train_edge_index[0], undi_train_edge_index[1], sprs_size)
    test_sparse_edge_index = utils.create_sparse_tensor(undi_test_edge_index[0], undi_test_edge_index[1], sprs_size)

    train_sparse_edge_index = train_sparse_edge_index.to(config.device)
    test_sparse_edge_index = test_sparse_edge_index.to(config.device)

    model = getattr(models, config.model)(num_users, num_items, config)
    model = model.to(config.device)
    optimizer = optim.Adam(model.parameters(), lr=config.lr)

    current_datetime = datetime.datetime.now()
    current_datetime = current_datetime.strftime("%Y-%m-%d-%H-%M-%S")
    try:
        checkpoint_filename = f"{model.name}_{dataset.split('/')[1]}_batch_{config.batch_size}_lr_{config.lr}_neg_" \
                              f"{config.negative}_topk_{config.topk}_lmbd_{config.beta}"
    except:
        checkpoint_filename = f"{model.name}_{dataset}_batch_{config.batch_size}_lr_{config.lr}_neg_" \
                              f"{config.negative}_topk_{config.topk}_lmbd_{config.beta}"

    tensorboard_filename = f"{model.name}_batch_{config.batch_size}_lr_{config.lr}_neg_{config.negative}" \
                           f"_topk_{config.topk}_lmbd_{config.beta}"

    save_file = f"./checkpoints/{checkpoint_filename}_date_{current_datetime}_ndcg_best_model.pth.tar"
    monitor = utils.Monitor(model, optimizer, 'higher', patience=20, filename=save_file)
    writer = SummaryWriter(f'./runs/{tensorboard_filename}_date_{current_datetime}/{dataset}')

    for epoch in tqdm(range(1, config.epochs + 1)):
        train_loss = train(model, optimizer, train_sparse_edge_index, di_train_edge_index, train_pos_items, config)
        print(f'EPOCH {epoch}/{config.epochs} loss: {train_loss}')
        if epoch % 1 == 0:
            model.eval()
            val_loss, ndcg, mrr, ma_p, pru, pri, pop_unpop_metrics = \
                evaluate(model, train_sparse_edge_index, di_test_edge_index, [di_train_edge_index], test_pos_items,
                         config, config.tail_res)

            utils.tensorboard_writer(writer, dataset, train_loss, val_loss, epoch=epoch, NDCG=ndcg, MRR=mrr, MAP=ma_p,
                                     PRU=pru, PRI=pri, Split=pop_unpop_metrics)
            print(f"Eval: Loss: {val_loss}, NDCG: {ndcg}, MRR: {mrr}, MAP: {ma_p}, PRU:{pru}, PRI:{pri}")

            if config.tail_res: print(pop_unpop_metrics)
            if not monitor.coninue_check(epoch, ndcg, val_loss=val_loss, ndcg=ndcg, MRR=mrr, MAP=ma_p, PRU=pru, PRI=pri,
                                         Split=pop_unpop_metrics):
                print("Training Stopped due to early stop mechanism")
                break

    monitor.save_best_model()
    writer.flush()
