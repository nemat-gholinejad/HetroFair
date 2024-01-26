from collections import Counter
from scipy import stats as ss
import numpy as np
import torch


# computes NDCG@K
def NDCGatK_r(groundTruth, r, k):
    """Computes Normalized Discounted Cumulative Gain (NDCG) @ k

    Args:
        groundTruth (list): list of lists containing highly rated items of each user
        r (list): list of lists indicating whether each top k item recommended to each user
            is a top k ground truth item or not
        k (int): determines the top k items to compute ndcg on

    Returns:
        float: ndcg @ k
    """
    assert len(r) == len(groundTruth)

    test_matrix = torch.zeros((len(r), k))

    for i, items in enumerate(groundTruth):
        length = min(len(items), k)
        test_matrix[i, :length] = 1
    max_r = test_matrix
    idcg = torch.sum(max_r * 1. / torch.log2(torch.arange(2, k + 2)), axis=1)
    dcg = r * (1. / torch.log2(torch.arange(2, k + 2)))
    dcg = torch.sum(dcg, axis=1)
    idcg[idcg == 0.] = 1.
    ndcg = dcg / idcg
    ndcg[torch.isnan(ndcg)] = 0.
    return round(torch.mean(ndcg).item(), 4)


def Coverage_AtK(top_K_items, category):
    coverage = stat(top_K_items, category)
    coverage = np.sum(list(map(lambda x: x.size, coverage)))

    return round(coverage / top_K_items.shape[0], 4)


def PRU_AtK(users, rating, test_user_pos_items, exclude_items):
    pru_pop = Counter(exclude_items)
    sp_coeff = []
    rating = rating.detach().cpu().numpy()
    for user in users:
        user_pru_pop = [pru_pop[x] for x in test_user_pos_items[user]]
        if len(user_pru_pop) > 1 and len(set(user_pru_pop)) != 1:
            ranking_list = np.argsort(rating[user])
            ranks = np.zeros_like(ranking_list)
            ranks[ranking_list] = np.arange(1, len(rating[user]) + 1)
            ranks = rating.shape[1] - ranks
            test_items_rank = ranks[test_user_pos_items[user]]
            sp_coeff.append(ss.spearmanr(user_pru_pop, test_items_rank)[0])

    pru = -sum(sp_coeff) / len(sp_coeff)
    pru = round(pru, 4)
    return pru


def PRI_AtK(rating, test_user_pos_items, exclude_items):
    item_ditribution = {}
    for index, profile in enumerate(test_user_pos_items):
        for item in profile:
            item_ditribution.setdefault(item, []).append(index)

    item_intraction_count = Counter(exclude_items)

    rating = rating.detach().cpu().numpy()
    ranking_list = ss.rankdata(rating, method='ordinal', axis=1)
    ranking_list = rating.shape[1] - ranking_list
    items_avg_rank = []
    for item, user_profiles in item_ditribution.items():
        avg_rank = ranking_list[user_profiles, item]
        items_avg_rank.append(np.sum(avg_rank) // len(user_profiles))
    pop = []
    for item in item_ditribution.keys():
        pop.append(item_intraction_count[item])

    pri = -ss.spearmanr(pop, items_avg_rank)[0]
    return round(pri, 4)


def MRR_AtK(r):
    reciprocal_rank = [(1 / (torch.argmax(x) + 1)).item() if x.sum() != 0 else 0 for x in r]
    mean_reciprocal_rank = np.sum(reciprocal_rank) / r.size()[0]
    return round(mean_reciprocal_rank, 4)


def MAP_AtK(groundTruth, r):
    r = r.numpy()
    if np.sum(r) == 0:
        return 0
    comulative_av = np.cumsum(r, axis=1) / np.arange(1, r.shape[1] + 1)
    user_av = []
    r = r.astype(np.bool)
    for i, (prediction_list, mask) in enumerate(zip(comulative_av, r)):
        av = np.sum(prediction_list[mask]) / len(groundTruth[i])
        user_av.append(av)

    mean_average_precision = np.sum(user_av) / len(groundTruth)
    return round(mean_average_precision, 4)
