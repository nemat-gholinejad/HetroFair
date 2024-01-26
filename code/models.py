from torch_geometric.nn.conv.gcn_conv import gcn_norm
from torch_geometric.nn.conv import MessagePassing
from torch_sparse import SparseTensor, matmul, mul
from torch_sparse import sum as sparsesum
from torch import nn, Tensor
import torch


class HetroFair(MessagePassing):
    def __init__(self, users_cnt: int, items_cnt: int, config, add_self_loops: bool = False):
        """Initializes HetroFair Model

        Args:
            users_cnt (int): Number of users
            items_cnt (int): Number of items
            config file: model config file
            add_self_loops (bool, optional): Whether to add self loops for message passing. Defaults to False.
        """

        super().__init__()
        self.name = 'HetroFair'
        self.num_users, self.num_items = users_cnt, items_cnt
        self.embedding_dim, self.K = config.embedding_size, config.layer
        self.add_self_loops = add_self_loops

        self.users_emb = nn.Embedding(num_embeddings=self.num_users, embedding_dim=self.embedding_dim)
        self.items_emb = nn.Embedding(num_embeddings=self.num_items, embedding_dim=self.embedding_dim)
        self.params = nn.ParameterDict()
        for layer in range(self.K):
            self.params.update({str(layer): nn.Parameter(nn.init.xavier_normal_(torch.empty(1, self.embedding_dim)))})
            self.params.update(
                {'b' + str(layer): nn.Parameter(nn.init.xavier_normal_(torch.empty(1, self.embedding_dim)))})

        self.coef = config.lmbda

        nn.init.xavier_normal_(self.users_emb.weight)
        nn.init.xavier_normal_(self.items_emb.weight)

    def forward(self, edge_index: SparseTensor):
        """Forward propagation of HetroFair Model.

        Args:
            edge_index (SparseTensor): adjacency matrix

        Returns:
            tuple (Tensor): h_u^K, h_i^K
        """

        edge_index_norm = gcn_norm(edge_index, add_self_loops=self.add_self_loops)

        emb_0 = torch.cat([self.users_emb.weight, self.items_emb.weight])  # H^0
        embs = [emb_0]
        emb_k = emb_0

        for i in range(self.K):
            dot = torch.sum(torch.mul(torch.index_select(emb_k, dim=0, index=edge_index.t().coo()[0]),
                                      torch.index_select(emb_k, dim=0, index=edge_index.t().coo()[1])), dim=1)

            dot = dot.unsqueeze(1)
            dot = torch.relu(torch.matmul(dot, self.params[str(i)] + self.params['b' + str(i)]))

            weight = 1 / (self.coef * torch.sigmoid(dot))
            norm_weight = torch.mul(edge_index_norm.t().coo()[2].unsqueeze(1), weight)
            emb_k = self.propagate(edge_index, x=emb_k, norm=norm_weight)

            embs.append(emb_k)

        layers = torch.stack(embs, dim=1)
        emb_final = torch.mean(layers, dim=1)

        users_emb_final, items_emb_final = torch.split(emb_final, [self.num_users, self.num_items])

        return users_emb_final, items_emb_final

    def message(self, x_j: Tensor, norm):
        return norm * x_j

    def cal_loss(self, users_embd: Tensor, items_embd: Tensor, config, *tensors):
        """Bayesian Personalized Ranking Loss as described in https://arxiv.org/abs/1205.2618
        """
        users_emb_K = users_embd[tensors[0]]
        pos_items_emb_K = items_embd[tensors[1]]
        neg_items_emb_K = items_embd[tensors[2]]
        users_emb_0 = self.users_emb(tensors[0])
        pos_items_emb_0 = self.items_emb(tensors[1])
        neg_items_emb_0 = self.items_emb(tensors[2])
        reg_loss = config.beta * (
                users_emb_0.norm(2).pow(2) + pos_items_emb_0.norm(2).pow(2) + neg_items_emb_0.norm(2).pow(2) + sum(
            torch.norm(param) ** 2 for param in self.params.values()))

        pos_scores = torch.mul(users_emb_K, pos_items_emb_K)
        pos_scores = torch.sum(pos_scores, dim=1)  # predicted scores of positive samples
        loss = 0
        for i in range(config.negative):
            neg_scores = torch.mul(users_emb_K, neg_items_emb_K[:, i, :])
            neg_scores = torch.sum(neg_scores, dim=1)  # predicted scores of negative samples

            loss += torch.mean(-nn.functional.logsigmoid(pos_scores - neg_scores))

        loss /= config.negative
        loss += config.weight_decay * reg_loss

        return loss
