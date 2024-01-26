import argparse


def parse_args():
    parser = argparse.ArgumentParser()
    parser.add_argument('--batch_size', type=int, default=2048, help='batch size')
    parser.add_argument('--embedding_size', type=int, default=128, help='the nodes embeddings size')
    parser.add_argument('--layer', type=int, default=4, help='number of propgation layer')
    parser.add_argument('--negative', type=int, default=1, help='number of negative samples')
    parser.add_argument('--topk', type=int, default=20, help='test on topk results')
    parser.add_argument('--epochs', type=int, default=1000, help='number of training epochs')
    parser.add_argument('--seed', type=int, default=2020, help='random seed')
    parser.add_argument('--lmbda', type=float, default=0.6, help='lambda in Equation 13')
    parser.add_argument('--lr', type=float, default=0.0005, help='the learning rate')
    parser.add_argument('--beta', type=float, default=0.0001, help='the regularization coefficient')
    parser.add_argument('--tail_res', type=bool, default=False, help='accuracy result for long-tail items')
    parser.add_argument('--weight_decay', type=float, default=0.0001, help='the weight decay for beta regularization')
    parser.add_argument('--model', type=str, default='HetroFair', help='model name')
    parser.add_argument('--dataset', type=str, default='amazon/Health',
                        help='dataset, example: [epinions, amazon/Health]')

    return parser.parse_args()
