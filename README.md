This is the official implementation for our HetroFair paper:
>Heterophily-Aware Fair Recommendation using Graph Convolutional Networks

## Requirements
`pip install -r requirements.txt`

## Reproducibility
run HetroFair on epinions dataset:

`python main.py --batch_size=2048 --embedding_size=128 --layer=4 --topk=20 --lmbda=0.6 --lr=0.0005 --seed=2020 --tail_res=False --dataset="epinions"`
