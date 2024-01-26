## Amazon k-core preprocess
# # Amazon K-Core dataset
# import pandas as pd
# import json
# from collections import Counter
#
# f = open("amazon/Beauty_5.json")
# # f = open("amazon/reviews_Beauty.json")
# # f = open("amazon/Electronics_10.json")
# review_id = list()
# review_item = list()
# data = list()
# new_data = list()
# cnt = 0
# k = 10
# for i, l in enumerate(f):
#     l = eval(l)
#     data.append(l)
#     review_id.append(l["reviewerID"])
#     review_item.append(l["asin"])
#     cnt += 1
#
# while True:
#     a = Counter(review_id)
#     aa = Counter(review_item)
#
#     b = list(filter(lambda x: a[x] < k, set(review_id)))
#     c = list(filter(lambda x: aa[x] < k, set(review_item)))
#     b = set(b)
#     c = set(c)
#     for i, record in enumerate(data):
#         if record["reviewerID"] not in b and record["asin"] not in c:
#             new_data.append(record)
#
#     data = []
#     data.extend(new_data)
#     new_data = []
#     review_id = [record["reviewerID"] for record in data]
#     review_item = [record["asin"] for record in data]
#
#     if len(b) + len(c) == 0:
#         break

# Copyright (c) Recommenders contributors.
# Licensed under the MIT License.


import os
import shutil
import pandas as pd
import gzip
import random
import logging
import _pickle as cPickle
import requests
import math
from tqdm import tqdm
import pandas as pd
import itertools
import numpy as np

random.seed(10)
np.random.seed(10)
logger = logging.getLogger()


def maybe_download(url, filename=None, work_directory=".", expected_bytes=None):
    """Download a file if it is not already downloaded.

    Args:
        filename (str): File name.
        work_directory (str): Working directory.
        url (str): URL of the file to download.
        expected_bytes (int): Expected file size in bytes.

    Returns:
        str: File path of the file downloaded.
    """
    if filename is None:
        filename = url.split("/")[-1]
    os.makedirs(work_directory, exist_ok=True)
    filepath = os.path.join(work_directory, filename)
    if not os.path.exists(filepath):
        r = requests.get(url, stream=True)
        if r.status_code == 200:
            # log.info(f"Downloading {url}")
            total_size = int(r.headers.get("content-length", 0))
            block_size = 1024
            num_iterables = math.ceil(total_size / block_size)
            with open(filepath, "wb") as file:
                for data in tqdm(
                        r.iter_content(block_size),
                        total=num_iterables,
                        unit="KB",
                        unit_scale=True,
                ):
                    file.write(data)
        else:
            # log.error(f"Problem downloading {url}")
            r.raise_for_status()
    else:
        pass
        # log.info(f"File {filepath} already downloaded")
    if expected_bytes is not None:
        statinfo = os.stat(filepath)
        if statinfo.st_size != expected_bytes:
            os.remove(filepath)
            raise IOError(f"Failed to verify {filepath}")

    return filepath


def get_review_data(reviews_file):
    """Downloads amazon review data (only), prepares in the required format
    and stores in the same location

    Args:
        reviews_file (str): Filename for downloaded reviews dataset.
    """
    _, reviews_name = os.path.split(reviews_file)
    if not os.path.exists(reviews_file):
        download_and_extract(reviews_name, reviews_file)


def data_preprocessing(reviews_file, meta_file):
    """Create data for training, validation and testing from original dataset

    Args:
        reviews_file (str): Reviews dataset downloaded from former operations.
        meta_file (str): Meta dataset downloaded from former operations.
    """
    if os.path.exists(os.path.join(os.path.split(reviews_file)[0], "instance_output.txt")):
        return os.path.join(os.path.split(reviews_file)[0], "instance_output.txt")

    reviews_output = reviews_preprocessing(reviews_file)
    meta_output = meta_preprocessing(meta_file)
    output = create_instance(reviews_output, meta_output)

    return output


def meta_preprocessing(meta_readfile):
    logger.info("start meta preprocessing...")
    meta_writefile = meta_readfile + "_output"
    meta_r = open(meta_readfile, "r")
    meta_w = open(meta_writefile, "w")
    for line in meta_r:
        line_new = eval(line)
        meta_w.write(line_new["asin"] + "\t" + line_new["categories"][0][-1] + "\n")
    meta_r.close()
    meta_w.close()
    return meta_writefile


def reviews_preprocessing(reviews_readfile):
    logger.info("start reviews preprocessing...")
    reviews_writefile = reviews_readfile + "_output"
    reviews_r = open(reviews_readfile, "r")
    reviews_w = open(reviews_writefile, "w")
    for line in reviews_r:
        line_new = eval(line.strip())
        reviews_w.write(
            str(line_new["reviewerID"])
            + "\t"
            + str(line_new["asin"])
            + "\t"
            + str(line_new["overall"])
            + "\t"
            + str(line_new["unixReviewTime"])
            + "\n"
        )
    reviews_r.close()
    reviews_w.close()
    return reviews_writefile


def create_instance(reviews_file, meta_file):
    logger.info("start create instances...")
    dirs, _ = os.path.split(reviews_file)
    output_file = os.path.join(dirs, "instance_output.txt")

    f_reviews = open(reviews_file, "r")
    user_dict = {}
    item_list = []
    for line in f_reviews:
        line = line.strip()
        reviews_things = line.split("\t")
        if reviews_things[0] not in user_dict:
            user_dict[reviews_things[0]] = []
        user_dict[reviews_things[0]].append((line, float(reviews_things[-1])))
        item_list.append(reviews_things[1])

    f_meta = open(meta_file, "r")
    meta_dict = {}
    for line in f_meta:
        line = line.strip()
        meta_things = line.split("\t")
        if meta_things[0] not in meta_dict:
            meta_dict[meta_things[0]] = meta_things[1]

    f_output = open(output_file, "w")
    for user_behavior in user_dict:
        sorted_user_behavior = sorted(user_dict[user_behavior], key=lambda x: x[1])
        for line, _ in sorted_user_behavior:
            user_things = line.split("\t")
            asin = user_things[1]
            if asin in meta_dict:
                f_output.write(line + "\t" + meta_dict[asin] + "\n")
            else:
                f_output.write(line + "\t" + "default_cat" + "\n")

    f_reviews.close()
    f_meta.close()
    f_output.close()

    return output_file


def create_training_files(file):
    dire, _ = os.path.split(file)
    data = []
    with open(file) as f:
        for line in f:
            data.append(line.strip().split("\t"))
    data = pd.DataFrame(data, columns=["userid", "itemid", "rating", "time", "category"])
    data = data[["userid", "itemid", "rating", "category"]]

    unique_users = data['userid'].unique()
    unique_items = data['itemid'].unique()
    unique_categories = data['category'].unique()

    user_maping = {user: index for index, user in enumerate(unique_users)}
    item_maping = {item: index for index, item in enumerate(unique_items)}
    category_maping = {category: index for index, category in enumerate(unique_categories)}

    data["userid"] = data["userid"].map(user_maping)
    data["itemid"] = data["itemid"].map(item_maping)
    data["category"] = data["category"].map(category_maping)

    item_category = sorted(data[["itemid", "category"]].to_numpy().tolist(), key=lambda k: k[0])
    item_category = list(item_category for item_category, _ in itertools.groupby(item_category))

    grouped_df = data.groupby('userid')['itemid'].apply(list).reset_index()
    users_itractions = grouped_df.to_numpy().tolist()

    with open(os.path.join(dire, "category.txt"), 'w') as f:
        for category, index in category_maping.items():
            f.write(':'.join([str(index), category]) + "\n")

    with open(os.path.join(dire, "item_category.txt"), 'w') as f:
        for itemid, category_id in item_category:
            f.write(','.join([str(itemid), str(category_id)]) + "\n")

    with open(os.path.join(dire, "users_inetractions.txt"), "w") as f:
        f.write("\n".join(str(x[0]) + " " + " ".join(map(str, x[1])) for x in users_itractions))


def download_and_extract(name, dest_path):
    """Downloads and extracts Amazon reviews and meta datafiles if they don’t already exist

    Args:
        name (str): Category of reviews.
        dest_path (str): File path for the downloaded file.

    Returns:
        str: File path for the extracted file.
    """
    dirs, _ = os.path.split(dest_path)
    if not os.path.exists(dirs):
        os.makedirs(dirs)

    file_path = os.path.join(dirs, name)
    if not os.path.exists(file_path):
        download_reviews(name, dest_path)
        extract_reviews(file_path, dest_path)

    return file_path


def download_reviews(name, dest_path):
    """Downloads Amazon reviews datafile.

    Args:
        name (str): Category of reviews
        dest_path (str): File path for the downloaded file
    """

    url = (
            "http://snap.stanford.edu/data/amazon/productGraph/categoryFiles/"
            + name
            + ".gz"
    )

    dirs, file = os.path.split(dest_path)
    maybe_download(url, file + ".gz", work_directory=dirs)


def extract_reviews(file_path, zip_path):
    """Extract Amazon reviews and meta datafiles from the raw zip files.

    To extract all files,
    use ZipFile's extractall(path) instead.

    Args:
        file_path (str): Destination path for datafile
        zip_path (str): zipfile path
    """
    with gzip.open(zip_path + ".gz", "rb") as zf, open(file_path, "wb") as f:
        shutil.copyfileobj(zf, f)


def write_data(path, data):
    with open(path, "w") as f:
        f.write("\n".join(" ".join(map(str, x)) for x in data))


if __name__ == "__main__":
    path = '../Datasets/amazon/'
    dataset_name = "Beauty"
    k_core = 10
    review_file_name = f"reviews_{dataset_name}_{k_core}.json"
    meta_file_name = f"meta_{dataset_name}.json"
    review_file_dir = os.path.join(path, dataset_name, review_file_name)
    meta_file_dir = os.path.join(path, dataset_name, meta_file_name)

    if not os.path.exists(os.path.join(path, dataset_name)):
        os.makedirs(os.path.join(path, dataset_name))

    get_review_data(review_file_dir)
    get_review_data(meta_file_dir)
    instance_file = data_preprocessing(review_file_dir, meta_file_dir)
    create_training_files(instance_file)
    users_intactions = []
    with open(os.path.join(path, dataset_name, "users_inetractions.txt")) as f:
        for line in f:
            users_intactions.append(list(map(int, line.strip().split(" ")[1:])))

    validation_data = []
    test_data = []
    test = 0.2
    val = 0.0
    for i in range(len(users_intactions)):
        validation_data.append([])
        test_data.append([])
    for i in range(len(users_intactions)):
        interactions = users_intactions[i]
        test_size = round(test * len(interactions))
        val_size = round(val * len(interactions))
        for ii in range(test_size):
            item = int(np.random.uniform(0, len(interactions)))
            test_data[i].append(interactions[item])
            interactions.pop(item)

        for ii in range(val_size):
            item = int(np.random.uniform(0, len(interactions)))
            validation_data[i].append(interactions[item])
            interactions.pop(item)

    for i in range(len(users_intactions)):
        users_intactions[i].insert(0, i)
        validation_data[i].insert(0, i)
        test_data[i].insert(0, i)

    dire = os.path.join(path, dataset_name)
    if not os.path.exists(dire):
        os.makedirs(dire)
    write_data(os.path.join(dire, "train.txt"), users_intactions)
    write_data(os.path.join(dire, "test.txt"), test_data)
