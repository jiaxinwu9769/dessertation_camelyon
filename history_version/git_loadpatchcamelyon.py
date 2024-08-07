"""PatchCamelyon(PCam) dataset
Small 96x96 patches from histopathology slides from the Camelyon16 dataset.

Please consider citing [1] when used in your publication:
- [1] B. S. Veeling, J. Linmans, J. Winkens, T. Cohen, M. Welling. "Rotation Equivariant CNNs for Digital Pathology". arXiv [cs.CV] (2018), (available at http://arxiv.org/abs/1806.03962).

Author: Bastiaan Veeling
Source: https://github.com/basveeling/pcam
"""

# Define data loading function
import os
import gzip
import shutil
import gdown
import h5py
import pandas as pd
import requests
import torch
from torch.utils.data import Dataset, DataLoader
import torchvision.transforms as transforms

def download_file_from_google_drive(file_id, dest_path):
    url = f"https://drive.google.com/drive/folders/1gHou49cA1s5vua2V5L98Lt8TiWA3FrKB"
    response = requests.get(url, stream=True)
    with open(dest_path, 'wb') as file:
        for chunk in response.iter_content(chunk_size=1024):
            if chunk:
                file.write(chunk)
    print(f"Downloaded {dest_path}")

def extract_gz(gz_path, dest_path):
    with gzip.open(gz_path, 'rb') as f_in:
        with open(dest_path, 'wb') as f_out:
            shutil.copyfileobj(f_in, f_out)
    print(f"Extracted {gz_path} to {dest_path}")

def download_and_extract_gz(file_id, dest_dir, file_name):
    gz_path = os.path.join(dest_dir, file_name)
    h5_path = gz_path[:-3]  # Remove .gz extension
    if not os.path.exists(gz_path):
        download_file_from_google_drive(file_id, gz_path)
    if not os.path.exists(h5_path):
        extract_gz(gz_path, h5_path)
    return h5_path

def download_and_read_csv(file_id, dest_dir, file_name):
    csv_path = os.path.join(dest_dir, file_name)
    if not os.path.exists(csv_path):
        download_file_from_google_drive(file_id, csv_path)
    return pd.read_csv(csv_path)

def load_data():
    dirname = os.path.join('datasets', 'pcam')
    os.makedirs(dirname, exist_ok=True)

    files = {
        'x_train': ('1Ka0XfEMiwgCYPdTI-vv6eUElOBnKFKQ2', 'camelyonpatch_level_2_split_train_x.h5.gz'),
        'y_train': ('1269yhu3pZDP8UYFQs-NYs3FPwuK-nGSG', 'camelyonpatch_level_2_split_train_y.h5.gz'),
        'x_valid': ('1hgshYGWK8V-eGRy8LToWJJgDU_rXWVJ3', 'camelyonpatch_level_2_split_valid_x.h5.gz'),
        'y_valid': ('1bH8ZRbhSVAhScTS0p9-ZzGnX91cHT3uO', 'camelyonpatch_level_2_split_valid_y.h5.gz'),
        'x_test': ('1qV65ZqZvWzuIVthK8eVDhIwrbnsJdbg_', 'camelyonpatch_level_2_split_test_x.h5.gz'),
        'y_test': ('17BHrSrwWKjYsOgTMmoqrIjDy6Fa2o_gP', 'camelyonpatch_level_2_split_test_y.h5.gz'),
        'meta_train': ('1XoaGG3ek26YLFvGzmkKeOz54INW0fruR', 'camelyonpatch_level_2_split_train_meta.csv'),
        'meta_valid': ('16hJfGFCZEcvR3lr38v3XCaD5iH1Bnclg', 'camelyonpatch_level_2_split_valid_meta.csv'),
        'meta_test': ('19tj7fBlQQrd4DapCjhZrom_fA4QlHqN4', 'camelyonpatch_level_2_split_test_meta.csv')
    }

    x_train_path = download_and_extract_gz(files['x_train'][0], dirname, files['x_train'][1])
    y_train_path = download_and_extract_gz(files['y_train'][0], dirname, files['y_train'][1])
    x_valid_path = download_and_extract_gz(files['x_valid'][0], dirname, files['x_valid'][1])
    y_valid_path = download_and_extract_gz(files['y_valid'][0], dirname, files['y_valid'][1])
    x_test_path = download_and_extract_gz(files['x_test'][0], dirname, files['x_test'][1])
    y_test_path = download_and_extract_gz(files['y_test'][0], dirname, files['y_test'][1])

    meta_train = download_and_read_csv(files['meta_train'][0], dirname, files['meta_train'][1])
    meta_valid = download_and_read_csv(files['meta_valid'][0], dirname, files['meta_valid'][1])
    meta_test = download_and_read_csv(files['meta_test'][0], dirname, files['meta_test'][1])

    with h5py.File(x_train_path, 'r') as f:
        x_train = f['x'][:]
    with h5py.File(y_train_path, 'r') as f:
        y_train = f['y'][:]
    with h5py.File(x_valid_path, 'r') as f:
        x_valid = f['x'][:]
    with h5py.File(y_valid_path, 'r') as f:
        y_valid = f['y'][:]
    with h5py.File(x_test_path, 'r') as f:
        x_test = f['x'][:]
    with h5py.File(y_test_path, 'r') as f:
        y_test = f['y'][:]

    return (x_train, y_train, meta_train), (x_valid, y_valid, meta_valid), (x_test, y_test, meta_test)

# 测试数据加载
if __name__ == '__main__':
    (x_train, y_train, meta_train), (x_valid, y_valid, meta_valid), (x_test, y_test, meta_test) = load_data()
    print(x_train.shape, y_train.shape)
