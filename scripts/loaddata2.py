import os
import requests
import gzip
import shutil
import pandas as pd
import h5py

def download_file(url, destination):
    """Download file from a URL to a destination."""
    response = requests.get(url, stream=True)
    if response.status_code == 200:
        with open(destination, 'wb') as f:
            f.write(response.content)
    else:
        raise Exception(f"Error downloading file: {response.status_code}")

def extract_gzip(source_path, dest_path):
    """Extract a gzip file."""
    with gzip.open(source_path, 'rb') as f_in:
        with open(dest_path, 'wb') as f_out:
            shutil.copyfileobj(f_in, f_out)

def load_data():
    """Loads PCam dataset using h5py.

    # Returns
        Tuple of Numpy arrays: `(x_train, y_train, meta_train), (x_valid, y_valid, meta_valid), (x_test, y_test, meta_test)`.
    """
    dirname = os.path.join('datasets', 'pcam')
    os.makedirs(dirname, exist_ok=True)
    base = 'https://drive.google.com/uc?export=download&id='
    
    # URLs for files
    files = {
        'camelyonpatch_level_2_split_train_y.h5.gz': '1269yhu3pZDP8UYFQs-NYs3FPwuK-nGSG',
        'camelyonpatch_level_2_split_valid_x.h5.gz': '1hgshYGWK8V-eGRy8LToWJJgDU_rXWVJ3',
        'camelyonpatch_level_2_split_valid_y.h5.gz': '1bH8ZRbhSVAhScTS0p9-ZzGnX91cHT3uO',
        'camelyonpatch_level_2_split_test_x.h5.gz': '1qV65ZqZvWzuIVthK8eVDhIwrbnsJdbg_',
        'camelyonpatch_level_2_split_test_y.h5.gz': '17BHrSrwWKjYsOgTMmoqrIjDy6Fa2o_gP',
        'camelyonpatch_level_2_split_train_meta.csv': '1XoaGG3ek26YLFvGzmkKeOz54INW0fruR',
        'camelyonpatch_level_2_split_valid_meta.csv': '16hJfGFCZEcvR3lr38v3XCaD5iH1Bnclg',
        'camelyonpatch_level_2_split_test_meta.csv': '19tj7fBlQQrd4DapCjhZrom_fA4QlHqN4',
        'camelyonpatch_level_2_split_train_x.h5.gz': '1Ka0XfEMiwgCYPdTI-vv6eUElOBnKFKQ2'
    }
    
    # Download and extract files if not already present
    for fname, file_id in files.items():
        file_path = os.path.join(dirname, fname)
        if not os.path.exists(file_path):
            print(f"Downloading {fname}...")
            download_file(base + file_id, file_path)
        # If file is a .gz file, decompress it
        if fname.endswith('.gz'):
            decompressed_path = file_path[:-3]  # Remove .gz extension
            if not os.path.exists(decompressed_path):
                print(f"Decompressing {fname}...")
                extract_gzip(file_path, decompressed_path)
    
    try:
        with h5py.File(os.path.join(dirname, 'camelyonpatch_level_2_split_train_y.h5'), 'r') as hf:
            y_train = hf['y'][:]
        with h5py.File(os.path.join(dirname, 'camelyonpatch_level_2_split_valid_x.h5'), 'r') as hf:
            x_valid = hf['x'][:]
        with h5py.File(os.path.join(dirname, 'camelyonpatch_level_2_split_valid_y.h5'), 'r') as hf:
            y_valid = hf['y'][:]
        with h5py.File(os.path.join(dirname, 'camelyonpatch_level_2_split_test_x.h5'), 'r') as hf:
            x_test = hf['x'][:]
        with h5py.File(os.path.join(dirname, 'camelyonpatch_level_2_split_test_y.h5'), 'r') as hf:
            y_test = hf['y'][:]

        meta_train = pd.read_csv(os.path.join(dirname, 'camelyonpatch_level_2_split_train_meta.csv'))
        meta_valid = pd.read_csv(os.path.join(dirname, 'camelyonpatch_level_2_split_valid_meta.csv'))
        meta_test = pd.read_csv(os.path.join(dirname, 'camelyonpatch_level_2_split_test_meta.csv'))
        
        with h5py.File(os.path.join(dirname, 'camelyonpatch_level_2_split_train_x.h5'), 'r') as hf:
            x_train = hf['x'][:]
            
    except OSError as e:
        print(f"Error reading HDF5 file: {e}")
        raise NotImplementedError('Direct download currently not working. Please go to https://drive.google.com/drive/folders/1gHou49cA1s5vua2V5L98Lt8TiWA3FrKB and press download all. Then place files (ungzipped) in ~/.keras/datasets/pcam.')
        
    return (x_train, y_train, meta_train), (x_valid, y_valid, meta_valid), (x_test, y_test, meta_test)

if __name__ == '__main__':
    (x_train, y_train, meta_train), (x_valid, y_valid, meta_valid), (x_test, y_test, meta_test) = load_data()
    print(len(x_train))  # Example of accessing the length of x_train
