import h5py
import numpy as np

# This function is used to extract a small part of the data , and run code on my own laptop (CPU)


def extract_subset(data_file, labels_file, output_data_file, output_labels_file, num_samples):
    # Read raw data and tags
    with h5py.File(data_file, 'r') as f:
        data = f['x'][:]
    
    with h5py.File(labels_file, 'r') as f:
        labels = f['y'][:]
    
    # Make sure that the num_samples does not exceed the dataset size
    num_samples = min(num_samples, data.shape[0])
    
    # 'num_samples' samples were randomly selected
    indices = np.random.choice(data.shape[0], num_samples, replace=False)
    data_subset = data[indices]
    labels_subset = labels[indices]
    
    # Save the extracted sample to a new HDF5 file
    with h5py.File(output_data_file, 'w') as f:
        f.create_dataset('x', data=data_subset)
    
    with h5py.File(output_labels_file, 'w') as f:
        f.create_dataset('y', data=labels_subset)



### use function to subset data

# original data file
data_file = 'camedata/camelyonpatch_level_2_split_test_x.h5'
labels_file = 'camedata/camelyonpatch_level_2_split_test_y.h5'

# subset data file
output_data_file = 'camedata/camelyonpatch_level_2_split_test_x_subset.h5'
output_labels_file = 'camedata/camelyonpatch_level_2_split_test_y_subset.h5'

num_samples = 10     # 10 samples were taken

extract_subset(data_file, labels_file, output_data_file, output_labels_file, num_samples)
