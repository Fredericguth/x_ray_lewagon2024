import splitfolders

# Define the input folder, which contains the subfolder 'non_fractured'
input_folder = 'data/Bone Break Classification/non_fractured'

# Define the output folder where the split datasets will be saved
output_folder = 'data/split_data'

# Split the data with 85% train and 15% test
splitfolders.ratio(input_folder, output=output_folder, seed=42, ratio=(0.85, 0.15), group_prefix=None)
