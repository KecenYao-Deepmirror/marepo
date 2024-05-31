import os
import shutil
import re
import argparse

def generate_dataset(dataset_name,source_dir):
    dst_dir = '/workspace/project/marepo/datasets/' + dataset_name
    train_dir = dst_dir + '/train'
    test_dir = dst_dir + '/test'

    # Define the subfolder names
    subfolders = ['calibration', 'poses', 'rgb']

    # Create the train and test directories if they don't exist
    os.makedirs(train_dir, exist_ok=True)
    os.makedirs(test_dir, exist_ok=True)

    # Create the subdirectories within train and test
    for subfolder in subfolders:
        os.makedirs(os.path.join(train_dir, subfolder), exist_ok=True)
        os.makedirs(os.path.join(test_dir, subfolder), exist_ok=True)

    test_sequence = range(20)
    train_sequence = range(20,60)
    # Function to determine the target directory based on the file name pattern
    def get_target_directory(file_name):
        match = re.match(r'CAMERA_BOTTOM_LEFT_(\d+)_', file_name)
        if match:
            number = int(match.group(1))
            if number in test_sequence:
                return 'test'
            elif number in train_sequence:
                return 'train'
        return None

    # Move the files based on the number in their names
    for subfolder in subfolders:
        source_subfolder = os.path.join(source_dir, subfolder)
        train_subfolder = os.path.join(train_dir, subfolder)
        test_subfolder = os.path.join(test_dir, subfolder)

        # Get a list of all files in the subfolder
        if subfolder == 'rgb':
            files = [file for file in os.listdir(source_subfolder) if 'color' in file]
        else:
            files = os.listdir(source_subfolder)
        for file_name in files:
            target_dir = get_target_directory(file_name)
            if target_dir:
                source_file = os.path.join(source_subfolder, file_name)
                if target_dir == 'train':
                    target_file = os.path.join(train_subfolder, file_name)
                else:
                    target_file = os.path.join(test_subfolder, file_name)
                shutil.copy(source_file, target_file)
                print(f"Copy {file_name} to {target_dir}")

    print("Files successfully split into train and test directories.")

if __name__ == '__main__':
    parser = argparse.ArgumentParser(
        description='Setup a customized dataset in 7Scenes format.',
        formatter_class=argparse.ArgumentDefaultsHelpFormatter)

    parser.add_argument('--source_dir', type=str, default='',
                        help='Path to the source dataset')

    parser.add_argument('--dataset_name', type=str, default='testBL',
                        help='name of your custome dataset')

    opt = parser.parse_args()
    # Define the source directory and target train/test directories
    source_dir = opt.source_dir
    name = '7scenes_' + opt.dataset_name #it should be named as '7scenes_*'
    generate_dataset(name,source_dir)
