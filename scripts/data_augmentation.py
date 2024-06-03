import os
import cv2
from PIL import Image, ImageOps

def augment_images(directory):
    '''
    Applies augmentation to files in the 'Bone Break Classification' dataset.
    Applies a blurring effect and a mirror effect.
    '''
    for root, dirs, files in os.walk(directory):  # cycles through the directory
        for dir_name in dirs:  # iterates over folders in the subdirectory
            if dir_name in ['Train', 'Test']:  # opens only 'Train' and 'Test'
                subdir = os.path.join(root, dir_name)  # makes path to 'Train' or 'Test'
                for subroot, subdirs, subfiles in os.walk(subdir):  # walks through 'Train'/'Test'
                    for file in subfiles:  # iterates over files
                        if file.endswith(('.png', '.jpg', '.jpeg', '.bmp')):  # if file is an image
                            file_path = os.path.join(subroot, file)  # gets file path
                            print(f'Processing file: {file_path}')  # print file path for debugging

                            if not os.path.exists(file_path):
                                print(f'Error: File does not exist - {file_path}')
                                continue

                            try:
                                '''
                                Augmentation already applied:

                                    img = cv2.imread(file_path)  # reads the image
                                    if img is None:
                                        print(f'Warning: Could not read image file {file_path}')
                                        continue

                                    # Blurring the image
                                    blurred = cv2.GaussianBlur(img, (15, 15), 0)  # applies a blur
                                    blurred_path = os.path.join(subroot, f'blurred_{file}')  # creates the path for the blurred image
                                    cv2.imwrite(blurred_path, blurred)  # saves the blurred image

                                    # Mirroring the image
                                    img_pil = Image.open(file_path)  # opens the image using PIL
                                    mirrored = ImageOps.mirror(img_pil)  # mirrors the image
                                    mirrored_path = os.path.join(subroot, f'mirrored_{file}')  # creates the path for the mirrored image
                                    mirrored.save(mirrored_path)  # saves the mirrored image
                                '''
                                for angle in [90, 180, 270]:
                                    rotated_img = cv2.Rotate(angle)
                                    rota

                                print(f'Processed {file}')  # prints the name of the processed file

                            except Exception as e:
                                print(f'Error processing file {file_path}: {e}')

if __name__ == '__main__':
    data_dir = '/Users/owenclary/code/Owencclary/x_ray_lewagon2024/data/Bone Break Classification'
    print('Script Running')
    augment_images(data_dir)
