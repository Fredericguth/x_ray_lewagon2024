import os
import cv2
import numpy as np
from PIL import Image, ImageOps, ImageEnhance

def augment_images(directory):
    '''
    Applies augmentation to files in the 'Bone Break Classification' dataset.
    Augmentations include blurring, mirroring, rotating, brightness adjustment and contrast adjustment
    '''
    for root, dirs, files in os.walk(directory):
        for dir_name in dirs:
            if dir_name in ['Train', 'Test']:
                subdir = os.path.join(root, dir_name)
                for subroot, subdirs, subfiles in os.walk(subdir):
                    for file in subfiles:
                        if file.endswith(('.png', '.jpg', '.jpeg', '.bmp')):

                            file_path = os.path.join(subroot, file)

                            print(f'Processing file: {file_path}')

                            if not os.path.exists(file_path):
                                print(f'Error: File does not exist - {file_path}')
                                continue

                            try:
                                img = cv2.imread(file_path)

                                if img is None:
                                    print(f'Warning: Could not read image file {file_path}')
                                    continue

                                # Blurring the image
                                blurred = cv2.GaussianBlur(img, (10, 10), 0)
                                blurred_path = os.path.join(subroot, f'blurred_{file}')
                                cv2.imwrite(blurred_path, blurred)

                                # Mirroring the image
                                img_pil = Image.open(file_path)
                                mirrored = ImageOps.mirror(img_pil)
                                mirrored_path = os.path.join(subroot, f'mirrored_{file}')
                                mirrored.save(mirrored_path)

                                # Rotating the image
                                def rotate_image(image, angle):
                                    (h, w) = image.shape[:2]
                                    center = (w / 2, h / 2)
                                    M = cv2.getRotationMatrix2D(center, angle, 1.0)
                                    rotated = cv2.warpAffine(image, M, (w, h))
                                    return rotated

                                for angle in [90, 180, 270]:
                                    rotated = rotate_image(img, angle)
                                    rotated_path = os.path.join(subroot, f'rotated_{angle}_{file}')
                                    cv2.imwrite(rotated_path, rotated)

                                # Brightness adjustment
                                enhancer = ImageEnhance.Brightness(img_pil)
                                brightened = enhancer.enhance(1)
                                brightened_path = os.path.join(subroot, f'brightened_{file}')
                                brightened.save(brightened_path)

                                # Contrast adjustment
                                enhancer = ImageEnhance.Contrast(img_pil)
                                contrasted = enhancer.enhance(1)
                                contrasted_path = os.path.join(subroot, f'contrasted_{file}')
                                contrasted.save(contrasted_path)

                                print(f'Processed {file}')
                            except Exception as e:
                                print(f'Error processing file {file_path}: {e}')

if __name__ == '__main__':
    data_dir = '/Users/owenclary/code/Owencclary/x_ray_lewagon2024/data/Bone Break Classification'
    print('Script Running')
    augment_images(data_dir)
