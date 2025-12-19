from Data_converter import Data_converter

import PIL.Image as Image
import os
import cv2
import numpy as np

if __name__ == "__main__":
    converter = Data_converter([768, 768], 1)
    converter.convert_all()

    for index, image_array in converter.converted_data.items():
        image = Image.fromarray(image_array.astype('uint8').squeeze(), mode='L')
        image.save(f"golf_hole_masks/{index}.png")

    # go over each golf hole and replace all pixels where mask == 255 with white
    for file_name in os.listdir("golf_hole_masks"):
        if file_name.endswith(".png"):
            mask_image = Image.open(os.path.join("golf_hole_masks", file_name))
            mask_array = np.array(mask_image)
            golf_hole = Image.open(os.path.join("golf_holes", file_name))
            golf_hole_array = np.array(golf_hole)
            golf_hole_array = np.where(mask_array[:, :, np.newaxis] == 255, 255, golf_hole_array)
            result_image = Image.fromarray(golf_hole_array.astype('uint8'), mode='RGB')
            result_image.save(os.path.join("golf_holes", file_name))