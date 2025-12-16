from Data_converter import Data_converter

import PIL.Image as Image
import os

if __name__ == "__main__":
    converter = Data_converter([768, 768], 1)
    converter.convert_all()

    for index, image_array in converter.converted_data.items():
        image = Image.fromarray(image_array.astype('uint8').squeeze(), mode='L')
        image.save(f"golf_hole_masks/{index}.png")