# Visualise the images in the testbench

import numpy as np
import matplotlib.pyplot as plt
import re
from pathlib import Path


def read_images(cpp_file):

    images = {}
    with open(cpp_file, 'r') as f:
        cpp_code = f.read()

    # Regex to extract all input_img arrays (input_img, input_img1, input_img2, ...)
    pattern = re.compile(
        r"float\s+(input_img\d*)\[n_inputs\]\s*=\s*\{([^}]*)\};", re.MULTILINE | re.DOTALL
    )

    matches = pattern.findall(cpp_code)

    if not matches:
        raise ValueError("No input_img arrays found in the file")

    for idx, (name, array_str) in enumerate(matches):

        array = [float(x) 
                for x in re.split(r",|\s", array_str)
                if x.strip() and not x.strip().startswith("//")]
        
        if len(array) != 100:
            print(f"Warning: {name} has {len(array)} elements, expected 100")
            continue

        images[name] = array

    return images

def visualise_images(images):
    for name, array in images.items():
        img = np.array(array).reshape((10,10))
        print(img)
        plt.figure(figsize=(10,10))
        plt.imshow(img, cmap='gray')
        plt.title(f"{name}")
        plt.axis('off')
        plt.savefig(f"{name}.png", bbox_inches='tight', pad_inches=0)
        plt.close()

if __name__ == "__main__":
    images = read_images("../hls/matmul_tb.cpp")
    #print(images)
    visualise_images(images)