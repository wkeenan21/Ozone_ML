import pandas as pd
import os
import matplotlib.pyplot as plt
from matplotlib.colors import Normalize
from matplotlib.cm import ScalarMappable
import matplotlib.image as mpimg
import numpy as np



import matplotlib as mpl
import matplotlib.pyplot as plt
import matplotlib.image as mpimg
import numpy as np
from PIL import Image


def rgb2gray(rgb):
    return np.dot(rgb[:,:,:3], [0.299, 0.587, 0.114])

def plt_test(imPath, savepath):
    img_src = mpimg.imread(imPath)
    im = rgb2gray(img_src)
    plt.axis('off')
    norm = Normalize(vmin=15, vmax=80)
    plt.imshow(im, cmap='viridis', norm=norm)
    scalar_map = ScalarMappable(norm=norm, cmap='viridis')
    plt.colorbar(scalar_map)
    plt.show()
    plt.savefig(savepath)
    plt.close()

# Load your TIFF file

for i in range(24):
    j = str(i).zfill(2)
    imPath = r'D:\Will_Git\Ozone_ML\Year2\results\ready_to_IDW\IDWs\2023_08_12_{}.tif'.format(j)
    name = imPath[-18:-4]
    plt_test(imPath, savepath=r'D:\Will_Git\Ozone_ML\Year2\results\interpolations\3bandJPGs\{}.jpg'.format(name))

import imageio
jpgDir = r'D:\Will_Git\Ozone_ML\Year2\results\interpolations\3bandJPGs'
with imageio.get_writer(r'D:\Will_Git\Ozone_ML\Year2\results\interpolations\3bandJPGs\8-12.gif', fps=2) as writer:
    for file in os.listdir(jpgDir):
        imPath = os.path.join(jpgDir, file)
        image = imageio.v2.imread(imPath)
        writer.append_data(image)


