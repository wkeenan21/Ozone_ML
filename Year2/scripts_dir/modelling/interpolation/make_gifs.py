import pandas as pd
import os
import matplotlib.pyplot as plt
from matplotlib.colors import Normalize
from matplotlib.cm import ScalarMappable
import matplotlib.image as mpimg
import numpy as np
import datetime as dt
from datetime import timedelta


import matplotlib as mpl
import matplotlib.pyplot as plt
import matplotlib.image as mpimg
import numpy as np
from PIL import Image, ImageDraw, ImageFont


def rgb2gray(rgb):
    return np.dot(rgb[:,:,:3], [0.299, 0.587, 0.114])

def plt_test(imPath, savepath, text, cs):
    img_src = mpimg.imread(imPath)
    im = rgb2gray(img_src)
    plt.axis('off')
    norm = Normalize(vmin=15, vmax=80)
    plt.imshow(im, cmap=cs, norm=norm)
    scalar_map = ScalarMappable(norm=norm, cmap=cs)
    plt.colorbar(scalar_map)
    plt.show()
    # Save the modified image
    plt.savefig(savepath)
    image = Image.open(savepath)
    draw = ImageDraw.Draw(image)
    # Choose a font (you'll need to specify the path to the font file)
    font = ImageFont.truetype("D:\Will_Git\Ozone_ML\open-sans\OpenSans-Regular.ttf", 36)
    # Calculate the position for the top middle of the image
    # Get image dimensions
    width, height = image.size
    text_width, text_height = draw.textsize(text, font=font)
    text_x = (width - text_width) // 2
    text_y = 0
    # Define the color of the text (R, G, B)
    text_color = (1, 1, 1)
    # Draw the text on the image
    draw.text((text_x, text_y), text, fill=text_color, font=font)
    image.save(savepath)

    plt.close()

# Load your TIFF file

for i in range(6,24):
    j = str(i).zfill(2)
    imPath = r'D:\Will_Git\Ozone_ML\Year2\nh_results\ready_to_interp_csvs\csvs\1Hour\2023_08_02\2023_08_02_{}_w2.tif'.format(j)
    name = imPath[-20:-7]
    datetime_name = dt.datetime(year=int(name[0:4]), month=int(name[5:7]), day=int(name[8:10]), hour=int(name[-2:]))
    datetime_name = datetime_name + timedelta(hours=-7)
    time = str(datetime_name)
    print(time)
    plt_test(imPath, savepath=r'D:\Will_Git\Ozone_ML\Year2\nh_results\gifs\8-2\{}.jpg'.format(name), text=time, cs='YlOrRd')

import imageio
jpgDir = r'D:\Will_Git\Ozone_ML\Year2\nh_results\gifs\8-2'
with imageio.get_writer(r'D:\Will_Git\Ozone_ML\Year2\nh_results\gifs\8-2\8-02.gif', fps=2) as writer:
    help(writer)
    for file in os.listdir(jpgDir):
        imPath = os.path.join(jpgDir, file)
        image = imageio.v2.imread(imPath)
        writer.append_data(image)




