import imageio
import numpy as np

def gifMaker(filenames, gifName, duration):
    images = []
    for filename in filenames:
        images.append(imageio.imread(filename))
    imageio.mimsave(gifName, images, format = 'GIF', duration = duration)

# filenames = []
# for i in np.arange(2000.5, 2021.5, 1):
#     filenames.append("Closer_EMF_Field{}.png".format(i))
# images = []
# for filename in filenames:
#     images.append(imageio.imread(filename))
# imageio.mimsave('closer.gif', images)
