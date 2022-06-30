import imageio
import numpy as np
filenames = []
for i in np.arange(2000.5, 2021.5, 1):
    filenames.append("Closer_EMF_Field{}.png".format(i))
images = []
for filename in filenames:
    images.append(imageio.imread(filename))
imageio.mimsave('closer.gif', images)