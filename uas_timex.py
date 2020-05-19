import glob
from PIL import Image
import numpy as np
import matplotlib.pyplot as plt
from skimage.io import imsave

files = sorted(glob.glob('./data/test/UAS/*.png'))
imgarray = np.zeros((1001, 501, 4, len(files)))
num = 0
for i in files:
    imgarray[:,:,:,num] = np.asarray(Image.open(i))
    num+=1
timex = np.mean(imgarray, axis=3)
print(np.shape(timex))
print(timex)
timex = timex.astype(int)
print(timex)
plt.imshow(timex)
plt.savefig('foo.png', bbox_inches='tight')
imsave('test.png', timex)