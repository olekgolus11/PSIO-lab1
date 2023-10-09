import matplotlib.image as mpimg
import matplotlib.pyplot as plt
from skimage import io
from skimage.transform import resize, rescale
from skimage.color import rgb2gray
import numpy as np

# ZAD 1

img_original = io.imread('./The_Rock.jpg')
img_flipud = img_original[::-1, :]
img_fliplr = img_original[:, ::-1]
img_gray = rgb2gray(img_original)
img_rot90l = img_gray.T
img_rot90r = img_gray.T[:, ::-1][::-1, :]

height = img_gray.shape[0]
width = img_gray.shape[1]
if height > width:
    pad = (height - width) // 2
    img_square_pad = np.pad(img_gray, ((0, 0), (pad, pad)))
    img_square_crop = img_gray[pad:-pad, :]
else:
    pad = (width - height) // 2
    img_square_pad = np.pad(img_gray, ((pad, pad), (0, 0)))
    img_square_crop = img_gray[:, pad:-pad]


def squareify(img):
    height = img.shape[0]
    width = img.shape[1]
    if height > width:
        pad = (height - width) // 2
        img_square_pad = np.pad(img, ((0, 0), (pad, pad)))
        img_square_crop = img[pad:-pad, :]
    else:
        pad = (width - height) // 2
        img_square_pad = np.pad(img, ((pad, pad), (0, 0)))
        img_square_crop = img[:, pad:-pad]
    return img_square_pad, img_square_crop


def split_and_shuffle_blocks(n_cuts, img):
    height = img.shape[0]
    width = img.shape[1]
    block_height = height // n_cuts
    block_width = width // n_cuts
    blocks = []
    for i in range(n_cuts):
        for j in range(n_cuts):
            blocks.append(img[i * block_height:(i + 1) * block_height, j * block_width:(j + 1) * block_width])

    # np.random.shuffle(blocks)
    blocks = np.reshape(np.array(blocks), (block_height * n_cuts, block_width * n_cuts))
    return blocks


# ZAD 2

lena_original = io.imread('./lena.png')
lena_gray = rgb2gray(lena_original)
lena_canvas = np.zeros((640, 480))
lena_canvas[lena_canvas.shape[0] // 2 - lena_gray.shape[0] // 2:lena_canvas.shape[0] // 2 + lena_gray.shape[0] // 2,
lena_canvas.shape[1] // 2 - lena_gray.shape[1] // 2:lena_canvas.shape[1] // 2 + lena_gray.shape[1] // 2] = lena_gray

# ZAD 3
pies_original = io.imread('./pies.jpg')
pies_gray = rgb2gray(pies_original)
[pies_square_pad, pies_square] = squareify(pies_gray)
puzzled_pies = split_and_shuffle_blocks(2, pies_square)

# print(img.shape)
fig, ax = plt.subplots(3, 4, figsize=(20, 10))
ax[0, 0].imshow(img_original)
ax[0, 0].set_xticks([])
ax[0, 0].set_yticks([])

ax[0, 1].imshow(img_flipud, cmap='gray')
ax[0, 1].set_xticks([])
ax[0, 1].set_yticks([])

ax[0, 2].imshow(img_fliplr, cmap='gray')
ax[0, 2].set_xticks([])
ax[0, 2].set_yticks([])

ax[1, 0].imshow(img_rot90l, cmap='gray')
ax[1, 0].set_xticks([])
ax[1, 0].set_yticks([])

ax[1, 1].imshow(img_rot90r, cmap='gray')
ax[1, 1].set_xticks([])
ax[1, 1].set_yticks([])

ax[1, 2].imshow(img_square_pad, cmap='gray')
ax[1, 2].set_xticks([])
ax[1, 2].set_yticks([])

ax[2, 0].imshow(img_square_crop, cmap='gray')
ax[2, 0].set_xticks([])
ax[2, 0].set_yticks([])

ax[2, 1].imshow(lena_gray, cmap='gray')
ax[2, 1].set_xticks([])
ax[2, 1].set_yticks([])

ax[2, 2].imshow(lena_canvas, cmap='gray')
ax[2, 2].set_xticks([])
ax[2, 2].set_yticks([])

ax[1, 3].imshow(puzzled_pies, cmap='gray')
ax[1, 3].set_xticks([])
ax[1, 3].set_yticks([])

plt.show()
