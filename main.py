import matplotlib.image as mpimg
import matplotlib.pyplot as plt
from skimage import io
from skimage.transform import resize, rescale
from skimage.color import rgb2gray
import numpy as np


def squareify(img):
    img_height = img.shape[0]
    img_width = img.shape[1]
    if img_height > img_width:
        pad = (img_height - img_width) // 2
        img_square_pad = np.pad(img, ((0, 0), (pad, pad), (0, 0)))
        img_square_crop = img[pad:-pad, :, :]
    else:
        pad = (img_width - img_height) // 2
        img_square_pad = np.pad(img, ((pad, pad), (0, 0), (0, 0)))
        img_square_crop = img[:, pad:-pad, :]
    return img_square_pad, img_square_crop


def split_and_shuffle_blocks(n_cuts, img):
    size = img.shape[0]
    block_size = size // n_cuts
    blocks = []
    for i in range(n_cuts):
        for j in range(n_cuts):
            blocks.append(img[j * block_size:(j + 1) * block_size, i * block_size:(i + 1) * block_size])
    np.random.shuffle(blocks)
    shuffled_image = np.zeros_like(img)
    for i in range(n_cuts):
        for j in range(n_cuts):
            shuffled_image[j * block_size:(j + 1) * block_size, i * block_size:(i + 1) * block_size] = blocks.pop(0)
    return shuffled_image

def put_image_on_canvas(img, canvas):
    img_height = img.shape[0]
    img_width = img.shape[1]
    canvas_height = canvas.shape[0]
    canvas_width = canvas.shape[1]
    if img_height > canvas_height or img_width > canvas_width:
        return None
    else:
        canvas[canvas_height // 2 - img_height // 2:canvas_height // 2 + img_height // 2,
        canvas_width // 2 - img_width // 2:canvas_width // 2 + img_width // 2] = img
        return canvas


# ZAD 1

the_rock_original = io.imread('./The_Rock.jpg')
the_rock_flip_ud = the_rock_original[::-1, :]
the_rock_flip_lr = the_rock_original[:, ::-1]
the_rock_gray = rgb2gray(the_rock_original)
the_rock_rot_90l = the_rock_gray.T
the_rock_rot_90r = the_rock_gray.T[:, ::-1][::-1, :]
[img_square_pad, img_square_crop] = squareify(the_rock_original)

# ZAD 2

lena_original = io.imread('./lena.png')
lena_gray = rgb2gray(lena_original)
lena_canvas = put_image_on_canvas(lena_gray, np.zeros((640, 480)))

# ZAD 3
pies_original = io.imread('./pies.jpg')
pies_gray = rgb2gray(pies_original)
[pies_square_pad, pies_square] = squareify(pies_original)
puzzled_pies = split_and_shuffle_blocks(5, pies_square)

# print(img.shape)
fig, ax = plt.subplots(3, 4, figsize=(20, 10))
ax[0, 0].imshow(the_rock_original)
ax[0, 0].set_xticks([])
ax[0, 0].set_yticks([])

ax[0, 1].imshow(the_rock_flip_ud, cmap='gray')
ax[0, 1].set_xticks([])
ax[0, 1].set_yticks([])

ax[0, 2].imshow(the_rock_flip_lr, cmap='gray')
ax[0, 2].set_xticks([])
ax[0, 2].set_yticks([])

ax[1, 0].imshow(the_rock_rot_90l, cmap='gray')
ax[1, 0].set_xticks([])
ax[1, 0].set_yticks([])

ax[1, 1].imshow(the_rock_rot_90r, cmap='gray')
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
