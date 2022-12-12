import torch
import PIL.Image
import requests
import numpy as np
import io
import os


def N_max_elements(xx, N):
    elements = [0 for x in range(N)]
    inds = [None for x in range(N)]
    for k, x in enumerate(xx):
        if x > elements[0]:
            for n in range(N - 1, -1, -1):
                if x >= elements[n]:
                    elements = elements[1 : n + 1] + [x] + elements[n + 1 :]
                    inds = inds[1 : n + 1] + [k] + inds[n + 1 :]
                    break
    return elements, inds


@torch.no_grad()
def get_furthest_apart(xx):
    # xx should be a num_samples x vec_dim matrix
    dist = 0
    for k in range(xx.shape[0] - 1):
        x0 = xx[k : k + 1, :]
        d = torch.norm(x0 - xx[k + 1 :, :], dim=1)
        ind = torch.argmax(d).item()
        if d[ind] > dist:
            dist = d[ind]
            max_d_inds = [k, k + 1 + ind]

    return max_d_inds


def imread(url, max_size=None, mode=None, min_size=None):
    if url.startswith(('http:', 'https:')):
        r = requests.get(url)
        f = io.BytesIO(r.content)
    else:
        f = url
    img = PIL.Image.open(f)
    if min_size:
        w, h = img.size
        r = max(w, h) / min(w, h)
        if w > h:
            img = img.resize((int(r * min_size), min_size))
        else:
            img = img.resize((min_size, int(r * min_size)))

    if max_size is not None:
        img = img.resize((max_size, max_size))
    if mode is not None:
        img = img.convert(mode)
    img = np.float32(img) / 255.0

    return img


def square_crop(img):
    h, w, c = img.shape
    j0 = abs(h - w) // 2
    if h > w:
        img = img[j0 : j0 + w, :, :]
    elif h < w:
        img = img[:, j0 : j0 + h, :]
    return img


def checked_path(filepath, extension):
    '''
    Check if filepath.extension already exists, and if so, change it
    '''
    try:
        k=int(filepath[-1])
    except ValueError:
        print("filepath string should end in a digit")
    
    extension = extension.replace('.','')

    while os.path.exists(f"{filepath}{k}.{extension}"):
        k += 1
    
    return f"{filepath}{k}.{extension}"