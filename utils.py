def normalization(data):
    range_ = np.max(data) - np.min(data)
    return (data - np.min(data)) / range_


import numpy as np
from matplotlib import pyplot as plt
import math


def show_cam(cam):
    num_rows = num_cols = 1
    _, axs = plt.subplots(nrows=num_rows, ncols=num_cols, squeeze=False, figsize=(num_cols * 5, num_rows * 5))

    cam = cam.detach()
    if cam.shape[0] == 1:
        cam = cam.squeeze(0)
    # cam = cam.cpu().numpy()
    # cam = np.sum(cam, axis=1)
    cam = cam[1:, :]
    hw, c = cam.shape[0], cam.shape[1]
    cam = cam.sum(dim=1)
    h = w = int(math.sqrt(hw))
    cam = cam.reshape(h, w)

    cam = normalization(cam.numpy())
    cm = plt.get_cmap("viridis")
    heatmap = cm(cam)
    heatmap = heatmap[:, :, :3]
    # Convert (H, W, C) to (C, H, W)

    axs[0, 0].imshow(heatmap)
    plt.show()
    return cam
