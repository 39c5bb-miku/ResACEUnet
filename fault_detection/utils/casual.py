import os
import numpy as np
import matplotlib.pyplot as plt
from skimage import util
from matplotlib.colors import ListedColormap,LinearSegmentedColormap
from seismic_canvas import (SeismicCanvas, volume_slices, XYZAxis, Colorbar)
from vispy.color import get_colormap, Colormap, Color
from vispy import app


def fault_inline(seismic_path,fault_path,inline):

    original_cmap = get_colormap('RdYeBuCy')
    alpha = np.linspace(0, 1, 128) # 128 color samples
    rgba = np.array([original_cmap.map(x) for x in alpha]).squeeze()
    rgba[:, -1] = alpha
    alpha = np.where(rgba[:, 0] > 0.5, 1.0, 0.0)
    rgba[:, 0] = np.where(alpha > 0, 1.0, 0.0)
    rgba[:, 1:4] = 0.0
    rgba[:, -1] = alpha
    custom_cmap = ListedColormap(rgba)

    a = np.load(seismic_path)
    b = np.load(fault_path)
    a = np.squeeze(a)
    b = np.squeeze(b)
    vmax = np.max(np.abs(a)) / 4
    a = a[inline, :, :]
    b = b[inline, :, :]
    a = np.rot90(a, k=1)
    b = np.rot90(b, k=1)

    fig, ax2 = plt.subplots(1, 1, figsize=(12, 12))

    img1 = ax2.imshow(a, cmap='bone', origin='lower', alpha=1, vmax=vmax, vmin=-vmax)#
    img2 = ax2.imshow(b, cmap=custom_cmap, origin='lower', alpha=1, vmax=1, vmin=0)#

    plt.colorbar(img2, ax=ax2, aspect=24, shrink=0.822, pad=0.02)
    plt.xlabel('Xline', fontsize=16)
    plt.ylabel('Time', fontsize=16)

    plt.show()
    plt.savefig('fault_inline.png')


def fault_xline(seismic_path,fault_path,xline):

    original_cmap = get_colormap('RdYeBuCy')
    alpha = np.linspace(0, 1, 128) # 128 color samples
    rgba = np.array([original_cmap.map(x) for x in alpha]).squeeze()
    rgba[:, -1] = alpha
    alpha = np.where(rgba[:, 0] > 0.5, 1.0, 0.0)
    rgba[:, 0] = np.where(alpha > 0, 1.0, 0.0)
    rgba[:, 1:4] = 0.0
    rgba[:, -1] = alpha
    custom_cmap = ListedColormap(rgba)

    a = np.load(seismic_path)
    b = np.load(fault_path)
    a = np.squeeze(a)
    b = np.squeeze(b)
    vmax = np.max(np.abs(a)) / 4
    a = a[:, xline, :]
    b = b[:, xline, :]
    a = np.rot90(a, k=1)
    b = np.rot90(b, k=1)

    fig, ax2 = plt.subplots(1, 1, figsize=(12, 12))

    img1 = ax2.imshow(a, cmap='bone', origin='lower', alpha=1, vmax=vmax, vmin=-vmax)#
    img2 = ax2.imshow(b, cmap=custom_cmap, origin='lower', alpha=1, vmax=1, vmin=0)#

    plt.colorbar(img2, ax=ax2, aspect=24, shrink=0.822, pad=0.02)
    plt.xlabel('Inline', fontsize=16)
    plt.ylabel('Time', fontsize=16)

    plt.show()
    plt.savefig('fault_xline.png')


def fault_time(seismic_path,fault_path,time):

    original_cmap = get_colormap('RdYeBuCy')
    alpha = np.linspace(0, 1, 128) # 128 color samples
    rgba = np.array([original_cmap.map(x) for x in alpha]).squeeze()
    rgba[:, -1] = alpha
    alpha = np.where(rgba[:, 0] > 0.5, 1.0, 0.0)
    rgba[:, 0] = np.where(alpha > 0, 1.0, 0.0)
    rgba[:, 1:4] = 0.0
    rgba[:, -1] = alpha
    custom_cmap = ListedColormap(rgba)

    a = np.load(seismic_path)
    b = np.load(fault_path)
    a = np.squeeze(a)
    b = np.squeeze(b)
    vmax = np.max(np.abs(a)) / 4
    a = a[:, :, time]
    b = b[:, :, time]
    a = np.rot90(a, k=1)
    b = np.rot90(b, k=1)

    fig, ax2 = plt.subplots(1, 1, figsize=(12, 12))

    img1 = ax2.imshow(a, cmap='bone', origin='lower', alpha=1, vmax=vmax, vmin=-vmax)#
    img2 = ax2.imshow(b, cmap=custom_cmap, origin='lower', alpha=1, vmax=1, vmin=0)#

    plt.colorbar(img2, ax=ax2, aspect=24, shrink=0.822, pad=0.02)
    plt.xlabel('Inline', fontsize=16)
    plt.ylabel('Xime', fontsize=16)

    plt.show()
    plt.savefig('fault_time.png')


def seismic_inline(seismic_path,inline):

    a = np.load(seismic_path)
    a = np.squeeze(a)
    print(a.shape)
    vmax = np.max(np.abs(a)) / 4
    a = a[inline, :, :]
    a = np.rot90(a, k=1)

    fig, ax2 = plt.subplots(1, 1, figsize=(12, 12))

    img2 = ax2.imshow(a, cmap='seismic', origin='lower', alpha=1, vmax=vmax, vmin=-vmax)#

    plt.colorbar(img2, ax=ax2, aspect=24, shrink=0.822, pad=0.02)
    plt.xlabel('Xline', fontsize=16)
    plt.ylabel('Time', fontsize=16)

    plt.show()
    plt.savefig('seismic_inline.png')


def seismic_xline(seismic_path,xline):

    a = np.load(seismic_path)
    a = np.squeeze(a)
    vmax = np.max(np.abs(a)) / 4
    a = a[:, xline, :]
    a = np.rot90(a, k=1)

    fig, ax2 = plt.subplots(1, 1, figsize=(12, 12))

    img2 = ax2.imshow(a, cmap='seismic', origin='lower', alpha=1, vmax=vmax, vmin=-vmax)#

    plt.colorbar(img2, ax=ax2, aspect=24, shrink=0.822, pad=0.02)
    plt.xlabel('Inline', fontsize=16)
    plt.ylabel('Time', fontsize=16)

    plt.show()
    plt.savefig('seismic_xline.png')


def seismic_time(seismic_path,time):

    a = np.load(seismic_path)
    a = np.squeeze(a)
    vmax = np.max(np.abs(a)) / 4
    a = a[:, :, time]
    a = np.rot90(a, k=1)

    fig, ax2 = plt.subplots(1, 1, figsize=(12, 12))

    img2 = ax2.imshow(a, cmap='seismic', origin='lower', alpha=1, vmax=vmax, vmin=-vmax)#

    plt.colorbar(img2, ax=ax2, aspect=24, shrink=0.822, pad=0.02)
    plt.xlabel('Inline', fontsize=16)
    plt.ylabel('Xline', fontsize=16)

    plt.show()
    plt.savefig('seismic_time.png')


def fault3D(seismic_path,fault_path):

    slicing = {'x_pos': 10, 'y_pos': 10, 'z_pos': 107}
    canvas_params = {'size': (900, 900),
                    'axis_scales': (0.5, 0.5, 0.5), # stretch z-axis
                    'colorbar_region_ratio': 0.1,
                    'fov': 30, 'elevation': 25, 'azimuth': 45,
                    'zoom_factor': 1.6}
    colorbar_size = (800, 20)

    seismic_vol = np.load(seismic_path)
    semblance_vol = np.load(fault_path)
    seismic_vol = np.squeeze(seismic_vol)
    semblance_vol = np.squeeze(semblance_vol)
    seismic_cmap = 'bone'
    vmax = np.max(np.abs(seismic_vol)) / 4
    seismic_range = (-vmax, vmax)
    original_cmap = get_colormap('RdYeBuCy')
    alpha = np.linspace(0, 1, 128) # 128 color samples
    rgba = np.array([original_cmap.map(x) for x in alpha]).squeeze()
    rgba[:, -1] = alpha
    semblance_cmap = Colormap(rgba)
    alpha = np.where(rgba[:, 0] > 0.5, 1.0, 0.0)
    rgba[:, 0] = np.where(alpha > 0, 1.0, 0.0)
    rgba[:, 1:4] = 0.0
    rgba[:, -1] = alpha
    semblance_cmap = Colormap(rgba)
    semblance_range = (0, 1.0)
    visual_nodes = volume_slices([seismic_vol, semblance_vol],
    cmaps=[seismic_cmap, semblance_cmap],
    clims=[seismic_range, semblance_range],
    interpolation='bilinear', **slicing)
    xyz_axis = XYZAxis()
    colorbar = Colorbar(cmap=semblance_cmap, clim=semblance_range,
                        label_str='Fault Semblance', size=colorbar_size)
    canvas2 = SeismicCanvas(title='Fault Semblance',
                            visual_nodes=visual_nodes,
                            xyz_axis=xyz_axis,
                            colorbar=colorbar,
                            **canvas_params)
    app.run()


def seismic3D(seismic_path):

    slicing = {'x_pos': 10, 'y_pos': 10, 'z_pos': 107}
    canvas_params = {'size': (900, 900),
                    'axis_scales': (0.5, 0.5, 0.5), # stretch z-axis
                    'colorbar_region_ratio': 0.1,
                    'fov': 30, 'elevation': 25, 'azimuth': 45,
                    'zoom_factor': 1.6}
    colorbar_size = (800, 20)

    seismic_vol = np.load(seismic_path)
    seismic_vol = np.squeeze(seismic_vol)
    seismic_cmap = 'bone'
    vmax = np.max(np.abs(seismic_vol)) / 4
    seismic_range = (-vmax, vmax)
    visual_nodes = volume_slices([seismic_vol],
    cmaps=[seismic_cmap],
    clims=[seismic_range],
    interpolation='bilinear', **slicing)
    xyz_axis = XYZAxis()
    colorbar = Colorbar(cmap=seismic_cmap, clim=seismic_range,
                        label_str='Fault Semblance', size=colorbar_size)
    canvas2 = SeismicCanvas(title='Fault Semblance',
                            visual_nodes=visual_nodes,
                            xyz_axis=xyz_axis,
                            colorbar=colorbar,
                            **canvas_params)
    app.run()


if __name__ == '__main__':
    seismic_path = ''
    fault_path = ''
    # seismic3D(seismic_path)
    # fault3D(seismic_path,fault_path)
    # seismic_inline(seismic_path,0)
    # seismic_xline(seismic_path,0)
    # seismic_time(seismic_path,0)
    # fault_inline(seismic_path,fault_path,30)
    # fault_xline(seismic_path,fault_path,30)
    # fault_time(seismic_path,fault_path,30)



