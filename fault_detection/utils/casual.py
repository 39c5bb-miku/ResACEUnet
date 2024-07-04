import os
import numpy as np
import matplotlib.pyplot as plt
from skimage import util
from matplotlib.colors import ListedColormap,LinearSegmentedColormap
from seismic_canvas import (SeismicCanvas, volume_slices, XYZAxis, Colorbar)
from vispy.color import get_colormap, Colormap, Color
from vispy import app

def cut_patch(input_folder, output_folder):
# 指定输入和输出文件夹的路径
    input_folder = "/home/miku/Documents/Python/python/fault-detection/datasets/train2/labels"  # 更改为您的输入文件夹路径
    output_folder = "/home/miku/Documents/Python/python/fault-detection/datasets/train/labels"  # 更改为您的输出文件夹路径

    #创建输出文件夹
    if not os.path.exists(output_folder):
        os.makedirs(output_folder)

    # 遍历输入文件夹中的所有npy文件
    for i in range(217):
        # 读取原始npy文件
        input_file_path = os.path.join(input_folder, str(i) + ".npy")
        data = np.load(input_file_path)
        
        # 切割原始数据成8个(64, 64, 64)的块
        for j in range(2):
            for k in range(2):
                for l in range(2):
                    start_x = j * 64
                    end_x = (j + 1) * 64
                    start_y = k * 64
                    end_y = (k + 1) * 64
                    start_z = l * 64
                    end_z = (l + 1) * 64
                    sub_data = data[start_x:end_x, start_y:end_y, start_z:end_z]
                    
                    # 保存切割后的数据为新的npy文件
                    output_file_path = os.path.join(output_folder, str(i * 8 + j * 4 + k * 2 + l) + ".npy")
                    np.save(output_file_path, sub_data)


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

    # 加载数据
    a = np.load(seismic_path)
    b = np.load(fault_path)
    a = np.squeeze(a)
    b = np.squeeze(b)
    vmax = np.max(np.abs(a)) / 4
    a = a[inline, :, :]
    b = b[inline, :, :]
    a = np.rot90(a, k=1)
    b = np.rot90(b, k=1)

    # 创建图像并显示
    fig, ax2 = plt.subplots(1, 1, figsize=(12, 12))

    img1 = ax2.imshow(a, cmap='bone', origin='lower', alpha=1, vmax=vmax, vmin=-vmax)#
    img2 = ax2.imshow(b, cmap=custom_cmap, origin='lower', alpha=1, vmax=1, vmin=0)#

    # 创建颜色条并将图像对象传递给 colorbar()
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

    # 加载数据
    a = np.load(seismic_path)
    b = np.load(fault_path)
    a = np.squeeze(a)
    b = np.squeeze(b)
    vmax = np.max(np.abs(a)) / 4
    a = a[:, xline, :]
    b = b[:, xline, :]
    a = np.rot90(a, k=1)
    b = np.rot90(b, k=1)

    # 创建图像并显示
    fig, ax2 = plt.subplots(1, 1, figsize=(12, 12))

    img1 = ax2.imshow(a, cmap='bone', origin='lower', alpha=1, vmax=vmax, vmin=-vmax)#
    img2 = ax2.imshow(b, cmap=custom_cmap, origin='lower', alpha=1, vmax=1, vmin=0)#

    # 创建颜色条并将图像对象传递给 colorbar()
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

    # 加载数据
    a = np.load(seismic_path)
    b = np.load(fault_path)
    a = np.squeeze(a)
    b = np.squeeze(b)
    vmax = np.max(np.abs(a)) / 4
    a = a[:, :, time]
    b = b[:, :, time]
    a = np.rot90(a, k=1)
    b = np.rot90(b, k=1)

    # 创建图像并显示
    fig, ax2 = plt.subplots(1, 1, figsize=(12, 12))

    img1 = ax2.imshow(a, cmap='bone', origin='lower', alpha=1, vmax=vmax, vmin=-vmax)#
    img2 = ax2.imshow(b, cmap=custom_cmap, origin='lower', alpha=1, vmax=1, vmin=0)#

    # 创建颜色条并将图像对象传递给 colorbar()
    plt.colorbar(img2, ax=ax2, aspect=24, shrink=0.822, pad=0.02)
    plt.xlabel('Inline', fontsize=16)
    plt.ylabel('Xime', fontsize=16)

    plt.show()
    plt.savefig('fault_time.png')


def seismic_inline(seismic_path,inline):

    # 加载数据
    a = np.load(seismic_path)
    a = np.squeeze(a)
    print(a.shape)
    vmax = np.max(np.abs(a)) / 4
    a = a[inline, :, :]
    a = np.rot90(a, k=1)

    # 创建图像并显示
    fig, ax2 = plt.subplots(1, 1, figsize=(12, 12))

    img2 = ax2.imshow(a, cmap='seismic', origin='lower', alpha=1, vmax=vmax, vmin=-vmax)#

    # 创建颜色条并将图像对象传递给 colorbar()
    plt.colorbar(img2, ax=ax2, aspect=24, shrink=0.822, pad=0.02)
    plt.xlabel('Xline', fontsize=16)
    plt.ylabel('Time', fontsize=16)

    plt.show()
    plt.savefig('seismic_inline.png')


def seismic_xline(seismic_path,xline):

    # 加载数据
    a = np.load(seismic_path)
    a = np.squeeze(a)
    vmax = np.max(np.abs(a)) / 4
    a = a[:, xline, :]
    a = np.rot90(a, k=1)

    # 创建图像并显示
    fig, ax2 = plt.subplots(1, 1, figsize=(12, 12))

    img2 = ax2.imshow(a, cmap='seismic', origin='lower', alpha=1, vmax=vmax, vmin=-vmax)#

    # 创建颜色条并将图像对象传递给 colorbar()
    plt.colorbar(img2, ax=ax2, aspect=24, shrink=0.822, pad=0.02)
    plt.xlabel('Inline', fontsize=16)
    plt.ylabel('Time', fontsize=16)

    plt.show()
    plt.savefig('seismic_xline.png')


def seismic_time(seismic_path,time):

    # 加载数据
    a = np.load(seismic_path)
    a = np.squeeze(a)
    vmax = np.max(np.abs(a)) / 4
    a = a[:, :, time]
    a = np.rot90(a, k=1)

    # 创建图像并显示
    fig, ax2 = plt.subplots(1, 1, figsize=(12, 12))

    img2 = ax2.imshow(a, cmap='seismic', origin='lower', alpha=1, vmax=vmax, vmin=-vmax)#

    # 创建颜色条并将图像对象传递给 colorbar()
    plt.colorbar(img2, ax=ax2, aspect=24, shrink=0.822, pad=0.02)
    plt.xlabel('Inline', fontsize=16)
    plt.ylabel('Xline', fontsize=16)

    plt.show()
    plt.savefig('seismic_time.png')


def check_nan(folder_path):
    # 加载数据
    folder_path = "/home/miku/Documents/Python/python/fault-detection/datasets/train1/labels"  # 请将路径替换为实际文件夹路径

    # 遍历文件夹中的.npy文件
    for i in range(1736):
        # 读取.npy文件
        file_path = os.path.join(folder_path, str(i) + ".npy")
        data = np.load(file_path)
        
        # 检查是否存在NaN值
        if np.isnan(data).any():
            print(f"文件 {i}.npy 中存在NaN值。")
    print("检查完毕。")



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
    alpha = np.where(rgba[:, 0] > 0.2, 1.0, 0.0)
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


def add_noise(input_path, out_path):
    # 指定包含npy文件的文件夹路径
    input_path = '/home/miku/Documents/Python/fault-detection/datasets/val/seismic'
    out_path = '/home/miku/Documents/Python/fault-detection/datasets/test'

    # 循环处理每个npy文件
    for filename in os.listdir(input_path):
        if filename.endswith(".npy"):
            file_path = os.path.join(input_path, filename)

            # 加载原始数据
            data = np.load(file_path)

            # 生成具有相同形状的高斯噪音
            noisy_data = util.random_noise(data, "gaussian", var = 1)        
            # # 生成具有相同形状的乘性噪音
            # noisy_data = util.random_noise(data, "speckle")
            # # 生成具有相同形状的椒盐噪音
            # noisy_data = util.random_noise(data, "s&p")
            # # 生成具有相同形状的高斯白噪声
            # noisy_data = util.random_noise(data, "localvar")

            # 保存带有高斯噪音的数据到.npy文件
            noisy_file_path = os.path.join(out_path, f'gauss {filename}')
            np.save(noisy_file_path, noisy_data)


def dat_npy(dat_file,npy_file):
    # 设置文件路径和名称
    dat_file = '/home/miku/Documents/Python/fault-detection/fault/0.dat'
    npy_file = '/home/miku/Documents/Python/fault-detection/fault/0.npy'

    # 从.dat文件中读取数据
    with open(dat_file, 'r') as f:
        data = np.fromfile(f, dtype=np.float32)
        data = data.reshape((128, 128, 128))

    # 将数据保存为.npy文件
    np.save(npy_file, data)


if __name__ == '__main__':
    seismic_path = "/data/data1/zph/fault-detection/datasets/135-34/val/seismic/20.npy"
    fault_path = "/data/data1/zph/fault-detection/datasets/135-34/val/fault/20.npy"
    # seismic_path = 'datasets/135-34/train/images/0.npy'
    # fault_path = 'datasets/135-34/train/labels/0.npy'
    # seismic3D(seismic_path)
    # fault3D(seismic_path,fault_path)
    # seismic_inline(seismic_path,0)
    # seismic_xline(seismic_path,0)
    # seismic_time(seismic_path,0)
    # fault_inline(seismic_path,fault_path,30)
    # fault_xline(seismic_path,fault_path,30)
    fault_time(seismic_path,fault_path,30)



