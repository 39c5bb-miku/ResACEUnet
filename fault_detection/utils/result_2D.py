import numpy as np
import matplotlib.pyplot as plt
from matplotlib.colors import ListedColormap

def seismic(seismic_path,type,num):
    a = np.load(seismic_path)
    a = np.squeeze(a)
    vmax,vmin = np.max(a)/3,np.min(a)/3
    if type == 0:
        a = a[num, :, :]
        a = np.rot90(a, k=1)
        fig, ax2 = plt.subplots(1, 1, figsize=(12, 12))
        img2 = ax2.imshow(a, cmap='seismic', origin='lower', alpha=1, vmax=vmax, vmin=vmin)
        fig.colorbar(img2, ax=ax2, aspect=24, shrink=0.45, pad=0.02)
        plt.xlabel('Xline', fontsize=16)
        plt.ylabel('Time', fontsize=16)
        plt.tight_layout()
        plt.show()
        plt.savefig('seismic_inline.png')
    elif type == 1:
        a = a[:, num, :]
        a = np.rot90(a, k=1)
        fig, ax2 = plt.subplots(1, 1, figsize=(12, 12))
        img2 = ax2.imshow(a, cmap='seismic', origin='lower', alpha=1, vmax=vmax, vmin=vmin)
        fig.colorbar(img2, ax=ax2, aspect=24, shrink=0.45, pad=0.02)
        plt.xlabel('Inline', fontsize=16)
        plt.ylabel('Time', fontsize=16)
        plt.tight_layout()
        plt.show()
        plt.savefig('seismic_xline.png')
    elif type == 2:
        a = a[:, :, num]
        a = np.rot90(a, k=1)
        fig, ax2 = plt.subplots(1, 1, figsize=(12, 12))
        img2 = ax2.imshow(a, cmap='seismic', origin='lower', alpha=1, vmax=vmax, vmin=vmin)
        fig.colorbar(img2, ax=ax2, aspect=24, shrink=0.45, pad=0.02)
        plt.xlabel('Inline', fontsize=16)
        plt.ylabel('Xline', fontsize=16)
        plt.tight_layout()
        plt.show()
        plt.savefig('seismic_time.png')


def fault(seismic_path,fault_path,type,num):
    colors = [(1, 1, 0, 0),
        (1, 1, 0, 1)]
    custom_cmap = ListedColormap(colors)
    a = np.load(seismic_path)
    b = np.load(fault_path)
    a = np.squeeze(a)
    b = np.squeeze(b)
    b = (b > 0.5).astype(np.float32)
    vmax,vmin = np.max(a)/3,np.min(a)/3
    if type == 0:
        a = a[num, :, :]
        b = b[num, :, :]
        a = np.rot90(a, k=1)
        b = np.rot90(b, k=1)
        fig, (ax1,ax2) = plt.subplots(1, 2, figsize=(12, 12))
        img1 = ax1.imshow(a, cmap='seismic', origin='lower', alpha=1, vmax=vmax, vmin=vmin)
        fig.colorbar(img1, ax=ax1, aspect=24, shrink=0.45, pad=0.02)
        ax2.imshow(a, cmap='seismic', origin='lower', alpha=1, vmax=vmax, vmin=vmin)
        img2 = ax2.imshow(b, cmap=custom_cmap, origin='lower', alpha=1)
        fig.colorbar(img2, ax=ax2, aspect=24, shrink=0.45, pad=0.02)
        plt.xlabel('Xline', fontsize=16)
        plt.ylabel('Time', fontsize=16)
        plt.tight_layout()
        plt.show()
        plt.savefig('fault_inline.png')
    elif type == 1:
        a = a[:, num, :]
        b = b[:, num, :]
        a = np.rot90(a, k=1)
        b = np.rot90(b, k=1)
        fig, (ax1,ax2) = plt.subplots(1, 2, figsize=(12, 12))
        img1 = ax1.imshow(a, cmap='seismic', origin='lower', alpha=1, vmax=vmax, vmin=vmin)
        fig.colorbar(img1, ax=ax1, aspect=24, shrink=0.45, pad=0.02)
        ax2.imshow(a, cmap='seismic', origin='lower', alpha=1, vmax=vmax, vmin=vmin)
        img2 = ax2.imshow(b, cmap=custom_cmap, origin='lower', alpha=1)
        fig.colorbar(img2, ax=ax2, aspect=24, shrink=0.45, pad=0.02)
        plt.xlabel('Inline', fontsize=16)
        plt.ylabel('Time', fontsize=16)
        plt.tight_layout()
        plt.show()
        plt.savefig('fault_xline.png')
    elif type == 2:
        a = a[:, :, num]
        b = b[:, :, num]
        a = np.rot90(a, k=1)
        b = np.rot90(b, k=1)
        fig, (ax1,ax2) = plt.subplots(1, 2, figsize=(12, 12))
        img1 = ax1.imshow(a, cmap='seismic', origin='lower', alpha=1, vmax=vmax, vmin=vmin)
        fig.colorbar(img1, ax=ax1, aspect=24, shrink=0.45, pad=0.02)
        ax2.imshow(a, cmap='seismic', origin='lower', alpha=1, vmax=vmax, vmin=vmin)
        img2 = ax2.imshow(b, cmap=custom_cmap, origin='lower', alpha=1)
        fig.colorbar(img2, ax=ax2, aspect=24, shrink=0.45, pad=0.02)
        plt.xlabel('Inline', fontsize=16)
        plt.ylabel('Xime', fontsize=16)
        plt.tight_layout()
        plt.show()
        plt.savefig('fault_time.png')

if __name__ == '__main__':
    seismic_path = r""
    fault_path = r""
    seismic(seismic_path,0,1)
    fault(seismic_path,fault_path,0,1)
