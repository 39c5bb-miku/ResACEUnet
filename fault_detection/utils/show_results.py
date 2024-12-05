import numpy as np
import matplotlib.pyplot as plt

def CreateImgAlpha(img_input):
    img_alpha = np.zeros([np.shape(img_input)[0], np.shape(img_input)[1], 4])
    img_alpha[:, :, 0] = 0       #red
    img_alpha[:, :, 1] = 0       #green
    img_alpha[:, :, 2] = 0      # b
    img_alpha[..., -1] = img_input
    return img_alpha


def main(file,type,num):
    resample=False #TODO
    image_path=''
    pred_path=''
    data1 = np.load(image_path+'/'+ file+'.npy')
    print(data1.shape)
    if resample:
        data2 = np.load(pred_path+'/'+file+'_resample.npy')
    else:
        data2 = np.load(pred_path+'/'+file+'.npy')
    vmax = np.max(np.abs(data1))/4
    fig = plt.figure(figsize=(24, 8),dpi=400)
    plt.subplots_adjust(left=None, bottom=None, right=None, top=None, wspace=None, hspace=0.5)
    ax1 = fig.add_subplot(131)
    ax1.set_title('seismic')
    ax2 = fig.add_subplot(132)
    ax2.set_title('label')
    ax3 = fig.add_subplot(133)
    ax3.set_title('overlay_label')
    if type==1:
        ax1.imshow(data1[num,:,:].T,cmap='seismic',vmax=vmax,vmin=-vmax)
        ax2.imshow(data2[num,:,:].T,cmap=plt.cm.bone,vmax=1,vmin=-1)
        ax3.imshow(data1[num,:,:].T, cmap='seismic',vmax=vmax,vmin=-vmax)
        ax3.imshow(CreateImgAlpha(data2[num,:,:].T), alpha=1)
        plt.savefig(file + '_inline' + str(num) + '.jpg')
    elif type==3:
        ax1.imshow(data1[:,:,num].T,cmap='seismic',vmax=vmax,vmin=-vmax)
        ax2.imshow(data2[:,:,num].T,cmap=plt.cm.bone,vmax=1,vmin=-1)
        ax3.imshow(data1[:,:,num].T, cmap='seismic',vmax=vmax,vmin=-vmax)
        ax3.imshow(CreateImgAlpha(data2[:,:,num].T), alpha=1)
        plt.savefig(file + '_time' + str(num) + '.jpg')
    elif type==2:
        ax1.imshow(data1[:,num,:].T,cmap='seismic',vmax=vmax,vmin=-vmax)
        ax2.imshow(data2[:,num,:].T,cmap=plt.cm.bone,vmax=1,vmin=-1)
        ax3.imshow(data1[:,num,:].T, cmap='seismic',vmax=vmax,vmin=-vmax)
        ax3.imshow(CreateImgAlpha(data2[:,num,:].T), alpha=1)
        plt.savefig(file + '_xline' + str(num) + '.jpg')
    plt.show()

if __name__ == '__main__':
    file = ''
    num=600
    type=2
    main(file,type,num)