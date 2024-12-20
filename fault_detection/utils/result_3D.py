import numpy as np
from seismic_canvas import (SeismicCanvas, volume_slices, XYZAxis, Colorbar)
from vispy.color import get_colormap, Colormap, Color
from vispy import app

def fault3D(seismic_path,fault_path):

    slicing = {'x_pos': 10, 'y_pos': 10, 'z_pos': 107}
    canvas_params = {'size': (900, 900),
                    'axis_scales': (0.5, 0.5, 0.5), # stretch z-axis
                    'colorbar_region_ratio': 0.1,
                    'fov': 30, 'elevation': 45, 'azimuth': 45,
                    'zoom_factor': 1.6}
    colorbar_size = (800, 20)

    seismic_vol = np.load(seismic_path)
    seismic_vol = np.squeeze(seismic_vol)
    seismic_cmap = 'bone'
    vmax = np.max(seismic_vol)/4
    vmin = np.min(seismic_vol)/4
    seismic_range = (vmin, vmax)
    semblance_vol = np.load(fault_path)
    semblance_vol = np.squeeze(semblance_vol)
    semblance_vol = (semblance_vol > 0.5).astype(np.float32)
    colors = [(1, 1, 0, 0),
            (1, 1, 0, 1)]
    semblance_cmap = Colormap(colors)
    semblance_range = (0, 1)
    visual_nodes = volume_slices([seismic_vol, semblance_vol],
        cmaps=[seismic_cmap, semblance_cmap],
        clims=[seismic_range, semblance_range],
        interpolation='bilinear', **slicing)
    xyz_axis = XYZAxis(seismic_coord_system=False)
    colorbar = Colorbar(cmap=semblance_cmap, 
                        clim=semblance_range,
                        label_str='Fault Semblance', 
                        size=colorbar_size)
    canvas = SeismicCanvas(title='Fault Semblance',
                            visual_nodes=visual_nodes,
                            xyz_axis=xyz_axis,
                            colorbar=colorbar,
                            **canvas_params)
    app.run()

def seismic3D(seismic_path):

    slicing = {'x_pos': 10, 'y_pos': 10, 'z_pos': 117}
    canvas_params = {'size': (900, 900),
                    'axis_scales': (0.5, 0.5, 0.5), # stretch z-axis
                    'colorbar_region_ratio': 0.1,
                    'fov': 30, 'elevation': 25, 'azimuth': 45,
                    'zoom_factor': 1.6}
    colorbar_size = (800, 20)

    seismic_vol = np.load(seismic_path)
    seismic_vol = np.squeeze(seismic_vol)
    print(seismic_vol.shape)
    seismic_vol = seismic_vol
    seismic_cmap = 'seismic'
    vmax = np.max(seismic_vol)/3
    vmin = np.min(seismic_vol)/3
    seismic_range = (vmin, vmax)
    visual_nodes = volume_slices([seismic_vol],
        cmaps=[seismic_cmap],
        clims=[seismic_range],
        interpolation='bilinear', **slicing)
    xyz_axis = XYZAxis(seismic_coord_system=False)
    colorbar = Colorbar(cmap=seismic_cmap, 
                        clim=seismic_range,
                        label_str='Fault Semblance', 
                        size=colorbar_size)
    canvas = SeismicCanvas(title='Fault Semblance',
                            visual_nodes=visual_nodes,
                            xyz_axis=xyz_axis,
                            colorbar=colorbar,
                            **canvas_params)
    app.run()

if __name__ == '__main__':
    seismic_path = r""
    fault_path = r""
    # seismic3D(seismic_path)
    fault3D(seismic_path,fault_path)
