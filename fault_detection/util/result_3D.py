import numpy as np
import cigvis
from cigvis import colormap


def seismic3D(seismic_path):
    canvas_params = {
        "size": (800, 800),
        "axis_scales": (1, 1, 1),  # stretch z-axis
        "fov": 30,
        "elevation": 45,
        "azimuth": 45,
        "zoom_factor": 1,
    }
    seismic = np.load(seismic_path)
    vmin, vmax = np.min(seismic) / 3, np.max(seismic) / 3
    pos = [0, 0, 0]
    nodes = cigvis.create_slices(
        volume=seismic, pos=pos, cmap="gray_r", line_width=0, clim=[vmin, vmax]
    )
    # nodes += cigvis.create_colorbar_from_nodes(
    #     nodes=nodes, select='slices', tick_size=24)
    nodes += cigvis.create_axis(
        shape=seismic.shape,
        mode="axis",
        axis_pos="auto",
        tick_nums=4,
        ticks_font_size=24,
        labels_font_size=24,
        intervals=[1, 1, 3],
        starts=[4600, 2135, 2100],
        axis_labels=["Inline", "Xline", "Time [ms]"],
        line_width=1,
        rotation=(0, 0, -90),
    )
    cigvis.plot3D(nodes, **canvas_params)


def fault3D(seismic_path, fault_path):
    canvas_params = {
        "size": (800, 800),
        "axis_scales": (1, 1, 1),  # stretch z-axis
        "fov": 30,
        "elevation": 45,
        "azimuth": 45,
        "zoom_factor": 1,
    }
    seismic = np.load(seismic_path)
    vmin, vmax = np.min(seismic) / 3, np.max(seismic) / 3
    fault = np.load(fault_path)
    fault = (fault > 0.5).astype(np.uint8)
    fg_cmap = colormap.set_alpha_except_min(cmap="viridis", alpha=1)
    pos = [0, 0, 127]
    nodes = cigvis.create_slices(
        volume=seismic, pos=pos, cmap="gray", line_width=0, clim=[vmin, vmax]
    )
    nodes = cigvis.add_mask(nodes, fault, cmaps=fg_cmap, interpolation="nearest")  # type: ignore
    # nodes += cigvis.create_colorbar_from_nodes(
    #     nodes, 'Amplitude', select='slices')
    # nodes += cigvis.create_axis(
    #     shape=seismic.shape,
    #     mode='axis',
    #     axis_pos='auto',
    #     tick_nums=4,
    #     ticks_font_size=24,
    #     labels_font_size=24,
    #     intervals=[1, 1, 1],
    #     starts=[525, 12, 229],
    #     axis_labels=['Inline', 'Xline', 'Time [ms]'],
    #     line_width=1,
    #     rotation=(50, -20, -90),
    # )
    cigvis.plot3D(nodes, **canvas_params)


if __name__ == "__main__":
    seismic_path = r"data/test/seismic/F31.npy"
    fault_path = r"data/test/fault/F31.npy"
    seismic3D(seismic_path)
    # fault3D(seismic_path, fault_path)
