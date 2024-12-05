import torch
from tqdm import tqdm
from typing import Sequence, Any, Callable, List, Sequence, Union, Tuple
import torch.nn.functional as F
from monai.data.utils import compute_importance_map, dense_patch_slices, get_valid_patch_size

def rescale_array(arr,minv= 0.0,maxv=1.0):
    mina=arr.min()
    maxa=arr.max()
    if maxa>1000:
        arr[torch.abs(arr) < 0.1] = 0
    if (maxa - mina)!=0:
        norm = (arr - mina) / (maxa - mina)
        return (norm * (maxv - minv)) + minv
    else:
        return arr


def sliding_window_inference(
    inputs: torch.Tensor,
    roi_size: Union[Sequence[int], int],
    sw_batch_size: int,
    predictor: Callable[..., torch.Tensor],
    null_value: float = 0.0,
    overlap: float = 0.25,
    padding_mode: str = 'constant',
    mode: str = 'gaussian',
    sigma_scale: Union[Sequence[float], float] = 0.125,
    device: Union[torch.device, str, None] = None,
    *args: Any,
    **kwargs: Any,
) -> torch.Tensor:
    """
    Sliding window inference on `inputs` with `predictor`.
    Args:
        inputs: input image to be processed (assuming NCHW[D])
        roi_size: the spatial window size for inferences.
            When its components have None or non-positives, the corresponding inputs dimension will be used.
            if the components of the `roi_size` are non-positive values, the transform will use the
            corresponding components of img size. For example, `roi_size=(32, -1)` will be adapted
            to `(32, 64)` if the second spatial dimension size of img is `64`.
        sw_batch_size: the batch size to run window slices.
        predictor: given input tensor `patch_data` in shape NCHW[D], `predictor(patch_data)`
            should return a prediction with the same spatial shape and batch_size, i.e. NMHW[D];
            where HW[D] represents the patch spatial size, M is the number of output channels, N is `sw_batch_size`.
        overlap: Amount of overlap between scans.
        device: device for the stitched output prediction.
            By default the device (and accordingly the memory) of the `inputs` is used. If for example
            set to device=torch.device('cpu') the gpu memory consumption is less and independent of the
            `inputs` and `roi_size`. Output is on the `device`.
        args: optional args to be passed to ``predictor``.
        kwargs: optional keyword args to be passed to ``predictor``.
    """
    num_spatial_dims = len(inputs.shape) - 2
    if overlap < 0 or overlap >= 1:
        raise AssertionError("overlap must be >= 0 and < 1.")

    # determine image spatial size and batch size
    # Note: all input images must have the same image size and batch size
    image_size_ = list(inputs.shape[2:])
    batch_size = inputs.shape[0]

    if device is None:
        device = inputs.device

    # in case that image size is smaller than roi size
    image_size = tuple(max(image_size_[i], roi_size[i]) for i in range(num_spatial_dims))
    pad_size = []
    for k in range(len(inputs.shape) - 1, 1, -1):
        diff = max(roi_size[k - 2] - inputs.shape[k], 0)
        half = diff // 2
        pad_size.extend([half, diff - half])

    inputs = F.pad(inputs, pad=pad_size, mode=padding_mode, value=0.0)
    scan_interval = _get_scan_interval(image_size, roi_size, num_spatial_dims, overlap)

    # Store all slices in list
    slices = dense_patch_slices(image_size, roi_size, scan_interval)
    num_win = len(slices)  # number of windows per image
    total_slices = num_win * batch_size  # total number of windows

    # Create window-level importance map
    importance_map = compute_importance_map(get_valid_patch_size(image_size, roi_size), mode=mode, sigma_scale=sigma_scale)
    # Perform predictions
    output_image, count_map = torch.tensor(0.0, device='cpu'), torch.tensor(0.0, device='cpu')
    _initialized = False
    for slice_g in tqdm(range(0, total_slices, sw_batch_size)):
        slice_range = range(slice_g, min(slice_g + sw_batch_size, total_slices))
        unravel_slice = [
            [slice(int(idx / num_win), int(idx / num_win) + 1), slice(None)] + list(slices[idx % num_win])
            for idx in slice_range
        ]
        window_data = torch.cat([inputs[win_slice] for win_slice in unravel_slice]).to(device)
        rescale_data = rescale_array(window_data)
        if torch.max(rescale_data)==torch.min(rescale_data):
            seg_prob=torch.zeros(rescale_data.shape,dtype=torch.float32)
        else:
            seg_prob = predictor(rescale_data, *args, **kwargs)
            seg_prob = check_output_type(seg_prob)
            seg_prob = seg_prob.to('cpu')
        if not _initialized:  # init. buffer at the first iteration
            output_classes = seg_prob.shape[1]
            output_shape = [batch_size, output_classes] + list(image_size)
            # allocate memory to store the full output and the count for overlapping parts
            output_image = torch.zeros(output_shape, dtype=torch.float32)
            count_map = torch.zeros(output_shape, dtype=torch.float32)
            _initialized = True

        # store the result in the proper location of the full output. Apply weights from importance map.
        for idx, original_idx in zip(slice_range, unravel_slice):
            window_data = window_data.to('cpu')
            window_data[window_data==null_value]=0
            window_data[window_data!=0]=1
            output_image[original_idx] += importance_map * window_data * seg_prob[idx - slice_g]
            count_map[original_idx] += importance_map * torch.ones(window_data.shape, dtype=torch.float32)

    # account for any overlapping sections
    output_image = output_image / count_map

    final_slicing: List[slice] = []
    for sp in range(num_spatial_dims):
        slice_dim = slice(pad_size[sp * 2], image_size_[num_spatial_dims - sp - 1] + pad_size[sp * 2])
        final_slicing.insert(0, slice_dim)
    while len(final_slicing) < len(output_image.shape):
        final_slicing.insert(0, slice(None))
    return output_image[final_slicing]


def check_output_type(inputs):
    if isinstance(inputs,torch.Tensor):
        return inputs
    else:
        return inputs[0]


def _get_scan_interval(
    image_size: Sequence[int], roi_size: Sequence[int], num_spatial_dims: int, overlap: float
) -> Tuple[int, ...]:
    """
    Compute scan interval according to the image size, roi size and overlap.
    Scan interval will be `int((1 - overlap) * roi_size)`, if interval is 0,
    use 1 instead to make sure sliding window works.

    """
    if len(image_size) != num_spatial_dims:
        raise ValueError("image coord different from spatial dims.")
    if len(roi_size) != num_spatial_dims:
        raise ValueError("roi coord different from spatial dims.")

    scan_interval = []
    for i in range(num_spatial_dims):
        if roi_size[i] == image_size[i]:
            scan_interval.append(int(roi_size[i]))
        else:
            interval = int(roi_size[i] * (1 - overlap))
            scan_interval.append(interval if interval > 0 else 1)
    return tuple(scan_interval)