import cv2
import torch
import matplotlib
import numpy as np
from tqdm import tqdm
from PIL import Image
from pathlib import Path
from time import time

def colorize(value, vmin=None, vmax=None, cmap='viridis', invalid_val=-99, invalid_mask=None, background_color=(128, 128, 128, 255), gamma_corrected=False, value_transform=None):
    """Converts a depth map to a color image.

    Args:
        value (torch.Tensor, numpy.ndarry): Input depth map. Shape: (H, W) or (1, H, W) or (1, 1, H, W). All singular dimensions are squeezed
        vmin (float, optional): vmin-valued entries are mapped to start color of cmap. If None, value.min() is used. Defaults to None.
        vmax (float, optional):  vmax-valued entries are mapped to end color of cmap. If None, value.max() is used. Defaults to None.
        cmap (str, optional): matplotlib colormap to use. Defaults to 'magma_r'.
        invalid_val (int, optional): Specifies value of invalid pixels that should be colored as 'background_color'. Defaults to -99.
        invalid_mask (numpy.ndarray, optional): Boolean mask for invalid regions. Defaults to None.
        background_color (tuple[int], optional): 4-tuple RGB color to give to invalid pixels. Defaults to (128, 128, 128, 255).
        gamma_corrected (bool, optional): Apply gamma correction to colored image. Defaults to False.
        value_transform (Callable, optional): Apply transform function to valid pixels before coloring. Defaults to None.

    Returns:
        numpy.ndarray, dtype - uint8: Colored depth map. Shape: (H, W, 4)
    """
    if isinstance(value, torch.Tensor):
        value = value.detach().cpu().numpy()

    value = value.squeeze()
    if invalid_mask is None:
        invalid_mask = value == invalid_val
    mask = np.logical_not(invalid_mask)

    # normalize
    vmin = np.percentile(value[mask],2) if vmin is None else vmin
    vmax = np.percentile(value[mask],85) if vmax is None else vmax
    if vmin != vmax:
        value = (value - vmin) / (vmax - vmin)  # vmin..vmax
    else:
        # Avoid 0-division
        value = value * 0.

    # squeeze last dim if it exists
    # grey out the invalid values

    value[invalid_mask] = np.nan
    cmapper = matplotlib.cm.get_cmap(cmap)
    if value_transform:
        value = value_transform(value)
        # value = value / value.max()
    value = cmapper(value, bytes=True)  # (nxmx4)

    # img = value[:, :, :]
    img = value[...]
    img[invalid_mask] = background_color

    #     return img.transpose((2, 0, 1))
    if gamma_corrected:
        # gamma correction
        img = img / 255
        img = np.power(img, 2.2)
        img = img * 255
        img = img.astype(np.uint8)
    return img

def create_video_from_depth(npy_file, output_video_path, frame_rate=10):

    # Load the numpy array
    depth_data = np.load(npy_file)

    # Get the dimensions (frame_count, height, width)
    frame_count, height, width = depth_data.shape

    # Create a video writer
    fourcc = cv2.VideoWriter_fourcc(*'mp4v')
    video_writer = cv2.VideoWriter(output_video_path, fourcc, frame_rate, (width, height))

    print(f"Creating video from depth data: {npy_file}")
    for frame in depth_data:
        # Use the provided colorize function
        frame_colored = colorize(frame)
        depth_bgr = cv2.cvtColor(frame_colored, cv2.COLOR_RGB2BGR)

        # Write the frame to the video
        video_writer.write(depth_bgr)

    # Release the video writer
    video_writer.release()
    print(f"Video saved to {output_video_path}")


class MonocularDepth:
    def __init__(self, device='cpu'):
        model_repo = "isl-org/ZoeDepth"
        depth_model = torch.hub.load(model_repo, "ZoeD_K", pretrained=True)
        self.zoe_depth = depth_model.to(device).eval()
        print("Depth Model Loaded...")

    def find_frame_depth(self, frame, colourize):
        depth_map = self.zoe_depth.infer_pil(frame)
        if colourize:
            return colorize(depth_map)
        else:
            return depth_map

    def image_depth(self, image_path, output_dir, colourize=True):
        image_path = Path(image_path)
        output_dir = Path(output_dir)

        image = Image.open(image_path)
        depth = self.find_frame_depth(image, colourize)

        print(depth.shape)

        filename = image_path.stem
        output_dir.mkdir(parents=True, exist_ok=True)

        if colourize:
            output_file = output_dir / f"{filename}_depth.png"
            Image.fromarray(depth).save(output_file)
        else:
            output_file = output_dir / f"{filename}_depth.npy"
            np.save(output_file, depth)

        print(f"Depth map saved to {output_file}")

    def video_depth(self, video_path, output_dir, colourize=True, return_depth=False):
        video_path = Path(video_path)
        output_dir = Path(output_dir)
        output_dir.mkdir(parents=True, exist_ok=True)

        video_capture = cv2.VideoCapture(str(video_path))

        frame_rate = video_capture.get(cv2.CAP_PROP_FPS)
        frame_count = int(video_capture.get(cv2.CAP_PROP_FRAME_COUNT))
        frame_width = int(video_capture.get(cv2.CAP_PROP_FRAME_WIDTH))
        frame_height = int(video_capture.get(cv2.CAP_PROP_FRAME_HEIGHT))

        if colourize:
            fourcc = cv2.VideoWriter_fourcc(*'mp4v')
            output_video_path = output_dir / f"{video_path.stem}_depth.mp4"
            video_writer = cv2.VideoWriter(str(output_video_path), fourcc, frame_rate, (frame_width, frame_height))
        else:
            video_writer = None

        print(f"Processing video: {video_path}")

        all_depths = []

        for frame_idx in tqdm(range(frame_count), desc="Processing frames"):
            ret, frame = video_capture.read()
            if not ret:
                break

            frame_pil = Image.fromarray(cv2.cvtColor(frame, cv2.COLOR_BGR2RGB))
            depth = self.find_frame_depth(frame_pil, colourize)

            if colourize:
                # Save colorized frame to video
                depth_bgr = cv2.cvtColor(depth, cv2.COLOR_RGB2BGR)
                video_writer.write(depth_bgr)
            else:
                all_depths.append(depth)


        # Release resources
        video_capture.release()

        if return_depth:
            return all_depths

        print(np.shape(all_depths))

        if video_writer:
            video_writer.release()
            print(f"Depth map video saved to {output_video_path}")
        else:
            npy_output_path = output_dir / f"{video_path.stem}_depth.npy"
            np.save(npy_output_path, np.array(all_depths))
            print(f"Depth map numpy array saved to {npy_output_path}")




if __name__ == "__main__":

    DEVICE = "cuda" if torch.cuda.is_available() else "cpu"
    mv = MonocularDepth(device=DEVICE)
    root = Path().resolve().parent.parent

    start_time = time()

    # file_name = '00011.jpg'
    # mv.image_depth(root / 'sample' / file_name, root / 'sample', colourize=False)

    video_name = '000001.mp4'
    mv.video_depth(video_path=root / 'sample' / video_name, output_dir=root / 'sample', colourize=False)

    # Create depht map video from previously saved numpy file
    # create_video_from_depth(npy_file=root / 'sample' / '000001_depth.npy', output_video_path=root / 'sample' / '000001_depth_from_np.mp4')

    print(f'Finished in {time() - start_time} seconds')



