import torch
from monocular import MonocularDepth


class ObjDepth:
    def __init__(self, device='cpu'):
        self.DEVICE = "cuda" if torch.cuda.is_available() else "cpu"
        self.model = MonocularDepth(device=self.DEVICE, )
        pass

    def get_video_depth(self, video_path):
        pass

    def get_frame_obj_depth(self, frame, frame_obj_detections):
        pass

    def get_video_obj_depth(self, video_path, obj_detection_path):
        # Read video
        # Read np file
        # Loop through each video frame and numpy file by video frame number
        #     get depth
        #     loop through detections in the frame
        #         find depth at centre point of each detection
        pass


if __name__ == "__main__":
    pass
    # Get video list and pass it to another function



