import torch
import numpy as np
from pathlib import Path
from monocular import MonocularDepth


class ObjDepth:
    def __init__(self, device='cpu'):
        self.DEVICE = device
        # self.model = MonocularDepth(device=self.DEVICE)

    def get_video_depth(self, video_path):

        # return self.model.video_depth(video_path)
        return np.load(Path(__file__).resolve().parent.parent.parent / 'sample' / '000001_depth.npy')


    def get_object_detections(self, detections_path):
        return np.load(detections_path)['det']

    def get_frame_obj_depth(self, frame_depth, frame_obj_detections):
        obj_depths = []
        for x1, y1, x2, y2, _, _ in frame_obj_detections:
            cx, cy = int((x1 + x2) // 2), int((y1 + y2) // 2)
            obj_depths.append(frame_depth[cy][cx].item())
        return obj_depths

    def get_video_obj_depth(self, video_path, obj_detection_path):
        video_object_depth = []
        depth = self.get_video_depth(video_path)
        objects = self.get_object_detections(obj_detection_path)
        print(type(objects))
        print(depth.shape, objects.shape)
        if len(objects) != len(depth):
            raise Exception(f'Number of frames in the video at {video_path} does not match with the object detections at {obj_detection_path}.')
        for f in range(len(objects)):
            video_object_depth.append(self.get_frame_obj_depth(depth[f], objects[f]))

        return video_object_depth



if __name__ == "__main__":
    root = Path(__file__).resolve().parent.parent.parent

    device = 'cuda' if torch.cuda.is_available() else 'cpu'
    v_f_obj = ObjDepth(device=device)

    video_objects = v_f_obj.get_video_obj_depth(video_path=root / 'sample' / '000001.mp4', obj_detection_path=root / 'sample' / '000001.npz')
    print(np.array(video_objects))

    np.save(root / 'sample' / '000001_obj_depth.npy', video_objects)



