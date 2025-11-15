import os
import mujoco
import cv2
import numpy as np
from lxml import etree
from tqdm import tqdm

def render(model_path):
    n_frames = 120
    height = 480
    width = 640

    method = 'Ours'

    fourcc = cv2.VideoWriter_fourcc(*'mp4v')
    out = cv2.VideoWriter(os.path.join('tmp', 'simu_video_out', method + '.mp4'), fourcc, 30.0,
                          (width*4, height*6))
    out_frames = []
    red_color = np.array([206, 51, 2])
    for obj_idx in tqdm(range(6)):
        hframes = []
        for fid in range(4):
            frames = []
            model = mujoco.MjModel.from_xml_path(os.path.join(model_path, method, f"{obj_idx*20+fid*5:04d}", "hand_model.xml"))
            data = mujoco.MjData(model)

            with mujoco.Renderer(model, height, width) as renderer:
                mujoco.mj_resetData(model, data)
                for i in range(n_frames):
                    while data.time < i / 30.0:
                        mujoco.mj_step(model, data)
                    disp = np.linalg.norm(data.joint('object2world').qpos.copy()[:3])
                    red = max(min((disp - 0.05) / 0.3, 1), 0)
                    renderer.update_scene(data, "cam")
                    frame = renderer.render()
                    frame = (frame * (1-red*0.9) + red_color * red*0.9).astype(np.uint8)
                    # out.write(frame[:, :, ::-1])
                    frames.append(frame)
                frames = np.stack(frames, axis=0)
            hframes.append(frames)
        hframes = np.stack(hframes, axis=0)
        _, n ,h, w, _ = hframes.shape
        hframes = hframes.transpose(1, 2, 0, 3, 4).reshape(n, h, 4*w, 3)
        out_frames.append(hframes)

    out_frames = np.concatenate(out_frames, axis=1)

    for f in out_frames:
        out.write(f[:, :, ::-1])

    out.release()


if __name__ == "__main__":
    render(f'tmp/grab')

