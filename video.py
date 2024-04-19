import imageio
import os
import numpy as np


class VideoRecorder(object):
    def __init__(self, dir_name, height=256, width=256, camera_id=0, fps=30):
        self.dir_name = dir_name
        self.height = height
        self.width = width
        self.camera_id = camera_id
        self.fps = fps
        self.frames = []

    def init(self, enabled=True):
        self.frames = []
        self.enabled = self.dir_name is not None and enabled

    def record_env(self, env):
        if self.enabled:
            frame = env.render(
                mode='rgb_array',
                height=self.height,
                width=self.width
            )
            self.frames.append(frame)

    def record_dataset(self, states):
        if self.enabled:
            # B,N,cameras,H,W,C
            for i in range(states.shape[0]):
                for j in range(states.shape[1]):
                    frame = states[i,j,self.camera_id]
                    self.frames.append(frame)

    def save(self, file_name):
        if self.enabled:
            path = os.path.join(self.dir_name, file_name)
            imageio.mimsave(path, self.frames, fps=self.fps)