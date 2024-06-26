from __future__ import division, print_function

import os
import pandas as pd
import numpy as np
import torch

from torch.utils.data import Dataset
# from torchvision.io import read_video, read_image

class EvalVideoDataset2(Dataset):
    '''
    EvalVideoDataset:
        This dataset takes in a list of videos and return all clips with the given length and stride
        Each item in the dataset is a dictionary with the keys:
            - "clip": a Tensor (dtype=torch.float) of the clip frames after applying transforms
            - "filename": the video filename
            - "is-last-clip": a flag to mark the last clip in the video
    '''

    def __init__(self, metadata_df, root_dir, clip_length, frame_rate, stride, transforms=None):
        '''
        Args:
            metadata_df (pandas.DataFrame): a DataFrame with the following video metadata columns:
                [filename, fps, video-frames].
            root_dir (string): Directory with all the video files.
            clip_length (int): The number of frames per clip.
            frame_rate (int): The effective frame rate (fps) to sample clips.
            stride (int): The number of frames (after resampling with frame_rate) between consecutive clips.
                For example, `stride`=1 will generate dense clips, while `stride`=`clip_length` will generate non-overlapping clips
            output_dir (string): Path to the directory where video features will be saved
            transforms (callable): A function/transform that takes in a TxHxWxC video
                and returns a transformed version.
        '''
        metadata_df = EvalVideoDataset2._append_root_dir_to_filenames_and_check_files_exist(metadata_df, root_dir)
        self.clip_metadata_df, self.vid_clip_table = EvalVideoDataset2._generate_clips_metadata(metadata_df, clip_length, frame_rate, stride)
        self.clip_length = clip_length
        self.frame_rate = frame_rate
        self.stride = stride
        # self.output_dir = output_dir
        self.transforms = transforms


        # Holds clip features for a given video until all clips are processed and the
        # full video features are ready to be saved to disk
        # self.saved_features = {}
        # self.saved_results = {}

    def __len__(self):
        return len(self.vid_clip_table)

    def __getitem__(self, idx):
        sample = {}
        start_row = self.vid_clip_table[idx][0]
        end_row = self.vid_clip_table[idx][1]
        # stack = []
        sample['segment'] = []
        for i in range(start_row, end_row + 1):

            row = self.clip_metadata_df.iloc[i]
            filename, fps, clip_t_start = row['filename'], row['fps'], row['clip-t-start']

            # compute clip_t_start and clip_t_end
            clip_length_in_sec = self.clip_length / self.frame_rate
            clip_t_end = clip_t_start + clip_length_in_sec

            sample['segment'].append((clip_t_start, clip_t_end))

        filename = self.clip_metadata_df.iloc[start_row]['filename']
        sample['filename'] = filename
        sample['action-label'] = None
        sample['gvf'] = None 
        sample['temporal-region-label'] = None
        sample['fps'] = fps

        return sample



    @staticmethod
    def _append_root_dir_to_filenames_and_check_files_exist(df, root_dir):
        df['filename'] = df['filename'].map(lambda f: os.path.join(root_dir, f))
        filenames = df.drop_duplicates('filename')['filename'].values
        for f in filenames:
            if not os.path.exists(f):
                raise ValueError(f'<EvalVideoDataset>: file={f} does not exists. '
                                 f'Double-check root_dir and metadata_df inputs')
        return df

    @staticmethod
    def _generate_clips_metadata(df, clip_length, frame_rate, stride):
        clip_metadata = {
            # 'video-name': [],
            'filename': [],
            'fps': [],
            'clip-t-start': [],
        }
        vid_clip_table = {}
        idx = 0
        start = 0
        for i, row in df.iterrows():
            total_frames_after_resampling = int(row['video-frames'] * (float(frame_rate) / row['fps']))
            idxs = EvalVideoDataset2._resample_video_idx(total_frames_after_resampling, row['fps'], frame_rate)
            if isinstance(idxs, slice):
                frame_idxs = np.arange(row['video-frames'])[idxs]
            else:
                frame_idxs = idxs.numpy()
            clip_t_start = list(frame_idxs[np.arange(0,frame_idxs.shape[0]-clip_length+1,stride)]/row['fps'])
            num_clips = len(clip_t_start)

            clip_metadata['filename'].extend([row['filename']]*num_clips)
            clip_metadata['fps'].extend([row['fps']]*num_clips)
            clip_metadata['clip-t-start'].extend(clip_t_start)
            # clip_metadata['video-name'].extend(row['filename'][-17:-4] * num_clips)
            vid_clip_table[idx] = (start, start + num_clips - 1)
            start += num_clips
            idx += 1

        return pd.DataFrame(clip_metadata), vid_clip_table

    @staticmethod
    def _resample_video_idx(num_frames, original_fps, new_fps):
        step = float(original_fps) / new_fps
        if step.is_integer():
            # optimization: if step is integer, don't need to perform
            # advanced indexing
            step = int(step)
            return slice(None, None, step)
        idxs = torch.arange(num_frames, dtype=torch.float32) * step
        idxs = idxs.floor().to(torch.int64)
        return idxs

