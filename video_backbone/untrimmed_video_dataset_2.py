from __future__ import division, print_function
# 18h00 1/3/2024
import os
import pandas as pd
import numpy as np
import torch
import h5py

from torch.utils.data import Dataset
# from torchvision.io import read_video, read_image


class UntrimmedVideoDataset2(Dataset):
    '''
    UntrimmedVideoDataset:
        This dataset takes in temporal segments from untrimmed videos and samples fixed-length
        clips from each segment. Each item in the dataset is a dictionary with the keys:
            - "clip": A Tensor (dtype=torch.float) of the clip frames after applying transforms
            - "label-Y": A label from the `label_columns` (one key for each label) or -1 if label is missing for that clip
            - "gvf": The global video feature (GVF) vector if `global_video_features` parameter is not None
    '''

    def __init__(self, csv_filename, root_dir, clip_length, frame_rate,
            label_columns, label_mappings, stride, transforms=None, global_video_features=None, debug=False):
        '''
        Args:
            csv_filename (string): Path to the CSV file with temporal segments information and annotations.
                The CSV file must include the columns [filename, fps, t-start, t-end, video-duration] and
                the label columns given by the parameter `label_columns`.
            root_dir (string): Directory with all the video files.
            clip_length (int): The number of frames per clip.
            frame_rate (int): The effective frame rate (fps) to sample clips.
            transforms (callable): A function/transform that takes in a TxHxWxC video
                and returns a transformed version.
            label_columns (list of string): A list of the label columns in the CSV file.
                If more than one column is specified, the sample return a label for each.
            label_mappings (list of dict): A list of dictionaries to map the corresponding label
                from `label_columns` from a category string to an integer ID value.
            global_video_features (string): Path to h5 file containing global video features (optional)
            debug (bool): If true, create a debug dataset with 100 samples.
        '''
        df = UntrimmedVideoDataset2._clean_df_and_remove_short_segments(pd.read_csv(csv_filename), clip_length, frame_rate)
        df = UntrimmedVideoDataset2._append_root_dir_to_filenames_and_check_files_exist(df, root_dir)
        self.clip_metadata_df, self.vid_clip_table = UntrimmedVideoDataset2._generate_clips_metadata(df, clip_length, frame_rate, stride)
        self.clip_length = clip_length
        self.frame_rate = frame_rate

        # self.temporal_jittering = temporal_jittering
        # self.rng = np.random.RandomState(seed=seed)
        # self.uniform_sampling = np.linspace(0, 1, clips_per_segment)

        self.transforms = transforms

        self.label_columns = label_columns
        self.label_mappings = label_mappings
        
        for label_column, label_mapping in zip(label_columns, label_mappings):
            self.clip_metadata_df[label_column] = self.clip_metadata_df[label_column].map(lambda x: -1 if pd.isnull(x) or x == '' else label_mapping[x])
        

        self.global_video_features = global_video_features
        self.debug = debug


    def __len__(self):
        return len(self.vid_clip_table)


    def __getitem__(self, idx):
        sample = {}
        start_row = self.vid_clip_table[idx][0]
        end_row = self.vid_clip_table[idx][1]
        # stack = []
        sample['action-label'] = []
        sample['segment'] = []
        sample['temporal-region-label'] = []
        for i in range(start_row, end_row + 1):

            row = self.clip_metadata_df.iloc[i]
            filename, fps, clip_t_start, action_label, temporal_region_label = row['filename'], row['fps'], row['clip-t-start'], row['action-label'], row['temporal-region-label']

            # compute clip_t_start and clip_t_end
            clip_length_in_sec = self.clip_length / self.frame_rate
            clip_t_end = clip_t_start + clip_length_in_sec

            sample['segment'].append((clip_t_start, clip_t_end))
            # get a tensor [clip_length, H, W, C] of the video frames between clip_t_start and clip_t_end seconds
            # vframes = UntrimmedVideoDataset2.my_read_video(filename=filename, start_pts=clip_t_start, end_pts=clip_t_end, pts_unit='sec')
            # idxs = UntrimmedVideoDataset2._resample_video_idx(self.clip_length, fps, self.frame_rate)
            # vframes = vframes[idxs][:self.clip_length] # [:self.clip_length] for removing extra frames if isinstance(idxs, slice)
            # if vframes.shape[0] != self.clip_length:
            #     raise RuntimeError(f'<EvalVideoDataset>: got clip of length {vframes.shape[0]} != {self.clip_length}.'
            #                        f'filename={filename}, clip_t_start={clip_t_start}, clip_t_end={clip_t_end}, '
            #                        f'fps={fps}')

            # stack.append(vframes)
            sample['action-label'].append(action_label)
            sample['temporal-region-label'].append(temporal_region_label)

        # sample['clip'] = [self.transforms(vframes) for vframes in stack]        # list of (C, T, H, W) 
        filename = self.clip_metadata_df.iloc[start_row]['filename']
        sample['filename'] = filename
        sample['action-label'] = torch.tensor(sample['action-label'])   
        sample['temporal-region-label'] = torch.tensor(sample['temporal-region-label'])
        sample['fps'] = fps

        if self.global_video_features:
            f = h5py.File(self.global_video_features, 'r')
            sample['gvf'] = torch.tensor(f[os.path.basename(filename).split('.')[0]][()])
            f.close()
        else:
            sample['gvf'] = None

        return sample




    @staticmethod
    def _clean_df_and_remove_short_segments(df, clip_length, frame_rate):
        # restrict all segments to be between [0, video-duration]
        df['t-end'] = np.minimum(df['t-end'], df['video-duration'])
        df['t-start'] = np.maximum(df['t-start'], 0)

        # remove segments that are too short to fit at least one clip
        segment_length = (df['t-end'] - df['t-start']) * frame_rate
        mask = segment_length >= clip_length
        num_segments = len(df)
        num_segments_to_keep = sum(mask)
        if num_segments - num_segments_to_keep > 0:
            df = df[mask].reset_index(drop=True)
            print(f'<UntrimmedVideoDataset>: removed {num_segments - num_segments_to_keep}='
                f'{100*(1 - num_segments_to_keep/num_segments):.2f}% from the {num_segments} '
                f'segments from the input CSV file because they are shorter than '
                f'clip_length={clip_length} frames using frame_rate={frame_rate} fps.')

        return df

    @staticmethod
    def _append_root_dir_to_filenames_and_check_files_exist(df, root_dir):
        df['filename'] = df['filename'].map(lambda f: os.path.join(root_dir, f))
        filenames = df.drop_duplicates('filename')['filename'].values
        for f in filenames:
            if not os.path.exists(f):
                raise ValueError(f'<UntrimmedVideoDataset>: file={f} does not exists. '
                                 f'Double-check root_dir and csv_filename inputs.')
        return df

    
    @staticmethod
    def my_iou(first_seg, second_seg):
        first_start = first_seg[0]
        first_end = first_seg[1]
        second_start = second_seg[0]
        second_end = second_seg[1]

        if first_end <= second_start or second_end <= first_start:
            return 0
        
        isLeftOverlapped = first_end > second_start and first_end <= second_end and first_start < second_start
        isRightOverlapped = first_start < second_end and first_start >= second_start and first_end > second_end
        isInside = first_start >= second_start and first_end <= second_end
        isOutside = first_start <= second_start and first_end >= second_end

        total = max(first_end, second_end) - min(first_start, second_start)
        
        if isLeftOverlapped:
            return (first_end - second_start) / total
        
        if isRightOverlapped:
            return (second_end - first_start) / total
        
        if isInside or isOutside:
            return 1


    @staticmethod
    def _generate_clips_metadata(df, clip_length, frame_rate, stride):
        clip_metadata = {
            # 'video-name': [],
            'filename': [],
            'fps': [],
            'clip-t-start': [],
            'action-label': [],
            'temporal-region-label': [],
        }
        vid_clip_table = {}
        idx = 0
        start = 0
        all_segment_of_a_vid = []
        for i, row in df.iterrows():

            if row['temporal-region-label'] == 'No action':
                break

            all_segment_of_a_vid.append((float(row['t-start']), float(row['t-end'])))

            if row['filename'] != df.loc[i + 1, 'filename']:
                total_frames_after_resampling = int(row['video-frames'] * (float(frame_rate) / row['fps']))
                idxs = _resample_video_idx(total_frames_after_resampling, row['fps'], frame_rate)
                if isinstance(idxs, slice):
                    frame_idxs = np.arange(row['video-frames'])[idxs]
                else:
                    frame_idxs = idxs.numpy()
                clip_length_in_sec = clip_length / frame_rate
                clip_t_start = list(frame_idxs[np.arange(0,frame_idxs.shape[0]-clip_length+1,stride)]/row['fps'])
                num_clips = len(clip_t_start)
        
                clip_t_end = [t_start + clip_length_in_sec for t_start in clip_t_start]
                ptr = 0

                for sta, en in zip(clip_t_start, clip_t_end):
                    clip_metadata['fps'].append(row['fps'])
                    clip_metadata['filename'].append(row['filename'])
                    clip_metadata['clip-t-start'].append(sta)
                    # clip_metadata['video-name'].append(row['video-name'])
                    
                    if ptr < len(all_segment_of_a_vid) and sta >= all_segment_of_a_vid[ptr][1] and en >= all_segment_of_a_vid[ptr][1]:
                        ptr += 1        
                    
                    if ptr < len(all_segment_of_a_vid):
                        first_seg = (sta, en)
                        second_seg = (all_segment_of_a_vid[ptr][0], all_segment_of_a_vid[ptr][1])

                        if sta <= en and UntrimmedVideoDataset2.my_iou(first_seg, second_seg) >= 0.6:
                            clip_metadata['action-label'].append(row['action-label'])
                            clip_metadata['temporal-region-label'].append(row['temporal-region-label'])
                            
                    
                        else:
                            clip_metadata['action-label'].append('')
                            clip_metadata['temporal-region-label'].append('No action')

                    else:
                        clip_metadata['action-label'].append('')
                        clip_metadata['temporal-region-label'].append('No action')
                        

                all_segment_of_a_vid = []

                vid_clip_table[idx] = (start, start + num_clips - 1)
                start += num_clips
                idx += 1

        
        return pd.DataFrame(clip_metadata), vid_clip_table

    
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