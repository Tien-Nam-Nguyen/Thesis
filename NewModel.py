import torch
import torchaudio.transforms as transforms
from torch import nn
from pdvc.pdvc import build
from TSPmodel import Model
from video_backbone.untrimmed_video_dataset_2 import _resample_video_idx
from torchvision.io import read_video

class NewModel(nn.Module):

    def __init__(self, backbone, num_classes, num_heads, args, concat_gvf, device, transforms_train=None, transforms_valid=None):

        super(NewModel, self).__init__()
        self.tspModel = Model(backbone=backbone, num_classes=num_classes, num_heads=num_heads, concat_gvf=concat_gvf)
        self.pdvcModel, self.pdvcCriterion, self.pdvcPostprocessor = build(args)
        self.feature_size = self.tspModel.feature_size
        self.args = args
        self.tspCriterion = nn.CrossEntropyLoss(ignore_index=-1)
        self.device = device
        self.transforms_train = transforms_train
        self.transforms_valid = transforms_valid
        self.reduce_sound_clip_feature = nn.Linear(26 * 90, 768)
        self.mha = nn.MultiheadAttention(768, 8, batch_first=True)


    def forward(self, x, alphas=None, eval_mode=False):

        dt = x
        del dt['video_action-label']
        del dt['video_temporal-region-label']
        del dt['video_gvf']
        
        x = dt['video_segment']     # [(start, end), ...]
        # video_feature = []
        # sound_features = []
        final_features = []
        T = len(x)
        filename = dt['video_filename']
        los = 0
        
        while(len(x) > 0):
            clips, sound_feature = self.get_clips(x[:self.args.in_batch_size], filename, dt['video_fps'], eval_mode)
            logits, clip_features = self.tspModel.forward(clips, gvf=None, return_features=True)     # (in_batch_size, 768)
            
            # video_feature.append(clip_features.detach())
            # sound_features.append(sound_feature.detach())
            reduced_sound_feature = self.reduce_sound_clip_feature(sound_feature)       # (in_batch_size, 768)
            x = x[self.args.in_batch_size:]

            final_feature = self.combine_tensors(reduced_sound_feature, clip_features)      # (in_batch_size, 2, 768) sound_clip tren, vid_clip duoi
            final_feature, _ = self.mha.forward(query=final_feature, key=final_feature, value=final_feature)        # (in_batch_size, 2, 768)
            final_feature = final_feature[:, 0, :]      # (in_batch_size, 768)
            final_features.append(final_feature.detach())


            # if not eval_mode:
            #     middle_target = [dt[f'video_{col}'][:self.args.in_batch_size].view(1).to(self.device) for col in self.args.label_columns]
            #     # middle_target = dt['video_action-label'][:self.args.in_batch_size].view(1).to(self.device)

            #     for outpt, target, alpha in zip(logits, middle_target, alphas):
            #         head_loss = self.tspCriterion(outpt, target)
            #         los += alpha * head_loss

            #     # remove in_batch_size label
            #     for col in self.args.label_columns:
            #         dt[f'video_{col}'] = dt[f'video_{col}'][self.args.in_batch_size:]

        
        
        dt['video_tensor'] = torch.vstack(final_features).unsqueeze(0)          # (1, T, 768)
        
        if not eval_mode:
            for param in self.tspModel.parameters():
                param.grad = None
                
        del dt['video_segment']
        
        output, loss = self.pdvcModel.forward(dt= dt, criterion= self.pdvcCriterion, transformer_input_type= self.args.transformer_input_type, eval_mode= eval_mode)
        
        return output, loss, los
        

    def get_clips(self, segments, filename, fps, eval_mode):
        '''
            Lay ra video_frames va mfcc
            segments: list of tuples (clip_t_start, clip_t_end)
            filename: ten video file
            eval_mode: True if dang validation
        '''
        lst_vid = []
        lst_audio = []

        for clip_t_start, clip_t_end in segments:
            # get a tensor [clip_length, H, W, C] of the video frames between clip_t_start and clip_t_end seconds
            vframes, sound_tensor, info = read_video(filename=filename, start_pts=clip_t_start, end_pts=clip_t_end, pts_unit='sec')
            sr = info['audio_fps']
            transform = transforms.MFCC(sample_rate=sr, n_mfcc=13, melkwargs={'n_fft': 2048, 'hop_length': 512, 'n_mels': 128, 'center': False})
            mfcc_feature = transform(sound_tensor.to(self.device))      # (2, 13, x)
            mfcc_feature = mfcc_feature.reshape(26 * 90)
            idxs = _resample_video_idx(self.args.clip_len, fps, self.args.frame_rate)
            vframes = vframes[idxs][:self.args.clip_len]

            
            if eval_mode:
                vframes = self.transforms_valid(vframes)
            
            else:
                vframes = self.transforms_train(vframes)

            lst_vid.append(vframes)
            lst_audio.append(mfcc_feature)

        return torch.stack(lst_vid).to(self.device), torch.stack(lst_audio)         # (in_batch_size, C, clip_length, H, W), (in_batch_size, 26 * 90)
    

    def combine_tensors(self, sound_clip_features, vid_clip_features):
        '''
            Ket hop hai tensor so le nhau
            sound_clip_feature: (in_batch_size, 768)
            sound_clip_feature: (in_batch_size, 768)

            return: (in_batch_size, 2, 768)
        '''

        combined_tensor = []
        ptr = 0
        for i in range(vid_clip_features.shape(0) * 2):
            tensor = None
            if i % 2 == 0:
                tensor = sound_clip_features[ptr]
            else:
                tensor = vid_clip_features[ptr]
                ptr += 1
            combined_tensor.append(tensor)

        combined_tensor = torch.stack(combined_tensor)
        return combined_tensor.reshape(-1, 2, 768)