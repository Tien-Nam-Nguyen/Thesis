import torch
import torchaudio.transforms as transforms
from torch import nn
from pdvc.pdvc import build
from TSPmodel import Model
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
        self.mha = nn.MultiheadAttention(768, 8, kdim=26 * 90, vdim=26 * 90, batch_first=True)


    def forward(self, x, alphas=None, eval_mode=False):

        del x['video_gvf']
        
        dt = x
        
        x = dt['video_segment']     # [(start, end), ...]
        video_feature = []
        sound_features = []
        T = len(x)
        filename = dt['video_filename']
        los = 0
        
        while(len(x) > 0):
            clips, sound_feature = self.get_clips(x[:self.args.in_batch_size], filename, eval_mode)
            logits, clip_features = self.tspModel.forward(clips, gvf=None, return_features=True)     # (in_batch_size, 768)
            
            video_feature.append(clip_features.detach())
            sound_features.append(sound_feature.detach())
            x = x[self.args.in_batch_size:]
            
            if not eval_mode:
                middle_target = [dt[f'video_{col}'][:self.args.in_batch_size].view(1).to(self.device) for col in self.args.label_columns]
                # middle_target = dt['video_action-label'][:self.args.in_batch_size].view(1).to(self.device)

                for outpt, target, alpha in zip(logits, middle_target, alphas):
                    head_loss = self.tspCriterion(outpt, target)
                    los += alpha * head_loss

                # remove in_batch_size label
                for col in self.args.label_columns:
                    dt[f'video_{col}'] = dt[f'video_{col}'][self.args.in_batch_size:]

        
        
        # dt['video_tensor'] = torch.vstack(video_feature).view(1, T, 768) 
        dt['video_tensor'] = torch.vstack(video_feature).unsqueeze(0)       # (1, T. 768)
        sound_features = torch.vstack(sound_features).unsqueeze(0).to(self.device)      # (1, T, 26 * 90)
        final_feature, _ = self.mha.forward(query=dt['video_tensor'], key=sound_features, value=sound_features)     # (1, T, 768)
        dt['video_tensor'] = final_feature


        
        if not eval_mode:
            for param in self.tspModel.parameters():
                param.grad = None
                

        del dt['video_action-label']
        del dt['video_segment']
        del dt['video_temporal-region-label']
        
        output, loss = self.pdvcModel.forward(dt= dt, criterion= self.pdvcCriterion, transformer_input_type= self.args.transformer_input_type, eval_mode= eval_mode)
        
        return output, loss, los
        

    def get_clips(self, segments, filename, eval_mode):
        lst_vid = []
        lst_audio = []

        for clip_t_start, clip_t_end in segments:
            # get a tensor [clip_length, H, W, C] of the video frames between clip_t_start and clip_t_end seconds
            vframes, sound_tensor, info = read_video(filename=filename, start_pts=clip_t_start, end_pts=clip_t_end, pts_unit='sec')
            sr = info['audio_fps']
            transform = transforms.MFCC(sample_rate=sr, n_mfcc=13, melkwargs={'n_fft': 2048, 'hop_length': 512, 'n_mels': 128, 'center': False})
            mfcc_feature = transform(sound_tensor.to(self.device))      # (2, 13, x)
            mfcc_feature = mfcc_feature.reshape(26 * 90)

            
            if eval_mode:
                vframes = self.transforms_valid(vframes)
            
            else:
                vframes = self.transforms_train(vframes)

            lst_vid.append(vframes)
            lst_audio.append(mfcc_feature)

        return torch.stack(lst_vid).to(self.device), torch.stack(lst_audio)         # (in_batch_size, C, clip_length, H, W), (in_batch_size, 26 * 90)