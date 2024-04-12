import argparse
import time
import yaml
import os
import numpy as np

RELEASED_GITHUB_MODELS = [
    # main TSP models
    'r2plus1d_34-tsp_on_activitynet',
    'r2plus1d_34-tsp_on_thumos14',

    # main TAC baseline models
    'r2plus1d_34-tac_on_activitynet',
    'r2plus1d_34-tac_on_thumos14',
    'r2plus1d_34-tac_on_kinetics',

    # other models from the GVF and backbone architecture ablation studies
    'r2plus1d_34-tsp_on_activitynet-avg_gvf',
    'r2plus1d_34-tsp_on_activitynet-no_gvf',

    'r2plus1d_18-tsp_on_activitynet',
    'r2plus1d_18-tac_on_activitynet',
    'r2plus1d_18-tac_on_kinetics',

    'r3d_18-tsp_on_activitynet',
    'r3d_18-tac_on_activitynet',
    'r3d_18-tac_on_kinetics',
]

# PDVC original
def parse_opts():
    parser = argparse.ArgumentParser()

    # configure of this run
    parser.add_argument('--cfg_path', type=str, required=True, help='config file')
    parser.add_argument('--id', type=str, default='', help='id of this run. Results and logs will saved in this folder ./save/id')
    parser.add_argument('--gpu_id', type=str, nargs='+', default=[])
    parser.add_argument('--disable_tqdm', action='store_true')
    parser.add_argument('--seed', type=int, default=777)
    parser.add_argument('--random_seed',  action='store_true', help='choose a random seed from {1,...,1000}')
    parser.add_argument('--disable_cudnn', type=int, default=0, help='disable cudnn may solve some unknown bugs')
    parser.add_argument('--debug', action='store_true', help='using mini-dataset for fast debugging')
    parser.add_argument('--device', default='cuda', choices=['cpu', 'cuda'], help='device to use for training / testing')

    #  ***************************** INPUT DATA PATH *****************************
    parser.add_argument('--train_caption_file', type=str,
                        default='data/anet/captiondata/train_modified.json', help='')
    parser.add_argument('--invalid_video_json', type=str, nargs='+', default=[])
    parser.add_argument('--val_caption_file', type=str, default='data/anet/captiondata/val_1.json')
    parser.add_argument('--visual_feature_folder', type=str, default='data/anet/resnet_bn')
    parser.add_argument('--gt_file_for_auc', type=str, nargs='+', default='data/anet/captiondata/val_all.json')
    parser.add_argument('--gt_file_for_eval', type=str, nargs='+', default=['data/anet/captiondata/val_1.json', 'data/anet/captiondata/val_2.json'])
    parser.add_argument('--gt_file_for_para_eval', type=str, nargs='+', default= ['data/anet/captiondata/para/anet_entities_val_1_para.json', 'data/anet/captiondata/para/anet_entities_val_2_para.json'])
    parser.add_argument('--dict_file', type=str, default='data/anet/vocabulary_activitynet.json', help='')
    parser.add_argument('--criteria_for_best_ckpt', type=str, default='dvc', choices=['dvc', 'pc'], help='for dense video captioning, use soda_c + METEOR as the criteria'
                                                                                                         'for paragraph captioning, choose the best para_METEOR+para_CIDEr+para_BLEU4')

    parser.add_argument('--visual_feature_type', type=str, default='c3d', choices=['c3d', 'resnet_bn', 'resnet'])
    parser.add_argument('--feature_dim', type=int, default=500, help='dim of frame-level feature vector')

    parser.add_argument('--start_from', type=str, default='', help='id of the run with incompleted training')
    parser.add_argument('--start_from_mode', type=str, choices=['best', 'last'], default="last")
    parser.add_argument('--pretrain', type=str, choices=['full', 'encoder', 'decoder'])
    parser.add_argument('--pretrain_path', type=str, default='', help='path of .pth')

    #  ***************************** DATALOADER OPTION *****************************
    parser.add_argument('--nthreads', type=int, default=4)
    parser.add_argument('--data_norm', type=int, default=0)
    parser.add_argument('--data_rescale', type=int, default=1)

    parser.add_argument('--feature_sample_rate', type=int, default=1)
    parser.add_argument('--train_proposal_sample_num', type=int,
                        default=24,
                        help='number of sampled proposals (or proposal sequence), a bigger value may be better')
    parser.add_argument('--gt_proposal_sample_num', type=int, default=10)
    # parser.add_argument('--train_proposal_type', type=str, default='', choices=['gt', 'learnt_seq', 'learnt'])


    #  ***************************** Caption Decoder  *****************************
    parser.add_argument('--vocab_size', type=int, default=5747)
    parser.add_argument('--wordRNN_input_feats_type', type=str, default='C', choices=['C', 'E', 'C+E'],
                        help='C:clip-level features, E: event-level features, C+E: both')
    parser.add_argument('--caption_decoder_type', type=str, default="light",
                        choices=['none','light', 'standard'])
    parser.add_argument('--rnn_size', type=int, default=512,
                        help='size of the rnn in number of hidden nodes in each layer')
    parser.add_argument('--num_layers', type=int, default=1, help='number of layers in the RNN')
    parser.add_argument('--input_encoding_size', type=int, default=512,
                        help='the encoding size of each token in the vocabulary')
    parser.add_argument('--att_hid_size', type=int, default=512, help='the hidden size of the attention MLP')
    parser.add_argument('--drop_prob', type=float, default=0.5, help='strength of dropout in the Language Model RNN')
    parser.add_argument('--max_caption_len', type=int, default=30, help='')

    #  ***************************** Transformer  *****************************
    parser.add_argument('--hidden_dim', type=int, default=512)
    parser.add_argument('--num_queries', type=int, default=100)
    parser.add_argument('--hidden_dropout_prob', type=float, default=0.5)
    parser.add_argument('--layer_norm_eps', type=float, default=1e-12)
    parser.add_argument('--caption_cost_type', type=str, default='loss')
    parser.add_argument('--set_cost_caption', type=float, default=0)
    parser.add_argument('--set_cost_class', type=float, default=1)
    parser.add_argument('--set_cost_bbox', type=float, default=5)
    parser.add_argument('--set_cost_giou', type=float, default=2)
    parser.add_argument('--cost_alpha', type=float, default=0.25)
    parser.add_argument('--cost_gamma', type=float, default=2)

    parser.add_argument('--bbox_loss_coef', default=5, type=float)
    parser.add_argument('--giou_loss_coef', default=2, type=float)
    parser.add_argument('--count_loss_coef', default=0, type=float)
    parser.add_argument('--caption_loss_coef', default=0, type=float)
    parser.add_argument('--eos_coef', default=0.1, type=float,
                        help="Relative classification weight of the no-object class")
    parser.add_argument('--num_classes', type=int, default=1)
    parser.add_argument('--dec_layers', type=int, default=6)
    parser.add_argument('--enc_layers', type=int, default=6)
    parser.add_argument('--transformer_ff_dim', type=int, default=2048)
    parser.add_argument('--transformer_dropout_prob', type=float, default=0.1)
    parser.add_argument('--frame_embedding_num', type=int, default = 100)
    parser.add_argument('--sample_method', type=str, default = 'nearest', choices=['nearest', 'linear'])
    parser.add_argument('--fix_xcw', type=int, default=0)


    #  ***************************** OPTIMIZER *****************************
    parser.add_argument('--training_scheme', type=str, default='all', choices=['cap_head_only', 'no_cap_head', 'all'])
    parser.add_argument('--epoch', type=int, default=30)
    parser.add_argument('--batch_size', type=int, default=1, help='batch_size')
    parser.add_argument('--batch_size_for_eval', type=int, default=1, help='')
    parser.add_argument('--grad_clip', type=float, default=100., help='clip gradients at this value')
    parser.add_argument('--optimizer_type', type=str, default='adam')
    parser.add_argument('--weight_decay', type=float, default=0, help='weight_decay')

    parser.add_argument('--lr', type=float, default=1e-4, help='1e-4 for resnet feature and 5e-5 for C3D feature')
    parser.add_argument('--learning_rate_decay_start', type=float, default=8)
    parser.add_argument('--learning_rate_decay_every', type=float, default=3)
    parser.add_argument('--learning_rate_decay_rate', type=float, default=0.5)

    #  ***************************** SAVING AND LOGGING *****************************
    parser.add_argument('--min_epoch_when_save', type=int, default=-1)
    parser.add_argument('--save_checkpoint_every', type=int, default=1)
    parser.add_argument('--save_all_checkpoint', action='store_true')
    parser.add_argument('--save_dir', type=str, default='save', help='directory to store checkpointed models')

    #  ***************************** For Deformable DETR *************************************
    parser.add_argument('--lr_backbone_names', default=["None"], type=str, nargs='+')
    parser.add_argument('--lr_backbone', default=2e-5, type=float)
    parser.add_argument('--lr_proj', default=0, type=int)
    parser.add_argument('--lr_linear_proj_names', default=['reference_points', 'sampling_offsets'], type=str, nargs='+')
    parser.add_argument('--lr_linear_proj_mult', default=0.1, type=float)

    # Variants of Deformable DETR
    parser.add_argument('--with_box_refine', default=False, action='store_true')
    parser.add_argument('--transformer_input_type', default='queries', choices=['gt_proposals', 'learnt_proposals', 'queries'])

    # * Backbone
    parser.add_argument('--backbone', default=None, type=str,
                        help="Name of the convolutional backbone to use")
    parser.add_argument('--dilation', action='store_true',
                        help="If true, we replace stride with dilation in the last convolutional block (DC5)")
    parser.add_argument('--position_embedding', default='sine', type=str, choices=('sine', 'learned'),
                        help="Type of positional embedding to use on top of the image features")
    parser.add_argument('--position_embedding_scale', default=2 * np.pi, type=float,
                        help="position / size * scale")
    parser.add_argument('--num_feature_levels', default=4, type=int, help='number of feature levels')

    # * Transformer

    parser.add_argument('--nheads', default=8, type=int,
                        help="Number of attention heads inside the transformer's attentions")
    parser.add_argument('--dec_n_points', default=4, type=int)
    parser.add_argument('--enc_n_points', default=4, type=int)

    parser.add_argument('--share_caption_head', type = int ,default=1)

    parser.add_argument('--cap_nheads', default=8, type=int)
    parser.add_argument('--cap_dec_n_points', default=4, type=int)
    parser.add_argument('--cap_num_feature_levels', default=4, type=int)
    parser.add_argument('--disable_mid_caption_heads', action='store_true')

    # Loss
    parser.add_argument('--no_aux_loss', dest='aux_loss', action='store_false',
                        help="Disables auxiliary decoding losses (loss at each layer)")


    # * Loss coefficients

    parser.add_argument('--cls_loss_coef', default=2, type=float)
    parser.add_argument('--focal_alpha', default=0.25, type=float)
    parser.add_argument('--focal_gamma', default=2., type=float)


    #***************************** Event counter *****************************
    parser.add_argument('--max_eseq_length', default=10, type=int)
    parser.add_argument('--lloss_gau_mask', default=1, type=int)
    parser.add_argument('--lloss_beta', default=1, type=float)

    # scheduled sampling
    parser.add_argument('--scheduled_sampling_start', type=int, default=-1,
                        help='at what iteration to start decay gt probability')
    parser.add_argument('--basic_ss_prob', type=float, default=0, help='initial ss prob')
    parser.add_argument('--scheduled_sampling_increase_every', type=int, default=2,
                        help='every how many iterations thereafter to gt probability')
    parser.add_argument('--scheduled_sampling_increase_prob', type=float, default=0.05,
                        help='How much to update the prob')
    parser.add_argument('--scheduled_sampling_max_prob', type=float, default=0.25,
                        help='Maximum scheduled sampling prob.')
    

    #***************************** TSP opts *****************************
    parser.add_argument('--root_dir', help='Path to root directory containing the videos files')
    
    parser.add_argument('--train_subdir', default='train', 
                        help='Training subdirectory inside the root directory (default: train)')
    
    parser.add_argument('--valid_subdir', default='valid',
                        help='Validation subdirectory inside the root directory (default: val)')
    
    parser.add_argument('--backbone_tsp', default='r2plus1d_34',
                        choices=['r2plus1d_34', 'r2plus1d_18', 'r3d_18', 'mvit_v2_s'],
                        help='Encoder backbone architecture (default r2plus1d_34). '
                             'Supported backbones are r2plus1d_34, r2plus1d_18, and r3d_18')

    parser.add_argument('--released_checkpoint', default='r2plus1d-34_tsp-on-activitynet_max-gvf',
                        choices=RELEASED_GITHUB_MODELS,
                        help='Model checkpoint name to load from the released GitHub pretrained models. '
                             'The backbone parameter is set automatically if loading from a released model. '
                             'If `local-checkpoint` flag is not None, then this parameter is ignored and '
                             'a checkpoint is loaded from the given `local-checkpoint` path on disk.')
    
    parser.add_argument('--local_checkpoint', default=None,
                        help='Path to checkpoint on disk. If set, then read checkpoint from local disk. '
                            'Otherwise, load checkpoint from the released GitHub models.')

    parser.add_argument('--clip_len', default=16, type=int,
                        help='Number of frames per clip (default: 16)')

    parser.add_argument('--frame_rate', default=15, type=int,
                        help='Frames-per-second rate at which the videos are sampled (default: 15)')

    parser.add_argument('--stride', default=16, type=int,
                        help='Number of frames (after resampling with frame-rate) between consecutive clips (default: 16)')
    
    parser.add_argument('--workers', default=6, type=int,
                        help='Number of data loading workers (default: 6)') 
    
    # parser.add_argument('--output_dir', required=True,
    #                     help='Path for saving features')

    parser.add_argument('--shard_id', default=0, type=int,
                        help='Shard id number. Must be between [0, num-shards)')
    
    parser.add_argument('--num_shards', default=1, type=int,
                        help='Number of shards to split the metadata-csv-filename')

    parser.add_argument('--backbone_lr', default=0.0001, type=float,
                        help='Backbone layers learning rate')
    
    parser.add_argument('--lr_warmup_epochs', default=2, type=int,
                        help='Number of warmup epochs')

    parser.add_argument('--momentum', default=0.9, type=float,
                        help='Momentum (default: 0.9)')
    
    # parser.add_argument('--metadata_csv_train', required=True,
    #                     help='Path to the metadata CSV train file') 
    
    parser.add_argument('--metadata_csv_valid', help='Path to the metadata CSV valid file') 
    
    parser.add_argument('--lr_milestones', nargs='+', default=[4, 6], type=int,
                        help='Decrease lr on milestone epoch')
    
    parser.add_argument('--lr_gamma', default=0.01, type=float,
                        help='Decrease lr by a factor of lr-gamma at each milestone epoch')
    
    parser.add_argument('--pretrained_tsp_path',default='', type=str,
                        help='Path to pretrained tsp .pth file')

    parser.add_argument('--loss_alphas', nargs='+', default=[1.0, 1.0], type=float,
                        help='A list of the scalar alpha with which to weight each label loss')

    parser.add_argument('--label_columns', nargs='+', help='Names of the label columns in the CSV files')

    parser.add_argument('--label_mapping_jsons', nargs='+', help='Path to the mapping of each label column')
        
    parser.add_argument('--train_csv_filename', help='Path to the training CSV file')
    
    parser.add_argument('--valid_csv_filename', help='Path to the validation CSV file')

    parser.add_argument('--global_video_features',
                        help='Path to the h5 file containing global video features (GVF). '
                             'If not given, then train without GVF.')

    parser.add_argument('--start_epoch', default=0, type=int,
                        help='Start epoch (default: 0)')
    
    parser.add_argument('--fc_lr', default=0.0001, type=float,
                        help='Fully-connected classifiers learning rate')

    parser.add_argument('--in_batch_size', default=4, type=int,
                        help='Middle batch used for forwarding tsp clips')
                        
    parser.add_argument('--in_batch_size_valid', default=26, type=int,
                        help='Middle batch used for forwarding tsp clips')
    
    parser.add_argument('--sound_lr', default=0.0001, type=float,
                        help='lr for sound')
 
    # reranking
    parser.add_argument('--ec_alpha', type=float, default=0.3)
    args = parser.parse_args()

    if args.cfg_path:
        import_cfg(args.cfg_path, vars(args))

    if args.random_seed:
        import random
        seed = int(random.random() * 1000)
        new_id = args.id + '_seed{}'.format(seed)
        save_folder = os.path.join(args.save_dir, new_id)
        while os.path.exists(save_folder):
            seed = int(random.random() * 1000)
            new_id = args.id + '_seed{}'.format(seed)
            save_folder = os.path.join(args.save_dir, new_id)
        args.id = new_id
        args.seed = seed

    if args.debug:
        args.id = 'debug_' + time.strftime("%Y-%m-%d_%H-%M-%S", time.localtime())
        args.save_checkpoint_every = 1
        args.shuffle = 0

    if args.caption_decoder_type == 'none':
        assert args.caption_loss_coef == 0
        assert args.set_cost_caption == 0

    print("args.id: {}".format(args.id))
    return args

def import_cfg(cfg_path, args):
    with open(cfg_path, 'r') as handle:
        yml = yaml.load(handle, Loader=yaml.FullLoader)
        if 'base_cfg_path' in yml:
            base_cfg_path = yml['base_cfg_path']
            import_cfg(base_cfg_path, args)
        args.update(yml)
    pass
if __name__ == '__main__':
    opt = parse_opts()
    print(opt)