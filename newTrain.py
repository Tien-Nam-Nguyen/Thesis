# coding:utf-8
from __future__ import absolute_import
from __future__ import division
from __future__ import print_function

import json
import time
import torch
import torchvision
import os
import sys
import collections
import pandas as pd
import numpy as np
from tqdm import tqdm
import torch.optim as optim
from torch.utils.data import DataLoader
from os.path import dirname, abspath
from video_backbone.TSP.common import utils
from video_backbone.TSP.common import transforms as T
# from torchvision.datasets.samplers import DistributedSampler
from itertools import chain
# from video_backbone.TSP.common.scheduler import WarmupMultiStepLR
from TSPmodel import Model
from NewModel import NewModel
from NewDataset import NewDataset 

pdvc_dir = dirname(abspath(__file__))
sys.path.insert(0, pdvc_dir)
sys.path.insert(0, os.path.join(pdvc_dir, 'densevid_eval3'))
sys.path.insert(0, os.path.join(pdvc_dir, 'densevid_eval3/SODA'))

from NewEval_utils import evaluate
import new_opts as opts
from tensorboardX import SummaryWriter
from misc.utils import print_alert_message, build_floder, create_logger, backup_envir, print_opt, set_seed
from NewDataset import collate_fn
from pdvc.pdvc import build
from collections import OrderedDict

MODEL_URLS = {
    # main TSP models
    'r2plus1d_34-tsp_on_activitynet'        : 'https://github.com/HumamAlwassel/TSP/releases/download/model_weights/r2plus1d_34-tsp_on_activitynet-max_gvf-backbone_lr_0.0001-fc_lr_0.002-epoch_5-0d2cf854.pth',
    'r2plus1d_34-tsp_on_thumos14'           : 'https://github.com/HumamAlwassel/TSP/releases/download/model_weights/r2plus1d_34-tsp_on_thumos14-max_gvf-backbone_lr_0.0001-fc_lr_0.004-epoch_4-e6a30b2f.pth',

    # main TAC baseline models
    'r2plus1d_34-tac_on_activitynet'        : 'https://github.com/HumamAlwassel/TSP/releases/download/model_weights/r2plus1d_34-tac_on_activitynet-backbone_lr_0.0001-fc_lr_0.002-epoch_5-98ccac94.pth',
    'r2plus1d_34-tac_on_thumos14'           : 'https://github.com/HumamAlwassel/TSP/releases/download/model_weights/r2plus1d_34-tac_on_thumos14-backbone_lr_0.00001-fc_lr_0.002-epoch_3-54b5c8aa.pth',
    'r2plus1d_34-tac_on_kinetics'           : 'https://github.com/HumamAlwassel/TSP/releases/download/model_weights/r2plus1d_34-tac_on_kinetics-0547130e.pth',

    # other models from the GVF and backbone architecture ablation studies
    'r2plus1d_34-tsp_on_activitynet-avg_gvf': 'https://github.com/HumamAlwassel/TSP/releases/download/model_weights/r2plus1d_34-tsp_on_activitynet-avg_gvf-backbone_lr_0.0001-fc_lr_0.004-epoch_5-8b74eaa2.pth',
    'r2plus1d_34-tsp_on_activitynet-no_gvf' : 'https://github.com/HumamAlwassel/TSP/releases/download/model_weights/r2plus1d_34-tsp_on_activitynet-no_gvf-backbone_lr_0.0001-fc_lr_0.004-epoch_5-fb38fdd2.pth',

    'r2plus1d_18-tsp_on_activitynet'        : 'https://github.com/HumamAlwassel/TSP/releases/download/model_weights/r2plus1d_18-tsp_on_activitynet-max_gvf-backbone_lr_0.0001-fc_lr_0.002-epoch_6-22835b73.pth',
    'r2plus1d_18-tac_on_activitynet'        : 'https://github.com/HumamAlwassel/TSP/releases/download/model_weights/r2plus1d_18-tac_on_activitynet-backbone_lr_0.0001-fc_lr_0.004-epoch_5-9f56941a.pth',
    'r2plus1d_18-tac_on_kinetics'           : 'https://github.com/HumamAlwassel/TSP/releases/download/model_weights/r2plus1d_18-tac_on_kinetics-76ce975c.pth',

    'r3d_18-tsp_on_activitynet'             : 'https://github.com/HumamAlwassel/TSP/releases/download/model_weights/r3d_18-tsp_on_activitynet-max_gvf-backbone_lr_0.0001-fc_lr_0.002-epoch_6-85584422.pth',
    'r3d_18-tac_on_activitynet'             : 'https://github.com/HumamAlwassel/TSP/releases/download/model_weights/r3d_18-tac_on_activitynet-backbone_lr_0.001-fc_lr_0.01-epoch_5-31fd6e95.pth',
    'r3d_18-tac_on_kinetics'                : 'https://github.com/HumamAlwassel/TSP/releases/download/model_weights/r3d_18-tac_on_kinetics-dcd952c6.pth',
}



'''
Build PDVC model

model, criterion, postprocessors = build(opt)
model.translator = train_dataset.translator
'''
 

# This is the main function of TSP

def main(args):
    
    set_seed(args.seed)
    save_folder = build_floder(args)
    logger = create_logger(save_folder, 'train.log')
    tf_writer = SummaryWriter(os.path.join(save_folder, 'tf_summary'))

    if not args.start_from:
        backup_envir(save_folder)
        logger.info('backup evironment completed !')

    saved_info = {'best': {}, 'last': {}, 'history': {}, 'eval_history': {}}

    if args.start_from:
        args.pretrain = False
        infos_path = os.path.join(save_folder, 'info.json')
        with open(infos_path) as f:
            logger.info('Load info from {}'.format(infos_path))
            saved_info = json.load(f)
            prev_opt = saved_info[args.start_from_mode[:4]]['opt']

            exclude_opt = ['start_from', 'start_from_mode', 'pretrain']
            for opt_name in prev_opt.keys():
                if opt_name not in exclude_opt:
                    vars(args).update({opt_name: prev_opt.get(opt_name)})
                if prev_opt.get(opt_name) != vars(args).get(opt_name):
                    logger.info('Change opt {} : {} --> {}'.format(opt_name, prev_opt.get(opt_name),
                                                                   vars(args).get(opt_name)))
                    
    
    epoch = saved_info[args.start_from_mode[:4]].get('epoch', 0)
    iteration = saved_info[args.start_from_mode[:4]].get('iter', 0)
    best_val_score = saved_info[args.start_from_mode[:4]].get('best_val_score', -1e5)
    val_result_history = saved_info['history'].get('val_result_history', {})
    loss_history = saved_info['history'].get('loss_history', {})
    lr_history = saved_info['history'].get('lr_history', {})
    args.current_lr = vars(args).get('current_lr', args.lr)

    
    print(args)
    utils.init_distributed_mode(args)
    print('TORCH VERSION: ', torch.__version__)
    print('TORCHVISION VERSION: ', torchvision.__version__)
    torch.backends.cudnn.benchmark = True

    device = torch.device(args.device)
    # os.makedirs(args.output_dir, exist_ok=True)

    train_dir = os.path.join(args.root_dir, args.train_subdir)
    valid_dir = os.path.join(args.root_dir, args.valid_subdir)

    print('LOADING DATA')
    label_mappings = []
    for label_mapping_json in args.label_mapping_jsons:
        with open(label_mapping_json) as fobj:
            label_mapping = json.load(fobj)
            label_mappings.append(dict(zip(label_mapping, range(len(label_mapping)))))


    normalize = T.Normalize(mean=[0.43216, 0.394666, 0.37645],
                            std=[0.22803, 0.22145, 0.216989])
    
    transform_train = None
    transform_valid = None
    if args.backbone_tsp != 'mvit_v2_s':

        transform_train = torchvision.transforms.Compose([
            T.ToFloatTensorInZeroOne(),
            T.Resize((128, 171)),
            T.RandomHorizontalFlip(),
            normalize,
            T.RandomCrop((112, 112))
        ])

        transform_valid = torchvision.transforms.Compose([
            T.ToFloatTensorInZeroOne(),
            T.Resize((128, 171)),
            normalize,
            T.CenterCrop((112, 112))
        ])

    else:
        transform_train = torchvision.transforms.Compose([
            T.ToFloatTensorInZeroOne(),
            T.Resize((224, 224)),
            T.RandomHorizontalFlip(),
            normalize,
            T.RandomCrop((224, 224))
        ])

        transform_valid = torchvision.transforms.Compose([
            T.ToFloatTensorInZeroOne(),
            T.Resize((224, 224)),
            T.RandomHorizontalFlip(),
            normalize,
            T.CenterCrop((224, 224))
        ])

    
    dataset_train = NewDataset(
        csv_filename=args.train_csv_filename,
        root_dir=train_dir,
        clip_length=args.clip_len,
        frame_rate=args.frame_rate,
        stride=args.stride,
        anno_file=args.train_caption_file,
        dict_file=args.dict_file,
        is_training=True,
        proposal_type='gt',
        opt=args,
        transforms=None,
        global_video_feature=None,
        label_columns=args.label_columns,
        label_mappings=label_mappings,
        dataset_type='train'
    )

    metadata_df = pd.read_csv(args.metadata_csv_valid)

    dataset_valid = NewDataset(
        csv_filename=metadata_df,
        root_dir=valid_dir,
        clip_length=args.clip_len,
        frame_rate=args.frame_rate,
        stride=args.stride,
        anno_file=args.val_caption_file,
        dict_file=args.dict_file,
        is_training=False,
        proposal_type='gt',
        opt=args,
        transforms=None,
        global_video_feature=None,
        label_columns=args.label_columns,
        label_mappings=label_mappings,
        dataset_type='valid'
    )



    print('CREATING DATA LOADERS')
    # sampler_train = DistributedSampler(dataset_train, shuffle=True) if args.distributed else None
    # sampler_valid = DistributedSampler(dataset_valid, shuffle=False) if args.distributed else None

    data_loader_train = DataLoader(
        dataset_train, batch_size=args.batch_size, shuffle=True,
        num_workers=args.workers, pin_memory=True, collate_fn=collate_fn)

    data_loader_valid = DataLoader(
        dataset_valid, batch_size=args.batch_size, shuffle=False,
        num_workers=args.workers, pin_memory=True, collate_fn=collate_fn)

    print('CREATING MODEL')

    model = NewModel(backbone=args.backbone_tsp, num_classes=[len(l) for l in label_mappings], num_heads=len(args.label_columns), concat_gvf=args.global_video_features is not None, 
                     device=device, args=args, transforms_valid=transform_valid, transforms_train=transform_train)
    model.pdvcModel.translator = dataset_train.translator


    # if args.distributed and args.sync_bn:
    #     model = torch.nn.SyncBatchNorm.convert_sync_batchnorm(model)

    # criterion = torch.nn.CrossEntropyLoss(ignore_index=-1) # targets with -1 indicate missing label

    backbone_params = None
    params = None

    # fc_params = model.tspModel.fc.parameters() if len(args.label_columns) == 1 \
    #                 else chain(model.tspModel.fc1.parameters(), model.fc2.parameters())

    if args.backbone_tsp != 'mvit_v2_s':
        backbone_params = chain(model.tspModel.features.layer1.parameters(),
                                model.tspModel.features.layer2.parameters(),
                                model.tspModel.features.layer3.parameters(),
                                model.tspModel.features.layer4.parameters())
        
        params = [
            {'params': model.tspModel.features.stem.parameters(), 'lr': 0, 'name': 'stem'},
            {'params': backbone_params, 'lr': args.backbone_lr * args.world_size, 'name': 'backbone'},
        ]

    else:
        params = [
            {'params': model.tspModel.features.parameters(), 'lr': args.backbone_lr * args.world_size, 'name': 'backbone'},
            {'params': model.tspModel.fc1.parameters(), 'lr': args.fc_lr * args.world_size, 'name': 'fc1'},
            {'params': model.tspModel.fc2.parameters(), 'lr': args.fc_lr * args.world_size, 'name': 'fc2'}
        ]


    if args.pretrained_tsp_path and (not args.start_from):
        print(f'Load a model from tsp pretrained weights')
        pretrained_state_dict_tsp = torch.load(args.pretrained_tsp_path, map_location='cpu')['model']
        pretrained_state_dict_tsp = {k: v for k,v in pretrained_state_dict_tsp.items() if 'fc' not in k}
        state_dict = model.tspModel.state_dict()
        pretrained_state_dict_tsp['fc1.weight'] = state_dict['fc1.weight']
        pretrained_state_dict_tsp['fc2.weight'] = state_dict['fc2.weight']
        pretrained_state_dict_tsp['fc1.bias'] = state_dict['fc1.bias']
        pretrained_state_dict_tsp['fc2.bias'] = state_dict['fc2.bias']
        model.tspModel.load_state_dict(pretrained_state_dict_tsp)
        model.tspModel.fc2 = Model._build_fc(model.tspModel.feature_size, model.tspModel.temporal_region_num_classes)

        params = [
            {'params': model.tspModel.features.parameters(), 'lr': args.backbone_lr * args.world_size, 'name': 'backbone'},
            {'params': model.tspModel.fc1.parameters(), 'lr': args.fc_lr * args.world_size, 'name': 'fc1'},
            {'params': model.tspModel.fc2.parameters(), 'lr': args.fc_lr * args.world_size, 'name': 'fc2'}
        ]

        # optimizer = torch.optim.SGD(
        #     params, momentum=args.momentum, weight_decay=args.weight_decay
        # )
        args.start_epoch = 0
        # warmup_iters = args.lr_warmup_epochs * len(data_loader_train)
        # lr_milestones = [len(data_loader_train) * m for m in args.lr_milestones]
        # lr_scheduler = WarmupMultiStepLR(
        #     optimizer, milestones=lr_milestones, gamma=args.lr_gamma,
        #     warmup_iters=warmup_iters, warmup_factor=1e-5)
    
    # Recover the pdvc parameters
    model_pth = None
    # if args.start_from and (not args.pretrain):
    #     if args.start_from_mode == 'best':
    #         model_pth = torch.load(os.path.join(save_folder, 'model-best.pth'))
    #     elif args.start_from_mode == 'last':
    #         model_pth = torch.load(os.path.join(save_folder, 'model-last.pth'))
    #     logger.info('Loading pth from {}, iteration:{}'.format(save_folder, iteration))
    #     model.pdvcModel.load_state_dict(model_pth['model'])

    # Load the pre-trained model
    if args.pretrain and (not args.start_from):
        logger.info('Load pre-trained parameters from {}'.format(args.pretrain_path))
        model_pth = torch.load(args.pretrain_path, map_location=torch.device(args.device))
        # query_weight = model_pth['model'].pop('query_embed.weight')
        if args.pretrain == 'encoder':
            encoder_filter = model.pdvcModel.get_filter_rule_for_encoder()
            encoder_pth = {k:v for k,v in model_pth['model'].items() if encoder_filter(k)}
            model.pdvcModel.load_state_dict(encoder_pth, strict=True)
        elif args.pretrain == 'decoder':
            encoder_filter = model.pdvcModel.get_filter_rule_for_encoder()
            decoder_pth = {k:v for k,v in model_pth['model'].items() if not encoder_filter(k)}
            model.pdvcModel.load_state_dict(decoder_pth, strict=True)
            pass
        elif args.pretrain == 'full':
            # model_pth = transfer(model, model_pth)
            model.pdvcModel.load_state_dict(model_pth['model'], strict=True)
        else:
            raise ValueError("wrong value of args.pretrain")

        params.append(
                {'params': model.pdvcModel.parameters(), 'lr': args.lr, 'name': 'decoder'},
        )

    # optimizer = torch.optim.SGD(
    #     params, momentum=args.momentum, weight_decay=args.weight_decay
    # )
    
    optimizer = None
    if args.optimizer_type == 'adam':
        optimizer = optim.Adam(params=params, amsgrad=True, weight_decay=args.weight_decay)

    elif args.optimizer_type == 'adamw':
        optimizer = optim.AdamW(params=params, amsgrad=True, weight_decay=args.weight_decay)
        
    # convert scheduler to be per iteration, not per epoch, for warmup that lasts
    # between different epochs
    # warmup_iters = args.lr_warmup_epochs * len(data_loader_train)
    # lr_milestones = [len(data_loader_train) * m for m in args.lr_milestones]
    # lr_scheduler = WarmupMultiStepLR(
    #     optimizer, milestones=lr_milestones, gamma=args.lr_gamma,
    #     warmup_iters=warmup_iters, warmup_factor=1e-5)

    milestone = [args.learning_rate_decay_start + args.learning_rate_decay_every * _ for _ in range(int((args.epoch - args.learning_rate_decay_start) / args.learning_rate_decay_every))]
    lr_scheduler = torch.optim.lr_scheduler.MultiStepLR(optimizer, milestone, gamma=args.learning_rate_decay_rate)
    
    
    # model_without_ddp = model
    # if args.distributed:
    #     model = torch.nn.parallel.DistributedDataParallel(model, device_ids=[args.gpu])
    #     model_without_ddp = model.module


    if args.start_from and (not args.pretrain):
        model.tspModel.fc2 = Model._build_fc(model.tspModel.feature_size, model.tspModel.temporal_region_num_classes)
        if args.start_from_mode == 'best':
            model_pth = torch.load(os.path.join(save_folder, 'model-best.pth'))
        elif args.start_from_mode == 'last':
            model_pth = torch.load(os.path.join(save_folder, 'model-last.pth'))
        logger.info('Loading pth from {}, iteration:{}'.format(save_folder, iteration))
        # model.pdvcModel.load_state_dict(model_pth['model'])
        print(f'Resuming from checkpoint.')
        # checkpoint = torch.load(args., map_location='cpu')
        model.load_state_dict(model_pth['model'])
        optimizer.load_state_dict(model_pth['optimizer'])
        lr_scheduler.step(epoch-1)
        # lr_scheduler.load_state_dict(checkpoint['lr_scheduler'])
        # args.start_epoch = checkpoint['epoch'] + 1

    

    model.to(device)
    

    # if args.start_from:
    #     optimizer.load_state_dict(model_pth['optimizer'])
    #     lr_scheduler.step(epoch-1)

    print_opt(args, model, logger)
    print_alert_message('Strat training !', logger)



    # if args.valid_only:
    #     epoch = args.start_epoch - 1 if  args.resume else args.start_epoch
    #     evaluate(model=model, criterion=criterion, data_loader=data_loader_valid, device=device, epoch=epoch,
    #         print_freq=args.print_freq, label_columns=args.label_columns, loss_alphas=args.loss_alphas,
    #         output_dir=args.output_dir)
    #     return


    print('START TRAINING')
    loss_sum = OrderedDict()
    bad_video_num = 0

    start = time.time()

    weight_dict = model.pdvcCriterion.weight_dict
    logger.info('loss type: {}'.format(weight_dict.keys()))
    logger.info('loss weights: {}'.format(weight_dict.values()))

    
    while True:
        if True:
            # scheduled sampling rate update
            if epoch > args.scheduled_sampling_start >= 0:
                frac = (epoch - args.scheduled_sampling_start) // args.scheduled_sampling_increase_every
                args.ss_prob = min(args.basic_ss_prob + args.scheduled_sampling_increase_prob * frac,
                                  args.scheduled_sampling_max_prob)
                model.pdvcModel.caption_head.ss_prob = args.ss_prob

            print('lr:{}'.format(float(args.current_lr)))

        
        # Batch-level iteration
        for dt in tqdm(data_loader_train, disable=args.disable_tqdm):
            if args.device=='cuda':
                torch.cuda.synchronize(args.device)
            if args.debug:
                # each epoch contains less mini-batches for debugging
                if (iteration + 1) % 5 == 0:
                    iteration += 1
                    break
            iteration += 1

            optimizer.zero_grad()
            dt = {key: _.to(args.device) if isinstance(_, torch.Tensor) else _ for key, _ in dt.items()}
            dt['video_target'] = [
                {key: _.to(args.device) if isinstance(_, torch.Tensor) else _ for key, _ in vid_info.items()} for vid_info in
                dt['video_target']]

            dt = collections.defaultdict(lambda: None, dt)

            # pdvc forward
            # output, loss = model(dt, criterion, opt.transformer_input_type)
            _, loss, tsp_head_loss = model.forward(dt, args.loss_alphas, eval_mode=False)

            final_loss = sum(loss[k] * weight_dict[k] for k in loss.keys() if k in weight_dict)
            (final_loss + 0.25 * tsp_head_loss).backward()
            torch.nn.utils.clip_grad_norm_(model.parameters(), args.grad_clip)

            optimizer.step()

            for loss_k,loss_v in loss.items():
                loss_sum[loss_k] = loss_sum.get(loss_k, 0)+ loss_v.item()
            loss_sum['total_loss'] = loss_sum.get('total_loss', 0) + final_loss.item() + tsp_head_loss.item() * 0.25

            if args.device=='cuda':
                torch.cuda.synchronize()

            losses_log_every = int(len(data_loader_train) / 10)

            if args.debug:
                losses_log_every = 6

            if iteration % losses_log_every == 0:
                end = time.time()
                for k in loss_sum.keys():
                    loss_sum[k] = np.round(loss_sum[k] /losses_log_every, 3).item()

                logger.info(
                    "ID {} iter {} (epoch {}), \nloss = {}, \ntime/iter = {:.3f}, bad_vid = {:.3f}"
                        .format(args.id, iteration, epoch, loss_sum,
                                (end - start) / losses_log_every, bad_video_num))

                tf_writer.add_scalar('lr', args.current_lr, iteration)
                for loss_type in loss_sum.keys():
                    tf_writer.add_scalar(loss_type, loss_sum[loss_type], iteration)
                loss_history[iteration] = loss_sum
                lr_history[iteration] = args.current_lr
                loss_sum = OrderedDict()
                start = time.time()
                bad_video_num = 0
                torch.cuda.empty_cache()

        # evaluation
        if (epoch % args.save_checkpoint_every == 0) and (epoch >= args.min_epoch_when_save):

            # Save model
            saved_pth = {'epoch': epoch,
                         'model': model.state_dict(),
                         'optimizer': optimizer.state_dict(), }

            if args.save_all_checkpoint:
                checkpoint_path = os.path.join(save_folder, 'model_iter_{}.pth'.format(iteration))
            else:
                checkpoint_path = os.path.join(save_folder, 'model-last.pth')

            torch.save(saved_pth, checkpoint_path)

            model.eval()
            result_json_path = os.path.join(save_folder, 'prediction',
                                         'num{}_epoch{}.json'.format(
                                             len(dataset_valid), epoch))
            eval_score, eval_loss = evaluate(model, model.pdvcCriterion, model.pdvcPostprocessor, data_loader_valid, result_json_path, logger=logger, alpha=args.ec_alpha, device=args.device, debug=args.debug)
            if args.caption_decoder_type == 'none':
                current_score = 2./(1./eval_score['Precision'] + 1./eval_score['Recall'])
            else:
                if args.criteria_for_best_ckpt == 'dvc':
                    current_score = np.array(eval_score['METEOR']).mean() + np.array(eval_score['soda_c']).mean()
                else:
                    current_score = np.array(eval_score['para_METEOR']).mean() + np.array(eval_score['para_CIDEr']).mean() + np.array(eval_score['para_Bleu_4']).mean()

            # add to tf summary
            for key in eval_score.keys():
                tf_writer.add_scalar(key, np.array(eval_score[key]).mean(), iteration)

            for loss_type in eval_loss.keys():
                tf_writer.add_scalar('eval_' + loss_type, eval_loss[loss_type], iteration)

            _ = [item.append(np.array(item).mean()) for item in eval_score.values() if isinstance(item, list)]
            print_info = '\n'.join([key + ":" + str(eval_score[key]) for key in eval_score.keys()])
            logger.info('\nValidation results of iter {}:\n'.format(iteration) + print_info)
            logger.info('\noverall score of iter {}: {}\n'.format(iteration, current_score))
            val_result_history[epoch] = {'eval_score': eval_score}
            logger.info('Save model at iter {} to {}.'.format(iteration, checkpoint_path))

            # save the model parameter and  of best epoch
            if current_score >= best_val_score:
                best_val_score = current_score
                best_epoch = epoch
                saved_info['best'] = {'opt': vars(args),
                                      'iter': iteration,
                                      'epoch': best_epoch,
                                      'best_val_score': best_val_score,
                                      'result_json_path': result_json_path,
                                      'avg_proposal_num': eval_score['avg_proposal_number'],
                                      'Precision': eval_score['Precision'],
                                      'Recall': eval_score['Recall']
                                      }

                # suffix = "RL" if sc_flag else "CE"
                torch.save(saved_pth, os.path.join(save_folder, 'model-best.pth'))
                logger.info('Save Best-model at iter {} to checkpoint file.'.format(iteration))

            saved_info['last'] = {'opt': vars(args),
                                  'iter': iteration,
                                  'epoch': epoch,
                                  'best_val_score': best_val_score,
                                  }
            saved_info['history'] = {'val_result_history': val_result_history,
                                     'loss_history': loss_history,
                                     'lr_history': lr_history,
                                     # 'query_matched_fre_hist': query_matched_fre_hist,
                                     }
            with open(os.path.join(save_folder, 'info.json'), 'w') as f:
                json.dump(saved_info, f)
            logger.info('Save info to info.json')

            model.train()

        epoch += 1
        lr_scheduler.step()
        args.current_lr = optimizer.param_groups[0]['lr']
        torch.cuda.empty_cache()
        # Stop criterion
        if epoch >= args.epoch:
            tf_writer.close()
            break




if __name__ == '__main__':
    # opt for PDVC
    opt = opts.parse_opts()
    if opt.gpu_id:
        os.environ["CUDA_VISIBLE_DEVICES"] = ",".join([str(i) for i in opt.gpu_id])
    if opt.disable_cudnn:
        torch.backends.cudnn.enabled = False

    os.environ['KMP_DUPLICATE_LIB_OK'] = 'True' # to avoid OMP problem on macos
    main(opt)
    # train(opt)

