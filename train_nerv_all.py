import argparse
import csv
import os
import random
import shutil
from copy import deepcopy
from datetime import datetime

import imageio
import numpy as np
import pandas as pd
import torch
import torch.backends.cudnn as cudnn
import torch.multiprocessing as mp
import torch.optim as optim
import torch.utils.data
from dahuffman import HuffmanCodec
from torch.utils.data import Subset
from torch.utils.tensorboard import SummaryWriter
from torchvision.utils import save_image
import yaml

from hnerv_utils import *
from model import SPP


def is_main_process(local_rank):
    return local_rank in [0, None]


def append_text(file_path, text):
    with open(file_path, 'a') as f:
        f.write(text + '\n')


def append_rank0_log(args, text):
    append_text('{}/rank0.txt'.format(args.outf), text)


def main():
    parser = argparse.ArgumentParser(
        description='Train/evaluate SPP-NeRV on a frame-sequence dataset.',
        formatter_class=argparse.ArgumentDefaultsHelpFormatter,
    )

    # Dataset parameters
    parser.add_argument('--split_num', type=int, default=2, help='Split each frame into N x N patches.')
    parser.add_argument('--data_path', type=str, default='', help='Path to input frame directory.')
    parser.add_argument('--vid', type=str, default='k400_train0', help='Video identifier used in logs/outputs.')
    parser.add_argument('--shuffle_data', action='store_true', help='Shuffle frame indices before data split.')
    parser.add_argument('--data_split', type=str, default='1_1_1', help='Split ratio as valid_train/total_train/period (example: 18_19_20).',)
    parser.add_argument('--crop_list', type=str, default='640_1280', help='Target crop size as H_W.')
    parser.add_argument('--resize_list', type=str, default='-1', help='Resize size as H_W, or -1 to keep default path.')

    # NERV architecture parameters
    # Embedding and encoding parameters
    parser.add_argument('--model', type=str, default='', help='Model name.')
    parser.add_argument('--embed', type=str, default='', help='Embedding setting (empty string disables explicit positional embedding).',)
    parser.add_argument('--ks', type=str, default='0_3_3', help='Kernel setting string for encoder/decoder.')
    parser.add_argument('--enc_blks', type=int, default=1, help='Number of encoder blocks.')
    parser.add_argument('--enc_strds', type=int, nargs='+', default=[], help='Encoder stride list.')
    parser.add_argument('--enc_dim', type=str, default='64_16', help='Encoder latent dim and embed ratio as C_R.')
    parser.add_argument('--modelsize', type=float, default=1.5, help='Target total model size (millions of parameters).',)
    parser.add_argument('--saturate_stages', type=int, default=-1, help='Saturation stages for model-size computation (-1 means all stages).',)

    # Decoding parameters: FC + Conv
    parser.add_argument('--lfreq', type=str, default='pi', help='Frequency basis setting for decoder.')
    parser.add_argument('--fc_dim', type=int, default=None, help='Decoder FC dimension (None uses internal sizing).')
    parser.add_argument('--fc_hw', type=str, default='9_16', help='Initial decoder feature-map size as H_W.')
    parser.add_argument('--reduce', type=float, default=1.2, help='Channel reduction ratio between decoder stages.')
    parser.add_argument('--lower_width', type=int, default=32, help='Minimum channel width in decoder.')
    parser.add_argument('--dec_strds', type=int, nargs='+', default=[5, 3, 2, 2, 2], help='Decoder stride list.')
    parser.add_argument('--dec_blks', type=int, nargs='+', default=[1, 1, 1, 1, 1], help='Number of blocks per decoder stage.')
    # parser.add_argument('--num_blks', type=str, default='1_1', help='block number for encoder and decoder')
    parser.add_argument('--conv_type', default=['convnext', 'pshuffel'], type=str, nargs='+', help='Convolution type pair: <encoder_type> <decoder_type>.', choices=['pshuffel', 'conv', 'convnext', 'interpolate', 'pshuffel_3x3'],)
    parser.add_argument('--norm', default='none', type=str, help='Normalization layer type.', choices=['none', 'bn', 'in'])
    parser.add_argument('--act', type=str, default='gelu', help='Activation function.', choices=['relu', 'leaky', 'leaky01', 'relu6', 'gelu', 'swish', 'softplus', 'hardswish', 'sin', 'ressin'],)
    parser.add_argument('--sft_block', type=str, default='none', help='SFT block type.')
    parser.add_argument('--ch_t', type=int, default=32, help='SFT input channel count.')
    parser.add_argument('--block_dim', type=int, default=128, help='Transformer block dimension (ENeRV-related).')

    # General training setups
    parser.add_argument('-j', '--workers', type=int, default=4, help='Number of data-loading workers.')
    parser.add_argument('-b', '--batchSize', type=int, default=1, help='Mini-batch size.')
    parser.add_argument('--start_epoch', type=int, default=-1, help='Manual start epoch (-1 uses resume checkpoint epoch).')
    parser.add_argument('--not_resume', action='store_true', help='Disable auto-resume from model_latest.pth.')
    parser.add_argument('-e', '--epochs', type=int, default=5, help='Total training epochs.')
    parser.add_argument('--block_params', type=str, default='1_1', help='Residual-block/percentile config string.')
    parser.add_argument('--lr', type=float, default=0.001, help='Initial learning rate.')
    parser.add_argument('--lr_type', type=str, default='cosine_0.1_1_0.1', help='Learning-rate schedule config string.')
    parser.add_argument('--loss', type=str, default='Fusion6', help='Loss function name.')
    parser.add_argument('--out_bias', default='tanh', type=str, help='Output activation/bias mode.')
    parser.add_argument('--optim_type', default='adan', type=str, help='Optimizer type (e.g., Adam, Adan).')
    parser.add_argument('--clip_max_norm', default=0.0, type=float, help='Gradient clip max norm (0 disables clipping).')
    parser.add_argument('--inpanting', default='none', type=str, help='Inpainting mode.')
    parser.add_argument('--interpolation', action='store_true', default=False, help='Enable interpolation mode.')
    parser.add_argument('--embed_inter', action='store_true', default=False, help='Enable embedding interpolation.')
    parser.add_argument('--cabac', action='store_true', default=False, help='Enable CABAC-related option.')

    # evaluation parameters
    parser.add_argument('--quant', action='store_true', default=False, help='Enable quantized evaluation branch.')
    parser.add_argument('--eval_only', action='store_true', default=False, help='Run evaluation only (skip training).')
    parser.add_argument('--eval_freq', type=int, default=10, help='Evaluate every N epochs.')
    parser.add_argument('--quant_model_bit', type=int, default=8, help='Model quantization bit-width (-1 disables quantized model copy).')
    parser.add_argument('--quant_embed_bit', type=int, default=6, help='Embedding quantization bit-width.')
    parser.add_argument('--quant_axis', type=int, default=0, help='Quantization axis (-1 means per-tensor).')
    parser.add_argument('--dump_images', action='store_true', default=False, help='Dump predicted images during evaluation.')
    parser.add_argument('--dump_videos', action='store_true', default=False, help='Create GIF video from dumped images.')
    parser.add_argument('--eval_fps', action='store_true', default=False, help='Run repeated forwards to benchmark FPS.')
    parser.add_argument('--encoder_file', default='', type=str, help='Path to external encoder/embedding file.')
    parser.add_argument('--dump_values', action='store_true', default=False, help='Dump intermediate values for debugging.')
    parser.add_argument('--dump_features', action='store_true', default=False, help='Dump intermediate features for debugging.')
    # distribute learning parameters
    parser.add_argument('--manualSeed', type=int, default=1, help='Random seed.')
    parser.add_argument('-d', '--distributed', action='store_true', default=False, help='Enable distributed training mode.')

    # logging, output directory,
    parser.add_argument('--debug', action='store_true', help='Debug mode (forces eval_freq=1 and output/debug path).')
    parser.add_argument('-p', '--print-freq', default=50, type=int, help='Print frequency (steps).')
    parser.add_argument('--weight', default='None', type=str, help='Checkpoint path for weight initialization/evaluation.')
    parser.add_argument('--overwrite', action='store_true', help='Remove existing output directory before running.')
    parser.add_argument('--outf', default='unify', help='Output directory name under ./output.')
    parser.add_argument('--suffix', default='', help='Optional experiment-name suffix string.')

    args = parser.parse_args()
    torch.set_printoptions(precision=4)
    if args.debug:
        args.eval_freq = 1
        args.outf = 'output/debug'
    else:
        args.outf = os.path.join('output', args.outf)

    args.enc_strd_str = ','.join(str(x) for x in args.enc_strds)
    args.dec_strd_str = ','.join(str(x) for x in args.dec_strds)
    args.quant_str = f'quant_M{args.quant_model_bit}_E{args.quant_embed_bit}'
    args.exp_id = f'{args.vid}/Size{args.modelsize}'

    if args.overwrite and os.path.isdir(args.outf):
        print('Will overwrite the existing output dir!')
        shutil.rmtree(args.outf)

    if not os.path.isdir(args.outf):
        os.makedirs(args.outf)

    port = hash(args.exp_id) % 20000 + 10000
    args.init_method = f'tcp://127.0.0.1:{port}'
    print(f'init_method: {args.init_method}', flush=True)

    torch.set_printoptions(precision=2)
    args.ngpus_per_node = torch.cuda.device_count()
    train(None, args)


def data_to_gpu(x, device):
    return x.to(device)

def train(local_rank, args):
    # cudnn.benchmark = True
    cudnn.benchmark = False
    cudnn.deterministic = True
    torch.manual_seed(args.manualSeed)
    np.random.seed(args.manualSeed)
    random.seed(args.manualSeed)

    args.metric_names = [
        'pred_seen_psnr', 'pred_seen_ssim', 'pred_unseen_psnr', 'pred_unseen_ssim',
        'quant_seen_psnr', 'quant_seen_ssim', 'quant_unseen_psnr', 'quant_unseen_ssim',]
    best_metric_list = [torch.tensor(0) for _ in range(len(args.metric_names))]

    # setup dataloader
    full_dataset = VideoDataSet(args)
    full_dataloader = torch.utils.data.DataLoader(
        full_dataset, batch_size=args.batchSize, shuffle=False, num_workers=args.workers, 
        pin_memory=True, sampler=None, drop_last=False, worker_init_fn=worker_init_fn,)
    args.final_size = full_dataset.final_size
    args.full_data_length = len(full_dataset)
    args.patches_per_frame = args.split_num * args.split_num
    if args.full_data_length % args.patches_per_frame != 0:
        raise ValueError(
            f'full_data_length ({args.full_data_length}) is not divisible by '
            f'patches_per_frame ({args.patches_per_frame}).'
        )
    args.num_frames = args.full_data_length // args.patches_per_frame
    args.total_pixels = args.final_size * args.num_frames

    split_num_list = [int(x) for x in args.data_split.split('_')]
    train_ind_list, args.val_ind_list = data_split(list(range(args.full_data_length)), split_num_list, args.shuffle_data, 0)
    print("train:", train_ind_list)
    print("val:", args.val_ind_list)
    args.dump_vis = (args.dump_images or args.dump_videos)

    #  Make sure the testing dataset is fixed for every run
    train_dataset = Subset(full_dataset, train_ind_list)
    train_dataloader = torch.utils.data.DataLoader(
        train_dataset, batch_size=args.batchSize, shuffle=True, num_workers=args.workers,
        pin_memory=True, sampler=None, drop_last=True, worker_init_fn=worker_init_fn,)

    total_enc_strds = np.prod(args.enc_strds)
    enc_dim1, embed_ratio = [float(x) for x in args.enc_dim.split('_')]
    embed_dim = embed_ratio
    embed_param = float(embed_dim) / total_enc_strds**2 * args.final_size * args.num_frames
    print(
        f'embed_dim: {embed_dim}, total_enc_strds: {total_enc_strds}, '
        f'final_size: {args.final_size}, full_data_length(patches): {args.full_data_length}, '
        f'num_frames: {args.num_frames}')
    args.fc_dim = 200
    model = SPP(args)
    while (model.decoder_params() + embed_param / 1e6 > args.modelsize):
        near_modelsize = model.decoder_params() + embed_param / 1e6
        args.fc_dim = args.fc_dim - 1
        model = SPP(args)
    print(f'b: {near_modelsize}, a: {model.decoder_params() + embed_param / 1e6}')
    args.fc_dim = (
        args.fc_dim
        if abs(args.modelsize - (model.decoder_params() + embed_param / 1e6)) < abs(args.modelsize - near_modelsize)
        else args.fc_dim + 1
    )

    # Building model
    if args.model == "SPP":
        model = SPP(args)
    else:
        exit()

    args_text = yaml.safe_dump(args.__dict__, default_flow_style=False)
    with open(os.path.join(args.outf, 'args.yaml'), 'w') as f:
        f.write(args_text)

    ##### get model params and flops #####
    encoder_param = sum([p.data.nelement() for p in model.encoder.parameters()]) / 1e6
    decoder_param = model.decoder_params()
    total_param = decoder_param + embed_param / 1e6
    args.encoder_param, args.decoder_param, args.total_param = encoder_param, decoder_param, total_param
    param_str = f'Encoder_{round(encoder_param, 2)}M_Decoder_{round(decoder_param, 4)}M_Total_{round(total_param, 4)}M'
    print(f'{args}\n {param_str}', flush=True)
    append_rank0_log(args, str(model) + '\n' + f'{param_str}')
    writer = SummaryWriter(os.path.join(args.outf, param_str, 'tensorboard'))

    # distrite model to gpu or parallel
    print(f'Use GPU: {local_rank} for training')
    model = model.cuda()

    if args.optim_type == "Adam":
        optimizer = optim.Adam(model.parameters(), lr=args.lr)
    elif args.optim_type == "Adan":
        from optimizer import Adan
        optimizer = Adan(model.parameters(), lr=args.lr)
    args.transform_func = TransformInput(args)

    # resume from args.weight
    checkpoint = None
    if args.weight != 'None':
        print(f"=> loading checkpoint '{args.weight}'")
        checkpoint_path = args.weight
        checkpoint = torch.load(checkpoint_path, map_location='cpu')
        orig_ckt = checkpoint['state_dict']
        new_ckt = {k.replace('blocks.0.', ''): v for k, v in orig_ckt.items()}
        if 'module' in list(orig_ckt.keys())[0] and not hasattr(model, 'module'):
            new_ckt = {k.replace('module.', ''): v for k, v in new_ckt.items()}
            model.load_state_dict(new_ckt, strict=False)
        elif 'module' not in list(orig_ckt.keys())[0] and hasattr(model, 'module'):
            model.module.load_state_dict(new_ckt, strict=False)
        else:
            model.load_state_dict(new_ckt, strict=False)
        print(f"=> loaded checkpoint '{args.weight}' (epoch {checkpoint['epoch']})")

    # resume from model_latest
    if not args.not_resume:
        checkpoint_path = os.path.join(args.outf, 'model_latest.pth')
        if os.path.isfile(checkpoint_path):
            checkpoint = torch.load(checkpoint_path, map_location='cpu')
            model.load_state_dict(checkpoint['state_dict'])
            print(f"=> Auto resume loaded checkpoint '{checkpoint_path}' (epoch {checkpoint['epoch']})")
        else:
            print(f"=> No resume checkpoint found at '{checkpoint_path}'")

    if args.start_epoch < 0:
        if checkpoint is not None:
            args.start_epoch = checkpoint['epoch']
        args.start_epoch = max(args.start_epoch, 0)

    if args.eval_only:
        print_str = 'Evaluation ... \n {} Results for checkpoint: {}\n'.format(
            datetime.now().strftime('%Y_%m_%d_%H_%M_%S'), args.weight
        )
        results_list, hw = evaluate(model, full_dataloader, local_rank, args, args.dump_vis, huffman_coding=True)
        print_str = f'PSNR for output {hw} for quant {args.quant_str}: '
        for i, (metric_name, best_metric_value, metric_value) in enumerate(zip(args.metric_names, best_metric_list, results_list)):
            best_metric_value = best_metric_value if best_metric_value > metric_value.max() else metric_value.max()
            cur_v = RoundTensor(best_metric_value, 2 if 'psnr' in metric_name else 4)
            print_str += f'best_{metric_name}: {cur_v} | '
            best_metric_list[i] = best_metric_value
        if is_main_process(local_rank):
            print(print_str, flush=True)
            with open('{}/eval.txt'.format(args.outf), 'a') as f:
                f.write(print_str + '\n\n')
            args.train_time, args.cur_epoch = 0, args.epochs
            Dump2CSV(args, best_metric_list, results_list, [torch.tensor(0)], 'eval.csv')

        return

    # Training
    start = datetime.now()
    time_list = []
    psnr_list = []
    for epoch in range(args.start_epoch, args.epochs):
        model.train()
        epoch_start_time = datetime.now()
        pred_psnr_list = []
        # iterate over dataloader
        device = next(model.parameters()).device
        for i, sample in enumerate(train_dataloader):
            img_data = data_to_gpu(sample['img'], device)
            img_idx = data_to_gpu(sample['img_idx'], device)
            norm_img_idx = data_to_gpu(sample['norm_img_idx'], device)
            part_img = data_to_gpu(sample['part'], device)
            part_idx = data_to_gpu(sample['part_idx'], device)
            norm_part_idx = data_to_gpu(sample['norm_part_idx'], device)

            # forward and backward
            part_img, img_gt = part_img, part_img
            cur_epoch = (epoch + float(i) / len(train_dataloader)) / args.epochs
            lr = adjust_lr(optimizer, cur_epoch, i, args)

            cur_input = img_data
            img_out, _, _ = model(cur_input, norm_idx=norm_img_idx, part_img=part_img, part_idx=norm_part_idx)
            final_loss = loss_fn(img_out, img_gt, args.loss, img_data=img_data, split_num=args.split_num)
            optimizer.zero_grad()
            final_loss.backward()
            if args.clip_max_norm > 0:
                torch.nn.utils.clip_grad_norm_(model.parameters(), args.clip_max_norm)
            optimizer.step()

            pred_psnr_list.append(psnr_fn_single(img_out.detach(), img_gt))
            if i == len(train_dataloader) - 1:
                pred_psnr = torch.cat(pred_psnr_list).mean()
                print_str = '[{}] Rank:{}, Epoch[{}/{}], Step [{}/{}], lr:{:.2e} pred_PSNR: {}'.format(
                    datetime.now().strftime('%Y/%m/%d %H:%M:%S'),
                    local_rank, epoch + 1, args.epochs, i + 1, len(train_dataloader), lr, RoundTensor(pred_psnr, 4),)
                print(print_str, flush=True)
                if is_main_process(local_rank):
                    append_rank0_log(args, print_str)

        # ADD train_PSNR TO TENSORBOARD
        epoch_end_time = datetime.now()
        h, w = img_out.shape[-2:]
        writer.add_scalar(f'Train/pred_PSNR_{h}X{w}', pred_psnr, epoch + 1)
        writer.add_scalar('Train/lr', lr, epoch + 1)
        time_list.append((epoch_end_time - epoch_start_time).total_seconds())

        # evaluation
        if (epoch + 1) % args.eval_freq == 0 or (args.epochs - epoch) in [1]:
            results_list, hw = evaluate(
                model, full_dataloader, local_rank, args,
                args.dump_vis if epoch == args.epochs - 1 else False,
                True if epoch == args.epochs - 1 else False,)
            # ADD val_PSNR TO TENSORBOARD
            print_str = f'Eval at epoch {epoch+1} for {hw}: '
            for i, (metric_name, best_metric_value, metric_value) in enumerate(zip(args.metric_names, best_metric_list, results_list)):
                best_metric_value = best_metric_value if best_metric_value > metric_value.max() else metric_value.max()
                if 'psnr' in metric_name:
                    writer.add_scalar(f'Val/{metric_name}_{hw}', metric_value.max(), epoch + 1)
                    writer.add_scalar(f'Val/best_{metric_name}_{hw}', best_metric_value, epoch + 1)
                    if metric_name == 'pred_seen_psnr':
                        psnr_list.append(metric_value.max())
                print_str += f'{metric_name}: {RoundTensor(metric_value, 4)} | '
                best_metric_list[i] = best_metric_value
            print(print_str, flush=True)
            append_rank0_log(args, print_str)

        state_dict = model.state_dict()
        save_checkpoint = {
            'epoch': epoch + 1,
            'state_dict': state_dict,
            'optimizer': optimizer.state_dict(),
        }
        torch.save(save_checkpoint, '{}/model_latest.pth'.format(args.outf))
        if (epoch + 1) % args.epochs == 0:
            args.cur_epoch = epoch + 1
            args.train_time = str(datetime.now() - start)
            Dump2CSV(args, best_metric_list, results_list, psnr_list, f'epoch{epoch + 1}.csv')

    print_str = "Training complete in: " + str(datetime.now() - start)
    total_time_seconds = torch.tensor(time_list).sum().item()
    total_time = convert(total_time_seconds)
    print_str += "\n Training wo evaluation complete in: {}, {}s".format(total_time, total_time_seconds)
    print(print_str)
    append_rank0_log(args, print_str)

def convert(seconds):
    seconds = seconds % (24 * 3600)
    hour = seconds // 3600
    seconds %= 3600
    minutes = seconds // 60
    seconds %= 60
    return "%d:%02d:%02d" % (hour, minutes, seconds)


# Writing final results in CSV file
def Dump2CSV(args, best_results_list, results_list, psnr_list, filename='results.csv'):
    result_dict = {
        'Vid': args.vid,
        'CurEpoch': args.cur_epoch,
        'Time': args.train_time,
        'FPS': args.fps,
        'Split': args.data_split,
        'Embed': args.embed,
        'Crop': args.crop_list,
        'Resize': args.resize_list,
        'Lr_type': args.lr_type,
        'LR (E-3)': args.lr * 1e3,
        'Batch': args.batchSize,
        'Size (M)': f'{round(args.encoder_param, 2)}_{round(args.decoder_param, 2)}_{round(args.total_param, 2)}',
        'ModelSize': args.modelsize,
        'Epoch': args.epochs,
        'Loss': args.loss,
        'Act': args.act,
        'Norm': args.norm,
        'FC': args.fc_hw,
        'Reduce': args.reduce,
        'ENC_type': args.conv_type[0],
        'ENC_strds': args.enc_strd_str,
        'KS': args.ks,
        'enc_dim': args.enc_dim,
        'DEC': args.conv_type[1],
        'DEC_strds': args.dec_strd_str,
        'lower_width': args.lower_width,
        'Quant': args.quant_str,
        'bits/param': args.bits_per_param,
        'bits/param w/ overhead': args.full_bits_per_param,
        'bits/pixel': args.total_bpp,
        f'PSNR_list_{args.eval_freq}': ','.join([RoundTensor(v, 2) for v in psnr_list]),
    }
    result_dict.update({f'best_{k}': RoundTensor(v, 4) for k, v in zip(args.metric_names, best_results_list)})
    result_dict.update({f'{k}': RoundTensor(v, 4) for k, v in zip(args.metric_names, results_list)})
    csv_path = os.path.join(args.outf, filename)
    print(f'results dumped to {csv_path}')
    pd.DataFrame(result_dict, index=[0]).to_csv(csv_path)


@torch.no_grad()
def evaluate(model, full_dataloader, local_rank, args, dump_vis=False, huffman_coding=False):
    img_embed_list = []
    model_list, quant_ckt = quant_model(model, args)
    metric_list = [[] for _ in range(len(args.metric_names))]

    for model_ind, cur_model in enumerate(model_list):
        time_list = []
        cur_model.eval()
        device = next(cur_model.parameters()).device
        if dump_vis:
            visual_dir = f'{args.outf}/visualize_model' + ('_quant' if model_ind else '_orig')
            print(f'Saving predictions to {visual_dir}...')
            if not os.path.isdir(visual_dir):
                os.makedirs(visual_dir)

        combine = []
        psnr_avg = []
        for i, sample in enumerate(full_dataloader):
            img_data = data_to_gpu(sample['img'], device)
            img_idx = data_to_gpu(sample['img_idx'], device)
            norm_img_idx = data_to_gpu(sample['norm_img_idx'], device)
            part_img = data_to_gpu(sample['part'], device)
            part_idx = data_to_gpu(sample['part_idx'], device)
            norm_part_idx = data_to_gpu(sample['norm_part_idx'], device)

            part_img, img_gt = part_img, part_img
            cur_input = img_data

            img_out, embed_list, dec_time = cur_model(cur_input, norm_idx=norm_img_idx, part_img=part_img, part_idx=norm_part_idx)
            if model_ind == 0:
                img_embed_list.append(embed_list[0])

            # collect decoding fps
            time_list.append(dec_time)
            if args.eval_fps:
                time_list.pop()
                for _ in range(100):
                    _, _, dec_time = cur_model(cur_input, embed_list[0], norm_idx=norm_img_idx, part_img=part_img, part_idx=norm_part_idx)
                    time_list.append(dec_time)

            # compute psnr and ms-ssim
            pred_psnr, pred_ssim = psnr_fn_batch([img_out], img_gt), msssim_fn_batch([img_out], img_gt)
            for metric_idx, cur_v in enumerate([pred_psnr, pred_ssim]):
                for batch_i, cur_img_idx in enumerate(img_idx):
                    metric_idx_start = 2 if cur_img_idx in args.val_ind_list else 0
                    metric_list[metric_idx_start + metric_idx + 4 * model_ind].append(cur_v[:, batch_i])

            # dump predictions
            if dump_vis:
                for batch_ind, cur_img_idx in enumerate(img_idx):
                    temp_psnr_list = ','.join([str(round(x[batch_ind].item(), 2)) for x in pred_psnr])
                    combine.append(img_out[batch_ind])
                    psnr_avg.append(pred_psnr.item())
                    if i % (args.split_num * args.split_num) == 3:
                        combine_img = combine_images(combine, args.split_num)
                        avg = str(round(sum(psnr_avg) / len(psnr_avg), 2))
                        save_image(combine_img, f'{visual_dir}/pred_{img_idx.item():04d}_{avg}.png')
                        combine = []
                        psnr_avg = []
                    temp_psnr_list = ','.join([str(round(x[batch_ind].item(), 2)) for x in pred_psnr])
                    temp_msssim_list = ','.join([str(round(x[batch_ind].item(), 4)) for x in pred_ssim])
                    pred_lpips = lpips_fn_single(img_out[batch_ind], img_gt[batch_ind])
                    temp_lpips = str(round(pred_lpips, 4))
                    with open('{}/psnr_ssim_lpips.txt'.format(args.outf), 'a') as f:
                        f.write(
                            str(img_idx.item()).zfill(3)
                            + ': psnr: '
                            + temp_psnr_list
                            + ', msssim: '
                            + temp_msssim_list
                            + ', lpips: '
                            + temp_lpips
                            + '\n'
                        )

            # print eval results and add to log txt
            if i == len(full_dataloader) - 1:
                avg_time = sum(time_list) / len(time_list)
                fps = args.batchSize / avg_time
                print_str = '[{}] Rank:{}, Eval at Step [{}/{}] , FPS {}, '.format(
                    datetime.now().strftime('%Y/%m/%d %H:%M:%S'),
                    local_rank,
                    i + 1,
                    len(full_dataloader),
                    round(fps, 1),
                )
                # metric_name = ('quant' if model_ind else 'pred') + '_seen_psnr'
                for v_name, v_list in zip(args.metric_names, metric_list):
                    # if metric_name in v_name:
                    cur_value = torch.stack(v_list, dim=-1).mean(-1) if len(v_list) else torch.zeros(1)
                    print_str += f'{v_name}: {RoundTensor(cur_value, 4)} | '
                if is_main_process(local_rank):
                    print(print_str, flush=True)
                    append_rank0_log(args, print_str)

        # embedding quantization
        if model_ind == 0:
            vid_embed = torch.cat(img_embed_list, 0)
            quant_embed, dequant_embed = quant_tensor(vid_embed, args.quant_embed_bit)
            dequant_vid_embed = dequant_embed.split(args.batchSize, dim=0)

        # Collect results from
        results_list = [torch.stack(v_list, dim=1).mean(1).cpu() if len(v_list) else torch.zeros(1) for v_list in metric_list]
        args.fps = fps
        h, w = img_data.shape[-2:]
        cur_model.train()
        if args.distributed and args.ngpus_per_node > 1:
            for cur_v in results_list:
                cur_v = all_reduce([cur_v.to(local_rank)])

        # Dump predictions and concat into videos
        # if dump_vis and args.dump_videos:
        #     gif_file = os.path.join(args.outf, 'gt_pred' + ('_quant.gif' if model_ind else '.gif'))

        #     with imageio.get_writer(gif_file, mode='I') as writer:
        #         for filename in sorted(os.listdir(visual_dir)):
        #             image = imageio.v2.imread(os.path.join(visual_dir, filename))
        #             writer.append_data(image)
        #     if not args.dump_images:
        #         shutil.rmtree(visual_dir)

        if dump_vis and args.dump_videos:
            gif_file = os.path.join(args.outf, 'gt_pred' + ('_quant.gif' if model_ind else '.gif'))
            frames = []
            for filename in sorted(os.listdir(visual_dir)):
                frames.append(imageio.imread(os.path.join(visual_dir, filename)))
            imageio.mimsave(gif_file, frames, 'GIF')
            if not args.dump_images:
                shutil.rmtree(visual_dir)

    # dump quantized checkpoint, and decoder
    if is_main_process(local_rank) and quant_ckt is not None:
        # huffman coding
        if huffman_coding:
            if 'HNeRV' in args.model:
                quant_v_list = quant_embed['quant'].flatten().tolist()
                tmin_scale_len = quant_embed['min'].nelement() + quant_embed['scale'].nelement()
            else:
                quant_v_list = []
                tmin_scale_len = 0
            for k, layer_wt in quant_ckt.items():
                quant_v_list.extend(layer_wt['quant'].flatten().tolist())
                tmin_scale_len += layer_wt['min'].nelement() + layer_wt['scale'].nelement()

            # get the element name and its frequency
            unique, counts = np.unique(quant_v_list, return_counts=True)
            num_freq = dict(zip(unique, counts))

            # generating HuffmanCoding table
            codec = HuffmanCodec.from_data(quant_v_list)
            sym_bit_dict = {}
            for k, v in codec.get_code_table().items():
                sym_bit_dict[k] = v[0]

            # total bits for quantized embed + model weights
            total_bits = 0
            for num, freq in num_freq.items():
                total_bits += freq * sym_bit_dict[num]
            args.bits_per_param = total_bits / len(quant_v_list)

            # including the overhead for min and scale storage,
            total_bits += tmin_scale_len * 16  # (16bits for float16)
            args.full_bits_per_param = total_bits / len(quant_v_list)

            # bits per pixel on original frames (full_data_length is patch count).
            print(
                f'final_size: {args.final_size}, full_data_length(patches): {args.full_data_length}, '
                f'num_frames: {args.num_frames}, patches_per_frame: {args.patches_per_frame}'
            )
            args.total_bpp = total_bits / args.total_pixels
            print_str = f'After quantization and encoding: \n bits per parameter: {round(args.full_bits_per_param, 2)}, bits per pixel: {round(args.total_bpp, 4)}'
            print(print_str, flush=True)
            append_rank0_log(args, print_str)

    return results_list, (h, w)


def quant_model(model, args):
    model_list = [deepcopy(model)]
    if args.quant_model_bit == -1:
        return model_list, None

    cur_model = deepcopy(model)
    quant_ckt, cur_ckt = [cur_model.state_dict() for _ in range(2)]
    encoder_k_list = []
    for k, v in cur_ckt.items():
        if 'encoder' in k:
            encoder_k_list.append(k)
        else:
            quant_v, new_v = quant_tensor(v, args.quant_model_bit)
            quant_ckt[k] = quant_v
            cur_ckt[k] = new_v
    for encoder_k in encoder_k_list:
        del quant_ckt[encoder_k]
    cur_model.load_state_dict(cur_ckt)
    model_list.append(cur_model)

    return model_list, quant_ckt


if __name__ == '__main__':
    main()
