import argparse
import torch
import torch.nn as nn
import torch.nn.functional as F
import math
import torch.distributions as dist
from einops import rearrange
from model_blocks import *
import time
from lib.quant_ops import CustomConv2d, CustomLinear

class SPP(nn.Module):
    def __init__(self, args):
        super().__init__()
        # Encoder LAYERS
        self.embed = args.embed
        ks_enc, ks_dec1, ks_dec2 = [int(x) for x in args.ks.split('_')]
        enc_blks = args.enc_blks        
        
        enc_dim1, enc_dim2 = [int(x) for x in args.enc_dim.split('_')]
        c_in_list, c_out_list = [enc_dim1] * len(args.enc_strds), [enc_dim1] * len(args.enc_strds)
        c_out_list[-1] = enc_dim2
        self.encoder = ConvNeXt(stage_blocks=enc_blks, strds=args.enc_strds, dims=c_out_list, drop_path_rate=0)

        # Decoder LAYERS
        # first part: position embedding for time index
        self.pe_embed_t = PositionEncoding(args.embed, args.lfreq) #PositionEncoding(lbase=args.lbase, levels=args.levels, lfreq=args.lfreq)
        mlp_dim_list = [int(self.pe_embed_t.embed_length)] + [int(args.ch_t*2)] + [args.ch_t]
        self.stem_t = NeRV_MLP(dim_list=mlp_dim_list, bias=True, act=args.act, omega=1, args=args)
        self.stem_part = NeRV_MLP(dim_list=mlp_dim_list, bias=True, act=args.act, omega=1, args=args)

        # second part: reconstruction module
        decoder_layers = []        
        ngf = args.fc_dim
        decoder_layer1 = NeRVBlock(dec_block=False, conv_type='conv', ngf=enc_dim2, new_ngf=ngf, ks=0, strd=1, 
                bias=True, norm=args.norm, act=args.act, sft_ngf=args.ch_t, args=args) 
        decoder_layers.append(decoder_layer1)

        for i, strd in enumerate(args.dec_strds):                  
            reduction = sqrt(strd) if args.reduce ==-1 else args.reduce
            new_ngf = int(max(round(ngf / reduction), args.lower_width))
            for j in range(args.dec_blks[i]):
                cur_blk = NeRVBlock(dec_block=True, conv_type=args.conv_type[1], ngf=ngf, new_ngf=new_ngf, 
                    ks=min(ks_dec1+2*i, ks_dec2), strd=1 if j else strd, bias=True, norm=args.norm, act=args.act, sft_ngf=args.ch_t, args=args)
                decoder_layers.append(cur_blk)
                ngf = new_ngf

        self.decoder = nn.ModuleList(decoder_layers)
        self.head_layer = CustomConv2d(ngf, 3, 3, 1, 1, args=args) 
        self.out_bias = args.out_bias
        if args.quant:
            self.embed_quantizer = quant_map[args.quantizer_e](args.quant_embed_bit, signed=False, per_channel=args.per_channel_e)
            self.bitrate_e_dict = {}
        else:
            self.embed_quantizer = None

        self.outf = args.outf

    def forward(self, input, input_embed=None, entropy_model=None, pre_img=None, post_img=None, norm_idx=None, part_img=None, part_idx=None):
        # bunny: input: torch.Size([1, 3, 720, 1280]), norm_idx: torch.Size([1, 1]), part_img: torch.Size([1, 3, 360, 640]), part_idx: torch.Size([1, 1])
        if input_embed != None:
            img_embed = input_embed
        else:
            img_embed = self.encoder(input) # bunny: [1, 16, 9, 16] 

        if self.embed_quantizer is not None:
            self.embed_quantizer.init_data(img_embed)
            code_e, quant_e, img_embed = self.embed_quantizer(img_embed)
            if entropy_model is not None:
                self.bitrate_e_dict.update(entropy_model.cal_bitrate(code_e, quant_e, self.training))
        
        embed_list = [img_embed]
        dec_start = time.time()
        t_embed = self.stem_t(self.pe_embed_t(norm_idx[:, None]).float()) # bunny: [1, 32, 1, 1]
        part_embed = self.stem_part(self.pe_embed_t(part_idx[:, None]).float())

        # output = self.decoder[0]((img_embed, t_embed))
        output = self.decoder[0]((img_embed, t_embed, t_embed))
        embed_list.append(output)
        for i, layer in enumerate(self.decoder[1:]):
            if i==5:
                output = layer((output, part_embed, part_embed))
            else:
                output = layer((output, t_embed, t_embed))
            embed_list.append(output)
        # for layer in self.decoder[1:]:
        #     output = layer((output, t_embed)) 
        #     embed_list.append(output)
        img_out = OutImg(self.head_layer(output), self.out_bias)
        if torch.cuda.is_available():
            torch.cuda.synchronize()
        dec_time = time.time() - dec_start
        return img_out, embed_list, dec_time


    def forward_encoder(self, input):
        img_embed = self.encoder(input)
        return img_embed

    def forward_embed_quant(self, img_embed, entropy_model=None):
        code, quant, img_embed = self.embed_quantizer(img_embed)
        if entropy_model is not None:
            self.bitrate_e_dict.update(entropy_model.cal_bitrate(code, quant, self.training))
        return code, quant, img_embed

    def forward_decoder(self, img_embed, norm_idx):
        embed_list = [img_embed]
        dec_start = time.time()
        t_embed = self.stem_t(self.pe_embed_t(norm_idx[:, None]).float())
        output = self.decoder[0]((img_embed, t_embed))
        embed_list.append(output)
        for layer in self.decoder[1:]:
            output = layer((output, t_embed)) 
            embed_list.append(output)
        img_out = OutImg(self.head_layer(output), self.out_bias)
        if torch.cuda.is_available():
            torch.cuda.synchronize()
        dec_time = time.time() - dec_start
        return img_out, embed_list, dec_time

    def decoder_params(self):
        decoder_param = (sum([p.data.nelement() for p in self.parameters()]) - sum([p.data.nelement() for p in self.encoder.parameters()])) /1e6
        return decoder_param

    def stage_params(self):
        model_params = self.decoder_params()
        stage0 = sum([p.data.nelement() for p in self.decoder[0].parameters()])/1e6/model_params
        ratio_list = [stage0, 0, 0, 0, 0, 0]
        index = 1
        for i, strd in enumerate(self.dec_strds): 
            for j in range(self.dec_blks[i]):
                ratio_list[i+1] += sum([p.data.nelement() for p in self.decoder[index].parameters()])/1e6/model_params
                index += 1
        return ratio_list
        #print("params distribution:", ratio_list)

    def cal_params(self, entropy_model=None):
        for m in self.modules():
            if type(m) in [CustomConv2d, CustomLinear]:
                code_w, quant_w, dequant_w = m.weight_quantizer(m.weight)
                m.dequant_w = dequant_w
                if m.bias is not None:
                    code_b, quant_b, dequant_b = m.bias_quantizer(m.bias)
                    m.dequant_b = dequant_b
                if entropy_model is not None:
                    m.bitrate_w_dict.update(entropy_model.cal_bitrate(code_w, quant_w, self.training))
                    if m.bias is not None:
                        m.bitrate_b_dict.update(entropy_model.cal_bitrate(code_b, quant_b, self.training))

    def get_bitrate_sum(self, name="bitrate"):
        sum = 0
        for m in self.modules():
            if type(m) in [CustomConv2d, CustomLinear]:
                sum += m.bitrate_w_dict[name]
                if name in m.bitrate_b_dict.keys():
                    sum += m.bitrate_b_dict[name]
        return sum

    def init_data(self):
        for m in self.modules():
            if type(m) in [CustomConv2d, CustomLinear]:
                m.weight_quantizer.init_data(m.weight)
                if m.bias is not None:
                    m.bias_quantizer.init_data(m.bias)

#######################################################################
from hnerv_utils import *
def debug(args):
    model = SPP(args)
    dataloader = VideoDataSet(args)
    sample = dataloader[0]
    img_data = sample['img'].unsqueeze(0)
    img_idx = sample['img_idx']
    norm_img_idx = sample['norm_img_idx']
    part_img = sample['part'].unsqueeze(0)
    part_idx = sample['part_idx']
    norm_part_idx = sample['norm_part_idx']

    norm_img_idx = torch.tensor(norm_img_idx).unsqueeze(0).unsqueeze(0)
    norm_part_idx = torch.tensor(norm_part_idx).unsqueeze(0).unsqueeze(0)
    img_out, _, _ = model(img_data, norm_idx=norm_img_idx, part_img=part_img, part_idx=norm_part_idx)

if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument('--split_num', type=str, default=2)
    parser.add_argument('--data_path', type=str, default='dataset/bunny')
    parser.add_argument('--vid', type=str, default='k400_train0')
    parser.add_argument('--shuffle_data', action='store_true')
    parser.add_argument('--data_split', type=str, default='1_1_1')
    parser.add_argument('--crop_list', type=str, default='640_1280')
    parser.add_argument('--resize_list', type=str, default='-1')
    parser.add_argument('--model', type=str, default='')
    parser.add_argument('--embed', type=str, default='')
    parser.add_argument('--ks', type=str, default='0_3_3')
    parser.add_argument('--enc_blks', type=int, default=1)
    parser.add_argument('--enc_strds', type=int, nargs='+', default=[])
    parser.add_argument('--enc_dim', type=str, default='64_16')
    parser.add_argument('--modelsize', type=float,  default=1.5)
    parser.add_argument('--saturate_stages', type=int, default=-1)
    parser.add_argument('--lfreq', type=str, default="pi")
    parser.add_argument('--fc_dim', type=int, default=None)
    parser.add_argument('--fc_hw', type=str, default='9_16')
    parser.add_argument('--reduce', type=float, default=1.2)
    parser.add_argument('--lower_width', type=int, default=32)
    parser.add_argument('--dec_strds', type=int, nargs='+', default=[5, 3, 2, 2, 2])
    parser.add_argument('--dec_blks', type=int, nargs='+',  default=[1, 1, 1, 1, 1])
    parser.add_argument("--conv_type", default=['convnext', 'pshuffel',], type=str, nargs="+")
    parser.add_argument('--norm', default='none', type=str)
    parser.add_argument('--act', type=str, default='gelu')
    parser.add_argument('--sft_block', type=str, default='none')
    parser.add_argument('--ch_t', type=int, default=32)
    parser.add_argument('--block_dim', type=int, default=128)
    parser.add_argument('-j', '--workers', type=int, default=4)
    parser.add_argument('-b', '--batchSize', type=int, default=1)
    parser.add_argument('--start_epoch', type=int, default=-1)
    parser.add_argument('--not_resume', action='store_true')
    parser.add_argument('-e', '--epochs', type=int, default=5)
    parser.add_argument('--block_params', type=str, default='1_1')
    parser.add_argument('--lr', type=float, default=0.001)
    parser.add_argument('--lr_type', type=str, default='cosine_0.1_1_0.1')
    parser.add_argument('--loss', type=str, default='Fusion6')
    parser.add_argument('--out_bias', default='tanh', type=str)
    parser.add_argument('--optim_type', default='adan', type=str)
    parser.add_argument('--clip_max_norm', default=0., type=float)
    parser.add_argument('--inpanting', default='none', type=str)
    parser.add_argument('--interpolation', action='store_true', default=False)
    parser.add_argument('--embed_inter', action='store_true', default=False)
    parser.add_argument('--cabac', action='store_true', default=False)
    parser.add_argument('--quant', action='store_true', default=False)
    parser.add_argument('--eval_only', action='store_true', default=False)
    parser.add_argument('--eval_freq', type=int, default=10)
    parser.add_argument('--quant_model_bit', type=int, default=8)
    parser.add_argument('--quant_embed_bit', type=int, default=6)
    parser.add_argument('--quant_axis', type=int, default=0)
    parser.add_argument('--dump_images', action='store_true', default=False)
    parser.add_argument('--dump_videos', action='store_true', default=False)
    parser.add_argument('--eval_fps', action='store_true', default=False)
    parser.add_argument('--encoder_file',  default='', type=str)
    parser.add_argument('--dump_values', action='store_true', default=False)
    parser.add_argument('--dump_features', action='store_true', default=False)
    parser.add_argument('--manualSeed', type=int, default=1)
    parser.add_argument('-d', '--distributed', action='store_true', default=False)
    parser.add_argument('--debug', action='store_true')  
    parser.add_argument('-p', '--print-freq', default=50, type=int)
    parser.add_argument('--weight', default='None', type=str)
    parser.add_argument('--overwrite', action='store_true')
    parser.add_argument('--outf', default='unify')
    parser.add_argument('--suffix', default='')
    args = parser.parse_args()
    args.model = 'SPP'
    args.sft_block = 'res_sft'
    args.ch_t = 32
    args.optim_type = 'Adan'
    args.conv_type = ['convnext', 'pshuffel_3x3']
    args.act = 'sin'
    args.norm = 'none'
    args.crop_list = '720_1280'
    args.resize_lsit = -1
    args.loss = 'Fusion10_freq'
    args.embed = 'pe_1.25_80'
    args.enc_strds = [5, 2, 2, 2, 2]
    args.enc_dim = '64_16'
    args.dec_strds = [5, 2, 2, 2, 2]
    args.ks = '0_1_5'
    args.reduce = 1.2
    args.dec_blks = [1, 1, 2, 2, 2]
    args.model_size = 1.275
    args.epochs = 300
    args.eval_freq = 30
    args.lower_width = 12
    args.batchSize = 1
    args.lr = 0.003
    args.enc_strd_str, args.dec_strd_str = ','.join([str(x) for x in args.enc_strds]), ','.join([str(x) for x in args.dec_strds])
    args.quant_str = f'quant_M{args.quant_model_bit}_E{args.quant_embed_bit}'
    args.exp_id = exp_id = f'{args.vid}/Size{args.modelsize}'
    args.ngpus_per_node = 0
    args.fc_dim = 60

    debug(args)