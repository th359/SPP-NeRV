data_path="./dataset/MCL-JCV/images"
out_path="MCL-JCV"

video_list=(01 02 03 04 05 06 07 08 09 10 11 12 13 14 15 16 17 18 19 20 21 22 23 24 25 26 27 28 29 30)
epoch="300"
model_size="1.5"

common_args=(
  --model SPP
  --sft_block res_sft
  --ch_t 32
  --optim_type Adan
  --conv_type convnext pshuffel_3x3
  --act sin
  --norm none
  --crop_list 640_1280
  --resize_list -1
  --loss spp
  --embed pe_1.25_80
  --enc_strds 5 4 4 2 2
  --enc_dim 64_16
  --dec_strds 5 4 2 2 2
  --ks 0_1_5
  --reduce 1.2
  --dec_blks 1 1 2 2 2
  --eval_freq 5000
  --lower_width 12
  -b 1
  --lr 0.003
  --split_num 2
  --dump_images
)

for video in "${video_list[@]}"; do
  run_name="MCL-JCV_${video}"
  python train_nerv_all.py \
    --outf "${out_path}/${video}" \
    --data_path "${data_path}/1080PAVCFQPvideoSRC${video}" \
    --vid "${run_name}" \
    -e "${epoch}" \
    --modelsize "${model_size}" \
    "${common_args[@]}"
done
