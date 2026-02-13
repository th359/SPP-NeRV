data_path="./dataset/DAVIS-data/DAVIS/JPEGImages/1080p"
out_path="DAVIS"

video_list=(
  bear blackswan bmx-bumps bmx-trees boat breakdance breakdance-flare bus camel
  car-roundabout car-shadow car-turn cows dance-jump dance-twirl dog dog-agility
  drift-chicane drift-straight drift-turn elephant flamingo goat hike hockey
  horsejump-high horsejump-low kite-surf kite-walk libby lucia mallard-fly
  mallard-water motocross-bumps motocross-jump motorbike paragliding
  paragliding-launch parkour rhino rollerblade scooter-black scooter-gray soapbox
  soccerball stroller surf swing tennis train
)
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
  run_name="DAVIS_${video}"
  python train_nerv_all.py \
    --outf "${out_path}/${video}" \
    --data_path "${data_path}/${video}" \
    --vid "${run_name}" \
    -e "${epoch}" \
    --modelsize "${model_size}" \
    "${common_args[@]}"
done
