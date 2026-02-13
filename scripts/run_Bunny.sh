# data_path="./dataset/bunny"
data_path="/home/watanabelab/Documents/dataset/bunny"
out_path="bunny"

epoch_list=(50 100 150 200 250 300)
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

for epoch in "${epoch_list[@]}"; do
    run_name="Bunny_epoch${epoch}"
    python train_nerv_all.py \
        --outf "${out_path}/${run_name}" \
        --data_path "${data_path}" \
        --vid "${run_name}" \
        -e "${epoch}" \
        --modelsize "${model_size}" \
        "${common_args[@]}"
done
