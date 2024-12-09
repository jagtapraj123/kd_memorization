#!/bin/bash

git clone https://github.com/NVlabs/stylegan2-ada-pytorch.git

cp generate_updated.py stylegan2-ada-pytorch/

cd stylegan2-ada-pytorch


python generate_updated.py --outdir=../../gen_data --trunc=1 --seeds=1-102400 \
    --network=https://nvlabs-fi-cdn.nvidia.com/stylegan2-ada-pytorch/pretrained/ffhq.pkl
