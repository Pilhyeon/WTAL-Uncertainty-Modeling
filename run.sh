model_path="./models/BMUE"
output_path="./outputs/BMUE"
log_path="./logs/BMUE"
seed=6

CUDA_VISIBLE_DEVICES=2 python -W ignore ./main.py --model_path ${model_path} --output_path ${output_path} --log_path ${log_path} --seed ${seed}