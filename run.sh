model_path="./models/UM"
output_path="./outputs/UM"
log_path="./logs/UM"

CUDA_VISIBLE_DEVICES=0 python -W ignore ./main.py --model_path ${model_path} --output_path ${output_path} --log_path ${log_path}
