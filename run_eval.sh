model_path="./models/BMUE_eval"
output_path="./outputs/BMUE_eval"
log_path="./logs/BMUE_eval"
model_file='./BMUE_model_best.pkl'

CUDA_VISIBLE_DEVICES=0 python -W ignore ./main_eval.py --model_path ${model_path} --output_path ${output_path} --log_path ${log_path} --model_file ${model_file}
