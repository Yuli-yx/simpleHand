train:
	CUDA_VISIBLE_DEVICES=1,2 torchrun --master_port=29501 --nproc_per_node 2 train.py 

eval:
	CUDA_VISIBLE_DEVICES=2 python3 infer_to_json_wo_vertice.py epoch_195
