train:
	CUDA_VISIBLE_DEVICES=2,3 torchrun --nproc_per_node 2 train.py 

eval:
	python3 infer_to_json.py epoch_200
