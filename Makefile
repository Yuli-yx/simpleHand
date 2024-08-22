train:
	CUDA_VISIBLE_DEVICES=0,1,2 torchrun --nproc_per_node 3 train.py --resume

eval:
	python3 infer_to_json.py epoch_200
