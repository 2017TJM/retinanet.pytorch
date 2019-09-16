train:
	CUDA_VISIBLE_DEVICES='0,1,2,3' \
	python train.py \
	--dataset coco \
	--coco_path /home/bumsoo/Data/coco \
	--optimizer Adam \
	--depth 50
