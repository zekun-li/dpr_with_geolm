python generate_dense_embeddings.py \
	model_file="/home/zekun/mapqa/DPR/outputs/2024-08-10/16-38-39/weights/dpr_biencoder.23"  \
  	encoder.encoder_model_type=hf_geolm \
	encoder.pretrained_model_cfg=zekun-li/geolm-base-cased \
	ctx_src=mapqa_socal_entity_ctx \
	shard_id=0 num_shards=1 \
	out_file=/home/zekun/mapqa/DPR/downloads/data/mapqa_retriever_results/geolm-2024-06-15-08-33-35-socal.pkl






# CUDA_VISIBLE_DEVICES='0' \
# python train_dense_encoder.py \
# encoder.encoder_model_type=hf_geolm \
# encoder.pretrained_model_cfg=zekun-li/geolm-base-cased \
# train_datasets=[mapqa_train_geolm_entity_0810_ratio20] \
# dev_datasets=[mapqa_val_geolm_entity_0810] \
# train=biencoder_local \
# output_dir=weights \
# train.batch_size=16

# CUDA_VISIBLE_DEVICES='1' \
# python train_dense_encoder.py \
# encoder.encoder_model_type=hf_geolm \
# encoder.pretrained_model_cfg=zekun-li/geolm-base-cased \
# train_datasets=[mapqa_train_geolm_entity_0810_ratio40] \
# dev_datasets=[mapqa_val_geolm_entity_0810] \
# train=biencoder_local \
# output_dir=weights \
# train.batch_size=16

# CUDA_VISIBLE_DEVICES='2' \
# python train_dense_encoder.py \
# encoder.encoder_model_type=hf_geolm \
# encoder.pretrained_model_cfg=zekun-li/geolm-base-cased \
# train_datasets=[mapqa_train_geolm_entity_0810_ratio60] \
# dev_datasets=[mapqa_val_geolm_entity_0810] \
# train=biencoder_local \
# output_dir=weights \
# train.batch_size=16

# CUDA_VISIBLE_DEVICES='3' \
# python train_dense_encoder.py \
# encoder.encoder_model_type=hf_geolm \
# encoder.pretrained_model_cfg=zekun-li/geolm-base-cased \
# train_datasets=[mapqa_train_geolm_entity_0810_ratio80] \
# dev_datasets=[mapqa_val_geolm_entity_0810] \
# train=biencoder_local \
# output_dir=weights \
# train.batch_size=16




# python dense_retriever.py \
#   model_file="/home/zekun/mapqa/DPR/outputs/2024-06-15/01-05-11/weights/dpr_biencoder.19"  \
# 	qa_dataset=mapqa_0609_socal_0 \
#   encoder.encoder_model_type=hf_geolm \
# 	encoder.pretrained_model_cfg=zekun-li/geolm-base-cased \
# 	ctx_datatsets=[mapqa_socal_entity_ctx] \
# 	encoded_ctx_files=["/home/zekun/mapqa/DPR/downloads/data/mapqa_retriever_results/geolm-2024-06-15-01-05-11-socal.pkl_0"] \
# 	out_file="output.json"


# python dense_retriever.py \
#   model_file="/home/zekun/mapqa/DPR/outputs/2024-06-15/01-05-11/weights/dpr_biencoder.19"  \
# 	qa_dataset=mapqa_0609_socal_1 \
#   encoder.encoder_model_type=hf_geolm \
# 	encoder.pretrained_model_cfg=zekun-li/geolm-base-cased \
# 	ctx_datatsets=[mapqa_socal_entity_ctx] \
# 	encoded_ctx_files=["/home/zekun/mapqa/DPR/downloads/data/mapqa_retriever_results/geolm-2024-06-15-01-05-11-socal.pkl_0"] \
# 	out_file="output.json"

# python dense_retriever.py \
#   model_file="/home/zekun/mapqa/DPR/outputs/2024-06-15/01-05-11/weights/dpr_biencoder.19"  \
# 	qa_dataset=mapqa_0609_socal_2 \
#   encoder.encoder_model_type=hf_geolm \
# 	encoder.pretrained_model_cfg=zekun-li/geolm-base-cased \
# 	ctx_datatsets=[mapqa_socal_entity_ctx] \
# 	encoded_ctx_files=["/home/zekun/mapqa/DPR/downloads/data/mapqa_retriever_results/geolm-2024-06-15-01-05-11-socal.pkl_0"] \
# 	out_file="output.json"

# python dense_retriever.py \
#   model_file="/home/zekun/mapqa/DPR/outputs/2024-06-15/01-05-11/weights/dpr_biencoder.19"  \
# 	qa_dataset=mapqa_0609_socal_3 \
#   encoder.encoder_model_type=hf_geolm \
# 	encoder.pretrained_model_cfg=zekun-li/geolm-base-cased \
# 	ctx_datatsets=[mapqa_socal_entity_ctx] \
# 	encoded_ctx_files=["/home/zekun/mapqa/DPR/downloads/data/mapqa_retriever_results/geolm-2024-06-15-01-05-11-socal.pkl_0"] \
# 	out_file="output.json"

# python dense_retriever.py \
#   model_file="/home/zekun/mapqa/DPR/outputs/2024-06-15/01-05-11/weights/dpr_biencoder.19"  \
# 	qa_dataset=mapqa_0609_socal_4 \
#   encoder.encoder_model_type=hf_geolm \
# 	encoder.pretrained_model_cfg=zekun-li/geolm-base-cased \
# 	ctx_datatsets=[mapqa_socal_entity_ctx] \
# 	encoded_ctx_files=["/home/zekun/mapqa/DPR/downloads/data/mapqa_retriever_results/geolm-2024-06-15-01-05-11-socal.pkl_0"] \
# 	out_file="output.json"

# python dense_retriever.py \
#   model_file="/home/zekun/mapqa/DPR/outputs/2024-06-15/01-05-11/weights/dpr_biencoder.19"  \
# 	qa_dataset=mapqa_0609_socal_5 \
#   encoder.encoder_model_type=hf_geolm \
# 	encoder.pretrained_model_cfg=zekun-li/geolm-base-cased \
# 	ctx_datatsets=[mapqa_socal_entity_ctx] \
# 	encoded_ctx_files=["/home/zekun/mapqa/DPR/downloads/data/mapqa_retriever_results/geolm-2024-06-15-01-05-11-socal.pkl_0"] \
# 	out_file="output.json"

# python dense_retriever.py \
#   model_file="/home/zekun/mapqa/DPR/outputs/2024-06-15/01-05-11/weights/dpr_biencoder.19"  \
# 	qa_dataset=mapqa_0609_socal_6 \
#   encoder.encoder_model_type=hf_geolm \
# 	encoder.pretrained_model_cfg=zekun-li/geolm-base-cased \
# 	ctx_datatsets=[mapqa_socal_entity_ctx] \
# 	encoded_ctx_files=["/home/zekun/mapqa/DPR/downloads/data/mapqa_retriever_results/geolm-2024-06-15-01-05-11-socal.pkl_0"] \
# 	out_file="output.json"

