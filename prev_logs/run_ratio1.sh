python dense_retriever.py \
  model_file="/home/zekun/mapqa/DPR/outputs/2024-06-15/08-33-35/weights/dpr_biencoder.8"  \
	qa_dataset=mapqa_0609_socal_0 \
  encoder.encoder_model_type=hf_geolm \
	encoder.pretrained_model_cfg=zekun-li/geolm-base-cased \
	ctx_datatsets=[mapqa_socal_entity_ctx] \
	encoded_ctx_files=["/home/zekun/mapqa/DPR/downloads/data/mapqa_retriever_results/geolm-2024-06-15-08-33-35-socal.pkl_0"] \
	out_file="output.json"

python dense_retriever.py \
  model_file="/home/zekun/mapqa/DPR/outputs/2024-06-15/08-33-35/weights/dpr_biencoder.8"  \
	qa_dataset=mapqa_0609_socal_1 \
  encoder.encoder_model_type=hf_geolm \
	encoder.pretrained_model_cfg=zekun-li/geolm-base-cased \
	ctx_datatsets=[mapqa_socal_entity_ctx] \
	encoded_ctx_files=["/home/zekun/mapqa/DPR/downloads/data/mapqa_retriever_results/geolm-2024-06-15-08-33-35-socal.pkl_0"] \
	out_file="output.json"

python dense_retriever.py \
  model_file="/home/zekun/mapqa/DPR/outputs/2024-06-15/08-33-35/weights/dpr_biencoder.8"  \
	qa_dataset=mapqa_0609_socal_2 \
  encoder.encoder_model_type=hf_geolm \
	encoder.pretrained_model_cfg=zekun-li/geolm-base-cased \
	ctx_datatsets=[mapqa_socal_entity_ctx] \
	encoded_ctx_files=["/home/zekun/mapqa/DPR/downloads/data/mapqa_retriever_results/geolm-2024-06-15-08-33-35-socal.pkl_0"] \
	out_file="output.json"

python dense_retriever.py \
  model_file="/home/zekun/mapqa/DPR/outputs/2024-06-15/08-33-35/weights/dpr_biencoder.8"  \
	qa_dataset=mapqa_0609_socal_3 \
  encoder.encoder_model_type=hf_geolm \
	encoder.pretrained_model_cfg=zekun-li/geolm-base-cased \
	ctx_datatsets=[mapqa_socal_entity_ctx] \
	encoded_ctx_files=["/home/zekun/mapqa/DPR/downloads/data/mapqa_retriever_results/geolm-2024-06-15-08-33-35-socal.pkl_0"] \
	out_file="output.json"

python dense_retriever.py \
  model_file="/home/zekun/mapqa/DPR/outputs/2024-06-15/08-33-35/weights/dpr_biencoder.8"  \
	qa_dataset=mapqa_0609_socal_4 \
  encoder.encoder_model_type=hf_geolm \
	encoder.pretrained_model_cfg=zekun-li/geolm-base-cased \
	ctx_datatsets=[mapqa_socal_entity_ctx] \
	encoded_ctx_files=["/home/zekun/mapqa/DPR/downloads/data/mapqa_retriever_results/geolm-2024-06-15-08-33-35-socal.pkl_0"] \
	out_file="output.json"

python dense_retriever.py \
  model_file="/home/zekun/mapqa/DPR/outputs/2024-06-15/08-33-35/weights/dpr_biencoder.8"  \
	qa_dataset=mapqa_0609_socal_5 \
  encoder.encoder_model_type=hf_geolm \
	encoder.pretrained_model_cfg=zekun-li/geolm-base-cased \
	ctx_datatsets=[mapqa_socal_entity_ctx] \
	encoded_ctx_files=["/home/zekun/mapqa/DPR/downloads/data/mapqa_retriever_results/geolm-2024-06-15-08-33-35-socal.pkl_0"] \
	out_file="output.json"

python dense_retriever.py \
  model_file="/home/zekun/mapqa/DPR/outputs/2024-06-15/08-33-35/weights/dpr_biencoder.8"  \
	qa_dataset=mapqa_0609_socal_6 \
  encoder.encoder_model_type=hf_geolm \
	encoder.pretrained_model_cfg=zekun-li/geolm-base-cased \
	ctx_datatsets=[mapqa_socal_entity_ctx] \
	encoded_ctx_files=["/home/zekun/mapqa/DPR/downloads/data/mapqa_retriever_results/geolm-2024-06-15-08-33-35-socal.pkl_0"] \
	out_file="output.json"
