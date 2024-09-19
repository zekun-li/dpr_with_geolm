


# CUDA_VISIBLE_DEVICES='0' python3 pipeline.py --task='train' --model_type='hf_bert'  --question_type='entity'

# CUDA_VISIBLE_DEVICES='1' python3 pipeline.py --task='train' --model_type='hf_geolm'  --question_type='entity'

# CUDA_VISIBLE_DEVICES='2,3' python3 pipeline.py --task='ratio' --model_type='hf_geolm'  --question_type='entity'


CUDA_VISIBLE_DEVICES='1,2' python3 pipeline.py --task='generate_embedding' --model_type='hf_geolm'  --question_type='entity' --model_file="2024-09-19/05-53-38/weights/dpr_biencoder.14" --region='socal'
CUDA_VISIBLE_DEVICES='1,2' python3 pipeline.py --task='generate_embedding' --model_type='hf_geolm'  --question_type='entity' --model_file="2024-09-19/05-53-38/weights/dpr_biencoder.14" --region='il'


CUDA_VISIBLE_DEVICES='1,2' python3 pipeline.py --task='generate_embedding' --model_type='hf_bert'  --question_type='entity' --model_file="2024-09-19/05-53-06/weights/dpr_biencoder.19 " --region='socal'
CUDA_VISIBLE_DEVICES='1,2' python3 pipeline.py --task='generate_embedding' --model_type='hf_bert'  --question_type='entity' --model_file="2024-09-19/05-53-06/weights/dpr_biencoder.19 " --region='il'


CUDA_VISIBLE_DEVICES='1,2' python3 pipeline.py --task='retrieval' --model_type='hf_geolm'  --question_type='entity' --model_file="2024-09-19/05-53-38/weights/dpr_biencoder.14" --region='socal'
CUDA_VISIBLE_DEVICES='1,2' python3 pipeline.py --task='retrieval' --model_type='hf_geolm'  --question_type='entity' --model_file="2024-09-19/05-53-38/weights/dpr_biencoder.14" --region='il'


CUDA_VISIBLE_DEVICES='1,2' python3 pipeline.py --task='retrieval' --model_type='hf_bert'  --question_type='entity' --model_file="2024-09-19/05-53-06/weights/dpr_biencoder.19 " --region='socal'
CUDA_VISIBLE_DEVICES='1,2' python3 pipeline.py --task='retrieval' --model_type='hf_bert'  --question_type='entity' --model_file="2024-09-19/05-53-06/weights/dpr_biencoder.19 " --region='il'





# # CUDA_VISIBLE_DEVICES='1,2,3' python3 pipeline.py --task='generate_embedding' --model_type='hf_geolm'  --question_type='entity' --model_file="2024-08-11/05-50-33/weights/dpr_biencoder.29" --region='socal'
# # CUDA_VISIBLE_DEVICES='1,2,3' python3 pipeline.py --task='generate_embedding' --model_type='hf_geolm'  --question_type='entity' --model_file="2024-08-11/06-06-12/weights/dpr_biencoder.22" --region='socal'
# # CUDA_VISIBLE_DEVICES='1,2,3' python3 pipeline.py --task='generate_embedding' --model_type='hf_geolm'  --question_type='entity' --model_file="2024-08-11/06-24-09/weights/dpr_biencoder.26" --region='socal'
# # CUDA_VISIBLE_DEVICES='1,2,3' python3 pipeline.py --task='generate_embedding' --model_type='hf_geolm'  --question_type='entity' --model_file="2024-08-11/06-44-31/weights/dpr_biencoder.16" --region='socal'
# # CUDA_VISIBLE_DEVICES='1,2,3' python3 pipeline.py --task='generate_embedding' --model_type='hf_geolm'  --question_type='entity' --model_file="2024-08-10/16-38-39/weights/dpr_biencoder.23" --region='socal'

# # CUDA_VISIBLE_DEVICES='1,2,3' python3 pipeline.py --task='retrieval' --model_type='hf_geolm'  --question_type='entity' --model_file="2024-08-11/05-50-33/weights/dpr_biencoder.29" --region='socal'
# CUDA_VISIBLE_DEVICES='1,2,3' python3 pipeline.py --task='retrieval' --model_type='hf_geolm'  --question_type='entity' --model_file="2024-08-11/06-06-12/weights/dpr_biencoder.22" --region='socal'
# CUDA_VISIBLE_DEVICES='1,2,3' python3 pipeline.py --task='retrieval' --model_type='hf_geolm'  --question_type='entity' --model_file="2024-08-11/06-24-09/weights/dpr_biencoder.26" --region='socal'
# CUDA_VISIBLE_DEVICES='1,2,3' python3 pipeline.py --task='retrieval' --model_type='hf_geolm'  --question_type='entity' --model_file="2024-08-11/06-44-31/weights/dpr_biencoder.16" --region='socal'
# CUDA_VISIBLE_DEVICES='1,2,3' python3 pipeline.py --task='retrieval' --model_type='hf_geolm'  --question_type='entity' --model_file="2024-08-10/16-38-39/weights/dpr_biencoder.23" --region='socal'


# # illinois geolm
# # CUDA_VISIBLE_DEVICES='0' python3 pipeline.py --task='generate_embedding' --model_type='hf_geolm'  --question_type='entity' --model_file="2024-08-10/16-38-39/weights/dpr_biencoder.23" --region='il'
# CUDA_VISIBLE_DEVICES='0' python3 pipeline.py --task='retrieval' --model_type='hf_geolm'  --question_type='entity' --model_file="2024-08-10/16-38-39/weights/dpr_biencoder.23" --region='il'


# # # socal bert
# # CUDA_VISIBLE_DEVICES='0' python3 pipeline.py --task='generate_embedding' --model_type='hf_bert'  --question_type='entity' --model_file="2024-08-10/16-53-48/weights/dpr_biencoder.18" --region='socal'
# CUDA_VISIBLE_DEVICES='0' python3 pipeline.py --task='retrieval' --model_type='hf_bert'  --question_type='entity' --model_file="2024-08-10/16-53-48/weights/dpr_biencoder.18"  --region='socal'


# # # illinois bert
# # CUDA_VISIBLE_DEVICES='0' python3 pipeline.py --task='generate_embedding' --model_type='hf_bert'  --question_type='entity' --model_file="2024-08-10/16-53-48/weights/dpr_biencoder.18" --region='il'
# CUDA_VISIBLE_DEVICES='0' python3 pipeline.py --task='retrieval' --model_type='hf_bert'  --question_type='entity' --model_file="2024-08-10/16-53-48/weights/dpr_biencoder.18"  --region='il'


