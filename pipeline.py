import os 
import glob 
import subprocess
import time
import argparse


def execute_command(command, if_print_command):
    t1 = time.time()

    if if_print_command:
        print(command)

    try:
        proc = subprocess.run(command, shell=True,check=True,  stdout = subprocess.PIPE) #stderr=subprocess.STDOUT)
        t2 = time.time()
        time_usage = t2 - t1 
        # for line in proc.stdout:
        #     print(line)
        return {'time_usage':time_usage}
    except subprocess.CalledProcessError as err:
        error = err.stderr.decode('utf8')
        # format error message to one line
        error  = error.replace('\n','\t')
        error = error.replace(',',';')
        return {'error': error}
    

def train_encoder(model_type, question_type, if_ratio=False):

    batch_size = 16
    model_cfg = "zekun-li/geolm-base-cased" if model_type == 'hf_geolm' else "bert-base-uncased"

    train_datasets = f"mapqa_train_geolm_{question_type}_0915" if model_type == 'hf_geolm' else f"mapqa_train_bert_{question_type}_0915" 

    
    dev_datasets = f"mapqa_val_geolm_{question_type}_0915" if model_type == 'hf_geolm' else f"mapqa_val_bert_{question_type}_0915" 
    
    if if_ratio:
        # for ratio in ['_ratio20', '_ratio40', '_ratio60', '_ratio80']:
        for ratio in ['_ratio40', '_ratio60', '_ratio80']:
            train_datasets += ratio 
            command = f"python train_dense_encoder.py  encoder.encoder_model_type={model_type} encoder.pretrained_model_cfg={model_cfg} train_datasets=[{train_datasets}] dev_datasets=[{dev_datasets}] train=biencoder_local  output_dir=weights train.batch_size={batch_size}" 
            try:
                execute_command(command, True)
            except Exception as e:
                print(e)
    else:
        command = f"python train_dense_encoder.py  encoder.encoder_model_type={model_type} encoder.pretrained_model_cfg={model_cfg} train_datasets=[{train_datasets}] dev_datasets=[{dev_datasets}] train=biencoder_local  output_dir=weights train.batch_size={batch_size}" 

        execute_command(command, True)


def generate_embedding(model_file, model_type, region, question_type):
    # model_file eg: 2024-06-15/08-36-08/weights/dpr_biencoder.26
    # model_type = hf_geolm or hf_bert 
    # region: socal or il 
    # question_type: amenity or entity

    assert model_file is not None 

    temp = model_file.split('/')[0] + '-' +  model_file.split('/')[1]
    model_file_dir = '/home/zekun/mapqa/DPR/outputs/'
    out_dir = '/home/zekun/mapqa/DPR/downloads/data/mapqa_retriever_results/'
    model_file = os.path.join(model_file_dir, model_file) 
    model_cfg = "zekun-li/geolm-base-cased" if model_type == 'hf_geolm' else "bert-base-uncased"
    ctx_src = f"mapqa_{region}_{question_type}_ctx" 
    out_file =os.path.join(out_dir, f"{model_type}-{temp}-{region}.pkl" )

    command = f"python generate_dense_embeddings.py model_file={model_file} encoder.encoder_model_type={model_type} encoder.pretrained_model_cfg={model_cfg} ctx_src={ctx_src} shard_id=0 num_shards=1 out_file={out_file}" 

    execute_command(command, True)

def retrieval(model_file, model_type, region, question_type ):
    temp = model_file.split('/')[0] + '-' +  model_file.split('/')[1]
    model_file_dir = '/home/zekun/mapqa/DPR/outputs/'

    model_file = os.path.join(model_file_dir, model_file) 
    model_cfg = "zekun-li/geolm-base-cased" if model_type == 'hf_geolm' else "bert-base-uncased"
    ctx_src = f"mapqa_{region}_{question_type}_ctx" 
    ctx_encoded =os.path.join('/home/zekun/mapqa/DPR/downloads/data/mapqa_retriever_results/', f"{model_type}-{temp}-{region}.pkl_0" )

    split_list = [0,1,4,5,6] if question_type == 'entity' else [2,3] ##########HERE
    for split in split_list:
        qa_dataset = f"mapqa_0915_{region}_{split}"

        command = f"python dense_retriever.py model_file={model_file} qa_dataset={qa_dataset} encoder.encoder_model_type={model_type} encoder.pretrained_model_cfg={model_cfg} ctx_datatsets=[{ctx_src}] encoded_ctx_files=[{ctx_encoded}] out_file='output.json'"

        execute_command(command, True)
    


if __name__ == "__main__":
    parser = argparse.ArgumentParser()

    parser.add_argument('--model_type', type=str, default='hf_geolm', choices = ['hf_geolm', 'hf_bert']) 
    parser.add_argument('--question_type', type=str, default='entity', choices=['entity','amenity'])

    parser.add_argument('--model_file', type=str, default=None)
    parser.add_argument('--region', type=str, default='socal', choices =['socal','il']) 
    


    parser.add_argument('--task', type=str, default=None, choices=['train','generate_embedding','retrieval','ratio'])

    args = parser.parse_args()
    print('\n')
    print(args)
    print('\n')

    if args.task == 'train':
        train_encoder(args.model_type, args.question_type)
    elif args.task == 'generate_embedding':
        generate_embedding(args.model_file, args.model_type, args.region, args.question_type)
    elif args.task == 'retrieval':
        retrieval(args.model_file, args.model_type, args.region, args.question_type)
    elif args.task == 'ratio':
        train_encoder(args.model_type, args.question_type, if_ratio=True)



