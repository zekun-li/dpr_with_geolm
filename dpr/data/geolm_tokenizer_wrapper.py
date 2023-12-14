import numpy as np
import torch
from torch.utils.data import Dataset
from pyproj import Transformer as projTransformer
from transformers import PreTrainedTokenizer
import pdb


class GeoLMTokenizer(PreTrainedTokenizer):
    def __init__(self, tokenizer , max_token_len ,  distance_norm_factor, sep_between_neighbors = True ):
        self.tokenizer = tokenizer
        self.max_token_len = max_token_len 
        self.distance_norm_factor = distance_norm_factor
        self.sep_between_neighbors = sep_between_neighbors
        self.ptransformer = projTransformer.from_crs("EPSG:4326", "EPSG:4087", always_xy=True) # https://epsg.io/4087, equidistant cylindrical projection
        

    def get_offset_mapping(self, nl_tokens):
        offset_mapping = nl_tokens['offset_mapping'][1:-1]
        flat_offset_mapping = np.array(offset_mapping).flatten()
        offset_mapping_dict_start = {}
        offset_mapping_dict_end = {}
        for idx in range(0,len(flat_offset_mapping),2):
            char_pos = flat_offset_mapping[idx]
            if char_pos == 0 and idx != 0:
                break
            token_pos = idx//2 + 1 
            offset_mapping_dict_start[char_pos] = token_pos 
        for idx in range(1,len(flat_offset_mapping),2):
            char_pos = flat_offset_mapping[idx]
            if char_pos == 0 and idx != 0:
                break
            token_pos = (idx-1)//2 + 1 +1
            offset_mapping_dict_end[char_pos] = token_pos

        return offset_mapping_dict_start, offset_mapping_dict_end

    def parse_linguistic_context(self, pivot_name, text, spatial_dist_fill=90000):
        nl_tokens = self.tokenizer(text,  padding="max_length", max_length=self.max_token_len, truncation = True, return_offsets_mapping = True)
        offset_mapping_dict_start, offset_mapping_dict_end = self.get_offset_mapping(nl_tokens)

        start_span = text.find(pivot_name)
        end_span = start_span + len(pivot_name) 

        if start_span not in offset_mapping_dict_start or end_span not in offset_mapping_dict_end:
            # pdb.set_trace()
            return None  # TODO: exceeds length. fix later

        token_start_idx = offset_mapping_dict_start[start_span]
        token_end_idx = offset_mapping_dict_end[end_span]


        input_ids = nl_tokens['input_ids']


        input_data = {}
        input_data['pivot_name'] = pivot_name
        input_data['pivot_token_idx'] = torch.tensor([[token_start_idx, token_end_idx]])
        input_data['sent_position_ids'] = torch.tensor(np.arange(0, len(input_ids)))
        input_data['attention_mask'] = torch.tensor(nl_tokens['attention_mask'])
        input_data['token_type_ids'] = torch.zeros(1,len(input_ids)).int()
        input_data['norm_lng_list'] = torch.tensor([0 for i in range(len(input_ids))]).to(torch.float32)
        input_data['norm_lat_list'] = torch.tensor([0 for i in range(len(input_ids))]).to(torch.float32)
        input_data['input_ids'] = input_ids

        return input_data


    def parse_spatial_context(self, pivot_name, pivot_pos, neighbor_name_list, neighbor_geometry_list, spatial_dist_fill,  pivot_dist_fill = 0):

        sep_token_id = self.tokenizer.convert_tokens_to_ids(self.tokenizer.sep_token)
        cls_token_id = self.tokenizer.convert_tokens_to_ids(self.tokenizer.cls_token)
        mask_token_id  = self.tokenizer.convert_tokens_to_ids(self.tokenizer.mask_token)
        pad_token_id = self.tokenizer.convert_tokens_to_ids(self.tokenizer.pad_token)
        max_token_len = self.max_token_len


        # process pivot
        pivot_name_tokens = self.tokenizer.convert_tokens_to_ids(self.tokenizer.tokenize(pivot_name))
        pivot_token_len = len(pivot_name_tokens)
            
        pivot_lng = pivot_pos[0]
        pivot_lat = pivot_pos[1]

        pivot_lng, pivot_lat =  self.ptransformer.transform(pivot_lng, pivot_lat)  


        # process neighbors
        neighbor_token_list = []
        neighbor_lng_list = []
        neighbor_lat_list = []

        # add separator between pivot and neighbor tokens
        # checking pivot_dist_fill is a trick to avoid adding separator token after the class name (for class name encoding of margin-ranking loss)
        if self.sep_between_neighbors and pivot_dist_fill==0: 
            neighbor_lng_list.append(spatial_dist_fill)
            neighbor_lat_list.append(spatial_dist_fill)
            neighbor_token_list.append(sep_token_id)

        for neighbor_name, neighbor_geometry in zip(neighbor_name_list, neighbor_geometry_list):

            if not neighbor_name[0].isalpha():
                # only consider neighbors starting with letters
                continue 

            neighbor_token = self.tokenizer.convert_tokens_to_ids(self.tokenizer.tokenize(neighbor_name))
            neighbor_token_len = len(neighbor_token)

            # compute the relative distance from neighbor to pivot,
            # normalize the relative distance by distance_norm_factor
            # apply the calculated distance for all the subtokens of the neighbor

            if 'coordinates' in neighbor_geometry: # to handle different json dict structures
                neighbor_lng , neighbor_lat = self.ptransformer.transform(neighbor_geometry['coordinates'][0], neighbor_geometry['coordinates'][1])  
                
            else:
                neighbor_lng , neighbor_lat = self.ptransformer.transform(neighbor_geometry[0], neighbor_geometry[1])  

            neighbor_lng_list.extend([(neighbor_lng - pivot_lng)/self.distance_norm_factor] * neighbor_token_len)
            neighbor_lat_list.extend([(neighbor_lat - pivot_lat)/self.distance_norm_factor] * neighbor_token_len)
            neighbor_token_list.extend(neighbor_token)


            if self.sep_between_neighbors:
                neighbor_lng_list.append(spatial_dist_fill)
                neighbor_lat_list.append(spatial_dist_fill)
                neighbor_token_list.append(sep_token_id)
                


        pseudo_sentence = pivot_name_tokens + neighbor_token_list 
        dist_lng_list = [pivot_dist_fill] * pivot_token_len + neighbor_lng_list 
        dist_lat_list = [pivot_dist_fill] * pivot_token_len + neighbor_lat_list 
        

        #including cls and sep
        sent_len = len(pseudo_sentence)

        max_token_len_middle = max_token_len -2 # 2 for CLS and SEP token

        # padding and truncation
        if sent_len > max_token_len_middle : 
            pseudo_sentence = [cls_token_id] + pseudo_sentence[:max_token_len_middle] + [sep_token_id] 
            dist_lat_list = [spatial_dist_fill] + dist_lat_list[:max_token_len_middle]+ [spatial_dist_fill]
            dist_lng_list = [spatial_dist_fill] + dist_lng_list[:max_token_len_middle]+ [spatial_dist_fill]
            attention_mask = [0] + [1] * max_token_len_middle + [0] # make sure SEP and CLS are not attented to
        else:
            pad_len = max_token_len_middle - sent_len
            assert pad_len >= 0 

            pseudo_sentence = [cls_token_id] + pseudo_sentence + [sep_token_id] + [pad_token_id] * pad_len 
            dist_lat_list = [spatial_dist_fill] + dist_lat_list + [spatial_dist_fill] + [spatial_dist_fill] * pad_len
            dist_lng_list = [spatial_dist_fill] + dist_lng_list + [spatial_dist_fill] + [spatial_dist_fill] * pad_len
            attention_mask = [0] + [1] * sent_len + [0] * pad_len + [0]



        norm_lng_list = np.array(dist_lng_list) 
        norm_lat_list = np.array(dist_lat_list) 


        
        input_data = {}
        input_data['pivot_name'] = pivot_name
        input_data['pivot_token_idx'] = torch.tensor([1,pivot_token_len+1])
        input_data['sent_position_ids'] = torch.tensor(np.arange(0, len(pseudo_sentence)))
        input_data['attention_mask'] = torch.tensor(attention_mask)
        input_data['norm_lng_list'] = torch.tensor(norm_lng_list).to(torch.float32)
        input_data['norm_lat_list'] = torch.tensor(norm_lat_list).to(torch.float32)
        input_data['token_type_ids'] = torch.ones(len(pseudo_sentence)).int()
        input_data['input_ids'] = torch.tensor(pseudo_sentence)

        return input_data

