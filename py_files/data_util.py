import torch
import torch.utils.data as data
import numpy as np
import pandas as pd
from PIL import Image
from pytorch_pretrained_bert.tokenization import BertTokenizer
import re
import os

def get_data_from_csv(path):
    data_csv = pd.read_csv(path, encoding='utf-8')
    pic_id_list = data_csv['pic_id'].values
    seg_id_list = data_csv['seg_id'].values
    object_list = data_csv['object'].values
    segment_list = data_csv['segment'].values
    adj_list = data_csv['adj'].values
    des_list = data_csv['des'].values

    return pic_id_list, seg_id_list, object_list, segment_list, adj_list, des_list
    
class AttDesDataset(data.Dataset):
    def __init__(self, data_root, dataset_name, dataset_split='train', transform=None,
                 bert_model='bert-base-chinese',
                 des_len=256, obj_len=8, tgt_len=32
                 ):
        self.images = []
        self.descriptions = []
        self.data_root = data_root
        self.dataset_name = dataset_name
        self.transform = transform
        self.img_root = r'data_files\small'
        self.tokenizer = BertTokenizer.from_pretrained(bert_model)
        self.des_len = des_len
        self.obj_len = obj_len
        self.tgt_len = tgt_len
        assert self.transform is not None
        
        self.pic_id_list, self.seg_id_list, self.object_list, self.segment_list, self.adj_list, self.des_list = \
            get_data_from_csv(self.data_root)

        for i in self.pic_id_list:
            self.images.append(self.get_img_from_id(i))

        for i in self.des_list:
            self.descriptions.append(self.encode_text_bert(i))
            
        self.data_csv = pd.read_csv(data_root, encoding='utf-8')

    def get_data_from_csv_by_id(self, id):
        pic_id_list = self.data_csv['pic_id'].values
        des_list = self.data_csv['des'].values
        for i in range(len(pic_id_list)):
            if str(pic_id_list[i]) == str(id):
                return des_list[i]
        return ""

    def get_img_from_id(self, img_id):
        img_filename = self.img_root
        img_filename = os.path.join(self.img_root, str(img_id) + '.jpg')
        img = Image.open(img_filename)
        if self.transform:
            img = self.transform(img)
        return img

    def encode_text_bert(self, text):
        tokens = ["[CLS]"]
        token_obj = self.tokenizer.tokenize(text)
        tokens.extend(token_obj)
        tokens.append("[SEP]")
        tokens = self.tokenizer.convert_tokens_to_ids(tokens)
        return tokens

    def get_all_from_id(self, img_id, obj_given):
        img_id = str(img_id)
        if img_id[0] == '#':
            des = ""
        else:
            des = self.get_data_from_csv_by_id(img_id)
        img = self.get_img_from_id(img_id)
        des = self.encode_text_bert(des)
        obj_given = self.encode_text_bert(obj_given)
        while len(des) < self.des_len:
            des.append(100)
        while len(obj_given) < self.obj_len:
            obj_given.append(0)
        assert len(des) == self.des_len
        return img, torch.from_numpy(np.array(des)), torch.from_numpy(np.array(obj_given))

    def __getitem__(self, idx):
        img_id = self.pic_id_list[idx]
        img = self.get_img_from_id(img_id)
        des = re.split('ï¼Œ|ï¼›', str(self.des_list[idx]))
        masked_des = ""                   
        for i in range(len(des)):
            if i != int(self.seg_id_list[idx]):
                masked_des = masked_des + des[i] + '  '

        obj = self.object_list[idx]
        segment = self.segment_list[idx]
        masked_des = self.encode_text_bert(masked_des)
        obj = self.encode_text_bert(obj)
        segment = self.encode_text_bert(segment)
        while len(masked_des) < self.des_len:
            masked_des.append(100)
        while len(obj) < self.obj_len:
            obj.append(0)
        while len(segment) < self.tgt_len:
            segment.append(0)
        assert len(masked_des) == self.des_len
        assert len(obj) == self.obj_len
        assert len(segment) == self.tgt_len
        self.images.append(img)
        self.descriptions.append(masked_des)
        return {'images': img, 'captions': np.array(masked_des), 'objects': np.array(obj), 'segments': np.array(segment), 'img_id': img_id}

    def __len__(self):
        return len(self.pic_id_list)