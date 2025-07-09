# +
import pickle
import numpy as np
from numpy import random
import torch
import torch.nn as nn
import torch.utils.data as data_utils
import datetime

from utils import *


# -

# ## Dataset

class MyDataset(data_utils.Dataset):
    def __init__(self, args, dataset, stylelst, textlst, salepricelst, MonthProduct, Log_file, mode = "train"):
        seed = args.myseed
        
        self.args = args
        self.data = dataset[0]
        self.label = dataset[1]
        self.user_count = dataset[2]
        self.item_count = dataset[3]
        self.buydate = dataset[4]
        
        self.styleset = stylelst
        self.textset = textlst
        if self.args.use_price:
            self.salepriceset = torch.FloatTensor(salepricelst)
        
        #get month product list
        ym_datetime = []
        yearmonth = MonthProduct.keys()
        for ym in yearmonth:
            ym_datetime.append(datetime.datetime(int(ym[0:4]), int(ym[4:6]), 1, 0, 0, 0, 0))
        self.mProducts_date = ym_datetime
        self.mProducts = list(MonthProduct.values())
        
        self.mode = mode
        self.max_len = args.max_len
        self.pad_id = 0
        self.y_pad_len = args.y_pad_len
        
        

        if self.mode == "train":
            negative_sampler = RandomNegativeSampler(self.data, self.label,
                                                          self.user_count, self.item_count, self.buydate,
                                                          self.mProducts_date, self.mProducts,
                                                          args.train_negative_sample_size, seed)
        else:
            negative_sampler = RandomNegativeSampler(self.data, self.label,
                                                          self.user_count, self.item_count, self.buydate,
                                                          self.mProducts_date, self.mProducts,
                                                          args.test_negative_sample_size, seed)       
        self.negative_samples = negative_sampler.generate_negative_samples()

    
    def __len__(self):
        return len(self.data)
    
    def __getitem__(self, index):
        return self.data[index], self.label[index], self.negative_samples[index]
    
    
    
    def collate_fn(self, batch):
        RESULT_DICT = {}
        
        #use_style/ use_text/ use_price/ negsamples
        x_list = []
        neg_list = []
        y_list = []
        
        #PAD
        x_list_pad = []
        y_list_pad = []
                
        x_list_style = []
        x_list_text = []
        x_list_saleprice =[]
        
        y_list_style = []
        y_list_text = []
        y_list_saleprice =[]
        
        neg_list_style = []
        neg_list_text = []
        neg_list_saleprice =[]
        

        for data, label, negative_samples in batch:
            if data == []:
                data = [0]
            else:   
                data = data

            x_list.append(data)
            y_list.append(label)
            neg_list.append(negative_samples)

        #pad to same size
        for i in range(len(x_list)):
            x_list_pad.append(pad_to_len(x_list[i], self.max_len, self.pad_id))
        #pad to same size (multilabels)
        for i in range(len(y_list)):
            y_list_pad.append(pad_to_len(y_list[i], self.y_pad_len, self.pad_id))

            
               
        RESULT_DICT["X"] = torch.LongTensor(x_list_pad)
        RESULT_DICT["Y"] = torch.LongTensor(y_list_pad)
        RESULT_DICT["neg_list"] = torch.LongTensor(neg_list)
    
        #style 
        if self.args.use_style:
            for i in range(len(x_list_pad)):
                tmp = self.styleset(torch.LongTensor(x_list_pad[i]))
                x_list_style.append(tmp)
                tmp = self.styleset(torch.LongTensor(y_list_pad[i]))
                y_list_style.append(tmp)
            RESULT_DICT["x_style"] = torch.stack(x_list_style, 0)
            RESULT_DICT["y_style"] = torch.stack(y_list_style, 0)
            
            for i in range(len(neg_list)):
                tmp = self.styleset(torch.LongTensor(neg_list[i]))
                neg_list_style.append(tmp)
            RESULT_DICT["neg_style"] = torch.stack(neg_list_style, 0)
        
        #text
        if self.args.use_text:
            x_text = []
            y_text = []
            for textemb in self.textset:
                for i in range(len(x_list_pad)):
                    tmp = textemb(torch.LongTensor(x_list_pad[i]))
                    x_list_text.append(tmp)
                    tmp = textemb(torch.LongTensor(y_list_pad[i]))
                    y_list_text.append(tmp)
                x_text.append(torch.stack(x_list_text, 0))
                y_text.append(torch.stack(y_list_text, 0))
                x_list_text = []
                y_list_text = []
            RESULT_DICT["x_text"] = torch.stack(x_text, dim=1)
            RESULT_DICT["y_text"] = torch.stack(y_text, dim=1)

            neg_text = []
            for textemb in self.textset:
                for i in range(len(neg_list)):
                    tmp = textemb(torch.LongTensor(neg_list[i]))
                    neg_list_text.append(tmp)
                neg_text.append(torch.stack(neg_list_text, 0))
                neg_list_text = []
            RESULT_DICT["neg_text"] = torch.stack(neg_text, dim=1)

        
        #price
        if self.args.use_price:
            for i in range(len(x_list_pad)):
                tmp = self.salepriceset.index_select(0, torch.LongTensor(x_list_pad[i]))   
                x_list_saleprice.append(tmp)
                tmp = self.salepriceset.index_select(0, torch.LongTensor(y_list_pad[i]))
                y_list_saleprice.append(tmp)
                RESULT_DICT["x_saleprice"] = torch.stack(x_list_saleprice, 0)
                RESULT_DICT["y_saleprice"] = torch.stack(y_list_saleprice, 0)
        
            for i in range(len(neg_list)):
                tmp = self.salepriceset.index_select(0, torch.LongTensor(neg_list[i]))
                neg_list_saleprice.append(tmp)
                RESULT_DICT["neg_saleprice"] = torch.stack(neg_list_saleprice, 0)
        
        return RESULT_DICT


#
# ## Random Negative Sampler

# +
class RandomNegativeSampler():
    def __init__(self, train, test, user_count, item_count, date, monthProducts_date, monthProducts, sample_size, seed):
        self.train = train
        self.test = test
        self.user_count = user_count
        self.item_count = item_count
        self.date = date
        self.monthproduct_date = monthProducts_date
        self.monthproduct = monthProducts
        self.sample_size = sample_size
        self.seed = seed

    def generate_negative_samples(self):
        assert self.seed is not None, 'Specify seed for random sampling'
        np.random.seed(int(self.seed))
        negative_samples = {}
        for user in range(0, self.user_count):           
            seen = set(self.train[user] + self.test[user])
            buydate = self.date[user]
            
            #the candidate product during the purchase date
            candidateproducts = []
            for k, ym in enumerate(self.monthproduct_date):
                if buydate <= ym and k != len(self.monthproduct_date)-1:
                    continue
                elif buydate > ym:
                    candidateproducts = self.monthproduct[k-1]
                else:
                    candidateproducts = self.monthproduct[k]
            
            samples = []
            
            #negative samples
            for _ in range(self.sample_size):
                item = candidateproducts[np.random.choice(len(candidateproducts))]
                while item in seen or item in samples:
                    item = candidateproducts[np.random.choice(len(candidateproducts))]
                samples.append(item)
                
            negative_samples[user] = samples

        return negative_samples


