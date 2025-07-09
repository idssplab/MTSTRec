import os
import pickle
import random
import numpy as np
from pathlib import Path
from datetime import date
from collections import defaultdict
import time
import logging
import torch
import torch.nn as nn
import torch.backends.cudnn as cudnn
import numpy as np


# ### Time

def get_time(start_time, end_time):
    time_g = int(end_time - start_time)
    hour = int(time_g / 3600)
    minu = int(time_g / 60) % 60
    secon = time_g % 60
    return hour, minu, secon


# ### Parameters

def count_parameters(model):
    total_params = sum(p.numel() for p in model.parameters())
    trainable_params = sum(p.numel() for p in model.parameters() if p.requires_grad)
    ratio = trainable_params / total_params
    percentage = ratio * 100
    return total_params, trainable_params, percentage


# ### Log

def setuplogger(dir_label, datasetname, log_paras, time_run):

    formatter = logging.Formatter("[%(levelname)s %(asctime)s] %(message)s")
    Log_file = logging.getLogger('Log_file')
    Log_screen = logging.getLogger('Log_screen')

    log_path = os.path.join('./logs', datasetname, dir_label)
    if not os.path.exists(log_path):
        os.makedirs(log_path)

    log_file_name = os.path.join(log_path, 'log_' + log_paras + time_run + '.log')
    Log_file.setLevel(logging.INFO)
    Log_screen.setLevel(logging.INFO)

    th = logging.FileHandler(filename=log_file_name, encoding='utf-8')
    th.setLevel(logging.INFO)
    th.setFormatter(formatter)
    Log_file.addHandler(th)

    handler = logging.StreamHandler()
    handler.setLevel(logging.INFO)
    handler.setFormatter(formatter)

    Log_screen.addHandler(handler)
    Log_file.addHandler(handler)
    return Log_file, Log_screen


# ### Dataset

def load(file_name):
    with open(file_name, "rb") as f:
        return pickle.load(f)


def save_pickle(dataset,file_path):
    with open(file_path,'wb') as file:
        pickle.dump(dataset,file)


def data_partition(min_len ,max_len, filename, datename, popfilename, Log_file):
    user_x = []
    user_y = []
    
    usernumpurchase = 0
    itemnumpurchase = 0
    User = defaultdict(list)

    Log_file.info('build dataset...')
    Log_file.info('Filename: {}'.format(filename))
    
    f = open(filename, 'rb')
    sessions = pickle.load(f)
    
    for session in sessions:
        user_x.append(session[0])
        user_y.append(session[1])
        
        #maximum item
        max_1 = int(max(session[0]))
        max_2 = int(max(session[1]))
        max_ = max(max_1,max_2)
        itemnumpurchase = max(max_, itemnumpurchase)

    usernum = len(user_x)
    itemnum = itemnumpurchase
    

    Log_file.info('Usernum is : {}'.format(usernum))
    Log_file.info('Itemnum is : {}'.format(itemnum))
    
    Log_file.info('build sessiondate dataset...')
    Log_file.info('Filename: {}'.format(datename))
    
    f = open(datename, 'rb')
    sessiondates = pickle.load(f)
    
    data_x = []
    data_y = []
    data_date = []
    count_minlen = 0
    
    #preprocssing for sessiondata: filter out sessions with fewer than min_len interactions 
    for user in range(usernum):
        if len(user_x[user]) < min_len:
            count_minlen+=1
            continue
        if len(user_x[user]) <= max_len:
            data_x.append(user_x[user])
            data_y.append(user_y[user])
            data_date.append(sessiondates[user])
        elif len(user_x[user]) > max_len:
            length = len(user_x[user])
            data_x.append(user_x[user][length-max_len:length])
            data_y.append(user_y[user])
            data_date.append(sessiondates[user])

    usernum = len(data_x)

    Log_file.info('Length <{} count : {}'.format(min_len, count_minlen))
    Log_file.info('new Usernum is : {}'.format(usernum))


    return [data_x, data_y, usernum, itemnum, data_date]


def pad_to_len(seq, to_len, padding):
    seq = seq[-to_len:]
    padding_len = to_len - len(seq)
    seq = [padding] * padding_len + seq
    return seq


# ### Product Feature Dataset

def read_style_embedding(filename, dim, Log_file):
    Log_file.info('Read Style Embedding...')
    Log_file.info('Filename: {}'.format(filename))
    
    #read file
    features = load(filename)
    
    #add index 0
    a = torch.zeros(1,dim)
    features = torch.cat((features, a), 0)
    Log_file.info('Style Embedding Size is : {}'.format(features.size()))
    
    
    result = nn.Embedding.from_pretrained(features)
    return result


def read_text_embedding(filename,dim, Log_file):
    Log_file.info('Read Text Embedding...')
    Log_file.info('Filename: {}'.format(filename))
    
    #read file
    features = load(filename)
    
    #add index 0
    a = torch.zeros(1,dim)
    features = torch.cat((features, a), 0)
    Log_file.info('Text Embedding Size is : {}'.format(features.size()))
            
    result = nn.Embedding.from_pretrained(features)
    return result


# ### Trainer & Evaluation

def fix_random_seed_as(random_seed):
    random.seed(random_seed)
    torch.manual_seed(random_seed)
    torch.cuda.manual_seed_all(random_seed)
    np.random.seed(random_seed)
    cudnn.deterministic = True
    cudnn.benchmark = False


def count_batch_metrics(metric_ks, seqs_len, predictions_batch, answers):
    #count NDCG, HR, MRR
    metrics = {}
    for k in sorted(metric_ks, reverse=True):
        hr = 0.0
        mrr = 0.0
        ndcg = 0.0

        for l in range(seqs_len):
            dcg = 0.0
            idcg = 0.0
            num_answers = (answers[l] != 0).sum().item()
            
            hr_per_answer = []
            mrr_per_answer = []

            #Count HR & MRR
            for answer in answers[l]:
                if answer == 0:  # ignore padding
                    continue
                    
                filtered_predictions = [pred for pred in predictions_batch[l] if pred == answer or pred not in answers[l]]
            
                in_count_single = 0.0
                mrr_single = 0.0
                
                for j in range(k):
                    if filtered_predictions[j] == answer:
                        in_count_single = 1.0
                        mrr_single = (1.0 / (j + 1.0))
                        break 
                
                hr_per_answer.append(in_count_single)
                mrr_per_answer.append(mrr_single)

            if len(hr_per_answer) > 0:
                hr += sum(hr_per_answer) / len(hr_per_answer)
                mrr += sum(mrr_per_answer) / len(mrr_per_answer)
            
            #Count NDCG
            for j in range(k):
                if predictions_batch[l][j] in answers[l]:
                    dcg += (1.0 / np.log2(j + 2.0))
            for i in range(min(k, num_answers)):
                idcg+= (1.0 / np.log2(i + 2.0))
            ndcg += dcg / idcg if idcg > 0 else 0.0

            
            
        metrics['NDCG@%d' % k] = ndcg / seqs_len
        metrics['MRR@%d' % k] = mrr / seqs_len
        metrics['HR@%d' % k] = hr / seqs_len

    return metrics


def count_batch_metrics_ndcg(metric_ks, seqs_len, predictions_batch, answers):
    #count only NDCG
    metrics = {}
    for k in sorted(metric_ks, reverse=True):
        ndcg = 0.0
        
        for l in range(seqs_len):
            dcg = 0.0
            idcg = 0.0
            num_answers = (answers[l] != 0).sum().item()
   
            for j in range(k):
                if predictions_batch[l][j] in answers[l]:
                    dcg += (1.0 / np.log2(j + 2.0))   
            for i in range(min(k, num_answers)):
                idcg+= (1.0 / np.log2(i + 2.0))
            ndcg += dcg / idcg if idcg > 0 else 0.0
            
            
        metrics['NDCG@%d' % k] = ndcg / seqs_len

    return metrics


def Metrics_LOG(mode, epoch, metrics, k, evaluation_name, Log_file):
    if mode == "Testing":
        Log_file.info('Testing Result')
    else:
        Log_file.info('Epoch {} : {}'.format(epoch, mode))

    k = [str(x) for x in sorted(k, reverse=False)]
    
    Log_file.info("{:<10} {:<10} {:<10} {:<10}".format("Metrics", k[0], k[1], k[2]))
    for eval_name in evaluation_name:
        Log_file.info("{:<10} {:<10} {:<10} {:<10}".format(eval_name, round(metrics[eval_name+'@'+k[0]],6), round(metrics[eval_name+'@'+k[1]],6), round(metrics[eval_name+'@'+k[2]],6)))
