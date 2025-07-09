# +
import argparse
import numpy as np
import torch
import torch.nn as nn
from torch.utils.data import DataLoader
from sklearn.preprocessing import MinMaxScaler
import re
import os
from os import path
os.environ['CUDA_LAUNCH_BLOCKING'] = '1'

from parameters import parse_args
from dataloader import *
from model import *
from trainer import *
from utils import *

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")


# -

# ### Main

if __name__ == '__main__':
    args = parse_args()  
    os.environ['CUDA_VISIBLE_DEVICES'] = args.device_idx
    start_time = time.time()
    
    #seed
    fix_random_seed_as(args.myseed)
    

    run_model = 'MTSTRec'
    
    # Building Log file
    text_emb_list = [str(item) for item in args.text_embedding.split(',')]
    text_emb = text_emb_list[0].split('_')
    args.text_embedding_len = len(text_emb_list)
    
    dir_label = ''
    log_paras = run_model+'_'
    
    if args.use_token:
        log_paras+='TOKEN_'
        dir_label+='TOKEN'
    if args.use_style:
        log_paras+='STYLE_'
        dir_label+='_STYLE'
    if args.use_text:
        if args.text_embedding_len == 1:
            log_paras+=f'TEXT_{text_emb_list[0]}_'
        else:
            log_paras+=f'TEXT_{args.text_embedding_len}_'
        dir_label+=f'_TEXT_{text_emb[-1]}'
    if args.use_price:
        log_paras+='PRICE_'
        dir_label+='_PRICE'
    
    log_paras += f'bs_{args.batch_size}' \
                f'_lr_{args.lr}_wd_{args.weight_decay}_ds_{args.decay_step}' \
                f'_dp_{args.transformer_dropout}_hdp_{args.transformer_hidden_dropout}' \
                f'_nl_{args.transformer_num_blocks}_hd_{args.hidden_dimension}_seed_{args.myseed}'
    time_run = time.strftime('-%Y%m%d-%H%M%S', time.localtime())
    args.label_screen = args.label_screen + time_run
    Log_file, Log_screen = setuplogger(dir_label, args.dataset, log_paras, time_run)
    Log_file.info(args)
     
    #Read Session Data   
    dataset_train = data_partition(args.min_len, args.max_len, os.path.join(args.root_data_dir, args.trainset), os.path.join(args.root_data_dir, args.traindate),"None",Log_file)
    dataset_val = data_partition(args.min_len, args.max_len, os.path.join(args.root_data_dir, args.validset), os.path.join(args.root_data_dir, args.validdate),"None",Log_file)
    dataset_test = data_partition(args.min_len, args.max_len, os.path.join(args.root_data_dir, args.testset), os.path.join(args.root_data_dir, args.testdate),"None",Log_file)
    Log_file.info("train num: {}, validate num: {}, test num: {}".format(dataset_train[2], dataset_val[2], dataset_test[2]))
   
    #Read Style Data
    if args.use_style:
        style_embedding_list = read_style_embedding(os.path.join(args.root_data_dir, "Product_Feature", args.style_embedding), args.style_dimension, Log_file)
    else:
        style_embedding_list = "None"
        
    #Read Text Data
    if args.use_text:
        text_embedding_list = []
        for text_embedding in text_emb_list:
            text_embedding_list.append(read_text_embedding(os.path.join(args.root_data_dir, "Product_Feature", text_embedding), args.text_dimension, Log_file))
    else:
        text_embedding_list =  "None"
        
    #Read Price Data
    if args.use_price:
        if args.dataset == 'hm':
            saleprice_list, saleprice_list = load(os.path.join(args.root_data_dir, "Product_Feature", args.price_file)) 
            saleprice_list = np.array(saleprice_list).reshape(-1, 1)

            scaler = MinMaxScaler()
            salepricemean = saleprice_list.mean()
            saleprice_list = scaler.fit_transform(saleprice_list).flatten().tolist()
            saleprice_list = [salepricemean] + saleprice_list + [salepricemean]    
            
        else:
            price_list, discount_list = load(os.path.join(args.root_data_dir, "Product_Feature", args.price_file)) 
            price_list = np.array(price_list).reshape(-1, 1)
            discount_list = np.array(discount_list).reshape(-1, 1) 
            #count saleprice
            saleprice_list = price_list * discount_list

            scaler = MinMaxScaler()
            salepricemean = saleprice_list.mean()
            saleprice_list = scaler.fit_transform(saleprice_list).flatten().tolist()
            saleprice_list = [salepricemean] + saleprice_list + [salepricemean] 
    else:
        saleprice_list = "None"
            
        
    #Read Month Product Data for precise recommendation
    dirPath = os.path.join(args.root_data_dir, "Monthly_Product")
    MonthProduct_filelist = [os.path.join(dirPath, f) for f in os.listdir(dirPath) if os.path.isfile(os.path.join(dirPath, f))]

    MonthProduct = {}
    for filename in sorted(MonthProduct_filelist):
        #file
        f = open(filename, 'rb')
        mproducts = pickle.load(f)
        #name
        substrLocation = [m.start() for m in re.finditer('_productlist', filename)]
        yearmonth = '20'+str(filename[substrLocation[0] -4 : substrLocation[0]])
        MonthProduct[yearmonth] = mproducts

    #total product num
    args.num_items = max(dataset_train[3], dataset_val[3], dataset_test[3])
    
    train_set = MyDataset(args, dataset_train, style_embedding_list, text_embedding_list, saleprice_list, MonthProduct, Log_file, mode = 'train') 
    val_set   = MyDataset(args, dataset_val, style_embedding_list, text_embedding_list, saleprice_list, MonthProduct, Log_file, mode = 'eval')
    test_set  = MyDataset(args, dataset_test, style_embedding_list, text_embedding_list, saleprice_list, MonthProduct, Log_file, mode = 'eval')
    
    train_loader = DataLoader(train_set, batch_size = args.batch_size, collate_fn = train_set.collate_fn, shuffle = True) 
    val_loader   = DataLoader(val_set  , batch_size = args.batch_size, collate_fn = val_set.collate_fn, shuffle = False) 
    test_loader  = DataLoader(test_set , batch_size=1         , collate_fn = test_set.collate_fn, shuffle = False) 
    
    
    Log_file.info("Start to Run model {}".format(run_model))
    
    model = MTSTRec(args)

    Log_file.info("{}".format(model))

    
    # count parameters
    total_params = sum(p.numel() for p in model.parameters())
    Log_file.info(f"Total number of parameters: {total_params}")
    
    
    # Training                            
    trainer = TransformerTrainer(args, model, train_loader, val_loader, test_loader, Log_file)
    trainer.train()
    trainer.test()
    
    end_time = time.time()
    hour, minu, secon = get_time(start_time, end_time)
    Log_file.info("##### (time) all: {} hours {} minutes {} seconds #####".format(hour, minu, secon))
