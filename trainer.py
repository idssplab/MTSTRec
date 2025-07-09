# +
import os
import torch
import torch.nn as nn
import torch.optim as optim
from tqdm import tqdm
from pathlib import Path
import math
import pickle
import numpy as np

from utils import *
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")


# -

# ## TransformerTrainer
# * calculate_loss
# * calculate_metrics
# * create_optimizer
# * modelpreprocess
# * get_gtneg_embedding
# * train
# * validation
# * test
#

class TransformerTrainer():
    def __init__(self, args, model, train_loader, val_loader, test_loader, Log_file):
        self.args = args
        
        self.use_token = args.use_token
        self.use_style = args.use_style
        self.use_text = args.use_text
        self.use_price = args.use_price
        
        
        if args.is_parallel:
            self.model = nn.DataParallel(model)
            self.model = model.to(device)
        else:
            self.model = model.to(device)    

        self.train_loader = train_loader
        self.val_loader = val_loader
        self.test_loader = test_loader
        
        self.optimizer = self.create_optimizer()
        
        #steplr
        self.lr_scheduler = optim.lr_scheduler.StepLR(self.optimizer, step_size = args.decay_step, gamma=args.gamma)

        self.num_epochs = args.epoch
        self.metric_ks = args.metric_ks
        self.eval_idx = ['NDCG','HR','MRR']
        self.eval_idx2 = ['NDCG']
        self.batch_size = args.batch_size
        self.max_len = args.max_len
        self.y_pad_len = args.y_pad_len
        
        self.cosine_similarity = nn.CosineSimilarity(dim = -1, eps=1e-8)

        self.best_ndcg = -1.0
        self.Log_file = Log_file
        self.early_stop_step = args.early_stop_step

    def create_optimizer(self):
        return optim.Adam(self.model.parameters(), lr = self.args.lr, weight_decay = self.args.weight_decay)          
        
    def modelpreprocess(self, batch):        
        #read batch session feature data
        seqs_all_parts = []
        seqs = batch['X'].to(device)
            
        if self.use_style:
            seqs_style_embeddings= batch['x_style'].type(torch.FloatTensor).to(device)
            seqs_all_parts.append(seqs_style_embeddings)
                
        if self.use_text:
            seqs_text_embeddings= batch['x_text'].type(torch.FloatTensor).to(device)
            for i in range(seqs_text_embeddings.size(1)):  
                single_text_embedding = seqs_text_embeddings[:, i, :, :]  # (batch_size, seq_len, text_embedding_dim)
                seqs_all_parts.append(single_text_embedding)

        if self.use_price:
            seqs_saleprice= batch['x_saleprice'].type(torch.FloatTensor).to(device)
            seqs_all_parts.append(seqs_saleprice.unsqueeze(2))
                          
        # concatenation
        if len(seqs_all_parts) > 0:
            seqs_all = torch.cat(seqs_all_parts, dim=2).to(device)
        else:
            seqs_all = torch.zeros((self.batch_size, self.max_len, 0), device = device)
        
        return seqs, seqs_all

    def get_gtneg_embedding(self, batch):
        
        #read batch ground truth and negative samples feature data
        answers = batch['Y'].to(device)
        negs = batch['neg_list'].to(device)
        
        with torch.no_grad():
            y = {}
            modality_fusion = []
            if self.use_token: 
                ans_token_emb = self.model.token(answers.clone().detach().long().to(device))
                neg_token_emb = self.model.token(negs.clone().detach().long().to(device))
                y['token'] = torch.cat([ans_token_emb, neg_token_emb], dim=1)
                modality_fusion.append('token')

            if self.use_style:
                if self.args.style_dimension != self.args.input_dimension:
                    ans_style_emb = self.model.style_embedding_layer(batch['y_style'].type(torch.FloatTensor).to(device))
                    neg_style_emb = self.model.style_embedding_layer(batch['neg_style'].type(torch.FloatTensor).to(device))
                else:
                    ans_style_emb = batch['y_style'].type(torch.FloatTensor).to(device)
                    neg_style_emb = batch['neg_style'].type(torch.FloatTensor).to(device)
                y['style'] = torch.cat([ans_style_emb, neg_style_emb], dim=1)
                modality_fusion.append('style')

            if self.use_text:
                ans_text_emb = batch['y_text'][:, 0].type(torch.FloatTensor).to(device)
                neg_text_emb = batch['neg_text'][:, 0].type(torch.FloatTensor).to(device)
                if self.args.text_dimension != self.args.input_dimension:
                    ans_text_emb = self.model.text_embedding_layer(ans_text_emb)
                    neg_text_emb = self.model.text_embedding_layer(neg_text_emb)
                y['text'] = torch.cat([ans_text_emb, neg_text_emb], dim=1)
                modality_fusion.append('text')


                if self.args.text_embedding_len != 1:
                    ans_text_embs = []
                    neg_text_embs = []
                    
                    for i in range(1,self.args.text_embedding_len):
                        ans_text_embs.append(batch['y_text'][:, i].type(torch.FloatTensor).to(device))
                        neg_text_embs.append(batch['neg_text'][:, i].type(torch.FloatTensor).to(device))

                    # concat prompt embedding
                    ans_text_concat = torch.cat(ans_text_embs, dim=2)  # (batch_size, seq_len, text_dim * text_embedding_len)
                    neg_text_concat = torch.cat(neg_text_embs, dim=2)  # (batch_size, seq_len, text_dim * text_embedding_len)

                    # gating prompt embedding
                    ans_gates = self.model.gate_text(ans_text_concat)  # (batch_size, seq_len, text_embedding_len)
                    neg_gates = self.model.gate_text(neg_text_concat)  # (batch_size, seq_len, text_embedding_len)
                    ans_gates = self.model.softmax_text(ans_gates)  
                    neg_gates = self.model.softmax_text(neg_gates)

                    weighted_ans_text_embs = []
                    weighted_neg_text_embs = []

                    for i in range(self.args.text_embedding_len-1):
                        weighted_ans_text_emb = ans_text_embs[i] * ans_gates[:, :, i:i + 1]
                        weighted_neg_text_emb = neg_text_embs[i] * neg_gates[:, :, i:i + 1]
                        weighted_ans_text_embs.append(weighted_ans_text_emb)
                        weighted_neg_text_embs.append(weighted_neg_text_emb)

                    ans_text_emb = torch.sum(torch.stack(weighted_ans_text_embs, dim=2), dim=2)
                    neg_text_emb = torch.sum(torch.stack(weighted_neg_text_embs, dim=2), dim=2)

                    # linear projection
                    if self.args.text_dimension != self.args.input_dimension:
                        ans_text_emb = self.model.prompt_embedding_layer(ans_text_emb)
                        neg_text_emb = self.model.prompt_embedding_layer(neg_text_emb)

                    y['prompt'] = torch.cat([ans_text_emb, neg_text_emb], dim=1)
                    modality_fusion.append('prompt')


            if self.use_price:
                ans_saleprice_emb = batch['y_saleprice'].type(torch.FloatTensor).to(device).unsqueeze(2).expand(-1, -1, self.model.input_dim)
                neg_saleprice_emb = batch['neg_saleprice'].type(torch.FloatTensor).to(device).unsqueeze(2).expand(-1, -1, self.model.input_dim)
                #saleprice = self.model.saleprice(answers.clone().detach().long().to(device))
                #ans_saleprice_emb = saleprice * ans_saleprice_emb
                #saleprice = self.model.saleprice(negs.clone().detach().long().to(device))
                #neg_saleprice_emb =  saleprice * neg_saleprice_emb
                y['saleprice'] = torch.cat([ans_saleprice_emb, neg_saleprice_emb], dim=1)
                modality_fusion.append('saleprice')


            for modality in modality_fusion:
                if modality == "token":
                    y[modality] = self.model.representation_linear_token(y[modality])
                elif modality == "style":
                    y[modality] = self.model.representation_linear_style(y[modality])
                elif modality == "text":
                    y[modality] = self.model.representation_linear_text(y[modality])
                elif modality == "prompt":
                    y[modality] = self.model.representation_linear_prompt(y[modality])
                elif modality == "saleprice":
                    y[modality] = self.model.representation_linear_saleprice(y[modality])
            gtneg = torch.cat([y[mod] for mod in modality_fusion], dim=2) 


        
        return gtneg, answers, negs
    
    def calculate_loss(self, batch):
        #get input for model
        seqs, seqs_all = self.modelpreprocess(batch) 
        
        #get final representation
        logits= self.model(seqs, seqs_all)
        
        #get groundtruth and negative samples embeddings
        gtneg, answers, negs= self.get_gtneg_embedding(batch)

        logits_expanded = logits.expand(-1, gtneg.size(1), -1)  # [64, 150, 512]

        #count cosine_similarity
        cos_sim = self.cosine_similarity(logits_expanded, gtneg)  # [64, 150]
        poss_cos_sim = torch.sigmoid(cos_sim[:, :self.y_pad_len])
        neg_cos_sim = torch.sigmoid(cos_sim[:, self.y_pad_len:])
        loss_poss_all = torch.log(poss_cos_sim)
        loss_neg_all = torch.log(1.0 - neg_cos_sim).sum(dim=1)

        #BCE LOSS
        total_loss = 0.0

        for i in range(len(seqs)):
            anslength = (answers[i] != 0).sum().item()
            loss_poss_seq = loss_poss_all[i][self.y_pad_len - anslength:]
            loss_neg = loss_neg_all[i]
            loss_poss = (torch.zeros(1)).cuda()

            for l in range(0, anslength):
                loss_poss += loss_poss_seq[l]

            loss_poss = torch.div(loss_poss, anslength) * self.args.train_negative_sample_size
            bce_loss = -1* (loss_poss + loss_neg)/ (anslength + self.args.train_negative_sample_size)
            total_loss += bce_loss

        return total_loss/ len(seqs)
        

    
    
    def calculate_metrics(self, batch, mode):
        #get input for model
        seqs, seqs_all = self.modelpreprocess(batch)
        
        #get final representation
        logits = self.model(seqs, seqs_all)

        #get groundtruth and negative samples embeddings
        gtneg, answers, negs = self.get_gtneg_embedding(batch)  

        logits_expanded = logits.expand(-1, gtneg.size(1), -1)  # [64, 150, 512]

        #count cosine_similarity
        cos_sim = self.cosine_similarity(logits_expanded, gtneg)  # [64, 150]

        #answers[[0,0,...,0,1]]
        answers_mask = (answers != 0).float()  # [64, 50]
        padding = torch.ones((answers_mask.size(0), self.args.test_negative_sample_size), device =device)  # [64, 100]
        answers_mask = torch.cat((answers_mask, padding), dim=1)  # [64, 150]

        # mask item score -> -inf
        cos_sim = cos_sim.masked_fill(answers_mask == 0, float('-inf'))  # [64, 150]

        # Get top-k predictions
        topk_values, topk_indices = torch.topk(cos_sim, 100, dim = -1, largest=True, sorted=True)

        #real label
        gtneg_idx = torch.concat((answers, negs), 1)  # [64, 150]

        #get prediction from top-k
        predictions_batch = torch.gather(gtneg_idx, 1, topk_indices).tolist()  # [64, 100]

        #count evaluation metrics
        if mode == "valid":
            metrics = count_batch_metrics_ndcg(self.metric_ks, len(seqs), predictions_batch, answers)
        else:
            metrics = count_batch_metrics(self.metric_ks, len(seqs), predictions_batch, answers)

        return metrics
    
 
    def train(self):
        self.validate(0)
        
        #Prepare Early Stop
        early_stop = 0
        current_ndcg = self.best_ndcg
        
        for epoch in range(self.num_epochs): 
            
            self.model.train()
            iterator = self.train_loader
            total_loss = 0.0
            accum_iter = 0

            for batch_idx, batch in tqdm(enumerate(iterator)):
                self.optimizer.zero_grad()
                loss = self.calculate_loss(batch)
                total_loss += loss.item()* batch['X'].size(0)
                accum_iter += batch['X'].size(0)
                loss.backward()
                self.optimizer.step()
            
            self.Log_file.info('Epoch {} : Training -> Total Loss: {}'.format(epoch, total_loss/accum_iter))

            self.validate(epoch)
            self.lr_scheduler.step()
            
            #get current learning rate
            current_lr = self.lr_scheduler.optimizer.param_groups[0]['lr']
            self.Log_file.info(f"Epoch {epoch}, Current learning rate: {current_lr}")
                
            if self.best_ndcg == current_ndcg:
                early_stop += 1
            else:
                early_stop = 0
            current_ndcg = self.best_ndcg
            
            if early_stop == self.early_stop_step:
                self.Log_file.info('No Improving, Early Stop!')
                self.Log_file.info('Current Epoch : {}, Current Best NDCG@5 {}'.format(epoch, self.best_ndcg))
                break            
        
    def validate(self, epoch):
        self.model.eval()
        
        metrics = {}
        for k in sorted(self.metric_ks, reverse=True):
            for name in self.eval_idx2:
                metrics[f'{name}@{k}'] = 0.0

        accum_iter = 0

        with torch.no_grad():
            iterator = self.val_loader
            textgate_weights_list = []
        
            for batch_idx, batch in enumerate(iterator):
                metrics_batch = self.calculate_metrics(batch, "valid")
                if self.args.text_embedding_len!=1:
                    textgate_weights_list.append(self.model.gates_text.detach().cpu())
                for k in sorted(self.metric_ks, reverse=True):
                    for name in self.eval_idx2:
                        metrics[f'{name}@{k}'] += metrics_batch[f'{name}@{k}'] * batch['X'].size(0)
                accum_iter += batch['X'].size(0)
            
            for k in sorted(self.metric_ks, reverse=True):
                for name in self.eval_idx2:
                    metrics[f'{name}@{k}'] = metrics[f'{name}@{k}'] / accum_iter

            #Saving the best model
            if metrics['NDCG@%d' % 5] > self.best_ndcg:
                self.Log_file.info('<Saving the Best Model!>...Epoch {}'.format(epoch))
                self.best_ndcg = metrics['NDCG@%d' % 5]

                model_checkpoint = self.args.model_checkpoint_dir
                model_state_dict = {'model_state_dict': self.model.state_dict(), 'optimizer_state_dict': self.optimizer.state_dict()}
                torch.save(model_state_dict, os.path.join(model_checkpoint, self.args.best_modelname))
                
                if self.args.text_embedding_len!=1:
                    gate_weights = torch.mean(torch.mean(torch.cat(textgate_weights_list, dim=0),dim = 0), dim=0)
                    self.Log_file.info('Prompt text Gating average weights: {}'.format(gate_weights))
                            
        Metrics_LOG("Validation", epoch, metrics, self.metric_ks, self.eval_idx2, self.Log_file)

    
    def test(self):
        self.Log_file.info('Testing best model with the test set!')
        best_model = torch.load(os.path.join(self.args.model_checkpoint_dir, self.args.best_modelname)).get('model_state_dict')
        self.model.load_state_dict(best_model)
        
        self.model.eval()
        seqs_all = []
        predictions_all = []
        metrics = {}
        for k in sorted(self.metric_ks, reverse=True):
            for name in self.eval_idx:
                metrics[f'{name}@{k}'] = 0.0

        accum_iter = 0
                    
        with torch.no_grad():
            iterator = self.test_loader

            for batch_idx, batch in enumerate(iterator):
                metrics_batch= self.calculate_metrics(batch, "test")
                for k in sorted(self.metric_ks, reverse=True):
                    for name in self.eval_idx:
                        metrics[f'{name}@{k}'] += metrics_batch[f'{name}@{k}'] * batch['X'].size(0)

                accum_iter += batch['X'].size(0)
                
                
            for k in sorted(self.metric_ks, reverse=True):
                for name in self.eval_idx:
                        metrics[f'{name}@{k}'] = metrics[f'{name}@{k}'] / accum_iter

        
            Metrics_LOG("Testing", 0 , metrics, self.metric_ks, self.eval_idx, self.Log_file)


class Inference():
    def __init__(self, args, model, test_loader, Log_file):
        self.args = args
        
        self.use_token = args.use_token
        self.use_style = args.use_style
        self.use_text = args.use_text
        self.use_price = args.use_price
        
        
        if args.is_parallel:
            self.model = nn.DataParallel(model)
            self.model = model.to(device)
        else:
            self.model = model.to(device)    

        self.test_loader = test_loader
        
        self.optimizer = self.create_optimizer()
        
        #steplr
        self.lr_scheduler = optim.lr_scheduler.StepLR(self.optimizer, step_size = args.decay_step, gamma=args.gamma)

        self.num_epochs = args.epoch
        self.metric_ks = args.metric_ks
        self.eval_idx = ['NDCG','HR','MRR']
        self.eval_idx2 = ['NDCG']
        self.batch_size = args.batch_size
        self.max_len = args.max_len
        self.y_pad_len = args.y_pad_len
        
        self.cosine_similarity = nn.CosineSimilarity(dim = -1, eps=1e-8)

        self.best_ndcg = -1.0
        self.Log_file = Log_file
        self.early_stop_step = args.early_stop_step

    def create_optimizer(self):
        return optim.Adam(self.model.parameters(), lr = self.args.lr, weight_decay = self.args.weight_decay)          
        
    def modelpreprocess(self, batch):        
        #read batch session feature data
        seqs_all_parts = []
        seqs = batch['X'].to(device)
            
        if self.use_style:
            seqs_style_embeddings= batch['x_style'].type(torch.FloatTensor).to(device)
            seqs_all_parts.append(seqs_style_embeddings)
                
        if self.use_text:
            seqs_text_embeddings= batch['x_text'].type(torch.FloatTensor).to(device)
            for i in range(seqs_text_embeddings.size(1)):  
                single_text_embedding = seqs_text_embeddings[:, i, :, :]  # (batch_size, seq_len, text_embedding_dim)
                seqs_all_parts.append(single_text_embedding)

        if self.use_price:
            seqs_saleprice= batch['x_saleprice'].type(torch.FloatTensor).to(device)
            seqs_all_parts.append(seqs_saleprice.unsqueeze(2))
                          
        # concatenation
        if len(seqs_all_parts) > 0:
            seqs_all = torch.cat(seqs_all_parts, dim=2).to(device)
        else:
            seqs_all = torch.zeros((self.batch_size, self.max_len, 0), device = device)
        
        return seqs, seqs_all

    def get_gtneg_embedding(self, batch):
        
        #read batch ground truth and negative samples feature data
        answers = batch['Y'].to(device)
        negs = batch['neg_list'].to(device)
        
        with torch.no_grad():
            y = {}
            modality_fusion = []
            if self.use_token: 
                ans_token_emb = self.model.token(answers.clone().detach().long().to(device))
                neg_token_emb = self.model.token(negs.clone().detach().long().to(device))
                y['token'] = torch.cat([ans_token_emb, neg_token_emb], dim=1)
                modality_fusion.append('token')

            if self.use_style:
                if self.args.style_dimension != self.args.input_dimension:
                    ans_style_emb = self.model.style_embedding_layer(batch['y_style'].type(torch.FloatTensor).to(device))
                    neg_style_emb = self.model.style_embedding_layer(batch['neg_style'].type(torch.FloatTensor).to(device))
                else:
                    ans_style_emb = batch['y_style'].type(torch.FloatTensor).to(device)
                    neg_style_emb = batch['neg_style'].type(torch.FloatTensor).to(device)
                y['style'] = torch.cat([ans_style_emb, neg_style_emb], dim=1)
                modality_fusion.append('style')

            if self.use_text:
                ans_text_emb = batch['y_text'][:, 0].type(torch.FloatTensor).to(device)
                neg_text_emb = batch['neg_text'][:, 0].type(torch.FloatTensor).to(device)
                if self.args.text_dimension != self.args.input_dimension:
                    ans_text_emb = self.model.text_embedding_layer(ans_text_emb)
                    neg_text_emb = self.model.text_embedding_layer(neg_text_emb)
                y['text'] = torch.cat([ans_text_emb, neg_text_emb], dim=1)
                modality_fusion.append('text')


                if self.args.text_embedding_len != 1:
                    ans_text_embs = []
                    neg_text_embs = []
                    
                    for i in range(1,self.args.text_embedding_len):
                        ans_text_embs.append(batch['y_text'][:, i].type(torch.FloatTensor).to(device))
                        neg_text_embs.append(batch['neg_text'][:, i].type(torch.FloatTensor).to(device))

                    # concat prompt embedding
                    ans_text_concat = torch.cat(ans_text_embs, dim=2)  # (batch_size, seq_len, text_dim * text_embedding_len)
                    neg_text_concat = torch.cat(neg_text_embs, dim=2)  # (batch_size, seq_len, text_dim * text_embedding_len)

                    # gating prompt embedding
                    ans_gates = self.model.gate_text(ans_text_concat)  # (batch_size, seq_len, text_embedding_len)
                    neg_gates = self.model.gate_text(neg_text_concat)  # (batch_size, seq_len, text_embedding_len)
                    ans_gates = self.model.softmax_text(ans_gates)  
                    neg_gates = self.model.softmax_text(neg_gates)

                    weighted_ans_text_embs = []
                    weighted_neg_text_embs = []

                    for i in range(self.args.text_embedding_len-1):
                        weighted_ans_text_emb = ans_text_embs[i] * ans_gates[:, :, i:i + 1]
                        weighted_neg_text_emb = neg_text_embs[i] * neg_gates[:, :, i:i + 1]
                        weighted_ans_text_embs.append(weighted_ans_text_emb)
                        weighted_neg_text_embs.append(weighted_neg_text_emb)

                    ans_text_emb = torch.sum(torch.stack(weighted_ans_text_embs, dim=2), dim=2)
                    neg_text_emb = torch.sum(torch.stack(weighted_neg_text_embs, dim=2), dim=2)

                    # linear projection
                    if self.args.text_dimension != self.args.input_dimension:
                        ans_text_emb = self.model.prompt_embedding_layer(ans_text_emb)
                        neg_text_emb = self.model.prompt_embedding_layer(neg_text_emb)

                    y['prompt'] = torch.cat([ans_text_emb, neg_text_emb], dim=1)
                    modality_fusion.append('prompt')


            if self.use_price:
                ans_saleprice_emb = batch['y_saleprice'].type(torch.FloatTensor).to(device).unsqueeze(2).expand(-1, -1, self.model.input_dim)
                neg_saleprice_emb = batch['neg_saleprice'].type(torch.FloatTensor).to(device).unsqueeze(2).expand(-1, -1, self.model.input_dim)
                #saleprice = self.model.saleprice(answers.clone().detach().long().to(device))
                #ans_saleprice_emb = saleprice * ans_saleprice_emb
                #saleprice = self.model.saleprice(negs.clone().detach().long().to(device))
                #neg_saleprice_emb =  saleprice * neg_saleprice_emb
                y['saleprice'] = torch.cat([ans_saleprice_emb, neg_saleprice_emb], dim=1)
                modality_fusion.append('saleprice')


            for modality in modality_fusion:
                if modality == "token":
                    y[modality] = self.model.representation_linear_token(y[modality])
                elif modality == "style":
                    y[modality] = self.model.representation_linear_style(y[modality])
                elif modality == "text":
                    y[modality] = self.model.representation_linear_text(y[modality])
                elif modality == "prompt":
                    y[modality] = self.model.representation_linear_prompt(y[modality])
                elif modality == "saleprice":
                    y[modality] = self.model.representation_linear_saleprice(y[modality])
            gtneg = torch.cat([y[mod] for mod in modality_fusion], dim=2) 


        
        return gtneg, answers, negs
    

    
    
    def calculate_metrics(self, batch, mode):
        #get input for model
        seqs, seqs_all = self.modelpreprocess(batch)
        
        #get final representation
        logits = self.model(seqs, seqs_all)

        #get groundtruth and negative samples embeddings
        gtneg, answers, negs = self.get_gtneg_embedding(batch)  

        logits_expanded = logits.expand(-1, gtneg.size(1), -1)  # [64, 150, 512]

        #count cosine_similarity
        cos_sim = self.cosine_similarity(logits_expanded, gtneg)  # [64, 150]

        #answers[[0,0,...,0,1]]
        answers_mask = (answers != 0).float()  # [64, 50]
        padding = torch.ones((answers_mask.size(0), self.args.test_negative_sample_size), device =device)  # [64, 100]
        answers_mask = torch.cat((answers_mask, padding), dim=1)  # [64, 150]

        # mask item score -> -inf
        cos_sim = cos_sim.masked_fill(answers_mask == 0, float('-inf'))  # [64, 150]

        # Get top-k predictions
        topk_values, topk_indices = torch.topk(cos_sim, 100, dim = -1, largest=True, sorted=True)

        #real label
        gtneg_idx = torch.concat((answers, negs), 1)  # [64, 150]

        #get prediction from top-k
        predictions_batch = torch.gather(gtneg_idx, 1, topk_indices).tolist()  # [64, 100]

        #count evaluation metrics
        if mode == "valid":
            metrics = count_batch_metrics_ndcg(self.metric_ks, len(seqs), predictions_batch, answers)
        else:
            metrics = count_batch_metrics(self.metric_ks, len(seqs), predictions_batch, answers)

        return metrics
    
 

    def test(self):
        self.Log_file.info('Testing best model with the test set!')
        best_model = torch.load(os.path.join(self.args.model_checkpoint_dir, self.args.best_modelname)).get('model_state_dict')
        self.model.load_state_dict(best_model)
        
        self.model.eval()
        seqs_all = []
        predictions_all = []
        metrics = {}
        for k in sorted(self.metric_ks, reverse=True):
            for name in self.eval_idx:
                metrics[f'{name}@{k}'] = 0.0

        accum_iter = 0
                    
        with torch.no_grad():
            iterator = self.test_loader

            for batch_idx, batch in enumerate(iterator):
                metrics_batch= self.calculate_metrics(batch, "test")
                for k in sorted(self.metric_ks, reverse=True):
                    for name in self.eval_idx:
                        metrics[f'{name}@{k}'] += metrics_batch[f'{name}@{k}'] * batch['X'].size(0)

                accum_iter += batch['X'].size(0)
                
                
            for k in sorted(self.metric_ks, reverse=True):
                for name in self.eval_idx:
                        metrics[f'{name}@{k}'] = metrics[f'{name}@{k}'] / accum_iter

        
            Metrics_LOG("Testing", 0 , metrics, self.metric_ks, self.eval_idx, self.Log_file)
