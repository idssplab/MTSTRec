# +
import torch
import torch.nn as nn
import torch.nn.init as init
import torch.nn.functional as F
import math
import numpy as np

from module import *
from utils import *
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")


# -

# # MTSTRec
# * deal with different features (seperate)
# * multi-type encoder
# * using time-algned share tokens for fusion

class MTSTRec(nn.Module):
    def __init__(self, args):
        super(MTSTRec, self).__init__()
        self.args = args
        self.max_len = args.max_len
        self.num_items = args.num_items
        self.dropout = args.transformer_dropout
        self.hidden_dropout = args.transformer_hidden_dropout
        self.pad_id = 0
        
        #for tst fusion
        self.fusion_num_blocks = args.fusion_num_blocks
        
        ##########Settings########
        self.use_token = args.use_token
        self.use_style = args.use_style
        self.use_text = args.use_text
        self.use_price = args.use_price
        
        ##########Dimension#######
        #512(token)
        self.input_dim = args.input_dimension
        self.output_dim = args.output_dimension
        #512
        self.style_dim = args.style_dimension
        #4096
        self.text_dim = args.text_dimension
        
        ##########MTST Fusion##########
        self.modality_fusion = []
        self.num_blocks_lst = {}
        self.num_heads_lst = {}
        self.hidden_dimension_lst = {}
        self.hidden_dropout_lst = {}
    

        if self.use_token:
            self.token = torch.nn.Embedding(self.num_items+2, self.input_dim, padding_idx=0)
            self.modality_fusion.append('token')
            self.num_blocks_lst['token'] = args.token_num_blocks
            self.num_heads_lst['token'] = args.token_num_heads
            self.hidden_dimension_lst['token'] = args.token_hidden_dimension
            self.hidden_dropout_lst['token'] = self.hidden_dropout
            self.representation_linear_token = nn.Linear(self.input_dim, self.output_dim)


        if self.use_style:
            if self.style_dim != self.input_dim:
                self.style_embedding_layer = nn.Linear(self.style_dim, self.input_dim)
            self.modality_fusion.append('style')
            self.num_blocks_lst['style'] = args.style_num_blocks
            self.num_heads_lst['style'] = args.style_num_heads
            self.hidden_dimension_lst['style'] = args.style_hidden_dimension
            self.hidden_dropout_lst['style'] = args.style_hidden_dropout
            self.representation_linear_style = nn.Linear(self.input_dim, self.output_dim)

            
        if self.use_text:
            if self.text_dim != self.input_dim:
                self.text_embedding_layer = nn.Linear(self.text_dim, self.input_dim)
            self.modality_fusion.append('text')
            self.num_blocks_lst['text'] = args.text_num_blocks
            self.num_heads_lst['text'] = args.text_num_heads
            self.hidden_dimension_lst['text'] = args.text_hidden_dimension
            self.hidden_dropout_lst['text'] = args.text_hidden_dropout
            self.representation_linear_text = nn.Linear(self.input_dim, self.output_dim)
            
            if args.text_embedding_len != 1:
                #gatetext_emb
                self.gate_text = nn.Linear(self.text_dim*(args.text_embedding_len-1), args.text_embedding_len-1)
                self.softmax_text = nn.Softmax(dim = -1)
                self.gates_text = None
                                
                if self.text_dim != self.input_dim:
                    self.prompt_embedding_layer = nn.Linear(self.text_dim, self.input_dim)
                
                self.modality_fusion.append('prompt')
                self.num_blocks_lst['prompt'] = args.text_num_blocks
                self.num_heads_lst['prompt'] = args.text_num_heads
                self.hidden_dimension_lst['prompt'] = args.text_hidden_dimension
                self.hidden_dropout_lst['prompt'] = args.text_hidden_dropout
                self.representation_linear_prompt = nn.Linear(self.input_dim, self.output_dim)
            
         
        if self.use_price:
            self.modality_fusion.append('saleprice')
            self.num_blocks_lst['saleprice'] = args.price_num_blocks
            self.num_heads_lst['saleprice'] = args.price_num_heads
            self.hidden_dimension_lst['saleprice'] = args.price_hidden_dimension
            self.hidden_dropout_lst['saleprice'] = args.price_hidden_dropout
            self.representation_linear_saleprice = nn.Linear(self.input_dim, 1)
            #self.saleprice = torch.nn.Embedding(self.num_items+2, self.input_dim, padding_idx=0)


            
        #cloze
        self.cloze = nn.ModuleDict()
        for modality in self.modality_fusion:
            self.cloze[modality] = CLOZEWrapper((1, 1, self.input_dim))
            
        #position
        self.pos_emb = nn.ModuleDict()
        for modality in self.modality_fusion:
            self.pos_emb[modality] = AddPositionEmbs(self.max_len+1, self.input_dim)
            
        
        #time-algned share token
        tst_num = 1 #time-algned share tokens per time
        self.tst = nn.Parameter(torch.zeros(1, (self.max_len+1) * tst_num, self.input_dim))
        nn.init.xavier_uniform_(self.tst)
        
        self.fusionencoder = MTSTEncoder(
            dmodel = self.input_dim,
            fusion_num_layers = self.fusion_num_blocks,
            num_layers_lst = self.num_blocks_lst,
            num_heads_lst = self.num_heads_lst,
            num_ffdim_lst = self.hidden_dimension_lst,
            hidden_dropout_lst = self.hidden_dropout_lst,
            seq_len = self.max_len+1,
            dropout_rate = self.dropout,
            modality_fusion = self.modality_fusion
        )
        

    def temporal_encode(self, x, modality): # shape = (bs, 20, 512)
        temporal_dims = x.shape[1]
        cloze = self.cloze[modality]().expand(x.shape[0], -1, -1)
        x = torch.cat([x, cloze], dim=1)
        x = self.pos_emb[modality](x)
        return x, temporal_dims
            
    
    def forward(self, seqs, seqs_all):
        seqs_all = seqs_all.to(device)

        
        #x[modality]
        x = {}
        if self.use_token:
            x['token'] = self.token(seqs.clone().detach().to(dtype=torch.long, device=device))

        if self.use_style:
            if self.style_dim != self.input_dim:
                 x['style'] = self.style_embedding_layer(seqs_all[:,:,0:self.style_dim])
            else:
                 x['style'] = seqs_all[:,:,0:self.style_dim]
              
                    
        if self.use_text:
            #have prompt text
            if self.args.text_embedding_len != 1:
                if self.use_style:
                    if self.text_dim != self.input_dim:
                        x['text'] = self.text_embedding_layer(seqs_all[:,:,self.style_dim:self.style_dim + self.text_dim])
                    else:
                        x['text'] = seqs_all[:,:,self.style_dim:self.style_dim + self.text_dim]
                else:
                    if self.text_dim != self.input_dim:
                        x['text'] = self.text_embedding_layer(seqs_all[:,:,0:self.text_dim])
                    else:
                        x['text'] = seqs_all[:,:,0:self.text_dim]
                
                
                # concat all prompt text embedding
                text_embeddings = []

                for i in range(1, self.args.text_embedding_len):
                    if self.use_style:
                        text_emb = seqs_all[:, :, self.style_dim + i * self.text_dim: self.style_dim + (i + 1) * self.text_dim]
                    else:
                        text_emb = seqs_all[:, :, i * self.text_dim: (i + 1) * self.text_dim]
                    text_embeddings.append(text_emb)

                text_concat = torch.cat(text_embeddings, dim=2)  # (batch_size, seq_len, text_dim * prompt_embedding_len)

                # gating 
                gates = self.gate_text(text_concat)
                gates = self.softmax_text(gates)
                self.gates_text = gates

                weighted_text_embeddings = []
                for i in range(self.args.text_embedding_len-1):
                    text_emb = text_embeddings[i] * gates[:, :, i:i+1]
                    weighted_text_embeddings.append(text_emb)

                if self.text_dim != self.input_dim:
                    x['prompt'] = self.prompt_embedding_layer(torch.sum(torch.stack(weighted_text_embeddings, dim=2), dim=2))
                else:
                    x['prompt'] = torch.sum(torch.stack(weighted_text_embeddings, dim=2), dim=2)  # (batch_size, seq_len, text_dim)
            
            #no prompt text
            else:
                if self.use_style:
                    if self.text_dim != self.input_dim:
                        x['text'] = self.text_embedding_layer(seqs_all[:,:,self.style_dim:self.style_dim + self.text_dim])
                    else:
                        x['text'] = seqs_all[:,:,self.style_dim:self.style_dim + self.text_dim]
                else:
                    if self.text_dim != self.input_dim:
                        x['text'] = self.text_embedding_layer(seqs_all[:,:,0:self.text_dim])
                    else:
                        x['text'] = seqs_all[:,:,0:self.text_dim]

        if self.use_price:
            x['saleprice'] = seqs_all[:,:,-1].unsqueeze(-1).type(torch.FloatTensor).to(device).expand(-1, -1, self.input_dim)
            #saleprice = self.saleprice(seqs.clone().detach().to(dtype=torch.long, device=device))
            #x['saleprice'] = saleprice * x['saleprice'] 


            
        #Add CLOZE & POSITION
        
        temporal_dims = {}
        for modality in self.modality_fusion:
            x[modality], temporal_dims[modality] = self.temporal_encode(x[modality], modality)
            
        # Create key_padding_mask for the original seqs
        key_padding_mask = seqs == self.pad_id
        # Add a column of False to accommodate the new cloze token
        key_padding_mask = torch.cat([key_padding_mask, torch.zeros(seqs.shape[0], 1, dtype=torch.bool, device=device)], dim=1)


        tst = self.tst.expand(seqs.shape[0], -1, -1)
 
        x = self.fusionencoder(x, tst, key_padding_mask)
        
        
        x_out = {}
        counter =  0
        #get z_cloze
        for modality in self.modality_fusion:
            x_out[modality] = x[:, counter+temporal_dims[modality]:counter+temporal_dims[modality]+1, :] 
            counter += temporal_dims[modality] + 1

        for modality in self.modality_fusion:
            if modality == "token":
                x_out[modality] = self.representation_linear_token(x_out[modality])
            elif modality == "style":
                x_out[modality] = self.representation_linear_style(x_out[modality])
            elif modality == "text":
                x_out[modality] = self.representation_linear_text(x_out[modality])
            elif modality == "prompt":
                x_out[modality] = self.representation_linear_prompt(x_out[modality])
            elif modality == "saleprice":
                x_out[modality] = self.representation_linear_saleprice(x_out[modality])
        
        final_emb = torch.cat([x_out[mod] for mod in self.modality_fusion], dim=2)        

        return final_emb

