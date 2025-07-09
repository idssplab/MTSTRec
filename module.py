# +
import math
import numpy as np
import torch
import torch.nn as nn
import torch.nn.init as init
import torch.nn.functional as F

from utils import *
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")


# -

# ## Transformer Encoder Modules
# * FeedForword
# * EncoderBlock

class FeedForward(torch.nn.Module):
    def __init__(self, d_model, d_hidden, dropout_rate):
        super(FeedForward, self).__init__()

        self.conv1 = torch.nn.Conv1d(d_model, d_hidden, kernel_size=1)
        self.dropout1 = torch.nn.Dropout(p=dropout_rate)
        self.gelu = torch.nn.GELU()
        self.conv2 = torch.nn.Conv1d(d_hidden, d_model, kernel_size=1)
        self.dropout2 = torch.nn.Dropout(p=dropout_rate)
        

    def forward(self, inputs):        
        outputs = self.dropout2(self.conv2(self.dropout1(self.gelu(self.conv1(inputs.transpose(-1, -2))))))
        outputs = outputs.transpose(-1, -2) # as Conv1D requires (N, C, Length)    
        outputs += inputs
        return outputs


class EncoderBlock(nn.Module):
    def __init__(self, d_model, hidden, attn_heads, dropout, hidden_dropout):
        super().__init__()
        self.attention_layernorm = torch.nn.LayerNorm(d_model, eps=1e-8)
        self.attention_layer = torch.nn.MultiheadAttention(d_model, attn_heads, dropout)
        self.forward_layernorm = torch.nn.LayerNorm(d_model, eps=1e-8)
        self.forward_layer = FeedForward(d_model, hidden, hidden_dropout)

    def forward(self, x, key_padding_mask, attention_mask = "none"):
        attn_mask = torch.nn.Transformer.generate_square_subsequent_mask(x.shape[1]).to(device)
        x = torch.transpose(x, 0, 1)
        Q = self.attention_layernorm(x)
        if attention_mask == "none":
            mha_outputs, _ = self.attention_layer(Q, x, x, key_padding_mask = key_padding_mask)
        else:
            mha_outputs, _ = self.attention_layer(Q, x, x, key_padding_mask = key_padding_mask, attn_mask = attention_mask)
        x = Q + mha_outputs     
        x = torch.transpose(x, 0, 1)
        x = self.forward_layernorm(x)
        x = self.forward_layer(x)
        return x


# ## MTST Modules
# * CLOZEWrapper
# * AddPositionEmbs
# * MTSTEncoder

class CLOZEWrapper(nn.Module):
    def __init__(self, size):
        super(CLOZEWrapper, self).__init__()
        self.param = nn.Parameter(torch.zeros(size))

    def forward(self):
        return self.param


class AddPositionEmbs(nn.Module):
    def __init__(self, num_embeddings, embedding_dim):
        super(AddPositionEmbs, self).__init__()
        self.pos_embedding = nn.Parameter(torch.zeros(1, num_embeddings, embedding_dim))
        nn.init.xavier_uniform_(self.pos_embedding)  

    def forward(self, x):
        return x + self.pos_embedding.to(device)


class MTSTEncoder(nn.Module):
    def __init__(self, dmodel, fusion_num_layers, num_layers_lst, num_heads_lst, num_ffdim_lst, hidden_dropout_lst, seq_len = 21, dropout_rate=0.1, 
                 modality_fusion=('token','style',)):
        super().__init__()
        self.dmodel = dmodel
        self.fusion_num_layers = fusion_num_layers
        self.num_layers_lst = num_layers_lst
        self.num_heads_lst = num_heads_lst
        self.num_ffddim_lst = num_ffdim_lst
        self.dropout_rate = dropout_rate
        self.hidden_dropout_lst = hidden_dropout_lst
        self.modality_fusion = modality_fusion

        self.encoders = nn.ModuleDict()
        self.num_layers_max = 0
        self.num_layers_total = {}
        
        
        #build encoder for different features
        for modality in modality_fusion:
            # fusion layers + self-attention layers
            self.num_layers_total[modality] = self.fusion_num_layers + self.num_layers_lst[modality]
            self.encoders[modality] = nn.ModuleList([EncoderBlock(
                        d_model = self.dmodel,
                        hidden = self.num_ffddim_lst[modality],
                        attn_heads = self.num_heads_lst[modality],
                        dropout = self.dropout_rate,
                        hidden_dropout = self.hidden_dropout_lst[modality]
                    ) for lyr in range(self.num_layers_total[modality])
                ])
            if self.num_layers_max < self.num_layers_total[modality]:
                self.num_layers_max = self.num_layers_total[modality]

        self.encoder_norm = nn.LayerNorm(normalized_shape = self.dmodel)
        
        

    def create_attention_mask(self, sequence_length, share_tokens_length, key_padding_mask, mode = "fusion"):
        #for pad masking
        combined_key_padding_mask = torch.cat([key_padding_mask,torch.zeros(key_padding_mask.shape[0], share_tokens_length, dtype=torch.bool, device=device)], dim=1)
        #how many share tokens per time step
        token_num = share_tokens_length // sequence_length
        
        tl = sequence_length + share_tokens_length
        attention_mask = torch.zeros((tl, tl), dtype=torch.bool, device=device)
        
        #every seq token has corrensponding time-algned share token 
        if mode == "first fusion":
            # seq tokens can see each other but not share tokens 
            attention_mask[:sequence_length, sequence_length:] = True
            # Share tokens can see their corresponding seq tokens but not each other
            attention_mask[sequence_length:, :] = True
            for i in range(sequence_length):
                attention_mask[sequence_length + token_num * i:sequence_length + token_num * (i + 1), i] = False
                attention_mask[sequence_length + token_num * i:sequence_length + token_num * (i + 1), sequence_length + token_num * i:sequence_length + token_num * (i + 1)] = False

        else:
            attention_mask[:sequence_length, sequence_length:] = True
            # seq tokens can see each other and their corresponding share tokens
            for i in range(sequence_length):
                attention_mask[i, sequence_length + token_num * i:sequence_length + token_num * (i + 1)] = False
                      
            attention_mask[sequence_length:, :] = True
            # Share tokens can see their corresponding seq tokens but not each other
            for i in range(sequence_length):
                attention_mask[sequence_length + token_num * i:sequence_length + token_num * (i + 1), i] = False
                attention_mask[sequence_length + token_num * i:sequence_length + token_num * (i + 1), sequence_length + token_num * i:sequence_length + token_num * (i + 1)] = False

    
        return combined_key_padding_mask, attention_mask

        
    def forward(self, x, tst, key_padding_mask):
        for lyr in range(self.num_layers_max):
            #self-attention encoder
            if lyr < self.num_layers_max - self.fusion_num_layers or len(self.modality_fusion) == 1:
                for modality in self.modality_fusion:
                    modality_lyr = self.num_layers_max - self.num_layers_total[modality]
                    if lyr < modality_lyr:
                        pass
                    else:
                        encoders = self.encoders[modality]
                        x[modality] = encoders[modality_lyr - lyr](x[modality], key_padding_mask)
            #fusion encoder with TST
            else:  
                tst_weight = []
                for modality in self.modality_fusion:
                    modality_lyr = self.num_layers_max - self.num_layers_total[modality]
                    encoders = self.encoders[modality]
                    t_mod = x[modality].shape[1]
                    in_mod = torch.cat([x[modality], tst], dim=1)
                    sequence_length = x[modality].shape[1]
                    share_tokens_length = tst.shape[1]
                    #first fusion
                    if lyr == self.num_layers_max - self.fusion_num_layers:
                        combined_key_padding_mask, attention_mask = self.create_attention_mask(sequence_length, share_tokens_length, key_padding_mask, mode = "first fusion")
                    #start from second layer of fusion
                    else:
                        combined_key_padding_mask, attention_mask = self.create_attention_mask(sequence_length, share_tokens_length, key_padding_mask, mode = "fusion")
                    out_mod = encoders[modality_lyr - lyr](in_mod, combined_key_padding_mask, attention_mask)
                    x[modality] = out_mod[:, :t_mod]
                    tst_weight.append(out_mod[:, t_mod:])                      
                tst = torch.mean(torch.stack(tst_weight, dim=-1), dim=-1)

        x_out = [x[modality] for modality in self.modality_fusion]
        x_out = torch.cat(x_out, dim=1)
        encoded = self.encoder_norm(x_out)
        return encoded
