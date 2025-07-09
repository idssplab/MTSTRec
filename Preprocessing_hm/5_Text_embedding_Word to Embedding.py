import pickle
import csv
import torch
import os
from tqdm import tqdm
import time
import transformers 
from transformers import AutoTokenizer, AutoModel, BertTokenizer, BertModel, RobertaTokenizer, RobertaModel


# +
###What is the dataset name?###
dname = "hm"

###What is the Dataset folder path?###
dataset_path = '../Dataset'

###What is the ProductDictName path?###
ProductDictFile = f'{dataset_path}/{dname}/Product/{dname}_productdetail_trousers'
    
###What is the IndexDictionary path?###
indexDictionaryFile = f'{dataset_path}/{dname}/Product/idtoindex_trousers_{dname}.csv'

###What is the Important Neighbors path?###
#importantneighborFile = f'{dataset_path}/{dname}/Product/importantneighbor'


#Which Type of LLM###
#MODE = 'gpt'
#MODE = 'llama3'
#MODE = 'llama3.1'
MODE = 'bert'
#MODE = 'roberta'

fileMODE = 'gpt'

    
#Which Type of Text Embedding###
'''
Title/ Descriptions
Basic – Paraphrase/ Tags/ Guess
Rec – Paraphrase / Tags
Engagement - Important Neighbors, Commonalities (based on session count)
Rec_Engagement
'''
#TYPE_list = ["Title_Description", "Basic", "Rec", "Engagement", "RecEngagement"]
#subTYPE_list = [["T","D","TD"],["Paraphrase", "Tags", "Guess"],["Paraphrase", "Tags"],["NoInst"],["NoInst"]]

TYPE_list = ["Title_Description"]
subTYPE_list = [["TD"]]


###Instruction Text?###

Inst = True

# -

def Read_Word_File(TYPE,subTYPE):
    #open Word File
    if Inst == True:
        if TYPE == "Title_Description":
            file = open(f'{dataset_path}/{dname}/Text_Word_{TYPE}_{subTYPE}_{fileMODE}', 'rb')
        elif TYPE != "Engagement" and TYPE != "RecEngagement":
            file = open(f'{dataset_path}/{dname}/Text_Word_{TYPE}_{subTYPE}_I_{fileMODE}', 'rb')
        else:
            file = open(f'{dataset_path}/{dname}/Text_Word_{TYPE}_I_{fileMODE}', 'rb')
    else:
        if TYPE != "Engagement" and TYPE != "RecEngagement":
            file = open(f'{dataset_path}/{dname}/Text_Word_{TYPE}_{subTYPE}_{fileMODE}', 'rb')
        else:
            file = open(f'{dataset_path}/{dname}/Text_Word_{TYPE}_{fileMODE}', 'rb')

    productWords = pickle.load(file)
    return productWords


# +
#########Product Index List from 1 + 'xxxxxxxx'product#########
#indexDictionary[product name] = index
## Product Index 
with open(indexDictionaryFile, newline='') as f:
    reader = csv.reader(f)
    idtoindex = list(reader)

indexDictionary = {}

for l in range(len(idtoindex)):
    indexDictionary[str(idtoindex[l][0])] = int(idtoindex[l][1])
print(len(indexDictionary))
# -


def LLM2Embeddings(MODE, product_texts, TYPE, subTYPE):
    if MODE == 'gpt':
        MODEL ='text-embedding-3-large'
        outputdim = 3072

        #text_emb
        text_emb=[]
        for i in tqdm(range(0, len(product_texts))):
            text = product_texts[i]
            response = client.embeddings.create(model=MODEL, input=text, encoding_format="float")
            embedding = response.data[0].embedding
            text_emb.append(torch.tensor(embedding))
        
    elif MODE == 'llama3':
        MODEL = 'meta-llama/Meta-Llama-3-8B'
        tokenizer = AutoTokenizer.from_pretrained(MODEL)
        model = AutoModel.from_pretrained(MODEL)
        outputdim = 4096
        
        #text_emb
        text_emb=[]
        for i in tqdm(range(0, len(product_texts))):
            text = product_texts[i]
            seq_ids = tokenizer(text, return_tensors='pt')["input_ids"]
            embedding = model(seq_ids)["last_hidden_state"].mean(axis=[0,1]).detach().numpy()
            text_emb.append(torch.tensor(embedding))
    
    elif MODE == 'llama3.1':
        MODEL = 'meta-llama/Meta-Llama-3.1-8B-Instruct'
        tokenizer = AutoTokenizer.from_pretrained(MODEL)
        model = AutoModel.from_pretrained(MODEL)
        outputdim = 4096
        
        #text_emb
        text_emb=[]
        for i in tqdm(range(0, len(product_texts))):
            text = product_texts[i]
            seq_ids = tokenizer(text, return_tensors='pt')["input_ids"]
            embedding = model(seq_ids)["last_hidden_state"].mean(axis=[0,1]).detach().numpy()
            text_emb.append(torch.tensor(embedding))
            
    elif MODE == 'bert':
        #   bert-base-chinese
        MODEL = 'bert-base-chinese'
        tokenizer = BertTokenizer.from_pretrained('bert-base-chinese')
        model = BertModel.from_pretrained(MODEL,return_dict = True)
        outputdim = 768
        
        #text_emb
        text_emb=[]
        for i in tqdm(range(0, len(product_texts))):
            text = product_texts[i]
            seq_ids = tokenizer(text, return_tensors='pt')
            embedding = model(**seq_ids).last_hidden_state.mean(axis=[0,1]).detach().numpy()
            text_emb.append(torch.tensor(embedding))

    elif MODE == 'roberta':
        # roberta-base-chinese
        MODEL = 'hfl/chinese-roberta-wwm-ext'
        tokenizer = BertTokenizer.from_pretrained(MODEL)
        model = BertModel.from_pretrained(MODEL, return_dict=True)
        outputdim = 768

        # text_emb
        text_emb = []
        for i in tqdm(range(0, len(product_texts))):
            text = product_texts[i]
            seq_ids = tokenizer(text, return_tensors='pt', padding=True, truncation=True, max_length=512)
            embedding = model(**seq_ids).last_hidden_state.mean(axis=[0,1]).detach().numpy()
            text_emb.append(torch.tensor(embedding))

            
    text_matrixes = torch.zeros([len(text_emb) + 1, outputdim])
    text_matrixes[0] = torch.zeros(outputdim)

    for l in range(len(text_emb)):
        text_matrixes[l+1] = text_emb[l]
    print(text_matrixes.size())
    
    #save
    if Inst == True:
        if TYPE == "Title_Description":
            file = open(f'{dataset_path}/{dname}/Text_Word2Embedding_{TYPE}_{subTYPE}_{MODE}', 'wb')
        elif TYPE != "Engagement" and TYPE != "RecEngagement":
            file = open(f'{dataset_path}/{dname}/Text_Word2Embedding_{TYPE}_{subTYPE}_I_{MODE}', 'wb')
        else:
            file = open(f'{dataset_path}/{dname}/Text_Word2Embedding_{TYPE}_I_{MODE}', 'wb')
    else:
        if TYPE != "Engagement" and TYPE != "RecEngagement":
            file = open(f'{dataset_path}/{dname}/Text_Word2Embedding_{TYPE}_{subTYPE}_{MODE}', 'wb')
        else:
            file = open(f'{dataset_path}/{dname}/Text_Word2Embedding_{TYPE}_{MODE}', 'wb')
            
        

    pickle.dump(text_matrixes, file)
    file.close()
    print("Inst = " + str(Inst))
    print(f'SAVE COMPLETE! {dataset_path}/{dname}/Text_Word2Embedding_{TYPE}_{subTYPE}_{MODE}')

# ## MAIN

for i, t in enumerate(TYPE_list):
    for st in subTYPE_list[i]:
        print(t)
        print(st)
        product_words = Read_Word_File(t,st)
        LLM2Embeddings(MODE, product_words, t, st)


