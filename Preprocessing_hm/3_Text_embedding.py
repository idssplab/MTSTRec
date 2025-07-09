import pickle
import csv
import torch
import transformers 
from transformers import AutoTokenizer, AutoModel, AutoModelForCausalLM, pipeline
import os
from huggingface_hub import notebook_login
from tqdm import tqdm


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
#MODE = 'llama3'
#MODE = 'twllama'
#MODE = 'mediatek'
#MODE = 'taide'
MODE = 'llama3.1'



###What Language?###
if MODE == 'llama3' or MODE == 'llama3.1':
    lg = "EN"
else:
    lg = "CH"
    

    
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

if Inst == True:
    if dname == "hm":
        if lg == "EN":
            Inst_text = 'H&M is a fashion brand offering a diverse range of products, from unique designer collaborations and functional sportswear to affordable wardrobe essentials, beauty products, and accessories. '
        else:
            Inst_text = 'H&M是一個提供從獨特的設計師聯名款、功能性運動裝，到實惠的基本款衣物、美容產品和配件的時尚品牌。'
else:
    Inst_text = ''


# HuggingFace Login for llama 3 & MediaTek
# notebook_login()

# +
#Read product detail file
#0_productid, 1_productimageurl, 2_producttitle, 3_productdescription, 4_price, 5_pricegap
file = open(ProductDictFile, 'rb')
productDetails = pickle.load(file)
productNames = list(productDetails[i][0] for i in range(len(productDetails)))

print(f'Total Product Number = {len(productDetails)}')
print(productNames[0])

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

def IE_filereader():
    file = open(importantneighborFile, 'rb')
    IE = pickle.load(file)
    IE_texts = {}
    for k, items in enumerate(IE):
        tmp = ''
        for item in items:
            if lg == "EN":
                tmp += f'Title: ‘{productDetails[int(item)-1][2]}’, Description: ‘{productDetails[int(item)-1][3]}’; '
            else:
                tmp += f'標題：「{productDetails[int(item)-1][2]}」，描述：「{productDetails[int(item)-1][3]}」; '
        IE_texts[k] = tmp
    return IE_texts


def product_text_dict(TYPE, subTYPE):
    product_texts = {}
    if TYPE == "Title_Description":
        if lg == 'EN':
            if subTYPE == 'TD':
                for l in range(len(productNames)):
                    product_texts[indexDictionary[productNames[l]]] = f'Title: {productDetails[l][2]}, Description: {productDetails[l][3]}.'
            elif subTYPE == 'T':
                for l in range(len(productNames)):
                    product_texts[indexDictionary[productNames[l]]] = f'Title: {productDetails[l][2]}.'
            elif subTYPE == 'D':
                for l in range(len(productNames)):
                    product_texts[indexDictionary[productNames[l]]] = f'Description: {productDetails[l][3]}.'
            else:
                product_texts = None
        else:
            #Chinese
            if subTYPE == 'TD':
                for l in range(len(productNames)):
                    product_texts[indexDictionary[productNames[l]]] = f'商品標題: {productDetails[l][2]}, 商品描述: {productDetails[l][3]}。'
            elif subTYPE == 'T':
                for l in range(len(productNames)):
                    product_texts[indexDictionary[productNames[l]]] = f'商品標題: {productDetails[l][2]}。'
            elif subTYPE == 'D':
                for l in range(len(productNames)):
                    product_texts[indexDictionary[productNames[l]]] = f'商品描述: {productDetails[l][3]}。'
            else:
                product_texts = None   
    elif TYPE == "Basic":
        if lg == 'EN':
            if subTYPE == 'Paraphrase':
                for l in range(len(productNames)):
                    product_texts[indexDictionary[productNames[l]]] = f'{Inst_text}The title and description of an product is as follows: Title:‘{productDetails[l][2]}’, Description:‘{productDetails[l][3]}’, paraphrase it.'
            elif subTYPE == 'Tags':
                for l in range(len(productNames)):
                    product_texts[indexDictionary[productNames[l]]] = f'{Inst_text}The title and description of an product is as follows: Title:‘{productDetails[l][2]}’, Description:‘{productDetails[l][3]}’, summarize it with tags.'
            elif subTYPE == 'Guess':
                for l in range(len(productNames)):
                    product_texts[indexDictionary[productNames[l]]] = f'{Inst_text}The title and description of an product is as follows: Title:‘{productDetails[l][2]}’, Description:‘{productDetails[l][3]}’. If a consumer purchases this product, what other products might they be interested in?'
            else:
                product_texts = None
        else:
            #Chinese
            if subTYPE == 'Paraphrase':
                for l in range(len(productNames)):
                    product_texts[indexDictionary[productNames[l]]] = f'{Inst_text}以下是商品的標題和描述：標題：「{productDetails[l][2]}」，描述：「{productDetails[l][3]}」，請改寫它。'
            elif subTYPE == 'Tags':
                for l in range(len(productNames)):
                    product_texts[indexDictionary[productNames[l]]] = f'{Inst_text}以下是商品的標題和描述：標題：「{productDetails[l][2]}」，描述：「{productDetails[l][3]}」，請用標籤總結它。'
            elif subTYPE == 'Guess':
                for l in range(len(productNames)):
                    product_texts[indexDictionary[productNames[l]]] = f'{Inst_text}以下是商品的標題和描述：標題：「{productDetails[l][2]}」，描述：「{productDetails[l][3]}」。如果消費者購買了這個商品，他還會對哪些其他商品感興趣呢？'
            else:
                product_texts = None    
    elif TYPE == "Rec":
        if lg == 'EN':
            if subTYPE == 'Paraphrase':
                for l in range(len(productNames)):
                    product_texts[indexDictionary[productNames[l]]] = f'{Inst_text}The title and description of an product is as follows: Title:‘{productDetails[l][2]}’, Description:‘{productDetails[l][3]}’, what else should I say if I want to recommend it to others?'
            elif subTYPE == 'Tags':
                for l in range(len(productNames)):
                    product_texts[indexDictionary[productNames[l]]] = f'{Inst_text}The title and description of an product is as follows: Title:‘{productDetails[l][2]}’, Description:‘{productDetails[l][3]}’, what tags should I use if I want to recommend it to others?'
            else:
                product_texts = None
        else:
            #Chinese
            if subTYPE == 'Paraphrase':
                for l in range(len(productNames)):
                    product_texts[indexDictionary[productNames[l]]] = f'{Inst_text}以下是商品的標題和描述：標題：「{productDetails[l][2]}」，描述：「{productDetails[l][3]}」，如果我想向別人推薦它，還應該說些什麼？'
            elif subTYPE == 'Tags':
                for l in range(len(productNames)):
                    product_texts[indexDictionary[productNames[l]]] = f'{Inst_text}以下是商品的標題和描述：標題：「{productDetails[l][2]}」，描述：「{productDetails[l][3]}」，如果我想向別人推薦它，應該使用哪些標籤？'
            else:
                product_texts = None  

    elif TYPE == "Engagement":
        IE_texts = IE_filereader()
        if lg == 'EN':
            for l in range(len(productNames)):
                product_texts[indexDictionary[productNames[l]]] = f'{Inst_text}Summarize the commonalities among the following titles and descriptions: Title:‘{productDetails[l][2]}’, Description:‘{productDetails[l][3]}’; {IE_texts[l]}.' 
        else:
            #Chinese
            for l in range(len(productNames)):
                product_texts[indexDictionary[productNames[l]]] = f'{Inst_text}概述以下標題和描述之間的共同點：標題：「{productDetails[l][2]}」，描述：「{productDetails[l][3]}」; {IE_texts[l]}。'

    elif TYPE == "RecEngagement":
        IE_texts = IE_filereader()
        if lg == 'EN':
            for l in range(len(productNames)):
                product_texts[indexDictionary[productNames[l]]] = f'{Inst_text}The title and description of an product is as follows: Title:‘{productDetails[l][2]}’, Description:‘{productDetails[l][3]}’, what else should I say if I want to recommend it to others? This content is considered to hold some similar attractive characteristics as the following titles and descriptions: {IE_texts[l]}.' 
        else:
            #Chinese
            for l in range(len(productNames)):
                product_texts[indexDictionary[productNames[l]]] = f'{Inst_text}以下是商品的標題和描述：標題：「{productDetails[l][2]}」，描述：「{productDetails[l][3]}」，如果我想向別人推薦它，還應該說些什麼？ 此內容被認為與以下標題和描述具有一些相似的吸引特點：{IE_texts[l]}。'
    return product_texts


# ## LLM

def LLM(MODE, product_texts, TYPE, subTYPE):
    if MODE == 'llama3':
        model_name = "meta-llama/Meta-Llama-3-8B"
    elif MODE == 'twllama':    
        model_name = "yentinglin/Llama-3-Taiwan-8B-Instruct"
    elif MODE == 'mediatek':
        model_name = "MediaTek-Research/Breeze-7B-Instruct-v0_1"
    elif MODE == 'taide':    
        model_name = "taide/TAIDE-LX-7B"
    elif MODE == 'llama3.1':
        model_name = "meta-llama/Meta-Llama-3.1-8B-Instruct"
    
    #tokenizer
    if MODE == 'taide':
        tokenizer = AutoTokenizer.from_pretrained(model_name, use_fast=False)
    else:
        tokenizer = AutoTokenizer.from_pretrained(model_name)
    
    #model
    model = AutoModel.from_pretrained(model_name)
    
    #text_emb
    text_emb=[]
    for i in tqdm(range(1, len(product_texts)+1)):
        text = product_texts[i]
        seq_ids = tokenizer(text, return_tensors='pt')["input_ids"]
        embedding = model(seq_ids)["last_hidden_state"].mean(axis=[0,1]).detach().numpy()
        text_emb.append(torch.tensor(embedding))
        
    text_matrixes = torch.zeros([len(text_emb) + 1, 4096])
    text_matrixes[0] = torch.zeros(4096)

    for l in range(len(text_emb)):
        text_matrixes[l+1] = text_emb[l]
    print(text_matrixes.size())
    
    #save
    if Inst == True:
        if TYPE == "Title_Description":
            file = open(f'{dataset_path}/{dname}/Text_Embedding_{TYPE}_{subTYPE}_{MODE}', 'wb')
        elif TYPE != "Engagement" and TYPE != "RecEngagement":
            file = open(f'{dataset_path}/{dname}/Text_Embedding_{TYPE}_{subTYPE}_I_{MODE}', 'wb')
        else:
            file = open(f'{dataset_path}/{dname}/Text_Embedding_{TYPE}_I_{MODE}', 'wb')
    else:
        if TYPE != "Engagement" and TYPE != "RecEngagement":
            file = open(f'{dataset_path}/{dname}/Text_Embedding_{TYPE}_{subTYPE}_{MODE}', 'wb')
        else:
            file = open(f'{dataset_path}/{dname}/Text_Embedding_{TYPE}_{MODE}', 'wb')

    pickle.dump(text_matrixes, file)
    file.close()
    print("Inst = " + str(Inst))
    print(f'SAVE COMPLETE! {dataset_path}/{dname}/Text_Embedding_{TYPE}_{subTYPE}_{MODE}')


# ## MAIN

for i, t in enumerate(TYPE_list):
    for st in subTYPE_list[i]:
        print(t)
        print(st)
        product_texts = product_text_dict(t, st)
        LLM(MODE, product_texts, t, st)
