# +
import pickle
import csv
import torch
import os
import openai
from openai import OpenAI
from tqdm import tqdm
import time
import backoff
import transformers 
from transformers import AutoTokenizer, AutoModel, pipeline
from huggingface_hub import notebook_login

# +
###What is the dataset name?###
dname = "hm"

###What is the Dataset folder path?###
dataset_path = '../Dataset'

###What is the ProductDictName path?###
ProductDictFile = f'{dataset_path}/{dname}/Product/{dname}_productdetail'

###What is the IndexDictionary path?###
indexDictionaryFile = f'{dataset_path}/{dname}/Product/idtoindex_{dname}.csv'

###What is the Important Neighbors path?###
#importantneighborFile = f'{dataset_path}/{dname}/Product/importantneighbor'


#Which Type of LLM###
#MODE = 'llama3'
MODE = 'llama3.1'
#MODE = 'gpt'


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

TYPE_list = ["Basic"]
subTYPE_list = [["Paraphrase", "Tags"]]


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

def get_time(start_time, end_time):
    time_g = int(end_time - start_time)
    hour = int(time_g / 3600)
    minu = int(time_g / 60) % 60
    secon = time_g % 60
    return hour, minu, secon


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
                    system = f'{Inst_text}You will be provided with the product title and description sold on this e-commerce website. Your task is to paraphrase them.'
                    user = f'Title:‘{productDetails[l][2]}’，Description:‘{productDetails[l][3]}’.'
                    product_texts[indexDictionary[productNames[l]]] = [system, user]
            elif subTYPE == 'Tags':
                for l in range(len(productNames)):
                    system = f'{Inst_text}You will be provided with the product title and description sold on this e-commerce website. Your task is to summarize this product using tags.'
                    user = f'Title:‘{productDetails[l][2]}’，Description:‘{productDetails[l][3]}’.'
                    product_texts[indexDictionary[productNames[l]]] = [system, user]            
            elif subTYPE == 'Guess':
                for l in range(len(productNames)):
                    system = f'{Inst_text}You will be provided with the product title and description sold on this e-commerce website. Your task is to infer what other products on the site a consumer might be interested in if they purchase this product.'
                    user = f'Title:‘{productDetails[l][2]}’，Description:‘{productDetails[l][3]}’.'
                    product_texts[indexDictionary[productNames[l]]] = [system, user]
            else:
                product_texts = None
        else:
            #Chinese
            if subTYPE == 'Paraphrase':
                for l in range(len(productNames)):
                    system = f'{Inst_text}您將獲得此電商網站販售的商品標題及敘述，您的任務是改寫它。'
                    user = f'標題：「{productDetails[l][2]}」，描述：「{productDetails[l][3]}」。'
                    product_texts[indexDictionary[productNames[l]]] = [system, user]
            elif subTYPE == 'Tags':
                for l in range(len(productNames)):
                    system = f'{Inst_text}您將獲得此電商網站販售的商品標題及敘述，您的任務是用關鍵字總結此商品。'
                    user = f'標題：「{productDetails[l][2]}」，描述：「{productDetails[l][3]}」。'
                    product_texts[indexDictionary[productNames[l]]] = [system, user]
            elif subTYPE == 'Guess':
                for l in range(len(productNames)):
                    system = f'{Inst_text}您將獲得此電商網站販售的商品標題及敘述，您的任務是推測如果消費者購買這個商品，他還會對網站哪些其他商品感興趣呢？'
                    user = f'標題：「{productDetails[l][2]}」，描述：「{productDetails[l][3]}」。'
                    product_texts[indexDictionary[productNames[l]]] = [system, user]
                    
            else:
                product_texts = None    
    elif TYPE == "Rec":
        if lg == 'EN':
            if subTYPE == 'Paraphrase':
                for l in range(len(productNames)):
                    system = f'{Inst_text}You will be provided with the product title and description sold on this e-commerce website. Your task is to tell me what else I should say if I want to recommend this product to someone.'
                    user = f'Title:‘{productDetails[l][2]}’，Description:‘{productDetails[l][3]}’.'
                    product_texts[indexDictionary[productNames[l]]] = [system, user]
            elif subTYPE == 'Tags':
                for l in range(len(productNames)):
                    system = f'{Inst_text}You will be provided with the product title and description sold on this e-commerce website. Your task is to tell me which tags should be used if I want to recommend this product to someone.'
                    user = f'Title:‘{productDetails[l][2]}’，Description:‘{productDetails[l][3]}’.'
                    product_texts[indexDictionary[productNames[l]]] = [system, user]
            else:
                product_texts = None
        else:
            #Chinese
            if subTYPE == 'Paraphrase':
                for l in range(len(productNames)):
                    system = f'{Inst_text}您將獲得此電商網站販售的商品標題及敘述，您的任務是告訴我如果我想向別人推薦此商品，還應該說些什麼？'
                    user = f'標題：「{productDetails[l][2]}」，描述：「{productDetails[l][3]}」。'
                    product_texts[indexDictionary[productNames[l]]] = [system, user]
                    
            elif subTYPE == 'Tags':
                for l in range(len(productNames)):
                    system = f'{Inst_text}您將獲得此電商網站販售的商品標題及敘述，您的任務是告訴我如果我想向別人推薦此商品，應該使用哪些標籤？'
                    user = f'標題：「{productDetails[l][2]}」，描述：「{productDetails[l][3]}」。'
                    product_texts[indexDictionary[productNames[l]]] = [system, user]            
            else:
                product_texts = None  

    elif TYPE == "Engagement":
        IE_texts = IE_filereader()
        if lg == 'EN':
            for l in range(len(productNames)):
                system = f'{Inst_text}You will be provided with the product title and description sold on this e-commerce website (the first product), as well as the titles and descriptions of related products (subsequent products). Your task is to summarize the commonalities between the titles and descriptions of these products.'
                user = f'Title:‘{productDetails[l][2]}’，Description:‘{productDetails[l][3]}’; {IE_texts[l]}.'
                product_texts[indexDictionary[productNames[l]]] = [system, user]   
        else:
            #Chinese
            for l in range(len(productNames)):
                system = f'{Inst_text}您將獲得此電商網站販售的商品標題及敘述(第一樣商品)，以及與此商品相關的商品標題及敘述(後續商品)，您的任務是概述以下商品們的標題和敘述之間的共同點。'
                user = f'標題：「{productDetails[l][2]}」，描述：「{productDetails[l][3]}」; {IE_texts[l]}。'
                product_texts[indexDictionary[productNames[l]]] = [system, user]

    elif TYPE == "RecEngagement":
        IE_texts = IE_filereader()
        if lg == 'EN':
            for l in range(len(productNames)):
                system = f'{Inst_text}You will be provided with the product title and description sold on this e-commerce website (the first product). Your task is to tell me what else I should say if I want to recommend this product to someone. This product is considered to have some similar appealing features as the subsequent products\' titles and descriptions.'
                user = f'Title:‘{productDetails[l][2]}’，Description:‘{productDetails[l][3]}’; {IE_texts[l]}.'
                product_texts[indexDictionary[productNames[l]]] = [system, user] 
        else:
            #Chinese
            for l in range(len(productNames)):
                system = f'{Inst_text}您將獲得此電商網站販售的商品標題及敘述(第一樣商品)，您的任務是告訴我如果我想向別人推薦此商品，還應該說些什麼？此商品內容被認為與後續商品們的標題和敘述具有一些相似的吸引特點。'
                user = f'標題：「{productDetails[l][2]}」，描述：「{productDetails[l][3]}」; {IE_texts[l]}。'
                product_texts[indexDictionary[productNames[l]]] = [system, user]
    return product_texts


# ## LLM
# ### h&m
# * BasicParaphrase
#     * [{'role': 'system', 'content': 'H&M is a fashion brand offering a diverse range of products, from unique designer collaborations and functional sportswear to affordable wardrobe essentials, beauty products, and accessories. You will be provided with the product title and description sold on this e-commerce website. Your task is to paraphrase them.'}, {'role': 'user', 'content': 'Title:‘Strap top’，Description:‘Jersey top with narrow shoulder straps.’.'}]
#     * The product title is: 'Strap top'\nThe product description is: 'Simple jersey top featuring delicate narrow shoulder straps.'\n or The product title is: 'Strap top'\nThe product description is: 'A casual, strap-supported jersey top with slender straps.'
# * BasicTags
#     * [{'role': 'system', 'content': 'H&M is a fashion brand offering a diverse range of products, from unique designer collaborations and functional sportswear to affordable wardrobe essentials, beauty products, and accessories. You will be provided with the product title and description sold on this e-commerce website. Your task is to summarize this product using tags.'}, {'role': 'user', 'content': 'Title:‘Strap top’，Description:‘Jersey top with narrow shoulder straps.’.'}]
#     * Tags: - Jersey Top=- Strap Top- Casual Wear- Summer Clothing- Women's Tops- Basic Tee- Fashion Essentials
# * BasicGuess
#     * [{'role': 'system', 'content': 'H&M is a fashion brand offering a diverse range of products, from unique designer collaborations and functional sportswear to affordable wardrobe essentials, beauty products, and accessories. You will be provided with the product title and description sold on this e-commerce website. Your task is to infer what other products on the site a consumer might be interested in if they purchase this product.'}, {'role': 'user', 'content': 'Title:‘Strap top’，Description:‘Jersey top with narrow shoulder straps.’.'}]
#     * 1. **High-waisted mom jeans**: A popular fashion combination that pairs well with a strap-top is high-waisted mom jeans. The loose-fitting jeans will create a relaxed, casual look with the strap top.2. **Sandals**: Strap tops are perfect for warm weather, so sandals would be a great match. A pair of flat sandals or wedges would complement the top nicely.3. **Distressed denim jacket**: Adding a denim jacket, especially a distressed one, can add an edgy touch to a strap-top outfit. It's a great layering piece for cooler evenings.4. **Shorts**: For a more casual look, consider pairing the strap top with a pair of shorts. The jersey fabric and straps will still be visible underneath the shorts, making it a great base layer for a summery outfit.5. **Heeled ankle boots**: For a dressier look, pair the strap top with heeled ankle boots. This will create a nice contrast between the relaxed top and the more polished shoes.6. **Bell-belted pants**: A flowy, bell-belted pant would create a nice silhouette with the strap top. The relaxed pants will balance out the fitted top.7. **Sunglasses**: To complete the summer look, add a pair of sunglasses. They'll add a stylish touch to the outfit and provide sun protection.These are just a few suggestions, but the possibilities are endless. The strap top is a versatile piece that can be styled in many ways, so feel free to experiment and find your own favorite combinations.
# * RecParaphrase
#     * [{'role': 'system', 'content': 'H&M is a fashion brand offering a diverse range of products, from unique designer collaborations and functional sportswear to affordable wardrobe essentials, beauty products, and accessories. You will be provided with the product title and description sold on this e-commerce website. Your task is to tell me what else I should say if I want to recommend this product to someone.'}, {'role': 'user', 'content': 'Title:‘Strap top’，Description:‘Jersey top with narrow shoulder straps.’.'}]
#     * If you want to recommend the 'Strap top' to someone, you might say:"If you're looking for a simple yet stylish top for your wardrobe, I would recommend the Strap top. It's made from soft and comfortable jersey material, making it perfect for casual wear. The narrow shoulder straps add a touch of elegance to this basic tee, making it suitable for both everyday life and a night out. Plus, it's a versatile piece that can be dressed up or down. I think this top would look great paired with high-waisted jeans for a relaxed look or with a flowy skirt for a more feminine look. Give it a try, I'm sure you'll love it!"
# * RecTags
#     * [{'role': 'system', 'content': 'H&M is a fashion brand offering a diverse range of products, from unique designer collaborations and functional sportswear to affordable wardrobe essentials, beauty products, and accessories. You will be provided with the product title and description sold on this e-commerce website. Your task is to tell me which tags should be used if I want to recommend this product to someone.'}, {'role': 'user', 'content': 'Title:‘Strap top’，Description:‘Jersey top with narrow shoulder straps.’.'}]
#     * Based on the product description, some relevant tags that can be used to recommend the product are:1. Tops2. Jersey tops3. Strap tops4. Summer clothing5. Casual wear6. Women's clothing (assuming the product is targeted towards women, based on the straps)


api_key ='your_api_key'
client = OpenAI(api_key = api_key)


def create_messages(prompt: str, system_prompt: str = None):
    messages = []
    if system_prompt:
        messages.append({"role": "system", "content": system_prompt})
    messages.append({"role": "user", "content": prompt})
    return messages


@backoff.on_exception(backoff.expo, openai.RateLimitError)
def completions_with_backoff(model,messages,temperature,max_tokens,top_p):
    response = client.chat.completions.create(model= model,messages=messages,temperature=temperature,max_tokens=max_tokens,top_p=top_p)
    return response


def LLM_word(MODE, product_texts, TYPE, subTYPE):
    if MODE == 'gpt':
        MODEL ='gpt-4o-mini'
    elif MODE == 'llama3.1':
        MODEL ="meta-llama/Meta-Llama-3.1-8B-Instruct"

    if TYPE == "Basic":
        if subTYPE == "Paraphrase":
            temp = 0.7
            maxt = 256
        elif subTYPE == "Tags":
            temp = 0.5
            maxt = 128
        elif subTYPE == "Guess":
            temp = 1
            maxt = 512
    
    elif TYPE == "Rec":
        if subTYPE == "Paraphrase":
            temp = 1
            maxt = 384
        elif subTYPE == "Tags":
            temp = 1
            maxt = 128
    else:
        temp = 1
        maxt = 384
    if MODE == 'gpt':
        answer_word=[]
        for i in tqdm(range(1, len(product_texts)+1)):
            messages = create_messages(product_texts[i][1],product_texts[i][0])
            response = completions_with_backoff(model= MODEL,messages=messages,temperature=temp,max_tokens=maxt,top_p=1)
            answer = response.choices[0].message.content
            answer_word.append(answer)
    else:
        
        pipeline = transformers.pipeline(
            "text-generation",
            model=MODEL,
            model_kwargs={"torch_dtype": torch.bfloat16},
            device_map="auto",
        )
        
        answer_word=[]
        
        for i in tqdm(range(1, len(product_texts)+1)):
            messages = create_messages(product_texts[i][1],product_texts[i][0])
            response = pipeline(
                messages,
                temperature=temp,
                max_new_tokens=maxt,
                top_p=1
            )
            answer = response[0]["generated_text"][-1]["content"]
            answer_word.append(answer)

    

    
    #save
    if Inst == True:
        if TYPE == "Title_Description":
            file = open(f'{dataset_path}/{dname}/Text_Word_{TYPE}_{subTYPE}_{MODE}', 'wb')
        elif TYPE != "Engagement" and TYPE != "RecEngagement":
            file = open(f'{dataset_path}/{dname}/Text_Word_{TYPE}_{subTYPE}_I_{MODE}', 'wb')
        else:
            file = open(f'{dataset_path}/{dname}/Text_Word_{TYPE}_I_{MODE}', 'wb')
    else:
        if TYPE != "Engagement" and TYPE != "RecEngagement":
            file = open(f'{dataset_path}/{dname}/Text_Word_{TYPE}_{subTYPE}_{MODE}', 'wb')
        else:
            file = open(f'{dataset_path}/{dname}/Text_Word_{TYPE}_{MODE}', 'wb')

    pickle.dump(answer_word, file)
    file.close()
    print("Inst = " + str(Inst))
    print(f'SAVE COMPLETE! {dataset_path}/{dname}/Text_Word_{TYPE}_{subTYPE}_{MODE}')


def TOword(MODE, product_texts, TYPE, subTYPE):
    answer_word=[]
    if TYPE == "Title_Description":
        for i in tqdm(range(1, len(product_texts)+1)):
            answer_word.append(product_texts[i])
    else:
        for i in tqdm(range(1, len(product_texts)+1)):
            answer_word.append(product_texts[i][0]+product_texts[i][1])
    
    #save
    if Inst == True:
        if TYPE == "Title_Description":
            file = open(f'{dataset_path}/{dname}/Text_Word_{TYPE}_{subTYPE}_{MODE}', 'wb')
        elif TYPE != "Engagement" and TYPE != "RecEngagement":
            file = open(f'{dataset_path}/{dname}/Text_Word_{TYPE}_{subTYPE}_I_{MODE}_base', 'wb')
        else:
            file = open(f'{dataset_path}/{dname}/Text_Word_{TYPE}_I_{MODE}_base', 'wb')
    else:
        if TYPE != "Engagement" and TYPE != "RecEngagement":
            file = open(f'{dataset_path}/{dname}/Text_Word_{TYPE}_{subTYPE}_{MODE}_base', 'wb')
        else:
            file = open(f'{dataset_path}/{dname}/Text_Word_{TYPE}_{MODE}_base', 'wb')

    pickle.dump(answer_word, file)
    file.close()
    print("Inst = " + str(Inst))
    print(f'SAVE COMPLETE! {dataset_path}/{dname}/Text_Word_{TYPE}_{subTYPE}_{MODE}_base')


# ## MAIN

for i, t in enumerate(TYPE_list):
    for st in subTYPE_list[i]:
        start_time = time.time()
        print(t)
        print(st)
        product_texts = product_text_dict(t, st)
        LLM_word(MODE, product_texts, t, st)
        end_time = time.time()
        hour, minu, secon = get_time(start_time, end_time)
        print('##### (time) all: '+ str(hour)+' hours '+ str(minu)+ ' minutes '+ str(secon)+ 'seconds #####')



