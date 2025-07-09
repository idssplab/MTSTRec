# +
import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim
import copy
from tqdm import tqdm
import pickle
import csv

from sklearn.preprocessing import RobustScaler, MinMaxScaler, StandardScaler 


# +
###What is the dataset name?###
dname = "hm"

###What is the Dataset folder path?###
dataset_path = '../Dataset'

###What is the ProductDictName path?###
ProductDictFile = f'{dataset_path}/{dname}/Product/{dname}_productdetail'

###What is the IndexDictionary path?###
indexDictionaryFile = f'{dataset_path}/{dname}/Product/idtoindex_trousers_{dname}.csv'


# +
##Product Names
file = open(ProductDictFile, 'rb')
productDictFinal = pickle.load(file)
productNames = list(productDictFinal[i][0] for i in range(len(productDictFinal)))

print(len(productDictFinal))
print(len(productNames))

# +
## Product Index 
with open(indexDictionaryFile, newline='') as f:
    reader = csv.reader(f)
    idtoindex = list(reader)

indexDictionary = {}
productNames = []

for l in range(len(idtoindex)):
    productNames.append(str(idtoindex[l][0]))
    indexDictionary[str(idtoindex[l][0])] = int(idtoindex[l][1])
print(len(indexDictionary))
print(len(productNames))

# +
price_dict = {i:[] for i in range(1,len(productNames)+1)}
pricediff_dict = {i:[] for i in range(1,len(productNames)+1)}

for i in productDictFinal:
    price_dict[indexDictionary[i[0]]] = i[4]
    pricediff_dict[indexDictionary[i[0]]] = i[5]
# -

price = list(price_dict.values())
pricediff = list(pricediff_dict.values())

price_data=(price, pricediff)
file = open(f'{dataset_path}/{dname}/PriceFeature', 'wb')
pickle.dump(price_data, file)
file.close()


