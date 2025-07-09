# +
import os
import csv
import pickle
import glob
import shutil
import datetime
import pandas as pd
import numpy as np
from tqdm import tqdm
import seaborn as sns
import matplotlib.pyplot as plt
from IPython.display import display_html

# +
###What is the dataset name?###
dname = "hm"

###What is the dataset folder path?###
dataset_path = '../Dataset'

# +
def adjust_id(x):
    '''Adjusts article ID code.'''
    x = str(x)
    if len(x) == 9:
        x = "0"+x
    
    return x

class clr:
    S = '\033[1m' + '\033[95m'
    E = '\033[0m'
    
my_colors = ["#AF0848", "#E90B60", "#CB2170", "#954E93", "#705D98", "#5573A8", "#398BBB", "#00BDE3"]
# -

# ## Filename
# * articles.csv - description features of all article_ids (105,542 datapoints)
# * customers.csv - description features of the customer profiles (1,371,980 datapoints)
# * transactions_train.csv - file containing the customer_id, the article that was bought and at what price (31,788,324 datapoints)

# Read in the data
articles = pd.read_csv(dataset_path + '/' + dname + '/RawData/articles.csv')
customers = pd.read_csv(dataset_path + '/' + dname + '/RawData/customers.csv')
transactions = pd.read_csv(dataset_path + '/' + dname + '/RawData/transactions_train.csv')
ss = pd.read_csv(dataset_path + '/' + dname + '/RawData/sample_submission.csv')

print(clr.S+"ARTICLES:"+clr.E, articles.shape)
display_html(articles.head(3))
print("\n", clr.S+"CUSTOMERS:"+clr.E, customers.shape)
display_html(customers.head(3))
print("\n", clr.S+"TRANSACTIONS:"+clr.E, transactions.shape)
display_html(transactions.head(3))
print("\n", clr.S+"SAMPLE_SUBMISSION:"+clr.E, ss.shape)
display_html(ss.head(3))

# ## Articles
# * There are more article_ids than actual images:
#     - unique article ids: 105,542
#     - unique images: 105,100
# * The path processing was taking too long, so the fastest (takes 1 second) way to do it was to create a variable that contains all article ids within the images folder (remember, set() is faster than a list), and then to correct any path that was invalid within the articles.csv file.
# * There are only 416 missing values within the desc column - product description

# +
# Replace missing values
articles.fillna(value="None", inplace=True)

# Adjust the article ID and product code to be string & add "0"
articles["article_id"] = articles["article_id"].apply(lambda x: adjust_id(x))
articles["product_code"] = articles["article_id"].apply(lambda x: x[:3])

# +
# Get all paths from the image folder
all_image_paths = glob.glob(dataset_path + '/' + dname + '/Images/*')

print(clr.S+"Number of unique article_ids within articles.csv:"+clr.E, len(articles), "\n"+
      clr.S+"Number of unique images within the image folder:"+clr.E, len(all_image_paths), "\n"+
      clr.S+"=> not all article_ids have a corresponding image!!!"+clr.E, "\n")

#Get all valid article ids and create a set to store the image_id
all_image_ids = set()

for path in tqdm(all_image_paths):
    article_id = path.split('/')[-1].split('.')[0]
    all_image_ids.add(article_id)


# Create full path to the article image
images_path = dataset_path + '/' + dname + '/Images/'
articles["path"] = images_path  + articles["article_id"] + ".jpg"

# Adjust the incorrect paths and set them to None
## **Important! Cuz not all the articles have corresponding images**
for k, article_id in tqdm(enumerate(articles["article_id"])):
    if article_id not in all_image_ids:
        articles.loc[k, "path"] = None


filtered_articles = articles[(articles['product_type_no'] == 272) & (articles['detail_desc'].notna())]
row_count = len(filtered_articles)
print(f"Matching rows: {row_count}")


articles = articles.dropna(subset=['path'])
uniqueProductNames = articles['article_id'].tolist()
titles = articles['prod_name'].tolist()
descriptions = articles['detail_desc'].tolist()

print(articles.shape[0])
print(len(uniqueProductNames))
print(len(titles))
print(len(descriptions))

# +
# Filter articles where product_type_no == 272
filtered_articles = articles[articles['product_type_no'] == 272]

# Find indices in indexDictionary that meet the condition
filtered_indices = [indexDictionary[article_id] for article_id in filtered_articles['article_id'].tolist() if article_id in indexDictionary]

print(f"Indices for product_type_no == 265: {filtered_indices}")
print(f"Total number of indices: {len(filtered_indices)}")
# -

file = open(f'{dataset_path}/{dname}/Trousers_idx', 'wb')
pickle.dump(filtered_indices, file)
file.close()

# +
#Creating the indexDictionary for conversion
indexDictionary = {}
for l in range(len(filtered_articles)):
    indexDictionary[filtered_articles['article_id'].tolist()[l]] = range(1,len(filtered_articles['article_id'].tolist())+1)[l]

print(f'index len = {len(indexDictionary)}')
# -

indexDictionary


##p_id to index
headers = ['Id', 'Index']
with open(f'{dataset_path}/{dname}/Product/idtoindex_trousers_{dname}.csv', 'w') as f:
    w = csv.writer(f)
    w.writerows(indexDictionary.items())

# ## Transactions

# +
print(clr.S+"Missing values within transactions dataset:"+clr.E)
print(transactions.isna().sum())

# Adjust article_id (as did for articles dataframe)
transactions["article_id"] = transactions["article_id"].apply(lambda x: adjust_id(x))
# -


# Calculate statistics from the original result
def calculate_stats(result):
    first_lists = [item[0] for item in result]
    second_lists = [item[1] for item in result]
    
    first_list_lengths = [len(lst) for lst in first_lists]
    second_list_lengths = [len(lst) for lst in second_lists]
    
    num_first_lists = len(first_lists)
    num_second_lists = len(second_lists)
    
    count_first_lists = sum(first_list_lengths)
    count_second_lists = sum(second_list_lengths)
    
    avg_first_length = count_first_lists / num_first_lists if num_first_lists > 0 else 0
    avg_second_length = count_second_lists / num_second_lists if num_second_lists > 0 else 0
    
    max_first_length = max(first_list_lengths, default=0)
    max_second_length = max(second_list_lengths, default=0)
    
    min_first_length = min(first_list_lengths, default=0)
    min_second_length = min(second_list_lengths, default=0)
    
    unique_customers = len(result)
    
    return count_first_lists, count_second_lists, avg_first_length, avg_second_length, max_first_length, max_second_length, min_first_length, min_second_length, unique_customers


indexDictionary.keys()

# +
filtered_transactions = transactions[transactions['article_id'].isin(indexDictionary.keys())]

# Check the number of records after filtering
print(f"Number of rows in filtered transactions: {filtered_transactions.shape[0]}")

# If needed, overwrite the original variable or export the filtered data to a file
transactions = filtered_transactions

# +
# Calculate the average price for each unique article_id
avg_price_per_article = transactions.groupby('article_id')['price'].mean()

# Construct price_list following the order specified by indexDictionary
price_list = []
for article_id in indexDictionary.keys():
    if article_id in avg_price_per_article:
        price_list.append(avg_price_per_article[article_id])
    else:
        price_list.append(None)  # If price data is missing, assign None or 0

# Check whether the length of price_list matches the length of indexDictionary
print(f"Price list length: {len(price_list)}")
print(f"IndexDictionary length: {len(indexDictionary)}")

# -

price_data=(price_list, price_list)
file = open(f'{dataset_path}/{dname}/PriceFeature_trousers', 'wb')
pickle.dump(price_data, file)
file.close()

# +
transactions['article_id'] = transactions['article_id'].map(indexDictionary)

# Check the result after replacement
print(transactions.head())

# +
# Convert t_dat to datetime format for easier sorting
transactions['t_dat'] = pd.to_datetime(transactions['t_dat'])

# Group transactions by customer_id
grouped = transactions.groupby('customer_id')

# Construct list of list
result = []
sessionsdate = []

for customer_id, group in grouped:
    group = group.sort_values(by='t_dat')
    
    last_t_dat = group['t_dat'].iloc[-1]
    articles_sequence = group[group['t_dat'] != last_t_dat]['article_id'].tolist()
    answers = group[group['t_dat'] == last_t_dat]['article_id'].tolist()
    
    if articles_sequence and answers:
        result.append([articles_sequence, answers])
        sessionsdate.append(last_t_dat)
    
# Calculate the statistics of the original data
original_stats = calculate_stats(result)
# -

file = open(f'{dataset_path}/{dname}/sessiondate_trousers', 'wb')
pickle.dump(sessionsdate, file)
file.close()
file = open(f'{dataset_path}/{dname}/purchase_trousers', 'wb')
pickle.dump(result, file)
file.close()

# +
# Filter article_ids that are not in uniqueProductNames and convert the valid ones to index values
filtered_result = []
filtered_sessiondate = []

for i, (seq, ans) in tqdm(enumerate(result)):
    filtered_seq = [indexDictionary[id] for id in seq if id in uniqueProductNames]
    filtered_ans = [indexDictionary[id] for id in ans if id in uniqueProductNames]
    
    if filtered_seq and filtered_ans:
        filtered_result.append([filtered_seq, filtered_ans])
        filtered_sessiondate.append(sessionsdate[i])
        
file = open(f'{dataset_path}/{dname}/filtered_sessiondate', 'wb')
pickle.dump(filtered_sessiondate, file)
file.close()
file = open(f'{dataset_path}/{dname}/filtered_purchase', 'wb')
pickle.dump(filtered_result, file)
file.close()


# Calculate the statistics of the filtered data
filtered_stats = calculate_stats(filtered_result)
# -

file = open(f'{dataset_path}/{dname}/filtered_sessiondate', 'rb')
filtered_sessiondate = pickle.load(file)
file.close()
file = open(f'{dataset_path}/{dname}/filtered_purchase', 'rb')
filtered_result = pickle.load(file)
file.close()

# +
np.random.seed(int(999))
count = 0
count2 = 0
result_filter = []
for seq,ans in tqdm(result):
    len_1 = len(ans)
    ans2 = list(set(ans))
    len_2 = len(ans2)
    if len_1 != len_2:
        count+=1
    if len_2 > 50:
        count2 += 1
        samples = []
        for _ in range(50):
            item = ans2[np.random.choice(len(ans2))]
            while item in samples:
                item = ans2[np.random.choice(len(ans2))]
            samples.append(item)
        ans2 = samples
        
    result_filter.append([seq,ans2])

    
# Calculate the statistics of the filtered data
filtered_stats = calculate_stats(result_filter)
# -

filtered_result = result_filter 

print(count)
print(count2)

# +
import matplotlib.pyplot as plt

# Assuming transactions is your dataframe and indexDictionary is already defined

# Count how many times each article_id was purchased
purchase_count = transactions['article_id'].value_counts()

# Get the article IDs purchased between 1-10 times
purchased_1_to_10 = purchase_count[(purchase_count <= 10)]

# Print the article IDs purchased between 1-10 times
print("Articles purchased 1-10 times:")
print(purchased_1_to_10)

# +
# Find the article IDs in indexDictionary that are not in purchase_count
not_in_purchase_count = set(indexDictionary.keys()) - set(purchase_count.index)

# Get the indices of purchased_1_to_10 items
purchased_1_to_10_indices = purchased_1_to_10.index.tolist()

# Combine both lists into a single list
combined_indices = list(not_in_purchase_count) + purchased_1_to_10_indices

print(len(combined_indices))
print(combined_indices[-1])

delete_item_idx = [indexDictionary[i] for i in combined_indices if i in indexDictionary.keys()]
print(len(delete_item_idx))
print(delete_item_idx[-1])
# -

file = open(f'{dataset_path}/{dname}/delete_product_idx', 'wb')
pickle.dump(delete_item_idx, file)
file.close()

# ### Divide into Train/ Val/ Test

# +
# Check if the dates in filtered_sessiondate are in chronological order
sorted_sessions_dates = sorted(sessionsdate)

# Get the total number of records in the dataset
total_sessions = len(sessionsdate)

# Determine split indices for training, validation, and test sets in a 75% / 12.5% / 12.5% ratio
train_split = int(0.75 * total_sessions)
val_split = int(0.875 * total_sessions) 
# -

print(train_split)
print(val_split)

sorted_sessions_dates[-1]

# +
#The dates for dividing train/val/test
dateTrain = datetime.datetime(2020, 7, 31, 0, 0, 0, 0)
dateVal = datetime.datetime(2020, 8, 31, 0, 0, 0, 0)

purchaseTrain = []
purchaseVal = []
purchaseTest = []

sessiondateTrain = []
sessiondateVal = []
sessiondateTest = []


for i in tqdm(range(len(filtered_result))):
    ts = sessionsdate[i]
    if ts < dateTrain:
        purchaseTrain.append(filtered_result[i])
        sessiondateTrain.append(ts)
    elif ts >= dateTrain and ts < dateVal:
        purchaseVal.append(filtered_result[i])
        sessiondateVal.append(ts)
    elif ts >=dateVal:
        purchaseTest.append(filtered_result[i])
        sessiondateTest.append(ts)
    else:
        print(i)
        
print(f'train len = {len(purchaseTrain)}')
print(f'val len = {len(purchaseVal)}')
print(f'test len = {len(purchaseTest)}')
# -


file = open(f'{dataset_path}/{dname}/sessiondateTrain', 'wb')
pickle.dump(sessiondateTrain, file)
file.close()
file = open(f'{dataset_path}/{dname}/sessiondateVal', 'wb')
pickle.dump(sessiondateVal, file)
file.close()
file = open(f'{dataset_path}/{dname}/sessiondateTest', 'wb')
pickle.dump(sessiondateTest, file)
file.close()

file = open(f'{dataset_path}/{dname}/purchaseTrain', 'wb')
pickle.dump(purchaseTrain, file)
file.close()
file = open(f'{dataset_path}/{dname}/purchaseVal', 'wb')
pickle.dump(purchaseVal, file)
file.close()
file = open(f'{dataset_path}/{dname}/purchaseTest', 'wb')
pickle.dump(purchaseTest, file)
file.close()


