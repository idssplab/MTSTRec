import os

# +
dataset = 'food'

#which modules
use_Token = True
use_Style = True
use_Text = True
use_Price = True
# -

# ## Hyper Parameters

# +
seed_list = [79]
epoch = 1
batch_size_list = [64]
lr_list = [0.00001]
weight_decay_list = [0.000005]
decay_step = 10
gamma = 0.9

dropout_list = [0.1]
hidden_dropout_list = [0.1]
num_heads_list = [4]
num_blocks_list = [2]
hidden_dimension_list = [512]
output_dimension = 512

text_dimension = 4096

# +
fusion_num_blocks = 3
token_num_blocks = 2
style_num_blocks = 4
text_num_blocks = 2
price_num_blocks = 2

token_num_heads = 4
style_num_heads = 8
text_num_heads = 8
price_num_heads = 1

token_hidden_dimension = 512
style_hidden_dimension = 2048
text_hidden_dimension = 1024
price_hidden_dimension = 512
# -

# ## Feature Embedding

# +
if use_Style == True:
    style_embedding = "Image_Style_Embedding"
else:
    style_embedding = "Nostyle"

text_dimension = 4096
if use_Text == True:
    text_embedding_list = "Text_Embedding_TitleDescription,Text_Embedding_Basic_Paraphrase,Text_Embedding_Basic_Tags,Text_Embedding_Basic_Guess,Text_Embedding_Rec_Paraphrase,Text_Embedding_Rec_Tags"
else:
    text_embedding_list = ""


# +
command_string = 'python3 ./mainfinal.py --dataset {} --myseed {} --epoch {}'\
                ' --batch_size {} --lr {} --weight_decay {} --decay_step {} --gamma {}'\
                ' --transformer_dropout {} --transformer_hidden_dropout {} --transformer_num_blocks {} --transformer_num_heads {}'\
                ' --fusion_num_blocks {} --token_num_blocks {} --style_num_blocks {} --text_num_blocks {} --price_num_blocks {}'\
                ' --token_num_heads {} --style_num_heads {} --text_num_heads {} --price_num_heads {}'\
                ' --token_hidden_dimension {} --style_hidden_dimension {} --text_hidden_dimension {} --price_hidden_dimension {}'\
                ' --hidden_dimension {} --output_dimension {} --style_embedding {} --text_embedding {} --text_dimension {} --label_screen {}'

if use_Token == True:
    command_string += " --use_token"
if use_Style == True:
    command_string += " --use_style"
if use_Text == True:
    command_string += " --use_text"
if use_Price == True:
    command_string += " --use_price"
# -

for seed in seed_list:
    for batch_size in batch_size_list:
        for lr in lr_list:
            for weight_decay in weight_decay_list:
                for drop_rate in dropout_list:
                    for hidden_drop_rate in hidden_dropout_list:
                        for num_blocks in num_blocks_list:
                            for num_heads in num_heads_list:
                                for hidden_dimension in hidden_dimension_list:
                                        label_screen = '{}_sd{}_bs{}_lr{}_wd{}_dp{}_hdp{}_nl{}_hd{}'.format(dataset, seed, batch_size, lr, weight_decay, drop_rate, hidden_drop_rate, num_blocks, hidden_dimension)
                                        run_py = command_string.format(
                                            dataset, seed, epoch, batch_size, lr, weight_decay, decay_step, gamma,
                                            drop_rate, hidden_drop_rate, num_blocks, num_heads, fusion_num_blocks,
                                            token_num_blocks, style_num_blocks, text_num_blocks, price_num_blocks,
                                            token_num_heads, style_num_heads, text_num_heads, price_num_heads,
                                            token_hidden_dimension, style_hidden_dimension, text_hidden_dimension, price_hidden_dimension,
                                            hidden_dimension, output_dimension, style_embedding, text_embedding_list, text_dimension, label_screen)
                                        print(run_py)
                                        os.system(run_py)
