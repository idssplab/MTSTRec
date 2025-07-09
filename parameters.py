import argparse


def parse_args():
    parser = argparse.ArgumentParser() 
    # ============== data_dir ==============    
    parser.add_argument("--root_data_dir", type=str, default="./Dataset")
    parser.add_argument("--dataset", type=str, default="food")
    parser.add_argument("--trainset", type=str, default="browse_seq_train")
    parser.add_argument("--validset", type=str, default="browse_seq_val")
    parser.add_argument("--testset", type=str, default="browse_seq_test") 
    parser.add_argument("--traindate", type=str, default="sessiondate_train")
    parser.add_argument("--validdate", type=str, default="sessiondate_val")
    parser.add_argument("--testdate", type=str, default="sessiondate_test") 
    parser.add_argument("--style_embedding", type=str, default="Image_Style_Embedding")
    parser.add_argument("--text_embedding", type=str, default="Text_Embedding_TitleDescription,Text_Embedding_Basic_Paraphrase,Text_Embedding_Basic_Tags,Text_Embedding_Basic_Guess,Text_Embedding_Rec_Paraphrase,Text_Embedding_Rec_Tags")
    parser.add_argument("--price_file", type=str, default="PriceFeature")
    parser.add_argument("--model_checkpoint_dir", type=str, default="./models")
    parser.add_argument("--best_modelname", type=str, default="best_model.pth")

    # ============== seed parameters ==============
    parser.add_argument("--myseed", type=int, default=79)

    # ============== train parameters ==============
    parser.add_argument("--batch_size", type=int, default=64)
    parser.add_argument("--epoch", type=int, default=100)
    parser.add_argument("--optimizer", type=str, default='adam')
    parser.add_argument("--lr", type=float, default=0.00001)
    parser.add_argument("--weight_decay", type=float, default=0.000005)
    parser.add_argument("--early_stop_step", type=int, default=10)
    parser.add_argument("--decay_step", type=int, default=10)
    parser.add_argument("--gamma", type=float, default=0.9)
    parser.add_argument("--transformer_dropout", type=float, default=0.1)
    parser.add_argument("--transformer_hidden_dropout", type=float, default=0.1)

    # ============== model parameters ==============
    parser.add_argument("--transformer_num_blocks", type=int, default=2)
    parser.add_argument("--transformer_num_heads", type=int, default=4)

    #for MTSTRec
    parser.add_argument("--style_hidden_dropout", type=float, default=0.2)
    parser.add_argument("--text_hidden_dropout", type=float, default=0.3)
    parser.add_argument("--price_hidden_dropout", type=float, default=0.2)
    
    parser.add_argument("--fusion_num_blocks", type=int, default=3)
    parser.add_argument("--token_num_blocks", type=int, default=2)
    parser.add_argument("--style_num_blocks", type=int, default=4)
    parser.add_argument("--text_num_blocks", type=int, default=2)
    parser.add_argument("--price_num_blocks", type=int, default=2)
    
    parser.add_argument("--token_num_heads", type=int, default=4)
    parser.add_argument("--style_num_heads", type=int, default=8)
    parser.add_argument("--text_num_heads", type=int, default=8)
    parser.add_argument("--price_num_heads", type=int, default=1)
    
    parser.add_argument("--input_dimension", type=int, default=512)
    parser.add_argument("--style_dimension", type=int, default=512)
    parser.add_argument("--text_dimension", type=int, default=4096)
    
    parser.add_argument("--hidden_dimension", type=int, default=512)
    parser.add_argument("--token_hidden_dimension", type=int, default=512)
    parser.add_argument("--style_hidden_dimension", type=int, default=2048)
    parser.add_argument("--text_hidden_dimension", type=int, default=1024)
    parser.add_argument("--price_hidden_dimension", type=int, default=512)
    parser.add_argument("--discount_hidden_dimension", type=int, default=512)
    
    parser.add_argument("--output_dimension", type=int, default= 512)
    
    parser.add_argument("--min_len", type=int, default=3)
    parser.add_argument("--max_len", type=int, default=20)
    parser.add_argument("--y_pad_len", type=int, default=50)
    parser.add_argument("--train_negative_sample_size", type=int, default=100) 
    parser.add_argument("--test_negative_sample_size", type=int, default=100) 
    
    #module
    parser.add_argument("--use_token", action='store_true')
    parser.add_argument("--use_style", action='store_true') 
    parser.add_argument("--use_text", action='store_true')
    parser.add_argument("--use_price", action='store_true') 
    
    #evaluation                      
    parser.add_argument("--metric_ks", type=list, default=[5, 10, 20])
    parser.add_argument("--pretrainmodel_path", type=str, default="./models/best_model.pth")

    # ============== other setting ==============
    parser.add_argument("--worker_number", type=int, default=4)
    parser.add_argument("--is_parallel", type=int, default=1)
    parser.add_argument("--device", type=str, default='cuda')
    parser.add_argument("--device_idx", type=str, default='0')
    parser.add_argument("--label_screen", type=str, default='None')


    args = parser.parse_args()

    return args


# ### Main

if __name__ == '__main__':
    args = parse_args()
