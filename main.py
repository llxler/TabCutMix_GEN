import torch
from utils import execute_function, get_args

if __name__ == '__main__':
    args = get_args()
    
    if torch.cuda.is_available():
        args.device = f'cuda:{args.gpu}'
    else:
        args.device = 'cpu'

    if not args.save_path:
        args.save_path = f'synthetic/{args.dataname}/{args.method}.csv'
        
    main_fn = execute_function(args.method, args.mode)

    main_fn(args)

# tabddpm
# python main.py --dataname shoppers --method tabddpm --mode train                                                                      # train model
# python main.py --dataname shoppers --method tabddpm --mode sample --save_path sample_end_csv/lxler_test.csv --task_name lxler_test    # sample from model

# sample 
# python main.py --dataname shoppers --method tabddpm --mode sample --save_path sample_end_csv/${TASK_NAME}.csv --task_name $TASK_NAME --eval_flag True

####################################################

# tabsyn
# train VAE first
# python main.py --dataname shoppers --method vae --mode train

# after the VAE is trained, train the diffusion model
# python main.py --dataname shoppers --method tabsyn --mode train

# sample
# python main.py --dataname shoppers --method tabsyn --mode sample --save_path sample_end_csv/eeeee.csv

