import pandas as pd
import json
import warnings
warnings.filterwarnings("ignore")

from eval.eval_mle import eval_mle
from eval.eval_density import eval_density
from eval.eval_dcr import eval_dcr
from eval.eval_detection import eval_detection

def eval_all(task_name, syn_data):
    train_data = pd.read_csv("synthetic/shoppers/real.csv")
    test_data = pd.read_csv("synthetic/shoppers/test.csv")
    
    # 打开要处理的文件
    with open(f"eval/result/{task_name}.txt", "a") as f:
        f.write("*********** Evaluation ***********\n\n")
        # 1. MLE
        print("start eval mle...")
        f.write(f"1. MLE: \n")
        overall_scores = eval_mle(syn_data.to_numpy(), test_data.to_numpy(), "shoppers")
        json.dump(overall_scores, f, indent=4, separators=(", ", ": "))
        
        # 2. density
        print("start eval density...")
        f.write(f"\n2. Density: \n")
        Shape, Trend, qual_report = eval_density(train_data, syn_data, "shoppers")
        f.write(f"Shape: {Shape}, Trend: {Trend}\n")
        
        # 3. dcr
        print("start eval dcr...")
        f.write(f"3. DCR: \n")
        score = eval_dcr(train_data, test_data, syn_data, "shoppers")
        f.write('DCR Score, a value closer to 0.5 is better\n')
        f.write(f'DCR Score = {score}\n')
        
        # 4. detection
        print("start eval detection...")
        f.write(f"4. Detection: \n")
        score = eval_detection(train_data, syn_data, "shoppers")
        f.write(f'Detection Score = {score}\n')
        
        # others
        # f.write("\n\n")
        
        # the end of this task, test the quality use bash
        print('start eval quality...')
        f.write('5. Quality: \n')
        # python eval/eval_quality.py ....