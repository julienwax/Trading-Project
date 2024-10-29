import os
import pandas as pd
from ib_insync import *
from DownloadData import download_data_ib
import importlib
### ALL THE STRATEGIES USED SHOULD HAVE FUNCTION TO TRANSFORM DATA INTO A CSV FILE AND A CLASS STRATEGY DEFINED ###

strategies = {1 : "ScalpingRSI_15min", 2 : "ScalpingBollinger_5min", 3 : "ScalpingLSTM_15min", 4 : "SwingLSTM_15min", 5 : "MomentumLSTM_5min" }

models = {1 : "NVDAmodel_15min_15_1", 2 : "AAPLmodel_15min_15_2", 3 : "SPYmodel_5min_15_2"}

def backtesting():
    strategy = input("Enter a strategy to backtest from the Strategy folder:\n 1 for ScalpingRSI_15min\n 2 for ScalpingBollinger_5min\n 3 for ScalpingLSTM_15min\n 4 for SwingLSTM_15min\n 5 for MomentumLSTM_5min\n")
    if not(os.path.exists("Strategy/"+strategies[int(strategy)]+".py")):
        print("The strategy does not exist.")
        return
    print("Downloading data...")
    file_name = download_data_ib(backtest=True)
    print("Data downloaded.")
    module = importlib.import_module("Strategy."+strategies[int(strategy)])
    if int(strategy) in [3,4,5]:
        model = input("Enter the model to use for the strategy:\n (Scalping <- 1 indicator | Swing,Momentum <- 2 indicators)\n 1 for NVDAmodel_15min_15_1\n 2 for AAPLmodel_15min_15_2\n 3 for SPYmodel_5min_15_2\n")
        if hasattr(module,'set_model_name'):
            module.set_model_name(models[int(model)])
            module.LoadModel()
    print("Data processing...")
    if hasattr(module, 'apply_save'):
        module.apply_save(file_name,backtest = True)
    else:
        print(f"The module {strategies[int(strategy)]} hasn't any function 'apply_save'.")
    print("Data processed.")
    print("Plotting signals...")
    df = pd.read_csv("Strategy/Data/"+file_name+".csv")    
    if hasattr(module, 'plot_signals'):
        module.plot_signals(df)
    else:
        print(f"The module {strategies[int(strategy)]} hasn't any function 'plot_signals'.")
    print("Signals plotted.")
    print("Running strategy...")
    cash = input("Enter the amount of cash you want to backtest with: (press enter for default: 100$)")
    margin = input("Enter the margin you want to use: (press enter for default: 1/50)")
    commission = input("Enter the commission you want to use: (press enter for default: 0.00$)")
    cash = 100 if cash == "" else float(cash); margin = 1/50 if margin == "" else float(margin); commission = 0.00 if commission == "" else float(commission)
    if hasattr(module, 'run_strategy') and hasattr(module,strategies[int(strategy)]):
        strat = getattr(module,strategies[int(strategy)])
        module.run_strategy(df,strat,filename = file_name,cash=cash,margin=margin,commission=commission,backtest = True)
    else:
        print(f"The module {strategies[int(strategy)]} hasn't any function 'run_strategy' or the class named: {strategies[int(strategy)]}.")
    print("Strategy processed.")
    if input("Do you want to optimize the parameters (slcoef,TPSLRatio) of the strategy (y/n)") == "y":
        if hasattr(module, 'optimize_strategy') and hasattr(module,strategies[int(strategy)]):
            strat = getattr(module,strategies[int(strategy)])
            module.optimize_strategy(df,strat,filename = file_name,cash=cash,margin=margin,commission=commission)
        else:
            print(f"The module {strategies[int(strategy)]} hasn't any function 'optimize_strategy' or the class named: {strategies[int(strategy)]}.")
    if input("Do you want to remove the csv file (y/n)") == "y":
        os.remove("Strategy/Data/"+file_name+".csv")
        print("The csv file has been removed.")
    

backtesting()