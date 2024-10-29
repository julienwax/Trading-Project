
from simulation_d_arima import *
from import_data import existe_fichier_nasdaq,h1,download2

ticker_list = ['AAPL','ABNB','ADBE','ADI','ADP','ADSK','AEP','AMAT','AMD',
'AMGN','AMZN','ANSS','ASML','AVGO','AZN','BIIB','BKNG','BKR','CCEP','CDNS','CDW',
'CEG','CHTR','CMCSA','COST','CPRT','CRWD','CSCO','CSGP','CSX','CTAS',
'CTSH','DASH','DDOG','DLTR','DXCM','EA','EXC','FANG','FAST','FTNT',
'GEHC','GFS','GILD','GOOG','GOOGL','HON','IDXX','ILMN','INTC','INTU','ISRG',
'KDP','KHC','KLAC','LRCX','LULU','MAR','MCHP','MDB','MDLZ','MELI','META','MNST',
'MRNA','MRVL','MSFT','MU','NFLX','NVDA','NXPI','ODFL','ON','ORLY','PANW',
'PAYX','PCAR','PDD','PEP','PYPL','QCOM','REGN','ROP','ROST','SBUX','SIRI','SNPS',
'SPLK','TEAM','TMUS','TSLA','TTD','TTWO','TXN','VRSK','VRTX','WBA','WBD',
'WDAY','XEL','ZS']

#d_ROI = {'ticker' : [], 'ROI': []}
nb_restant = 101

"""
if __name__ == "__main__":
    for ticker1 in ticker_list:
        try:
            nb_restant-=1
            if not existe_fichier_nasdaq(ticker1):
                download2(ticker1)
            bot=Trading_bot(sim=True,ticker = ticker1)  
            bot.prepare_chart_dict()      
            ROI = bot.pool("Ordre_courant",10000,False);  print(ROI); d_ROI['ticker'].append(ticker1); d_ROI['ROI'].append(ROI)
            plt.figure(figsize=(10, 8)); plt.subplot(2,1,2)
            df = pd.read_csv("nasdaq_data/"+ticker1+".P1D.csv", index_col= "Unnamed: 0", parse_dates=True); df['price'].plot()
            plt.scatter(bot.historique_d,bot.historique_p,marker = '+',c = 'r'); plt.title("Cours de : " + str(ticker1))
            plt.subplot(2,1,1); df1 = pd.Series(bot.capitaux,bot.historique_d);df1.index = pd.to_datetime(df1.index).date
            df1.plot(c='darkblue')
            plt.title("Evolution du portefeuille, ROI = {:.2f}".format(ROI) + "  |  (p,d,q) = "+str(order))
            plt.savefig('nasdaq_plot/'+ticker1+'.jpeg')
            print(nb_restant)
        except Exception as e:
            print("Une erreur s'est produite avec le ticker: " + ticker1)
    dfs = pd.DataFrame(d_ROI); dfs.to_csv('nasdaq_arima_ROI.csv', index = False)
"""                  

df_roi = pd.read_csv('nasdaq_arima_ROI.csv')

def moy_roi():
    s = 0
    for i in range(len(df_roi)):
        s+=df_roi.iloc[i,1]
    return s/len(df_roi)


print(moy_roi())