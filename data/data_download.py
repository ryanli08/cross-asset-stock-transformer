import pandas as pd
import pandas_datareader as pdr
from pathlib import Path

default_tickers = [
    "AAPL.US",    
    "MSFT.US",    
    "GOOGL.US",   
    "AMZN.US",    
    "META.US",    
    "NVDA.US",    
    "TSLA.US",    
    "ADBE.US",    
    "SPY.US",     
    "QQQ.US",     
    "IWM.US",     
    "DIA.US",     
    "XLK.US",     
    "XLE.US",     
    "XLY.US",     
    "JNJ.US",
    "JPM.US",     
    "PG.US",      
    "V.US",       
    "MA.US",      
    "HD.US",      
    "PEP.US",     
    "COST.US",
    "ABBV.US",    
    "MRK.US",     
    "LLY.US",     
    "AVGO.US",    
]

all_tickers = default_tickers + [
    "CRM.US",     
    "ORCL.US",    
    "INTC.US",    
    "AMD.US",     
    "NFLX.US",
    "BAC.US",     
    "WFC.US",     
    "GS.US",      
    "MS.US",      
    "UNH.US",
    "PFE.US",     
    "TMO.US",
    "KO.US",      
    "WMT.US",     
    "DIS.US",     
    "NKE.US",     
    "BA.US",      
    "CAT.US",     
    "GE.US",      
]

raw_data_path = Path(__file__).parent.parent / "data" / "raw"

def download_ticker(ticker):
    #https://pandas-datareader.readthedocs.io/en/latest/readers/stooq.html
    df_raw = pdr.stooq.StooqDailyReader(ticker, start='2012-01-01').read()
    
    if df_raw.empty:
        print(f"Failed to download: {ticker}")
        return
    df = df_raw.reset_index().rename(columns={
        'Date': 'date', 'Open': 'open', 'High': 'high', 
        'Low': 'low', 'Close': 'close', 'Volume': 'volume'
    })
    raw_data_path.mkdir(parents=True, exist_ok=True)
    df[['date', 'open', 'high', 'low', 'close', 'volume']].sort_values('date').to_csv(
        raw_data_path / f"{ticker.replace('.US', '')}.csv", index=False
    )

if __name__ == "__main__":
    for ticker in default_tickers:
        download_ticker(ticker)