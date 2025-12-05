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
# 100 stocks 
# default_tickers = [
# "NVDA.US",  # Electronic Technology
# "AAPL.US",  # Electronic Technology
# "AVGO.US",  # Electronic Technology
# "AMD.US",   # Electronic Technology
# "GE.US",    # Electronic Technology
# "CSCO.US",  # Electronic Technology
# "MU.US",    # Electronic Technology
# "RTX.US",   # Electronic Technology
# "INTC.US",  # Electronic Technology
# "QCOM.US",  # Electronic Technology
# "APH.US",   # Electronic Technology
# "AMAT.US",  # Electronic Technology
# "LRCX.US",  # Electronic Technology
# "TXN.US",   # Electronic Technology
# "KLAC.US",  # Electronic Technology
# "BA.US",    # Electronic Technology
# "ARM.US",   # Electronic Technology
# "ADI.US",   # Electronic Technology
# "HON.US",   # Electronic Technology
# "LMT.US",   # Electronic Technology
# "DELL.US",  # Electronic Technology
# "GD.US",  # Electronic Technology
# "MRVL.US",  # Electronic Technology
# "HWM.US",  # Electronic Technology
# "NOC.US",  # Electronic Technology
# "GOOG.US", # Technology Services
# "MSFT.US", # Technology Services
# "META.US", # Technology Services
# "ORCL.US", # Technology Services
# "NFLX.US", # Technology Services
# "PLTR.US", # Technology Services
# "IBM.US",  # Technology Services
# "CRM.US",  # Technology Services
# "APP.US",  # Technology Services
# "INTU.US", # Technology Services
# "NOW.US",  # Technology Services
# "ACN.US",  # Technology Services
# "ADBE.US", # Technology Services
# "PANW.US", # Technology Services
# "CRWD.US", # Technology Services
# "SNOW.US", # Technology Services
# "ADP.US",  # Technology Services
# "NTES.US",  # Technology Services
# "CDNS.US", # Technology Services
# "SNPS.US", # Technology Services
# "NET.US", # Technology Services
# "ADSK.US", # Technology Services
# "RBLX.US", # Technology Services
# "FTNT.US", # Technology Services
# "WDAY.US", # Technology Services
# "JPM.US",   # Finance
# "V.US",     # Finance
# "MA.US",    # Finance
# "BAC.US",   # Finance
# "MS.US",    # Finance
# "WFC.US",   # Finance
# "AXP.US",   # Finance
# "GS.US",    # Finance
# "C.US",     # Finance
# "BX.US",    # Finance
# "BLK.US",   # Finance
# "SCHW.US",  # Finance
# "COF.US",   # Finance
# "WELL.US",   # Finance
# "PGR.US",   # Finance
# "PLD.US",   # Finance
# "CB.US",   # Finance
# "BN.US",  # Finance
# "HOOD.US",   # Finance
# "KKR.US",    # Finance
# "IBKR.US",   # Finance
# "CME.US",   # Finance
# "MMC.US",  # Finance
# "ICE.US",    # Finance
# "NU.US",   # Finance
# "AMZN.US",  # Retail Trade
# "WMT.US",   # Retail Trade
# "COST.US",  # Retail Trade
# "BABA.US",  # Retail Trade
# "HD.US",    # Retail Trade
# "TJX.US",   # Retail Trade
# "PDD.US",  # Retail Trade
# "LOW.US",   # Retail Trade
# "MELI.US",  # Retail Trade
# "CVS.US",  # Retail Trade
# "ORLY.US",  # Retail Trade
# "CVNA.US",   # Retail Trade
# "SE.US",   # Retail Trade
# "AZO.US",  # Retail Trade
# "ROST.US",  # Retail Trade
# "CPNG.US",  # Retail Trade
# "KR.US",  # Retail Trade
# "TGT.US",  # Retail Trade
# "JD.US",  # Retail Trade
# "CPRT.US",  # Retail Trade
# "EBAY.US",  # Retail Trade
# "TSCO.US",  # Retail Trade
# "ULTA.US",  # Retail Trade
# "DG.US",  # Retail Trade
# "TPR.US",  # Retail Trade
# "XLK.US",     
# "XLE.US",     
# "XLY.US", 
# ]

# 305 stocks
# default_tickers = [
# "NVDA.US",  # Electronic Technology
# "AAPL.US",  # Electronic Technology
# "AVGO.US",  # Electronic Technology
# "AMD.US",   # Electronic Technology
# "GE.US",    # Electronic Technology
# "CSCO.US",  # Electronic Technology
# "MU.US",    # Electronic Technology
# "RTX.US",   # Electronic Technology
# "INTC.US",  # Electronic Technology
# "QCOM.US",  # Electronic Technology
# "APH.US",   # Electronic Technology
# "AMAT.US",  # Electronic Technology
# "LRCX.US",  # Electronic Technology
# "TXN.US",   # Electronic Technology
# "KLAC.US",  # Electronic Technology
# "BA.US",    # Electronic Technology
# "ARM.US",   # Electronic Technology
# "ADI.US",   # Electronic Technology
# "HON.US",   # Electronic Technology
# "LMT.US",   # Electronic Technology
# "DELL.US",  # Electronic Technology
# "GD.US",  # Electronic Technology
# "MRVL.US",  # Electronic Technology
# "HWM.US",  # Electronic Technology
# "NOC.US",  # Electronic Technology
# "TDG.US",  # Electronic Technology
# "EMR.US",  # Electronic Technology
# "GLW.US",  # Electronic Technology
# "VRT.US",  # Electronic Technology
# "TEL.US",  # Electronic Technology
# "MSI.US",  # Electronic Technology
# "STX.US",  # Electronic Technology
# "WDC.US",  # Electronic Technology
# "NXPI.US",  # Electronic Technology
# "LHX.US",  # Electronic Technology
# "MPWR.US",  # Electronic Technology
# "AXON.US",  # Electronic Technology
# "GRMN.US",  # Electronic Technology
# "HEI.US",  # Electronic Technology
# "KEYS.US",  # Electronic Technology
# "UI.US",  # Electronic Technology
# "PSTG.US",  # Electronic Technology
# "MCHP.US",  # Electronic Technology
# "SNDK.US",  # Electronic Technology
# "TER.US",  # Electronic Technology
# "HPE.US",  # Electronic Technology
# "CIEN.US",  # Electronic Technology
# "FSLR.US",  # Electronic Technology
# "COHR.US",  # Electronic Technology
# "BE.US",  # Electronic Technology
# "ALAB.US",  # Electronic Technology
# "TDY.US",  # Electronic Technology
# "NTAP.US",  # Electronic Technology
# "RKLB.US",  # Electronic Technology
# "FLEX.US",  # Electronic Technology
# "ON.US",  # Electronic Technology
# "GFS.US",  # Electronic Technology
# "SMCI.US",  # Electronic Technology
# "FTAI.US",  # Electronic Technology
# "NVT.US",  # Electronic Technology
# "FTV.US",  # Electronic Technology
# "IONQ.US",  # Electronic Technology
# "BWXT.US",  # Electronic Technology
# "FN.US",  # Electronic Technology
# "TXT.US",  # Electronic Technology
# "MTSI.US",  # Electronic Technology
# "AVAV.US",  # Electronic Technology
# "TSEM.US",  # Electronic Technology
# "NXT.US",  # Electronic Technology
# "ZBRA.US",  # Electronic Technology
# "JOBY.US",  # Electronic Technology
# "HII.US",  # Electronic Technology
# "KTOS.US",  # Electronic Technology
# "CR.US",  # Electronic Technology
# "GOOG.US", # Technology Services
# "MSFT.US", # Technology Services
# "META.US", # Technology Services
# "ORCL.US", # Technology Services
# "NFLX.US", # Technology Services
# "PLTR.US", # Technology Services
# "IBM.US",  # Technology Services
# "CRM.US",  # Technology Services
# "APP.US",  # Technology Services
# "INTU.US", # Technology Services
# "NOW.US",  # Technology Services
# "ACN.US",  # Technology Services
# "ADBE.US", # Technology Services
# "PANW.US", # Technology Services
# "CRWD.US", # Technology Services
# "SNOW.US", # Technology Services
# "ADP.US",  # Technology Services
# "NTES.US",  # Technology Services
# "CDNS.US", # Technology Services
# "SNPS.US", # Technology Services
# "NET.US", # Technology Services
# "ADSK.US", # Technology Services
# "RBLX.US", # Technology Services
# "FTNT.US", # Technology Services
# "WDAY.US", # Technology Services
# "DDOG.US", # Technology Services
# "MSTR.US", # Technology Services
# "EA.US", # Technology Services
# "ROP.US", # Technology Services
# "TTWO.US", # Technology Services
# "RDDT.US", # Technology Services
# "MSCI.US", # Technology Services
# "BIDU.US", # Technology Services
# "TEAM.US", # Technology Services
# "PAYX.US", # Technology Services
# "VEEV.US", # Technology Services
# "ZS.US", # Technology Services
# "CRWV.US", # Technology Services
# "CTSH.US", # Technology Services
# "FIS.US", # Technology Services
# "MDB.US", # Technology Services
# "CRDO.US", # Technology Services
# "VRSK.US", # Technology Services
# "FWONA.US", # Technology Services
# "CSGP.US", # Technology Services
# "BR.US", # Technology Services
# "ZM.US", # Technology Services
# "NBIS.US", # Technology Services
# "LDOS.US", # Technology Services
# "VRSN.US", # Technology Services
# "HPQ.US", # Technology Services
# "CYBR.US", # Technology Services
# "IOT.US", # Technology Services
# "PTC.US", # Technology Services
# "SSNC.US", # Technology Services
# "CHKP.US", # Technology Services
# "TOST.US", # Technology Services
# "TYL.US", # Technology Services
# "TWLO.US", # Technology Services
# "TRMB.US", # Technology Services
# "HUBS.US", # Technology Services
# "TTD.US", # Technology Services
# "U.US", # Technology Services
# "CDW.US", # Technology Services
# "GWRE.US", # Technology Services
# "CRCL.US", # Technology Services
# "PINS.US", # Technology Services
# "FIG.US", # Technology Services
# "Z.US", # Technology Services
# "GDDY.US", # Technology Services
# "IT.US", # Technology Services
# "GEN.US", # Technology Services
# "J.US", # Technology Services
# "ROKU.US", # Technology Services
# "OKTA.US", # Technology Services
# "FFIV.US", # Technology Services
# "DOCU.US", # Technology Services
# "RBRK.US", # Technology Services
# "JPM.US",   # Finance
# "V.US",     # Finance
# "MA.US",    # Finance
# "BAC.US",   # Finance
# "MS.US",    # Finance
# "WFC.US",   # Finance
# "AXP.US",   # Finance
# "GS.US",    # Finance
# "C.US",     # Finance
# "BX.US",    # Finance
# "BLK.US",   # Finance
# "SCHW.US",  # Finance
# "COF.US",   # Finance
# "WELL.US",   # Finance
# "PGR.US",   # Finance
# "PLD.US",   # Finance
# "CB.US",   # Finance
# "BN.US",  # Finance
# "HOOD.US",   # Finance
# "KKR.US",    # Finance
# "IBKR.US",   # Finance
# "CME.US",   # Finance
# "MMC.US",  # Finance
# "ICE.US",    # Finance
# "NU.US",   # Finance
# "BAM.US",  # Finance
# "AMT.US",    # Finance
# "BK.US",    # Finance
# "USB.US",    # Finance
# "APO.US",    # Finance
# "PNC.US",    # Finance
# "AON.US",    # Finance
# "EQIX.US",    # Finance
# "COIN.US",    # Finance
# "TRV.US",    # Finance
# "AJG.US",    # Finance
# "SPG.US",    # Finance
# "TFC.US",    # Finance
# "AFL.US",    # Finance
# "RKT.US",    # Finance
# "ALL.US",    # Finance
# "DLR.US",    # Finance
# "O.US",    # Finance
# "URI.US",    # Finance
# "NDAQ.US",    # Finance
# "MET.US",    # Finance
# "PSA.US",    # Finance
# "CBRE.US",    # Finance
# "AMP.US",    # Finance
# "AIG.US",    # Finance
# "VTR.US",    # Finance
# "CCI.US",    # Finance
# "PRU.US",    # Finance
# "HIG.US",    # Finance
# "SOFI.US",    # Finance
# "ARES.US",    # Finance
# "ACGL.US",    # Finance
# "STT.US",    # Finance
# "RJF.US",    # Finance
# "VICI.US",    # Finance
# "WTW.US",    # Finance
# "MTB.US",    # Finance
# "FITB.US",    # Finance
# "LPLA.US",    # Finance
# "SYF.US",    # Finance
# "EXR.US",    # Finance
# "WRB.US",    # Finance
# "BRO.US",    # Finance
# "CBOE.US",    # Finance
# "HBAN.US",    # Finance
# "MKL.US",    # Finance
# "CINF.US",    # Finance
# "AVB.US",    # Finance
# "TW.US",    # Finance
# "NTRS.US",    # Finance
# "AMZN.US",  # Retail Trade
# "WMT.US",   # Retail Trade
# "COST.US",  # Retail Trade
# "BABA.US",  # Retail Trade
# "HD.US",    # Retail Trade
# "TJX.US",   # Retail Trade
# "PDD.US",  # Retail Trade
# "LOW.US",   # Retail Trade
# "MELI.US",  # Retail Trade
# "CVS.US",  # Retail Trade
# "ORLY.US",  # Retail Trade
# "CVNA.US",   # Retail Trade
# "SE.US",   # Retail Trade
# "AZO.US",  # Retail Trade
# "ROST.US",  # Retail Trade
# "CPNG.US",  # Retail Trade
# "KR.US",  # Retail Trade
# "TGT.US",  # Retail Trade
# "JD.US",  # Retail Trade
# "CPRT.US",  # Retail Trade
# "EBAY.US",  # Retail Trade
# "TSCO.US",  # Retail Trade
# "ULTA.US",  # Retail Trade
# "DG.US",  # Retail Trade
# "TPR.US",  # Retail Trade
# "DLTR.US",  # Retail Trade
# "WSM.US",  # Retail Trade
# "CASY.US",  # Retail Trade
# "DKS.US",  # Retail Trade
# "BBY.US",  # Retail Trade
# "BURL.US",  # Retail Trade
# "CHWY.US",  # Retail Trade
# "W.US",  # Retail Trade
# "BLDR.US",  # Retail Trade
# "BJ.US",  # Retail Trade
# "DDS.US",  # Retail Trade
# "PAG.US",  # Retail Trade
# "GME.US",  # Retail Trade
# "GAP.US",  # Retail Trade
# "VIPS.US",  # Retail Trade
# "ACI.US",  # Retail Trade
# "FIVE.US",  # Retail Trade
# "SFM.US",  # Retail Trade
# "LAD.US",  # Retail Trade
# "AN.US",  # Retail Trade
# "OLLI.US",  # Retail Trade
# "MUSA.US",  # Retail Trade
# "URBN.US",  # Retail Trade
# "GLBE.US",  # Retail Trade
# "FND.US",  # Retail Trade
# "M.US",  # Retail Trade
# "BOOT.US",  # Retail Trade
# "KMX.US",  # Retail Trade
# "ETSY.US",  # Retail Trade
# "GPI.US",  # Retail Trade
# "ANF.US",  # Retail Trade
# "ABG.US",  # Retail Trade
# "RUSHA.US",  # Retail Trade
# "PSMT.US",  # Retail Trade
# "TBBB.US",  # Retail Trade
# "BBWI.US",  # Retail Trade
# "PLBL.US",  # Retail Trade
# "SIG.US",  # Retail Trade
# "AEO.US",  # Retail Trade
# "VSCO.US",  # Retail Trade
# "ASO.US",  # Retail Trade
# "AAP.US",  # Retail Trade
# "CPRI.US",  # Retail Trade
# "RH.US",  # Retail Trade
# "BKE.US",  # Retail Trade
# "KSS.US",  # Retail Trade
# "WRBY.US",  # Retail Trade
# "EYE.US",  # Retail Trade
# "SAH.US",  # Retail Trade
# "GRDN.US",  # Retail Trade
# "RVLV.US",  # Retail Trade
# "WMK.US",  # Retail Trade
# "REAL.US",  # Retail Trade
# "XLK.US",     
# "XLE.US",     
# "XLY.US", 
# ]

# all_tickers = default_tickers + [
#     "CRM.US",     
#     "ORCL.US",    
#     "INTC.US",    
#     "AMD.US",     
#     "NFLX.US",
#     "BAC.US",     
#     "WFC.US",     
#     "GS.US",      
#     "MS.US",      
#     "UNH.US",
#     "PFE.US",     
#     "TMO.US",
#     "KO.US",      
#     "WMT.US",     
#     "DIS.US",     
#     "NKE.US",     
#     "BA.US",      
#     "CAT.US",     
#     "GE.US",      
# ]

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