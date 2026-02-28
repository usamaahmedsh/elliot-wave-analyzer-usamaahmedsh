#!/usr/bin/env python3
"""
Additional Market Data Fetcher for Elliott Wave Neural Network Training

Fetches MORE diverse market data NOT included in the first dataset:
- Small/Mid cap stocks (Russell 2000, NASDAQ 100)
- More cryptocurrencies (altcoins, DeFi tokens)
- International ETFs and stocks from emerging markets
- More forex crosses and exotic pairs
- Agricultural and soft commodities
- REITs and specialty funds

Timeframes: 1h, 4h (synthetic), 1d, 1wk
"""

import os
import sys
import json
import time
import logging
from pathlib import Path
from datetime import datetime, timedelta
from typing import Dict, List, Optional, Tuple
import warnings

warnings.filterwarnings('ignore')

import pandas as pd
import numpy as np
import yfinance as yf
from dotenv import load_dotenv

# Configure logging
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(levelname)s - %(message)s',
    handlers=[
        logging.StreamHandler(),
        logging.FileHandler('data_fetch_additional.log')
    ]
)
logger = logging.getLogger(__name__)

# Load environment variables
load_dotenv()

# ============================================================================
# ADDITIONAL TICKER UNIVERSE - Different from first dataset
# ============================================================================

# Russell 2000 / Small Cap stocks (NOT in S&P 500)
SMALL_CAP_STOCKS = [
    "AAON", "ACIW", "AGCO", "AIT", "ALIT", "AMED", "AMWD", "ANIK",
    "AOSL", "AOS", "APOG", "ARCB", "ARCO", "ASGN", "ASIX", "ASTE",
    "ATEC", "ATKR", "AZEK", "BANC", "BANF", "BCPC", "BDC", "BEAM",
    "BECN", "BHE", "BJRI", "BKE", "BLKB", "BMBL", "BMI", "BPMC",
    "BRKR", "BSIG", "BTU", "CALM", "CARG", "CASA", "CASS", "CBRL",
    "CCOI", "CENX", "CFFN", "CHCT", "CHEF", "CHX", "CIEN", "CIGI",
    "CIVB", "CKH", "CLBK", "CLF", "CLOV", "CLSK", "CMC", "CMPR",
    "CNMD", "CNNE", "CNOB", "CODI", "COHU", "COLB", "CONN", "COOP",
    "CORE", "CORT", "CPRX", "CRAI", "CRGY", "CRI", "CRMD", "CROX",
    "CRUS", "CRVL", "CSAN", "CSR", "CSWI", "CTBI", "CTRE", "CUTR",
    "CVBF", "CVI", "CVLT", "CWT", "CYTK", "DCPH", "DDD", "DENN",
    "DFIN", "DGII", "DIOD", "DLX", "DNLI", "DNOW", "DOCN", "DOOR",
    "DRH", "DSGX", "DY", "EAT", "EBC", "EEFT", "EGAN", "EGP",
    "ELVT", "ENTA", "EOLS", "EPRT", "ERII", "ESNT", "ETD", "EVBG",
    "EVER", "EVRI", "EXLS", "EXPO", "EZPW", "FARO", "FBNC", "FCFS",
    "FELE", "FFIN", "FIBK", "FISI", "FLGT", "FLR", "FN", "FNKO",
    "FORM", "FORR", "FOUR", "FRME", "FRPT", "FRSH", "FSS", "FSTR",
    "FTDR", "FUL", "GBCI", "GBX", "GCMG", "GEF", "GEO", "GERN",
    "GFF", "GHC", "GIII", "GMS", "GNRC", "GOGL", "GOLF", "GOSS",
    "GPI", "GPRE", "GRBK", "GRND", "GTN", "GYRE", "HA", "HAIN",
    "HAYW", "HBI", "HBM", "HCC", "HCI", "HCKT", "HEAR", "HEES",
    "HGV", "HIBB", "HLIO", "HLX", "HMN", "HOMB", "HQY", "HRI",
    "HRMY", "HSII", "HTH", "HUBG", "HUN", "HY", "IAC", "IART",
    "IBCP", "IBKR", "IBTX", "ICL", "ICUI", "IDCC", "IDYA", "IIPR",
    "IMAX", "IMKTA", "IMMR", "INDB", "INGR", "INN", "INSM", "INST",
]

# NASDAQ 100 components not in S&P 500
NASDAQ_COMPONENTS = [
    "ABNB", "ADSK", "AEP", "ALGN", "AMAT", "ANSS", "ASML", "ATVI",
    "BKNG", "CDNS", "CEG", "CHTR", "CPRT", "CRWD", "CSGP", "DDOG",
    "DLTR", "DXCM", "EA", "EBAY", "EXC", "FANG", "FAST", "FTNT",
    "GFS", "IDXX", "ILMN", "KDP", "KHC", "LCID", "LULU", "MAR",
    "MCHP", "MDLZ", "MELI", "MNST", "MRNA", "MRVL", "NFLX", "NXPI",
    "ODFL", "OKTA", "ORLY", "PANW", "PAYX", "PCAR", "PDD", "PYPL",
    "REGN", "RIVN", "ROST", "SGEN", "SIRI", "SNPS", "SPLK", "TEAM",
    "TMUS", "TTWO", "VRSK", "VRSN", "VRTX", "WBA", "WBD", "WDAY",
    "XEL", "ZM", "ZS",
]

# More Cryptocurrencies - DeFi, Layer 2s, Gaming, Memecoins
MORE_CRYPTO = [
    # DeFi Tokens
    "COMP-USD", "MKR-USD", "SNX-USD", "YFI-USD", "SUSHI-USD",
    "1INCH-USD", "BAL-USD", "UMA-USD", "REN-USD", "BAND-USD",
    "KNC-USD", "PERP-USD", "DYDX-USD", "GMX-USD", "RDNT-USD",
    
    # Layer 2 / Scaling
    "OP-USD", "ARB-USD", "IMX-USD", "LRC-USD", "BOBA-USD",
    "METIS-USD", "CELR-USD", "SKL-USD", "CTSI-USD", "CELO-USD",
    
    # Gaming / Metaverse
    "ENS-USD", "APE-USD", "ILV-USD", "MAGIC-USD", "PRIME-USD",
    "BLUR-USD", "LOOKS-USD", "RARE-USD", "SUPER-USD", "ALICE-USD",
    "GALA-USD", "WAXP-USD", "PYR-USD", "GODS-USD", "VRA-USD",
    
    # Infrastructure / Oracles
    "GRT-USD", "API3-USD", "OCEAN-USD", "FET-USD", "AGIX-USD",
    "RNDR-USD", "ROSE-USD", "AR-USD", "SC-USD", "STORJ-USD",
    
    # Privacy Coins
    "SCRT-USD", "DUSK-USD", "BEAM-USD", "FIRO-USD", "PIVX-USD",
    
    # Memecoins
    "PEPE-USD", "FLOKI-USD", "BONK-USD", "WIF-USD", "BOME-USD",
    "MEW-USD", "TURBO-USD", "LADYS-USD", "WOJAK-USD", "BABYDOGE-USD",
    
    # Other Altcoins
    "INJ-USD", "SUI-USD", "SEI-USD", "TIA-USD", "PYTH-USD",
    "JUP-USD", "WLD-USD", "STRK-USD", "MANTA-USD", "DYM-USD",
    "ORDI-USD", "STX-USD", "RUNE-USD", "KAVA-USD", "OSMO-USD",
    "JUNO-USD", "KUJI-USD", "LUNC-USD", "USTC-USD", "LUNA-USD",
    "FTT-USD", "RAY-USD", "SRM-USD", "STEP-USD", "FIDA-USD",
    "MNGO-USD", "ORCA-USD", "SBR-USD", "TULIP-USD", "COPE-USD",
]

# Emerging Markets ETFs and Stocks
EMERGING_MARKETS = [
    # Country ETFs
    "THD", "VNM", "EPOL", "TUR", "ECH", "EWM", "EIDO", "EPU",
    "GXG", "ARGT", "QAT", "UAE", "KSA", "EFNL", "NORW", "EDEN",
    "EIS", "PGAL", "GREK", "PAK", "NGE", "AFK", "FM", "EMQQ",
    
    # India focused
    "INDA", "SMIN", "INDY", "EPI", "PIN", "INDL",
    
    # China/Asia focused
    "KWEB", "CQQQ", "ASHR", "GXC", "PGJ", "MCHI", "CNYA", "KBA",
    "FXI", "YINN", "YANG", "CHIQ", "CXSE", "CNXT", "HAO", "TAO",
    
    # Brazil
    "FLBR", "BRZU", "BRF", "EWZS",
    
    # Individual Emerging Market Stocks (ADRs)
    "BABA", "JD", "PDD", "BIDU", "NIO", "XPEV", "LI", "BILI",
    "IQ", "VIPS", "TME", "TCOM", "YUMC", "HTHT", "BZUN", "QFIN",
    "YY", "HUYA", "DOYU", "ZTO", "ATHM", "VNET", "GDS", "KC",
    # Indian ADRs
    "INFY", "WIT", "HDB", "IBN", "SIFY", "RDY", "TTM", "VEDL",
    # Latin America
    "NU", "STNE", "PAGS", "XP", "MELI", "GLOB", "DESP", "CAAP",
    # Other
    "GRAB", "SE", "CPNG", "COUPN", "BEKE",
]

# More Forex - Exotic pairs and crosses
MORE_FOREX = [
    # Scandinavian
    "USDSEK=X", "USDNOK=X", "USDDKK=X", "EURSEK=X", "EURNOK=X", "EURDKK=X",
    # Eastern European
    "USDPLN=X", "USDCZK=X", "USDHUF=X", "USDRON=X", "EURPLN=X", "EURCHF=X",
    # Asian
    "USDSGD=X", "USDHKD=X", "USDKRW=X", "USDTWD=X", "USDTHB=X", "USDINR=X",
    "USDMYR=X", "USDIDR=X", "USDPHP=X", "USDVND=X", "USDCNY=X",
    # Latin America
    "USDBRL=X", "USDMXN=X", "USDARS=X", "USDCLP=X", "USDCOP=X", "USDPEN=X",
    # Middle East / Africa
    "USDTRY=X", "USDZAR=X", "USDILS=X", "USDAED=X", "USDSAR=X", "USDEGP=X",
    # Oceania
    "AUDNZD=X", "NZDCAD=X", "NZDCHF=X", "AUDSGD=X", "AUDHKD=X",
    # Crosses
    "GBPNOK=X", "GBPSEK=X", "CHFNOK=X", "CHFSEK=X", "CADNOK=X", "CADSEK=X",
]

# REITs and Specialty Funds
REITS_AND_SPECIALTY = [
    # Data Center REITs
    "DLR", "EQIX", "COR", "QTS", "CONE", "INXN",
    # Healthcare REITs
    "WELL", "VTR", "PEAK", "OHI", "HR", "DOC", "SBRA", "CTRE",
    # Industrial REITs
    "PLD", "DRE", "FR", "STAG", "GTY", "TRNO", "COLD", "IIPR",
    # Retail REITs
    "SPG", "MAC", "SKT", "KIM", "REG", "FRT", "WRI", "ROIC",
    # Office REITs
    "BXP", "SLG", "VNO", "KRC", "DEI", "CLI", "OFC", "HIW",
    # Residential REITs
    "AVB", "EQR", "ESS", "UDR", "CPT", "MAA", "AIV", "NXRT",
    # Specialty REITs
    "AMT", "CCI", "SBAC", "UNIT", "LAMR", "OUT", "CCU", "GLPI",
    # Mortgage REITs
    "NLY", "AGNC", "STWD", "BXMT", "KREF", "TRTX", "RC", "DX",
]

# Thematic ETFs
THEMATIC_ETFS = [
    # Clean Energy
    "ICLN", "TAN", "QCLN", "PBW", "FAN", "ACES", "CTEC", "RAYS",
    # Cybersecurity
    "CIBR", "HACK", "BUG", "IHAK", "WCBR",
    # Robotics/AI
    "BOTZ", "ROBO", "IRBO", "ARKQ", "THNQ", "AIQ", "ROBT", "UBOT",
    # Biotech/Genomics
    "IBB", "XBI", "ARKG", "LABU", "LABD", "GNOM", "IDNA", "HELX",
    # Cannabis
    "MSOS", "MJ", "YOLO", "THCX", "CNBS", "POTX",
    # Space
    "UFO", "ARKX", "ROKT", "SPCE",
    # Blockchain/Crypto
    "BLOK", "BKCH", "BITQ", "LEGR", "DAPP", "BITO", "BTF", "XBTF",
    # ESG/Impact
    "ESGU", "ESGV", "ESGE", "SUSA", "SUSL", "SNPE", "KRMA", "VOTE",
    # Infrastructure
    "PAVE", "IFRA", "NFRA", "IGF", "TOLZ", "GRID", "PBS",
    # Water
    "PHO", "FIW", "CGW", "AQWA", "PIO",
    # Aging/Healthcare
    "OLD", "AGED", "BFIT", "GERM",
]

# Additional Indices
MORE_INDICES = [
    "^TWII",    # Taiwan Weighted
    "^JKSE",    # Jakarta
    "^KLSE",    # Kuala Lumpur
    "^BSESN",   # BSE Sensex
    "^NSEI",    # Nifty 50
    "^IPSA",    # Chile IPSA
    "^MERV",    # Argentina Merval
    "^TA125.TA",  # Tel Aviv 125
    "^XU100.IS",  # Istanbul 100
    "^CASE30",  # Egypt EGX 30
    "^MOEX",    # Moscow Exchange
    "^KS11",    # KOSPI
    "^NZ50",    # NZX 50
    "^OMXS30",  # Stockholm
    "^OMXC25",  # Copenhagen
    "^OMXH25",  # Helsinki
    "^ATX",     # Vienna
    "^BFX",     # Brussels
    "^AEX",     # Amsterdam
    "^SSMI",    # Swiss Market
]

# Bonds and Fixed Income ETFs
FIXED_INCOME_ETFS = [
    # Treasury
    "SHY", "IEI", "IEF", "TLH", "TLT", "EDV", "ZROZ", "VGSH", "VGIT", "VGLT",
    # Corporate
    "LQD", "VCSH", "VCIT", "VCLT", "IGIB", "IGLB", "SPIB", "SPLB",
    # High Yield
    "HYG", "JNK", "USHY", "SHYG", "HYLD", "ANGL", "FALN", "SJNK",
    # Municipal
    "MUB", "VTEB", "TFI", "HYD", "HYMB", "CMF", "NYF", "NXR",
    # International
    "BNDX", "IAGG", "EMB", "PCY", "VWOB", "EMLC", "IGOV", "BWX",
    # Inflation Protected
    "TIP", "SCHP", "STIP", "VTIP", "PBTP",
    # Floating Rate
    "FLOT", "FLRN", "USFR", "TFLO",
]

# Leveraged and Inverse ETFs for pattern diversity
LEVERAGED_INVERSE = [
    # US Equity
    "SSO", "SDS", "UPRO", "SPXU", "UDOW", "SDOW", "QLD", "QID",
    "TNA", "TZA", "URTY", "SRTY", "SPXS", "SPXL", "TECL", "TECS",
    # Sector
    "LABU", "LABD", "CURE", "FAS", "FAZ", "FNGU", "FNGD", "SOXL", "SOXS",
    "NUGT", "DUST", "JNUG", "JDST", "ERX", "ERY", "GUSH", "DRIP",
    # International
    "YINN", "YANG", "EDC", "EDZ", "INDL", "BRAZ",
    # Fixed Income
    "TMF", "TMV", "TYD", "TYO", "TBT", "TBF",
    # Volatility
    "UVXY", "SVXY", "VXX", "VIXY", "VIXM",
    # Currency
    "UUP", "UDN", "FXE", "FXY", "FXB", "FXA", "FXC", "FXS",
]

# More Commodities - Softs, Grains, Meats
MORE_COMMODITIES = [
    # Precious Metals
    "GC=F", "SI=F", "PL=F", "PA=F", "HG=F",
    # Energy
    "CL=F", "BZ=F", "NG=F", "RB=F", "HO=F",
    # Grains
    "ZC=F", "ZW=F", "ZS=F", "ZM=F", "ZL=F", "ZO=F", "ZR=F",
    # Softs
    "KC=F", "SB=F", "CC=F", "CT=F", "OJ=F", "LBS=F",
    # Meats
    "LE=F", "HE=F", "GF=F",
    # Commodity ETFs
    "DJP", "GSG", "DBC", "PDBC", "USCI", "BCD", "GCC",
    "CPER", "JJC", "DBB", "DBE", "DBO", "UNG", "UNL",
    "DBA", "JJA", "JJG", "CORN", "WEAT", "SOYB", "CANE", "NIB", "JO",
    "COW", "MOO", "COWZ",
]

# Timeframe configurations
TIMEFRAME_CONFIG = {
    "1h": {"interval": "1h", "period": "730d", "description": "Hourly (2 years)"},
    "1d": {"interval": "1d", "period": "max", "description": "Daily (max history)"},
    "1wk": {"interval": "1wk", "period": "max", "description": "Weekly (max history)"},
}


def fetch_single_ticker(
    ticker: str, 
    interval: str, 
    period: str,
    retries: int = 3
) -> Optional[pd.DataFrame]:
    """Fetch OHLCV data for a single ticker with retries."""
    for attempt in range(retries):
        try:
            df = yf.download(
                ticker,
                period=period,
                interval=interval,
                progress=False,
                auto_adjust=True,
                prepost=False,
                threads=False,
            )
            
            if df.empty:
                return None
            
            if isinstance(df.columns, pd.MultiIndex):
                df.columns = df.columns.droplevel(1)
            
            df.columns = [c.lower() for c in df.columns]
            
            required = ['open', 'high', 'low', 'close', 'volume']
            if not all(col in df.columns for col in required):
                return None
            
            df = df[required].copy()
            df = df.dropna()
            
            df['ticker'] = ticker
            df['interval'] = interval
            
            df = df.reset_index()
            df.rename(columns={'index': 'datetime', 'Date': 'datetime', 'Datetime': 'datetime'}, inplace=True)
            
            if 'datetime' not in df.columns:
                df = df.reset_index()
                df.columns = ['datetime'] + list(df.columns[1:])
            
            return df
            
        except Exception as e:
            if attempt < retries - 1:
                time.sleep(1 * (attempt + 1))
                continue
            logger.debug(f"Failed to fetch {ticker} ({interval}): {e}")
            return None
    
    return None


def resample_to_4h(df_1h: pd.DataFrame) -> Optional[pd.DataFrame]:
    """Resample 1h data to 4h timeframe."""
    if df_1h is None or df_1h.empty:
        return None
    
    try:
        df = df_1h.copy()
        ticker = df['ticker'].iloc[0]
        
        df['datetime'] = pd.to_datetime(df['datetime'])
        df = df.set_index('datetime')
        
        df_4h = df[['open', 'high', 'low', 'close', 'volume']].resample('4h').agg({
            'open': 'first',
            'high': 'max',
            'low': 'min',
            'close': 'last',
            'volume': 'sum'
        }).dropna()
        
        if df_4h.empty:
            return None
        
        df_4h['ticker'] = ticker
        df_4h['interval'] = '4h'
        df_4h = df_4h.reset_index()
        
        return df_4h
        
    except Exception as e:
        logger.debug(f"Failed to resample to 4h: {e}")
        return None


def fetch_ticker_all_timeframes(ticker: str) -> Dict[str, pd.DataFrame]:
    """Fetch all timeframes for a single ticker."""
    results = {}
    
    for tf_name, tf_config in TIMEFRAME_CONFIG.items():
        df = fetch_single_ticker(ticker, tf_config["interval"], tf_config["period"])
        if df is not None and len(df) >= 50:
            results[tf_name] = df
    
    if "1h" in results:
        df_4h = resample_to_4h(results["1h"])
        if df_4h is not None and len(df_4h) >= 50:
            results["4h"] = df_4h
    
    return results


def fetch_all_data(
    tickers: List[str],
    category: str,
    output_dir: Path,
    progress_callback=None
) -> Dict[str, int]:
    """Fetch data for all tickers in a category."""
    stats = {"total": len(tickers), "success": 0, "failed": 0, "bars": {}}
    category_dir = output_dir / category
    category_dir.mkdir(parents=True, exist_ok=True)
    
    all_data = {tf: [] for tf in ["1h", "4h", "1d", "1wk"]}
    
    logger.info(f"Fetching {category}: {len(tickers)} tickers...")
    
    for i, ticker in enumerate(tickers):
        try:
            data = fetch_ticker_all_timeframes(ticker)
            
            if data:
                stats["success"] += 1
                for tf, df in data.items():
                    all_data[tf].append(df)
                    if tf not in stats["bars"]:
                        stats["bars"][tf] = 0
                    stats["bars"][tf] += len(df)
            else:
                stats["failed"] += 1
                
        except Exception as e:
            stats["failed"] += 1
            logger.debug(f"Error fetching {ticker}: {e}")
        
        if (i + 1) % 10 == 0:
            time.sleep(0.5)
            
        if progress_callback and (i + 1) % 50 == 0:
            progress_callback(category, i + 1, len(tickers))
    
    for tf, dfs in all_data.items():
        if dfs:
            combined = pd.concat(dfs, ignore_index=True)
            output_path = category_dir / f"{category}_{tf}.parquet"
            combined.to_parquet(output_path, engine="pyarrow", compression="snappy")
            logger.info(f"Saved {output_path}: {len(combined)} rows, {combined['ticker'].nunique()} tickers")
    
    return stats


def upload_to_huggingface(
    data_dir: Path,
    repo_name: str = "elliott-wave-market-data-extended",
    private: bool = False
) -> bool:
    """Upload dataset to HuggingFace Hub."""
    try:
        from huggingface_hub import HfApi, create_repo
        
        hf_token = os.getenv("HF_TOKEN")
        if not hf_token:
            logger.error("HF_TOKEN not found in environment variables")
            return False
        
        api = HfApi(token=hf_token)
        
        user_info = api.whoami()
        username = user_info.get("name", user_info.get("fullname", "user"))
        repo_id = f"{username}/{repo_name}"
        
        logger.info(f"Creating/updating HuggingFace repo: {repo_id}")
        
        try:
            create_repo(
                repo_id=repo_id,
                repo_type="dataset",
                private=private,
                token=hf_token,
                exist_ok=True
            )
        except Exception as e:
            logger.warning(f"Repo creation note: {e}")
        
        for parquet_file in data_dir.rglob("*.parquet"):
            relative_path = parquet_file.relative_to(data_dir)
            logger.info(f"Uploading {relative_path}...")
            
            api.upload_file(
                path_or_fileobj=str(parquet_file),
                path_in_repo=str(relative_path),
                repo_id=repo_id,
                repo_type="dataset",
                token=hf_token,
            )
        
        readme_content = generate_dataset_readme(data_dir)
        readme_path = data_dir / "README.md"
        with open(readme_path, "w") as f:
            f.write(readme_content)
        
        api.upload_file(
            path_or_fileobj=str(readme_path),
            path_in_repo="README.md",
            repo_id=repo_id,
            repo_type="dataset",
            token=hf_token,
        )
        
        metadata_path = data_dir / "metadata.json"
        if metadata_path.exists():
            api.upload_file(
                path_or_fileobj=str(metadata_path),
                path_in_repo="metadata.json",
                repo_id=repo_id,
                repo_type="dataset",
                token=hf_token,
            )
        
        logger.info(f"✅ Dataset uploaded successfully to: https://huggingface.co/datasets/{repo_id}")
        return True
        
    except ImportError:
        logger.error("huggingface_hub not installed. Run: pip install huggingface-hub")
        return False
    except Exception as e:
        logger.error(f"Failed to upload to HuggingFace: {e}")
        return False


def generate_dataset_readme(data_dir: Path) -> str:
    """Generate README.md for the HuggingFace dataset."""
    
    stats = {}
    total_rows = 0
    total_tickers = set()
    
    for parquet_file in data_dir.rglob("*.parquet"):
        try:
            df = pd.read_parquet(parquet_file)
            category = parquet_file.parent.name
            tf = parquet_file.stem.split("_")[-1]
            
            if category not in stats:
                stats[category] = {}
            
            stats[category][tf] = {
                "rows": len(df),
                "tickers": df['ticker'].nunique() if 'ticker' in df.columns else 0
            }
            total_rows += len(df)
            if 'ticker' in df.columns:
                total_tickers.update(df['ticker'].unique())
                
        except Exception as e:
            logger.debug(f"Error reading {parquet_file}: {e}")
    
    readme = f"""---
license: mit
task_categories:
  - time-series-forecasting
tags:
  - finance
  - stocks
  - crypto
  - forex
  - commodities
  - elliott-wave
  - technical-analysis
  - ohlcv
  - small-cap
  - emerging-markets
  - defi
  - altcoins
pretty_name: Elliott Wave Market Data Extended
size_categories:
  - 1M<n<10M
---

# Elliott Wave Market Data - Extended Dataset

**Additional** comprehensive OHLCV market data for training Elliott Wave pattern recognition neural networks.

⚠️ **This is a companion dataset** - contains instruments NOT included in the primary dataset.

## Dataset Description

This extended dataset contains historical OHLCV data for additional asset classes and instruments, specifically curated to complement the primary Elliott Wave dataset.

### Asset Classes (Different from Primary Dataset)
- **Small/Mid Cap Stocks**: Russell 2000, NASDAQ 100 components not in S&P 500
- **More Cryptocurrencies**: DeFi tokens, Layer 2s, Gaming coins, Memecoins, New launches
- **Emerging Markets**: Country ETFs, ADRs from China, India, Latin America, etc.
- **More Forex**: Exotic pairs, Scandinavian, Asian, Latin American currencies
- **REITs & Specialty**: Data centers, Healthcare, Industrial, Mortgage REITs
- **Thematic ETFs**: Clean energy, Cybersecurity, AI/Robotics, Cannabis, Space
- **Fixed Income**: Treasury, Corporate, High Yield, Municipal bonds
- **Leveraged/Inverse**: 2x/3x ETFs for pattern diversity

### Timeframes
- **1h**: Hourly data (up to 2 years history)
- **4h**: 4-hour data (resampled from 1h)
- **1d**: Daily data (maximum available history)
- **1wk**: Weekly data (maximum available history)

## Dataset Statistics

- **Total Rows**: {total_rows:,}
- **Total Unique Tickers**: {len(total_tickers):,}
- **Generated**: {datetime.now().strftime('%Y-%m-%d %H:%M UTC')}

### By Category

| Category | 1h Bars | 4h Bars | 1d Bars | 1wk Bars | Tickers |
|----------|---------|---------|---------|----------|---------|
"""
    
    for category in sorted(stats.keys()):
        row = f"| {category} |"
        tickers = 0
        for tf in ["1h", "4h", "1d", "1wk"]:
            if tf in stats[category]:
                row += f" {stats[category][tf]['rows']:,} |"
                tickers = max(tickers, stats[category][tf].get('tickers', 0))
            else:
                row += " - |"
        row += f" {tickers} |"
        readme += row + "\n"
    
    readme += """
## Data Schema

Each Parquet file contains the following columns:

| Column | Type | Description |
|--------|------|-------------|
| datetime | datetime64 | Timestamp of the bar |
| open | float64 | Opening price |
| high | float64 | Highest price |
| low | float64 | Lowest price |
| close | float64 | Closing price |
| volume | float64 | Trading volume |
| ticker | string | Ticker symbol |
| interval | string | Timeframe interval |

## Usage

```python
import pandas as pd

# Load specific file
df = pd.read_parquet("hf://datasets/YOUR_USERNAME/elliott-wave-market-data-extended/small_cap/small_cap_1d.parquet")

# Combine with primary dataset
df_primary = pd.read_parquet("hf://datasets/YOUR_USERNAME/elliott-wave-market-data/stocks/stocks_1d.parquet")
df_extended = pd.read_parquet("hf://datasets/YOUR_USERNAME/elliott-wave-market-data-extended/small_cap/small_cap_1d.parquet")
df_combined = pd.concat([df_primary, df_extended], ignore_index=True)
```

## Data Sources

- Primary source: Yahoo Finance (via yfinance)
- All data is adjusted for splits and dividends

## License

MIT License - Free for academic and commercial use.
"""
    
    return readme


def progress_callback(category: str, current: int, total: int):
    """Print progress updates."""
    pct = (current / total) * 100
    logger.info(f"[{category}] Progress: {current}/{total} ({pct:.1f}%)")


def main():
    """Main entry point for additional data fetching."""
    start_time = time.time()
    
    output_dir = Path("data/hf_dataset_extended")
    output_dir.mkdir(parents=True, exist_ok=True)
    
    logger.info("=" * 60)
    logger.info("ELLIOTT WAVE MARKET DATA FETCHER - EXTENDED")
    logger.info("=" * 60)
    
    all_stats = {}
    
    # 1. Small Cap Stocks
    logger.info(f"\n📈 Small Cap Stocks: {len(SMALL_CAP_STOCKS)} tickers")
    all_stats["small_cap"] = fetch_all_data(
        SMALL_CAP_STOCKS, "small_cap", output_dir,
        progress_callback=progress_callback
    )
    
    # 2. NASDAQ Components
    logger.info(f"\n💻 NASDAQ Components: {len(NASDAQ_COMPONENTS)} tickers")
    all_stats["nasdaq"] = fetch_all_data(
        NASDAQ_COMPONENTS, "nasdaq", output_dir,
        progress_callback=progress_callback
    )
    
    # 3. More Crypto
    logger.info(f"\n₿ Additional Crypto: {len(MORE_CRYPTO)} tickers")
    all_stats["crypto_extended"] = fetch_all_data(
        MORE_CRYPTO, "crypto_extended", output_dir,
        progress_callback=progress_callback
    )
    
    # 4. Emerging Markets
    logger.info(f"\n🌍 Emerging Markets: {len(EMERGING_MARKETS)} tickers")
    all_stats["emerging_markets"] = fetch_all_data(
        EMERGING_MARKETS, "emerging_markets", output_dir,
        progress_callback=progress_callback
    )
    
    # 5. More Forex
    logger.info(f"\n💱 Additional Forex: {len(MORE_FOREX)} pairs")
    all_stats["forex_extended"] = fetch_all_data(
        MORE_FOREX, "forex_extended", output_dir,
        progress_callback=progress_callback
    )
    
    # 6. REITs & Specialty
    logger.info(f"\n🏢 REITs & Specialty: {len(REITS_AND_SPECIALTY)} tickers")
    all_stats["reits"] = fetch_all_data(
        REITS_AND_SPECIALTY, "reits", output_dir,
        progress_callback=progress_callback
    )
    
    # 7. Thematic ETFs
    logger.info(f"\n🎯 Thematic ETFs: {len(THEMATIC_ETFS)} tickers")
    all_stats["thematic_etfs"] = fetch_all_data(
        THEMATIC_ETFS, "thematic_etfs", output_dir,
        progress_callback=progress_callback
    )
    
    # 8. More Indices
    logger.info(f"\n📉 Additional Indices: {len(MORE_INDICES)} indices")
    all_stats["indices_extended"] = fetch_all_data(
        MORE_INDICES, "indices_extended", output_dir,
        progress_callback=progress_callback
    )
    
    # 9. Fixed Income ETFs
    logger.info(f"\n📊 Fixed Income ETFs: {len(FIXED_INCOME_ETFS)} tickers")
    all_stats["fixed_income"] = fetch_all_data(
        FIXED_INCOME_ETFS, "fixed_income", output_dir,
        progress_callback=progress_callback
    )
    
    # 10. Leveraged/Inverse ETFs
    logger.info(f"\n⚡ Leveraged/Inverse ETFs: {len(LEVERAGED_INVERSE)} tickers")
    all_stats["leveraged"] = fetch_all_data(
        LEVERAGED_INVERSE, "leveraged", output_dir,
        progress_callback=progress_callback
    )
    
    # 11. More Commodities
    logger.info(f"\n🛢️ Additional Commodities: {len(MORE_COMMODITIES)} tickers")
    all_stats["commodities_extended"] = fetch_all_data(
        MORE_COMMODITIES, "commodities_extended", output_dir,
        progress_callback=progress_callback
    )
    
    # Save metadata
    metadata = {
        "generated_at": datetime.now().isoformat(),
        "stats": all_stats,
        "timeframes": ["1h", "4h", "1d", "1wk"],
        "categories": list(all_stats.keys()),
        "total_tickers": {
            "small_cap": len(SMALL_CAP_STOCKS),
            "nasdaq": len(NASDAQ_COMPONENTS),
            "crypto_extended": len(MORE_CRYPTO),
            "emerging_markets": len(EMERGING_MARKETS),
            "forex_extended": len(MORE_FOREX),
            "reits": len(REITS_AND_SPECIALTY),
            "thematic_etfs": len(THEMATIC_ETFS),
            "indices_extended": len(MORE_INDICES),
            "fixed_income": len(FIXED_INCOME_ETFS),
            "leveraged": len(LEVERAGED_INVERSE),
            "commodities_extended": len(MORE_COMMODITIES),
        }
    }
    
    with open(output_dir / "metadata.json", "w") as f:
        json.dump(metadata, f, indent=2, default=str)
    
    # Print summary
    elapsed = time.time() - start_time
    logger.info("\n" + "=" * 60)
    logger.info("FETCH COMPLETE")
    logger.info("=" * 60)
    
    total_success = sum(s.get("success", 0) for s in all_stats.values())
    total_failed = sum(s.get("failed", 0) for s in all_stats.values())
    total_bars = sum(
        sum(s.get("bars", {}).values()) 
        for s in all_stats.values()
    )
    
    logger.info(f"Total tickers fetched: {total_success}")
    logger.info(f"Total tickers failed: {total_failed}")
    logger.info(f"Total data bars: {total_bars:,}")
    logger.info(f"Time elapsed: {elapsed/60:.1f} minutes")
    logger.info(f"Output directory: {output_dir}")
    
    # Upload to HuggingFace
    logger.info("\n" + "=" * 60)
    logger.info("UPLOADING TO HUGGINGFACE")
    logger.info("=" * 60)
    
    success = upload_to_huggingface(output_dir)
    
    if success:
        logger.info("🎉 All done! Extended dataset available on HuggingFace Hub.")
    else:
        logger.warning("⚠️ HuggingFace upload failed. Data saved locally.")
    
    return 0


if __name__ == "__main__":
    sys.exit(main())
