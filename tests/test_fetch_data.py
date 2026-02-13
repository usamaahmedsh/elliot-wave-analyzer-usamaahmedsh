import pytest
import pandas as pd
from unittest.mock import patch, MagicMock
import os

from models import helpers
import scripts.fetch_data as fd


def test_get_top_crypto_coins_mock(monkeypatch):
    # Patch rank_equities_by_volume to avoid live yfinance calls
    monkeypatch.setattr(fd, 'rank_equities_by_volume', lambda tickers, lookback, top_n, chunk_size: ['BTC-USD', 'ETH-USD'])
    res = fd.get_top_crypto_coins(2)
    assert isinstance(res, list)
    assert res[0] == 'BTC-USD'


def test_batch_fetch_yfinance_empty():
    df = fd.batch_fetch_yfinance([], start=None)
    assert isinstance(df, pd.DataFrame)
    assert df.empty

@patch('fetch_data.yf')
def test_rank_equities_by_volume_mock(yf_mock):
    # mock yf.download to return a DataFrame with Volume
    idx = pd.date_range('2025-01-01', periods=2)
    cols = pd.MultiIndex.from_product([['A','B'], ['Open','High','Low','Close','Volume']])
    data = pd.DataFrame([[1,1,1,1,100,2,2,2,2,200],[1,1,1,1,100,2,2,2,2,200]], index=idx, columns=cols)
    def download(symbols, period, interval, group_by, threads, progress):
        # return DataFrame with MultiIndex columns as yfinance does
        return data
    yf_mock.download.side_effect = download
    res = fd.rank_equities_by_volume(['A','B'], lookback='1mo', top_n=2, chunk_size=2)
    assert isinstance(res, list)

def test_write_out_json(tmp_path):
    df = pd.DataFrame([{'ticker':'A','Date':pd.Timestamp('2020-01-01'),'Open':1,'High':2,'Low':1,'Close':2,'Volume':10}])
    out = tmp_path / 'test.jsonl'
    fd.write_out(df, str(out), fmt='jsonl')
    assert out.exists()
    txt = out.read_text()
    assert 'ticker' in txt

