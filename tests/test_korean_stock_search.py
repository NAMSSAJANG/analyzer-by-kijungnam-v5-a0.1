import pandas as pd

from korean_stock_search import contains_hangul, load_krx_listing, search_krx_listing


EXPECTED = {
    "삼성전자": "005930.KS",
    "SK하이닉스": "000660.KS",
    "현대차": "005380.KS",
    "기아": "000270.KS",
    "NAVER": "035420.KS",
    "카카오": "035720.KS",
}


def test_required_korean_names_map_to_yahoo_symbols():
    fallback = {symbol: name for name, symbol in EXPECTED.items()}
    listing = load_krx_listing(lambda _: (_ for _ in ()).throw(RuntimeError("offline")), fallback=fallback)
    for name, symbol in EXPECTED.items():
        assert search_krx_listing(name, listing)[0]["symbol"] == symbol


def test_listing_fallback_order_and_kosdaq_suffix():
    calls = []

    def stock_listing(source):
        calls.append(source)
        if source in {"KRX", "KOSPI", "KOSDAQ", "KOSPI-DESC"}:
            raise RuntimeError(source)
        if source == "KOSDAQ-DESC":
            return pd.DataFrame({"Symbol": ["035720"], "Name": ["카카오"]})
        raise AssertionError("KRX-DESC must not run after a successful DESC stage")

    listing = load_krx_listing(stock_listing, fallback={})
    assert calls == ["KRX", "KOSPI", "KOSDAQ", "KOSPI-DESC", "KOSDAQ-DESC"]
    assert search_krx_listing("카카오", listing)[0]["symbol"] == "035720.KQ"


def test_all_remote_failures_use_fallback_and_hangul_is_detected():
    listing = load_krx_listing(
        lambda _: (_ for _ in ()).throw(RuntimeError("offline")),
        fallback={"005930.KS": "삼성전자"},
    )
    assert search_krx_listing("삼성전자", listing)[0]["symbol"] == "005930.KS"
    assert search_krx_listing("없는회사", listing) == []
    assert contains_hangul("없는회사")
    assert not contains_hangul("AAPL")
