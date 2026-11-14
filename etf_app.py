import io
import re
from concurrent.futures import ThreadPoolExecutor, as_completed
from datetime import datetime, timedelta
from typing import Any

import pandas as pd
import requests
import streamlit as st

st.set_page_config(
    page_title="국내 ETF 구성종목 비교기",
    page_icon="📊",
    layout="wide",
)

SHEET_ID = "1N7pHQgFqBcgNXIup-tFj40Al2jczS-f0eZ8XgW0wIMU"
SHEET_GID = "280536315"
SHEET_EDIT_URL = f"https://docs.google.com/spreadsheets/d/{SHEET_ID}/edit?gid={SHEET_GID}#gid={SHEET_GID}"
SHEET_CSV_URL = f"https://docs.google.com/spreadsheets/d/{SHEET_ID}/export?format=csv&gid={SHEET_GID}"

HEADERS = {
    "User-Agent": (
        "Mozilla/5.0 (Windows NT 10.0; Win64; x64) "
        "AppleWebKit/537.36 (KHTML, like Gecko) Chrome/120.0.0.0 Safari/537.36"
    ),
    "Referer": "https://m.stock.naver.com/",
}


def parse_weight_pct(raw_val: Any) -> float:
    """'15.4%', '15.4', 15.4 등의 비중 문자열/숫자를 백분율(%) float로 변환합니다."""
    if pd.isna(raw_val):
        return 0.0
    text = str(raw_val).strip().replace(",", "").replace("%", "")
    if not text:
        return 0.0
    return float(text)


@st.cache_data(ttl=300, show_spinner=False)
def load_portfolio_from_sheet() -> list[dict[str, Any]]:
    """구글 시트(퇴직연금v1 / Code 시트)에서 ETF 종목코드와 포트폴리오 비중(%)을 읽어옵니다."""
    resp = requests.get(SHEET_CSV_URL, headers={"User-Agent": HEADERS["User-Agent"]}, timeout=10)
    if resp.status_code != 200:
        raise ValueError(
            f"구글 시트를 불러오지 못했습니다 (HTTP {resp.status_code}). "
            "시트 공유 설정이 '링크가 있는 모든 사용자(뷰어)'인지 확인해 주세요."
        )
    resp.encoding = "utf-8"
    df = pd.read_csv(io.StringIO(resp.text), dtype=str)
    df.columns = [str(c).strip() for c in df.columns]

    if "Code" not in df.columns or "비중" not in df.columns:
        raise ValueError(f"시트에 'Code' 또는 '비중' 컬럼이 없습니다. 현재 컬럼: {list(df.columns)}")

    portfolio: dict[str, dict[str, Any]] = {}
    for _, row in df.iterrows():
        raw_code = str(row.get("Code", "")).strip()
        if not raw_code or raw_code.lower() == "nan":
            continue
        code = raw_code.zfill(6) if raw_code.isdigit() else raw_code.upper()
        weight_pct = parse_weight_pct(row.get("비중"))
        sheet_name = str(row.get("종목", "")).strip()
        if sheet_name.lower() == "nan":
            sheet_name = ""

        if code in portfolio:
            portfolio[code]["portfolio_weight"] += weight_pct
        else:
            portfolio[code] = {
                "code": code,
                "sheet_name": sheet_name,
                "portfolio_weight": weight_pct,
            }

    return list(portfolio.values())


@st.cache_data(ttl=600, show_spinner=False)
def fetch_etf_data(code: str, portfolio_weight: float) -> dict[str, Any]:
    """네이버 금융 모바일 API에서 ETF 기본 정보와 상위 구성종목(CU)을 조회합니다."""
    basic_url = f"https://m.stock.naver.com/api/etf/{code}/basic"
    const_url = f"https://m.stock.naver.com/api/etf/{code}/constituent"

    basic_resp = requests.get(basic_url, headers=HEADERS, timeout=10)
    if basic_resp.status_code != 200:
        raise ValueError(f"ETF 기본 정보를 찾을 수 없습니다 (HTTP {basic_resp.status_code}).")
    basic_data = basic_resp.json()
    if basic_data.get("stockEndType") != "etf" and not basic_data.get("stockName"):
        raise ValueError("유효한 ETF 종목코드가 아닙니다.")

    const_resp = requests.get(const_url, headers=HEADERS, timeout=10)
    if const_resp.status_code != 200:
        raise ValueError(f"구성종목 정보를 가져오지 못했습니다 (HTTP {const_resp.status_code}).")
    const_data = const_resp.json()

    etf_name = basic_data.get("stockName", code)
    holdings = []
    for idx, item in enumerate(const_data.get("result", []), start=1):
        is_domestic = bool(item.get("itemCode"))
        stock_code = item.get("itemCode") or item.get("reutersCode") or ""
        stock_name = (item.get("itemName") or "").strip()
        weight = float(item.get("constituentWeight") or 0.0)
        actual_weight = round((portfolio_weight / 100.0) * weight, 4)
        cu_qty = item.get("cuUnitQuantity")
        cu_unit = item.get("cuUnit") or ""
        holdings.append(
            {
                "ETF코드": code,
                "ETF명": etf_name,
                "ETF 비중(%)": round(portfolio_weight, 2),
                "순위": idx,
                "구성종목명": stock_name,
                "구성종목코드": stock_code,
                "보유비중(%)": weight,
                "실제 기여 비중(%)": actual_weight,
                "1개월 수익률(%)": None,
                "CU단위수량": f"{cu_qty:,.2f} {cu_unit}".strip() if cu_qty is not None else "-",
                "_match_key": stock_code if stock_code else stock_name,
                "_is_domestic": is_domestic,
            }
        )

    return {
        "code": code,
        "name": etf_name,
        "portfolio_weight": round(portfolio_weight, 2),
        "close_price": basic_data.get("closePrice", "-"),
        "change_rate": basic_data.get("fluctuationsRatio", "-"),
        "nav": basic_data.get("nav", "-"),
        "total_count": const_data.get("totalCount", len(holdings)),
        "total_weight": const_data.get("totalWeight"),
        "holdings": holdings,
    }


@st.cache_data(ttl=1800, show_spinner=False)
def fetch_1m_return(stock_code: str, is_domestic: bool) -> float | None:
    """네이버 금융 일별 차트 API를 사용해 최근 1개월(약 30일) 수익률(%)을 계산합니다."""
    if not stock_code:
        return None
    kind = "domestic" if is_domestic else "foreign"
    end_dt = datetime.now()
    start_dt = end_dt - timedelta(days=45)
    url = (
        f"https://api.stock.naver.com/chart/{kind}/item/{stock_code}/day"
        f"?startDateTime={start_dt.strftime('%Y%m%d0000')}&endDateTime={end_dt.strftime('%Y%m%d2359')}"
    )
    try:
        resp = requests.get(url, headers=HEADERS, timeout=8)
        if resp.status_code != 200:
            return None
        rows = resp.json()
        if not isinstance(rows, list) or len(rows) < 2:
            return None
        valid = [r for r in rows if r.get("localDate") and r.get("closePrice") is not None]
        if len(valid) < 2:
            return None

        latest = valid[-1]
        end_date = datetime.strptime(latest["localDate"], "%Y%m%d")
        target_str = (end_date - timedelta(days=30)).strftime("%Y%m%d")

        before_target = [r for r in valid[:-1] if r["localDate"] <= target_str]
        base = before_target[-1] if before_target else valid[0]

        p_start = float(base["closePrice"])
        p_end = float(latest["closePrice"])
        if p_start <= 0:
            return None
        return round((p_end - p_start) / p_start * 100.0, 2)
    except Exception:
        return None


def enrich_holdings_with_returns(etf_infos: list[dict[str, Any]]) -> None:
    """모든 구성종목의 고유 티커를 병렬 조회하여 1개월 수익률(%)을 채웁니다."""
    targets: dict[tuple[str, bool], float | None] = {}
    for info in etf_infos:
        for h in info["holdings"]:
            code = h["구성종목코드"]
            if code:
                targets[(code, bool(h["_is_domestic"]))] = None

    with ThreadPoolExecutor(max_workers=8) as executor:
        future_map = {
            executor.submit(fetch_1m_return, code, is_dom): (code, is_dom)
            for code, is_dom in targets
        }
        for future in as_completed(future_map):
            key = future_map[future]
            try:
                targets[key] = future.result()
            except Exception:
                targets[key] = None

    for info in etf_infos:
        for h in info["holdings"]:
            key = (h["구성종목코드"], bool(h["_is_domestic"]))
            h["1개월 수익률(%)"] = targets.get(key)


def build_overlap_table(
    all_df: pd.DataFrame, etf_infos: list[dict[str, Any]], min_etf_count: int = 2
) -> pd.DataFrame:
    """각 ETF의 포트폴리오 비중을 반영해 개별 구성종목의 전체 실제 비중(%) 요약표를 생성합니다."""
    if all_df.empty:
        return pd.DataFrame()

    etf_labels = {
        info["code"]: f"{info['name']} ({info['code']}, {info['portfolio_weight']:.1f}%)"
        for info in etf_infos
    }

    meta = (
        all_df.groupby("_match_key", as_index=False)
        .agg(
            구성종목명=("구성종목명", "first"),
            구성종목코드=("구성종목코드", "first"),
            수익률_1M=("1개월 수익률(%)", "first"),
            편입_ETF_수=("ETF코드", "nunique"),
            전체_실제_비중=("실제 기여 비중(%)", "sum"),
        )
    )
    meta = meta[meta["편입_ETF_수"] >= min_etf_count].copy()
    if meta.empty:
        return pd.DataFrame()

    meta["전체_실제_비중"] = meta["전체_실제_비중"].round(2)

    pivot = all_df.pivot_table(
        index="_match_key",
        columns="ETF코드",
        values="보유비중(%)",
        aggfunc="sum",
        fill_value=0.0,
    )

    ordered_codes = [info["code"] for info in etf_infos if info["code"] in pivot.columns]
    pivot = pivot[ordered_codes]

    merged = meta.set_index("_match_key").join(pivot, how="inner").reset_index(drop=True)
    merged = merged.rename(
        columns={
            "편입_ETF_수": "편입 ETF 수",
            "수익률_1M": "1개월 수익률(%)",
            "전체_실제_비중": "전체 실제 비중(%)",
        }
    )
    merged = merged.rename(columns=etf_labels)

    etf_col_names = [etf_labels[c] for c in ordered_codes]
    final_cols = [
        "구성종목명",
        "구성종목코드",
        "1개월 수익률(%)",
        "편입 ETF 수",
        "전체 실제 비중(%)",
        *etf_col_names,
    ]
    merged = merged[final_cols].sort_values(
        by=["전체 실제 비중(%)", "편입 ETF 수"], ascending=[False, False]
    )
    return merged.reset_index(drop=True)


def to_csv_bytes(df: pd.DataFrame) -> bytes:
    """Windows Excel에서 한글이 깨지지 않도록 UTF-8 BOM(utf-8-sig) CSV 바이트로 변환합니다."""
    return df.to_csv(index=False).encode("utf-8-sig")


# --- UI ---
header_col, btn_col = st.columns([5, 1])
with header_col:
    st.title("📊 퇴직연금 ETF 구성종목 및 실제 비중 분석")
    st.caption(
        f"구글 시트(`퇴직연금v1` · `Code` 시트)에 등록된 ETF 종목코드와 포트폴리오 비중을 자동으로 불러와 "
        f"상위 구성종목, 1개월 수익률, 개별 종목의 전체 실제 비중(%)을 계산합니다. "
        f"([시트 열기]({SHEET_EDIT_URL}))"
    )
with btn_col:
    st.write("")
    st.write("")
    if st.button("🔄 시트 새로고침", use_container_width=True, type="primary"):
        st.cache_data.clear()
        st.rerun()

try:
    with st.spinner("구글 시트에서 포트폴리오 목록을 불러오는 중입니다..."):
        portfolio_items = load_portfolio_from_sheet()
except Exception as exc:
    st.error(f"구글 시트 로드 오류: {exc}")
    st.stop()

if not portfolio_items:
    st.warning("구글 시트(`Code`)에 등록된 ETF 종목이 없습니다.")
    st.stop()

etf_infos: list[dict[str, Any]] = []
errors: list[str] = []

with st.spinner("각 ETF의 최신 구성종목 데이터를 불러오는 중입니다..."):
    for item in portfolio_items:
        code = item["code"]
        weight = item["portfolio_weight"]
        if not re.fullmatch(r"\d{6}", code):
            errors.append(f"`{code}`: 6자리 국내 종목코드 형식이 아닙니다.")
            continue
        try:
            info = fetch_etf_data(code, weight)
            if not info["holdings"]:
                errors.append(f"`{code}` ({info['name']}): 조회된 구성종목 데이터가 없습니다.")
            else:
                etf_infos.append(info)
        except Exception as exc:
            errors.append(f"`{code}`: {exc}")

for err in errors:
    st.warning(err)

if not etf_infos:
    st.error("조회된 ETF 데이터가 없습니다. 시트의 종목코드를 확인해 주세요.")
    st.stop()

with st.spinner("구성종목별 지난 1개월 수익률을 조회하는 중입니다..."):
    enrich_holdings_with_returns(etf_infos)

# 전체 데이터프레임 생성
all_rows = [row for info in etf_infos for row in info["holdings"]]
all_df = pd.DataFrame(all_rows)
export_df = all_df.drop(columns=["_match_key", "_is_domestic"])

# 1. ETF 요약 정보
st.subheader("1. 조회된 ETF 개요 (구글 시트 기준)")
total_etf_weight = sum(info["portfolio_weight"] for info in etf_infos)
st.caption(f"조회된 ETF **{len(etf_infos)}개** · 시트 내 ETF 비중 합계: **{total_etf_weight:.2f}%**")

summary_rows = []
for info in etf_infos:
    top_weight_sum = sum(h["보유비중(%)"] for h in info["holdings"])
    summary_rows.append(
        {
            "종목코드": info["code"],
            "ETF명": info["name"],
            "포트폴리오 비중(%)": info["portfolio_weight"],
            "현재가(원)": info["close_price"],
            "NAV(원)": info["nav"],
            "등락률(%)": info["change_rate"],
            "총 편입종목 수": info["total_count"],
            "상위 10종목 비중 합계(%)": round(top_weight_sum, 2),
        }
    )
st.dataframe(
    pd.DataFrame(summary_rows),
    use_container_width=True,
    hide_index=True,
    column_config={
        "포트폴리오 비중(%)": st.column_config.NumberColumn("포트폴리오 비중(%)", format="%.2f%%"),
        "상위 10종목 비중 합계(%)": st.column_config.NumberColumn(
            "상위 10종목 비중 합계(%)", format="%.2f%%"
        ),
    },
)

# 2. ETF별 상위 구성종목 리스트 및 한눈에 보는 표
st.subheader("2. ETF별 상위 구성종목 및 보유 비중(%)")

tab_side, tab_all = st.tabs(["📌 ETF별 나란히 보기", "📋 전체 통합 표 보기"])

with tab_side:
    cols_per_row = min(len(etf_infos), 3)
    for start in range(0, len(etf_infos), cols_per_row):
        cols = st.columns(cols_per_row)
        for col, info in zip(cols, etf_infos[start : start + cols_per_row]):
            with col:
                st.markdown(
                    f"**{info['name']}** (`{info['code']}` · 비중 `{info['portfolio_weight']:.1f}%`)"
                )
                etf_df = pd.DataFrame(info["holdings"])[
                    ["순위", "구성종목명", "구성종목코드", "보유비중(%)", "1개월 수익률(%)"]
                ]
                st.dataframe(
                    etf_df,
                    use_container_width=True,
                    hide_index=True,
                    column_config={
                        "보유비중(%)": st.column_config.ProgressColumn(
                            "보유비중(%)",
                            format="%.2f%%",
                            min_value=0.0,
                            max_value=float(max(etf_df["보유비중(%)"].max(), 1.0)),
                        ),
                        "1개월 수익률(%)": st.column_config.NumberColumn(
                            "1개월 수익률(%)",
                            format="%+.2f%%",
                        ),
                    },
                )

with tab_all:
    st.dataframe(
        export_df,
        use_container_width=True,
        hide_index=True,
        column_config={
            "ETF 비중(%)": st.column_config.NumberColumn("ETF 비중(%)", format="%.2f%%"),
            "보유비중(%)": st.column_config.NumberColumn("보유비중(%)", format="%.2f%%"),
            "실제 기여 비중(%)": st.column_config.NumberColumn("실제 기여 비중(%)", format="%.4f%%"),
            "1개월 수익률(%)": st.column_config.NumberColumn("1개월 수익률(%)", format="%+.2f%%"),
        },
    )

# 3. 공통 중복 편입 종목 및 전체 실제 비중 요약
st.subheader("3. ETF 간 공통 중복 편입 종목 요약 (전체 실제 비중 계산)")
st.caption(
    "계산식: **전체 실제 비중(%) = Σ (ETF 포트폴리오 비중(%) × ETF 내 구성종목 비중(%) ÷ 100)**"
)

if len(etf_infos) < 2:
    overlap_df = build_overlap_table(all_df, etf_infos, min_etf_count=1)
else:
    only_overlap = st.checkbox(
        "2개 이상 ETF에 중복 편입된 종목만 보기 (체크 해제 시 전체 상위 구성종목 표시)",
        value=True,
    )
    min_count_option = 2 if only_overlap else 1
    overlap_df = build_overlap_table(all_df, etf_infos, min_etf_count=min_count_option)

if overlap_df.empty:
    st.info("선택한 조건에 해당하는 구성종목이 없습니다.")
else:
    st.caption(
        f"총 **{len(overlap_df)}개** 종목이 표시됩니다. "
        "(우측 각 ETF 컬럼의 수치는 해당 ETF 내부에서의 보유 비중 %입니다.)"
    )
    etf_col_names = [
        f"{info['name']} ({info['code']}, {info['portfolio_weight']:.1f}%)"
        for info in etf_infos
    ]
    col_cfg = {
        "1개월 수익률(%)": st.column_config.NumberColumn("1개월 수익률(%)", format="%+.2f%%"),
        "전체 실제 비중(%)": st.column_config.NumberColumn("전체 실제 비중(%)", format="%.2f%%"),
    }
    for col_name in etf_col_names:
        if col_name in overlap_df.columns:
            col_cfg[col_name] = st.column_config.NumberColumn(col_name, format="%.2f%%")

    st.dataframe(
        overlap_df,
        use_container_width=True,
        hide_index=True,
        column_config=col_cfg,
    )

    chart_df = overlap_df.set_index("구성종목명")[["전체 실제 비중(%)"]]
    st.bar_chart(chart_df, height=360)

# 4. CSV 다운로드 버튼
st.subheader("4. 데이터 다운로드 (Excel 호환 CSV)")
timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
dl_col1, dl_col2 = st.columns(2)

with dl_col1:
    st.download_button(
        label="📥 전체 구성종목 데이터 CSV 다운로드",
        data=to_csv_bytes(export_df),
        file_name=f"etf_constituents_{timestamp}.csv",
        mime="text/csv",
        use_container_width=True,
        type="primary",
    )

with dl_col2:
    if not overlap_df.empty:
        st.download_button(
            label="📥 종목별 실제 비중 요약 CSV 다운로드",
            data=to_csv_bytes(overlap_df),
            file_name=f"etf_actual_weight_summary_{timestamp}.csv",
            mime="text/csv",
            use_container_width=True,
        )
