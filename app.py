import streamlit as st
import pymannkendall as mk
import yfinance as yf
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import mplfinance as mpf

st.set_page_config(
    page_title="Phân tích cổ phiếu VCB",
    layout="wide"
)

st.title("📈 PHÂN TÍCH CỔ PHIẾU VCB")

# Sidebar
st.sidebar.header("Tùy chọn")

ticker = st.sidebar.text_input("Mã cổ phiếu", "VCB.VN")
start_date = st.sidebar.date_input(
    "Ngày bắt đầu",
    pd.to_datetime("2026-01-01")
)
end_date = st.sidebar.date_input(
    "Ngày kết thúc",
    pd.to_datetime("2026-06-27")
)

if st.sidebar.button("Phân tích"):

    # Tải dữ liệu
    df = yf.download(
        ticker,
        start=start_date,
        end=end_date,
        progress=False
    )

    if df.empty:
        st.error("Không tìm thấy dữ liệu!")
    else:

        # Nếu dữ liệu có MultiIndex
        if isinstance(df.columns, pd.MultiIndex):
            df.columns = df.columns.droplevel(1)

        # Điền ngày thiếu
        full_date_range = pd.date_range(
            start=df.index.min(),
            end=df.index.max(),
            freq='D'
        )

        df = df.reindex(full_date_range)
        df = df.ffill()

        # Tính lợi nhuận
        df['simple_ret'] = df['Close'].pct_change()
        df['log_ret'] = np.log(
            df['Close'] / df['Close'].shift(1)
        )

        st.subheader("📋 Dữ liệu cổ phiếu")
        st.dataframe(df.tail())

        # ===== BIỂU ĐỒ GIÁ ĐÓNG CỬA + LOG RETURN =====
        st.subheader("📊 Biểu đồ giá và Log Return")

        fig, ax = plt.subplots(
            2, 1,
            figsize=(10, 7),
            sharex=True
        )

        # Giá đóng cửa
        ax[0].plot(
            df.index,
            df['Close'],
            color='red',
            label='Giá đóng cửa'
        )

        ax[0].set_title(
            f'GIÁ ĐÓNG CỬA CỦA {ticker}'
        )

        ax[0].set_ylabel('VND')
        ax[0].legend()
        ax[0].grid(True)

        # Log Return
        ax[1].plot(
            df.index,
            df['log_ret'],
            color='green',
            label='Log Return'
        )

        ax[1].set_title(f'{ticker} Log Return')
        ax[1].set_ylabel('Log Return')
        ax[1].set_xlabel('Date')
        ax[1].legend()
        ax[1].grid(True)

        plt.tight_layout()

        st.pyplot(fig)

        # ===== BIỂU ĐỒ NẾN =====
        st.subheader("🕯️ Biểu đồ nến")

        fig2, axlist = mpf.plot(
            df,
            type='candle',
            mav=(10, 20),
            volume=True,
            style='yahoo',
            title=f'GIÁ CỔ PHIẾU {ticker}',
            figsize=(10, 6),
            returnfig=True
        )

        st.pyplot(fig2)

        # ===== KIỂM ĐỊNH MANN-KENDALL =====
        st.subheader("📈 Kiểm định Mann-Kendall")

        close_prices = df["Close"].dropna().reset_index(drop=True)

        result = mk.original_test(close_prices)

        col1, col2, col3, col4 = st.columns(4)

        col1.metric("Trend", result.trend)
        col2.metric("p-value", f"{result.p:.4f}")
        col3.metric("Tau", f"{result.Tau:.4f}")
        col4.metric("Var(S)", f"{result.var_s:.2f}")

        st.write("### Diễn giải")

        if result.p < 0.05:
            st.success(
                "Có xu hướng đáng kể về mặt thống kê."
            )
        else:
            st.warning(
                "Không có xu hướng rõ ràng."
            )
