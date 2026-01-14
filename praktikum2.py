import streamlit as st
import pandas as pd

# Judul aplikasi
st.title("Cek Judul dan Status SKP")

# 1. Membaca file Excel
df = pd.read_excel("data_skp_2.xlsx")

# 2. Membuat kolom status (default: Lulus)
df["status"] = "Lulus"

# 3. Membuat kolom status_judul
#    Cek apakah kolom JUDUL kosong atau tidak
df["status_judul"] = df["JUDUL"].apply(
    lambda x: "Judul Kosong" if pd.isnull(x) or x == "" else "Ada Judul"
)

# 4. Menampilkan data di Streamlit
st.subheader("Data SKP")
st.dataframe(df)