import streamlit as st
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import statsmodels.api as sm
from sklearn.cluster import KMeans
from sklearn.preprocessing import StandardScaler

# ==========================================================
# KONFIGURASI HALAMAN
# ==========================================================
st.set_page_config(page_title="Dashboard Kepuasan Pegawai", layout="wide")
st.title("📊 Dashboard Mini Kepuasan Layanan Kepegawaian")
st.markdown("Analisis berbasis data untuk mendukung rekomendasi kebijakan")

# ==========================================================
# LOAD DATA
# ==========================================================
df = pd.read_excel("Data_Survei_Kepuasan_Layanan_Kepegawaian.xlsx")

# Ambil indikator V1–V5 dan pastikan numerik
indikator = df.iloc[:, 1:6].apply(pd.to_numeric, errors="coerce")

# ==========================================================
# KPI KEPUASAN (IKM)
# ==========================================================
mean_scores = indikator.mean()
ikm = (mean_scores.mean() / 5) * 100

def kategori_ikm(x):
    if x >= 81: return "Sangat Baik"
    elif x >= 66: return "Baik"
    elif x >= 51: return "Cukup"
    else: return "Kurang"

col1, col2, col3 = st.columns(3)
col1.metric("📈 Indeks Kepuasan (IKM)", f"{ikm:.2f}%")
col2.metric("🏷️ Kategori", kategori_ikm(ikm))
col3.metric("👥 Responden", len(df))

st.divider()

# ==========================================================
# 3️⃣ ANALISIS GAP
# ==========================================================
st.header("3️⃣ Analisis GAP (Expectation vs Performance)")

gap_scores = 5 - mean_scores
prioritas_gap = gap_scores.idxmax()

fig_gap, ax_gap = plt.subplots(figsize=(6,4))
ax_gap.bar(gap_scores.index, gap_scores.values, color=plt.cm.Set2(range(len(gap_scores))))
ax_gap.set_ylabel("Nilai GAP")
ax_gap.set_title("GAP Kepuasan per Indikator")
ax_gap.grid(axis="y", linestyle="--", alpha=0.6)

for i, v in enumerate(gap_scores.values):
    ax_gap.text(i, v + 0.03, f"{v:.2f}", ha="center", fontweight="bold")

st.pyplot(fig_gap)
st.info(f"📌 Prioritas perbaikan tercepat: **{prioritas_gap}**")

st.divider()

# ==========================================================
# 4️⃣ ANALISIS KORELASI
# ==========================================================
st.header("4️⃣ Korelasi Antar Indikator")

corr = indikator.corr()

fig_corr, ax_corr = plt.subplots(figsize=(6,5))
im = ax_corr.imshow(corr, cmap="coolwarm", vmin=-1, vmax=1)
plt.colorbar(im, ax=ax_corr)

ax_corr.set_xticks(range(len(corr.columns)))
ax_corr.set_yticks(range(len(corr.columns)))
ax_corr.set_xticklabels(corr.columns, rotation=45, ha="right")
ax_corr.set_yticklabels(corr.columns)

for i in range(len(corr)):
    for j in range(len(corr)):
        ax_corr.text(j, i, f"{corr.iloc[i, j]:.2f}", ha="center", va="center")

ax_corr.set_title("Heatmap Korelasi Pearson")
st.pyplot(fig_corr)

corr_v5 = corr.iloc[:-1, -1].sort_values(ascending=False)
st.subheader("📊 Ranking Faktor Berpengaruh")
st.dataframe(corr_v5.to_frame("Koefisien Korelasi"))

st.divider()

# ==========================================================
# 5️⃣ ANALISIS REGRESI
# ==========================================================
st.header("5️⃣ Analisis Regresi Linear Berganda")

X = sm.add_constant(indikator.iloc[:, 0:4])
y = indikator.iloc[:, 4]

model = sm.OLS(y, X, missing="drop").fit()

coef = model.params[1:]
r2 = model.rsquared

fig_reg, ax_reg = plt.subplots(figsize=(6,4))
ax_reg.bar(coef.index, coef.values, color="#3498db")
ax_reg.axhline(0, linestyle="--", color="black")
ax_reg.set_title("Koefisien Regresi")

st.pyplot(fig_reg)
st.info(f"📈 Nilai R²: **{r2:.2f}**")
st.success(f"🔑 Faktor dominan: **{coef.abs().idxmax()}**")

st.divider()

# ==========================================================
# 6️⃣ SEGMENTASI KEPUASAN (FIX PALING AMAN)
# ==========================================================
st.header("6️⃣ Segmentasi Kepuasan Pegawai")

scaler = StandardScaler()
X_scaled = scaler.fit_transform(indikator.fillna(indikator.mean()))

kmeans = KMeans(n_clusters=3, random_state=42, n_init=10)
cluster_label = kmeans.fit_predict(X_scaled)

# 🔥 GROUPBY PALING AMAN (HANYA DATA NUMERIK)
indikator_cluster = indikator.copy()
indikator_cluster["Cluster"] = cluster_label

cluster_mean = indikator_cluster.groupby("Cluster").mean()
cluster_mean = cluster_mean.sort_values(by=indikator.columns[-1], ascending=False)

cluster_mean["Segment"] = ["Sangat Puas", "Cukup Puas", "Tidak Puas"]

# Radar Chart
labels = indikator.columns.tolist()
angles = np.linspace(0, 2*np.pi, len(labels), endpoint=False).tolist()
angles += angles[:1]

fig_rad = plt.figure(figsize=(6,6))
ax_rad = plt.subplot(polar=True)

colors = ["#2ecc71", "#f1c40f", "#e74c3c"]

for i, row in cluster_mean.iterrows():
    values = row[labels].tolist() + [row[labels].tolist()[0]]
    ax_rad.plot(angles, values, label=row["Segment"], color=colors[i])
    ax_rad.fill(angles, values, alpha=0.25)

ax_rad.set_thetagrids(np.degrees(angles[:-1]), labels)
ax_rad.set_ylim(0, 5)
ax_rad.set_title("Radar Segmentasi Kepuasan")
ax_rad.legend(loc="upper right")

st.pyplot(fig_rad)

st.success("📌 Segmentasi berhasil – siap untuk rekomendasi kebijakan")
