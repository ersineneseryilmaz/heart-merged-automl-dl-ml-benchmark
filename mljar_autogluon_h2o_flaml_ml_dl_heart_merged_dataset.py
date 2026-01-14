# -*- coding: utf-8 -*-
"""MLJar-AutoGluon-H2O-FLAML-ML-DL_Heart_merged_dataset.ipynb
Coded and edited by Ersin Enes Eryılmaz.
Original file is located at
    https://colab.research.google.com/drive/19cZc2jJdWZWIqZNwuvS9wmNu4AGT6WZo
"""

pip freeze > requirements.txt

"""**0-ÖNİŞLEME:** Boşlukları medyan mod doldur, gerekli önişleme işlemleri yap"""

import pandas as pd
import numpy as np

def preprocess_heart_csv(input_csv_path: str, output_csv_path: str = "heart_clean.csv") -> pd.DataFrame:
    # 1) Oku: '?' vb. eksik değerleri NaN yap
    df = pd.read_csv(
        input_csv_path,
        na_values=["?", "??", "???", "????", " ?"],
        skipinitialspace=True,
        engine="python"
    )

    # 2) Kolon adlarını temizle (baş/son boşlukları sil)
    df.columns = [c.strip() for c in df.columns]

    # 3) Beklenen kolonları kontrol et (opsiyonel ama önerilir)
    expected = ["age","sex","cp","trestbps","chol","fbs","restecg","thalach","exang","oldpeak","slope","ca","thal","target"]
    missing_cols = [c for c in expected if c not in df.columns]
    if missing_cols:
        raise ValueError(f"CSV'de beklenen kolon(lar) yok: {missing_cols}\nMevcut kolonlar: {list(df.columns)}")

    # 4) Tüm sütunları sayısala çevir (sayısal olmayanlar NaN olur)
    for c in expected:
        df[c] = pd.to_numeric(df[c], errors="coerce")

    # 5) target'ı binary yap: 0 -> 0, 1-4 -> 1
    # target eksik olan satırları at (istersen median ile doldurmak yerine)
    df = df[df["target"].notna()].copy()
    df["target"] = (df["target"] > 0).astype(int)

    # 6) Feature eksiklerini kolon medyanı ile doldur
    feature_cols = [c for c in expected if c != "target"]
    medians = df[feature_cols].median(numeric_only=True)
    df[feature_cols] = df[feature_cols].fillna(medians)

    # (Opsiyonel) Hâlâ NaN kaldıysa (tamamen boş kolon gibi), uyarı ver
    remaining_nan = df[feature_cols].isna().sum().sum()
    if remaining_nan > 0:
        print(f"Uyarı: Doldurma sonrası hâlâ {remaining_nan} adet NaN kaldı (tamamen boş kolon olabilir).")

    # 7) Kaydet
    df.to_csv(output_csv_path, index=False)
    print(f"Kaydedildi: {output_csv_path} | Shape: {df.shape}")

    return df


# Kullanım:
df_clean = preprocess_heart_csv("cleveland_hungarian_long-beach-va_switzerland.csv", "heart_merged_clean.csv")

"""**Önişleme yapılmadan önce veri kümesinin özellikleri**"""

# ============================================================
# Preprocess ÖNCESİ (ham CSV) için 900x900 görseller
# - Dosya: cleveland_hungarian_long-beach-va_switzerland.csv
# - '?' vb. değerleri NaN'a çevirir, kolon adlarını strip eder
# - DÖNÜŞÜM / DOLDURMA YOK (median fill yok)
# - 900x900 px (dpi=150 -> 6x6 inch)
# - Çıktılar ./figures_raw_900/ altına kaydedilir (PNG)
# ============================================================

import os
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt


def load_raw_heart_csv(raw_csv_path: str) -> pd.DataFrame:
    """Ham dosyayı, preprocessing yapmadan (sadece NaN okuma + kolon trim) yükler."""
    df = pd.read_csv(
        raw_csv_path,
        na_values=["?", "??", "???", "????", " ?"],
        skipinitialspace=True,
        engine="python"
    )
    df.columns = [c.strip() for c in df.columns]
    return df


def _save_900(fig, out_path: str, dpi: int = 150):
    # 900px = 6 inch * 150 dpi
    fig.set_size_inches(6, 6, forward=True)
    fig.tight_layout()
    fig.savefig(out_path, dpi=dpi, bbox_inches="tight")
    plt.close(fig)


def make_peerj_figures_raw_900(
    raw_csv_path: str = "cleveland_hungarian_long-beach-va_switzerland.csv",
    target_col: str = "target",
    out_dir: str = "figures_raw_900"
):
    os.makedirs(out_dir, exist_ok=True)
    df_raw = load_raw_heart_csv(raw_csv_path)

    # Beklenen kolonlar (varsa bunlarla çalış; yoksa mevcut kolonlarla devam)
    expected = ["age","sex","cp","trestbps","chol","fbs","restecg","thalach","exang","oldpeak","slope","ca","thal","target"]
    cols = [c for c in expected if c in df_raw.columns]
    if not cols:
        cols = list(df_raw.columns)

    # Sayısal görseller için (ham veri -> sadece numeric'e çevir; DOLDURMA YOK)
    df_num = df_raw.copy()
    for c in cols:
        df_num[c] = pd.to_numeric(df_num[c], errors="coerce")

    feature_cols = [c for c in cols if c != target_col and c in df_num.columns]

    # ========================================================
    # FIG-R1: Kolonların eksik değer sayısı (ham veri)
    # ========================================================
    fig, ax = plt.subplots()
    nan_counts = df_num[cols].isna().sum().sort_values(ascending=False)
    ax.bar(nan_counts.index.astype(str), nan_counts.values)
    ax.set_title("Raw Data: Missing Values per Column")
    ax.set_xlabel("Column")
    ax.set_ylabel("Missing count")
    ax.tick_params(axis="x", rotation=90)
    _save_900(fig, os.path.join(out_dir, "FigR1_raw_missing_values_900.png"))

    # ========================================================
    # FIG-R2: Data type özeti (sayısal/nesne) + non-null sayıları
    # (Ham veri karakter içerdiği için rapor figürü gibi düşün)
    # ========================================================
    fig, ax = plt.subplots()
    non_null = df_raw[cols].notna().sum().sort_values(ascending=False)
    ax.bar(non_null.index.astype(str), non_null.values)
    ax.set_title("Raw Data: Non-null Counts per Column")
    ax.set_xlabel("Column")
    ax.set_ylabel("Non-null count")
    ax.tick_params(axis="x", rotation=90)
    _save_900(fig, os.path.join(out_dir, "FigR2_raw_nonnull_counts_900.png"))

    # ========================================================
    # FIG-R3: Target ham dağılımı (numeric'e çevrilmiş haliyle)
    # Not: ham target 0-4 olabilir; NaN olanlar ayrı kalır
    # ========================================================
    if target_col in df_num.columns:
        fig, ax = plt.subplots()
        t = df_num[target_col]
        # NaN'ları ayrı gösterelim
        counts = t.value_counts(dropna=False).sort_index()
        labels = []
        values = []
        for idx, v in counts.items():
            if pd.isna(idx):
                labels.append("NaN")
            else:
                # tam sayıysa daha temiz yaz
                labels.append(str(int(idx)) if float(idx).is_integer() else str(idx))
            values.append(int(v))

        ax.bar(labels, values)
        ax.set_title("Raw Data: Target Distribution (Original)")
        ax.set_xlabel("Target value")
        ax.set_ylabel("Count")
        for i, v in enumerate(values):
            ax.text(i, v, str(v), ha="center", va="bottom")
        _save_900(fig, os.path.join(out_dir, "FigR3_raw_target_distribution_900.png"))

    # ========================================================
    # FIG-R4: Feature completeness heatmap (satır bazlı doluluk)
    # - Seaborn yok; matplotlib imshow
    # - İlk 200 satırı göster (çok satır varsa görsel şişmesin)
    # ========================================================
    max_rows = 200
    show_cols = feature_cols[:13]  # 13 özellik tipik; daha fazlaysa ilk 13
    if show_cols:
        mat = df_num[show_cols].notna().astype(int).head(max_rows).values  # 1=var, 0=eksik
        fig, ax = plt.subplots()
        im = ax.imshow(mat, aspect="auto")
        ax.set_title(f"Raw Data: Feature Completeness (first {min(len(df_num), max_rows)} rows)")
        ax.set_xlabel("Feature")
        ax.set_ylabel("Row index")
        ax.set_xticks(range(len(show_cols)))
        ax.set_xticklabels([str(c) for c in show_cols], rotation=90)
        fig.colorbar(im, ax=ax, fraction=0.046, pad=0.04)
        _save_900(fig, os.path.join(out_dir, "FigR4_raw_completeness_heatmap_900.png"))

    # ========================================================
    # FIG-R5: Ham sayısal dağılım (boxplot) - NaN'lar otomatik elenir
    # - Çok NaN varsa yine de fikir verir
    # ========================================================
    if feature_cols:
        # çok fazla feature olursa boxplot kalabalıklaşır: ilk 13 ile sınırla
        fcols = feature_cols[:13]
        plot_data = [df_num[c].dropna().values for c in fcols]

        fig, ax = plt.subplots()
        ax.boxplot(plot_data, showfliers=False)
        ax.set_title("Raw Data: Feature Distributions (Boxplot)")
        ax.set_xlabel("Feature")
        ax.set_ylabel("Value")
        ax.set_xticks(range(1, len(fcols) + 1))
        ax.set_xticklabels([str(c) for c in fcols], rotation=90)
        _save_900(fig, os.path.join(out_dir, "FigR5_raw_boxplot_900.png"))

    # ========================================================
    # FIG-R6: Korelasyon (ham numeric) - NaN içeren satırlar düşer
    # - Pairwise correlation için pandas corr zaten NaN'i pairwise ele alır
    # ========================================================
    if feature_cols and target_col in df_num.columns:
        corr = df_num[feature_cols + [target_col]].corr(numeric_only=True)

        fig, ax = plt.subplots()
        im = ax.imshow(corr.values, aspect="auto")
        ax.set_title("Raw Data: Correlation Matrix (Numeric Only)")

        ax.set_xticks(range(corr.shape[1]))
        ax.set_yticks(range(corr.shape[0]))
        ax.set_xticklabels(corr.columns.astype(str), rotation=90)
        ax.set_yticklabels(corr.index.astype(str))
        fig.colorbar(im, ax=ax, fraction=0.046, pad=0.04)
        _save_900(fig, os.path.join(out_dir, "FigR6_raw_correlation_matrix_900.png"))

    print(f"\nBitti. Ham veri için 900x900 görselleri kaydedildi: ./{out_dir}/")
    print("Üretilen dosyalar:")
    for fn in sorted(os.listdir(out_dir)):
        if fn.lower().endswith(".png"):
            print(" -", fn)


# ----------------- Çalıştırma -----------------
make_peerj_figures_raw_900(
    raw_csv_path="cleveland_hungarian_long-beach-va_switzerland.csv",
    target_col="target",
    out_dir="figures_raw_900"
)

"""**Önişlenmiş veri kümesinin özellikleri**"""

# ============================================================
# 900x900 görseller (dataset özelliklerine göre)
# - 900x900 px (dpi=150 -> 6x6 inch)
# - Matplotlib ( default)
# - Çıktılar ./figures_900/ altına kaydedilir (PNG)
# ============================================================

import os
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt

# ---------- 900x900 fig kaydedici ----------
def _save_900(fig, out_path: str, dpi: int = 150):
    # 900px = 6 inch * 150 dpi
    fig.set_size_inches(6, 6, forward=True)
    fig.tight_layout()
    fig.savefig(out_path, dpi=dpi, bbox_inches="tight")
    plt.close(fig)


def make_peerj_figures_900(
    clean_csv_path: str = "heart_merged_clean.csv",
    target_col: str = "target",
    out_dir: str = "figures_900"
):
    os.makedirs(out_dir, exist_ok=True)
    df = pd.read_csv(clean_csv_path)

    # güvenlik: target int olsun
    df[target_col] = df[target_col].astype(int)

    feature_cols = [c for c in df.columns if c != target_col]

    # ========================================================
    # FIG-1: Target dağılımı (Class balance)
    # ========================================================
    fig, ax = plt.subplots()
    counts = df[target_col].value_counts().sort_index()
    ax.bar([str(i) for i in counts.index], counts.values)
    ax.set_title("Target Class Distribution")
    ax.set_xlabel("Class (target)")
    ax.set_ylabel("Count")
    for i, v in enumerate(counts.values):
        ax.text(i, v, str(v), ha="center", va="bottom")
    _save_900(fig, os.path.join(out_dir, "Fig1_target_distribution_900.png"))

    # ========================================================
    # FIG-2: Özelliklerin eksik değer sayısı (preprocess öncesi için faydalı)
    # Not: clean_csv zaten doldurulmuş olabilir; yine de rapor olur.
    # ========================================================
    fig, ax = plt.subplots()
    nan_counts = df[feature_cols].isna().sum().sort_values(ascending=False)
    ax.bar(nan_counts.index.astype(str), nan_counts.values)
    ax.set_title("Missing Values per Feature")
    ax.set_xlabel("Feature")
    ax.set_ylabel("Missing count")
    ax.tick_params(axis="x", rotation=90)
    _save_900(fig, os.path.join(out_dir, "Fig2_missing_values_900.png"))

    # ========================================================
    # FIG-3: Özelliklerin dağılımı (Boxplot; target'a göre)
    # ========================================================
    fig, ax = plt.subplots()
    # target 0 ve 1 için ayrı kutu grafiği: her feature için iki kutu
    data0 = df[df[target_col] == 0][feature_cols]
    data1 = df[df[target_col] == 1][feature_cols]

    # boxplot için uygun format: [feat0_class0, feat0_class1, feat1_class0, feat1_class1, ...]
    plot_data = []
    positions = []
    labels = []
    pos = 1
    for f in feature_cols:
        plot_data.append(data0[f].values)
        plot_data.append(data1[f].values)
        positions.extend([pos, pos + 0.35])
        labels.append(f)
        pos += 1

    ax.boxplot(plot_data, positions=positions, widths=0.25, showfliers=False)
    ax.set_title("Feature Distributions by Target (Boxplot)")
    ax.set_xlabel("Feature (pairs: 0 then 1)")
    ax.set_ylabel("Value")
    ax.set_xticks([i + 0.175 for i in range(1, len(feature_cols) + 1)])
    ax.set_xticklabels([str(f) for f in feature_cols], rotation=90)

    # küçük bir açıklama
    ax.text(0.02, 0.98, "Each feature has two boxes: target=0 (left), target=1 (right)",
            transform=ax.transAxes, ha="left", va="top")
    _save_900(fig, os.path.join(out_dir, "Fig3_boxplot_by_target_900.png"))

    # ========================================================
    # FIG-4: Korelasyon Isı Haritası (Pearson)
    # - Seaborn kullanmadan matplotlib imshow
    # ========================================================
    corr = df[feature_cols + [target_col]].corr(numeric_only=True)

    fig, ax = plt.subplots()
    im = ax.imshow(corr.values, aspect="auto")
    ax.set_title("Correlation Matrix (Pearson)")

    ax.set_xticks(range(corr.shape[1]))
    ax.set_yticks(range(corr.shape[0]))
    ax.set_xticklabels(corr.columns.astype(str), rotation=90)
    ax.set_yticklabels(corr.index.astype(str))

    # colorbar
    fig.colorbar(im, ax=ax, fraction=0.046, pad=0.04)

    # değerleri yazdırmak istersen (kalabalık olabilir):
    # for i in range(corr.shape[0]):
    #     for j in range(corr.shape[1]):
    #         ax.text(j, i, f"{corr.values[i,j]:.2f}", ha="center", va="center", fontsize=6)

    _save_900(fig, os.path.join(out_dir, "Fig4_correlation_matrix_900.png"))

    # ========================================================
    # FIG-5: İki güçlü klinik özellik: age vs thalach (scatter), target'a göre
    # - Bu ikisi genelde iyi anlatıyor; istersen değiştir.
    # ========================================================
    x_col, y_col = "age", "thalach"
    if x_col in df.columns and y_col in df.columns:
        fig, ax = plt.subplots()
        df0 = df[df[target_col] == 0]
        df1 = df[df[target_col] == 1]
        ax.scatter(df0[x_col], df0[y_col], label="target=0", alpha=0.8)
        ax.scatter(df1[x_col], df1[y_col], label="target=1", alpha=0.8)
        ax.set_title(f"{x_col} vs {y_col} by Target")
        ax.set_xlabel(x_col)
        ax.set_ylabel(y_col)
        ax.legend()
        _save_900(fig, os.path.join(out_dir, "Fig5_scatter_age_thalach_900.png"))

    print(f"\nBitti. 900x900 px görselleri kaydedildi: ./{out_dir}/")
    print("Üretilen dosyalar:")
    for fn in sorted(os.listdir(out_dir)):
        if fn.lower().endswith(".png"):
            print(" -", fn)


# ----------------- Çalıştırma -----------------
# 1) Preprocess (ham dosyadan temiz CSV üret)
df_clean = preprocess_heart_csv(
    "cleveland_hungarian_long-beach-va_switzerland.csv",
    "heart_merged_clean.csv"
)

# 2) 900x900 görselleri üret
make_peerj_figures_900(
    clean_csv_path="heart_merged_clean.csv",
    target_col="target",
    out_dir="figures_900"
)

pip freeze | egrep "scikit|xgboost|lightgbm|catboost|mljar|flaml|numpy|pandas"

# ============================================================
# Figure 3: End-to-end experimental workflow (900x900 px)
# - PeerJ-friendly: 900x900 (dpi=150 -> 6x6 inch)
# - Uses matplotlib patches (no seaborn)
# - Output: ./figures_900/Fig3_workflow_900.png
# ============================================================

import os
import matplotlib.pyplot as plt
from matplotlib.patches import FancyBboxPatch, FancyArrowPatch

def _save_900(fig, out_path: str, dpi: int = 150):
    fig.set_size_inches(6, 6, forward=True)  # 6*150=900 px
    fig.tight_layout()
    fig.savefig(out_path, dpi=dpi, bbox_inches="tight")
    plt.close(fig)

def _box(ax, xy, w, h, text, fontsize=9):
    x, y = xy
    patch = FancyBboxPatch(
        (x, y), w, h,
        boxstyle="round,pad=0.02,rounding_size=0.02",
        linewidth=1.2,
        facecolor="white"
    )
    ax.add_patch(patch)
    ax.text(x + w/2, y + h/2, text, ha="center", va="center", fontsize=fontsize)
    return patch

def _arrow(ax, p1, p2):
    ax.add_patch(FancyArrowPatch(
        p1, p2, arrowstyle="->", mutation_scale=12, linewidth=1.2
    ))

def make_fig3_workflow(out_dir="figures_900"):
    os.makedirs(out_dir, exist_ok=True)

    fig, ax = plt.subplots()
    ax.set_axis_off()
    ax.set_xlim(0, 1)
    ax.set_ylim(0, 1)

    # Layout (normalized coordinates)
    W, H = 0.82, 0.10
    X = 0.09

    b1 = _box(ax, (X, 0.86), W, H, "Raw merged CSV\n(Cleveland + Hungary + Switzerland + VA Long Beach)")
    b2 = _box(ax, (X, 0.72), W, H, "Unified preprocessing\n(“?”→NaN, numeric casting, target binarization,\nmedian imputation for features)")
    b3 = _box(ax, (X, 0.58), W, H, "Stratified split\nOuter: Train/Test\nInner: Train/Val (early stopping / selection)")
    b4 = _box(ax, (X, 0.44), W, H, "Train (two comparable regimes)\nML: Light vs Full\nDL: Light vs Full\nAutoML: Light vs Full")
    b5 = _box(ax, (X, 0.30), W, H, "Evaluate on held-out test set\nAccuracy, AUC, F1, Precision, Recall, LogLoss,\nConfusion Matrix")
    b6 = _box(ax, (X, 0.16), W, H, "Resource logging\nRuntime (wall-clock), RAM (process RSS & system delta),\nModel size (serialized file MB)")
    b7 = _box(ax, (X, 0.02), W, H, "Report results\nTables/Figures + exact configs for reproducibility")

    # Arrows (center-bottom to center-top)
    def mid_bottom(box):  # (x,y,w,h) from patch
        x, y = box.get_x(), box.get_y()
        w, h = box.get_width(), box.get_height()
        return (x + w/2, y)
    def mid_top(box):
        x, y = box.get_x(), box.get_y()
        w, h = box.get_width(), box.get_height()
        return (x + w/2, y + h)

    for a, b in [(b1, b2), (b2, b3), (b3, b4), (b4, b5), (b5, b6), (b6, b7)]:
        _arrow(ax, mid_bottom(a), mid_top(b))

    ax.set_title("Figure 3. End-to-end experimental workflow", fontsize=11)

    out_path = os.path.join(out_dir, "Fig3_workflow_900.png")
    _save_900(fig, out_path)
    print("Saved:", out_path)

# Run
make_fig3_workflow(out_dir="figures_900")

# Commented out IPython magic to ensure Python compatibility.
# %pip install mljar-supervised

"""**MLJAR FULL**"""

# -*- coding: utf-8 -*-
"""mljar_heart_full.py"""

import logging
logging.getLogger("matplotlib.font_manager").setLevel(logging.ERROR)


import os
import time
import psutil
import joblib
import pandas as pd
from supervised.automl import AutoML
from sklearn.model_selection import train_test_split
from sklearn.metrics import (
    accuracy_score, roc_auc_score, f1_score,
    precision_score, recall_score, log_loss, confusion_matrix
)

# import matplotlib.pyplot as plt
# plt.rcParams['font.family'] = 'sans-serif']
import numpy as np # Import numpy for np.unique

# === Sistem RAM ölçüm fonksiyonu ===
def get_system_ram():
    mem = psutil.virtual_memory()
    total = mem.total / (1024**3)
    used = mem.used / (1024**3)
    free = mem.available / (1024**3)
    percent = mem.percent
    return total, used, free, percent

# === Başlangıç ölçümleri ===
process = psutil.Process(os.getpid())
start_proc_mem = process.memory_info().rss / (1024 * 1024)  # MB
start_sys_total, start_sys_used, start_sys_free, start_sys_percent = get_system_ram()
start_time = time.time()

# 1. Veri kümesini yükle
data = pd.read_csv("heart_merged_clean.csv")
target_col = "target"
X = data.drop(columns=[target_col])
y = data[target_col]

# 2. Eğitim/Test bölme
X_train, X_test, y_train, y_test = train_test_split(
    X, y, test_size=0.2, random_state=42, stratify=y
)

# 3. MLJAR AutoML ayarları (tam/Compete sürüm)
automl = AutoML(
    mode="Compete",
    algorithms=[
        "LightGBM", "Xgboost", "CatBoost",
        "Random Forest", "Extra Trees", "Neural Network"
    ],
    total_time_limit=180,
    model_time_limit=100,
    start_random_models=5,
    hill_climbing_steps=3,
    stack_models=True,
    golden_features=True,
    features_selection=True,
    explain_level=2,
    random_state=42,
)

# 4. Model eğitimi
automl.fit(X_train, y_train)

# 5. Test tahmini
y_pred = automl.predict(X_test)
y_pred_proba = automl.predict_proba(X_test)[:, 1]

# 6. Performans ölçümleri
acc = accuracy_score(y_test, y_pred)
auc = roc_auc_score(y_test, y_pred_proba)
f1 = f1_score(y_test, y_pred)
precision = precision_score(y_test, y_pred)
recall = recall_score(y_test, y_pred)
ll = log_loss(y_test, y_pred_proba)
cm = confusion_matrix(y_test, y_pred)

# === Bitiş ölçümleri ===
end_time = time.time()
end_proc_mem = process.memory_info().rss / (1024 * 1024)
end_sys_total, end_sys_used, end_sys_free, end_sys_percent = get_system_ram()

runtime = end_time - start_time
proc_mem_usage = end_proc_mem - start_proc_mem
sys_used_diff = end_sys_used - start_sys_used

# === En iyi model dosya boyutu (joblib üzerinden) ===
best_model = automl._best_model  # MLJAR içindeki en iyi model nesnesi
best_model_file = "mljar_heart_full_best_model.pkl"
joblib.dump(best_model, best_model_file)
model_size_mb = os.path.getsize(best_model_file) / (1024 * 1024)

# === Sonuçlar ===
print("\n=== MLJAR Normal (Compete) Model Performansı (Test) ===")
print(f"Accuracy:  {acc*100:.2f}%")
print(f"AUC:       {auc*100:.2f}%")
print(f"F1 Score:  {f1*100:.2f}%")
print(f"Precision: {precision*100:.2f}%")
print(f"Recall:    {recall*100:.2f}%")
print(f"LogLoss:   {ll:.2f}")

print("\nConfusion Matrix:")
print(cm)

print("\n=== Özet Bilgiler ===")
print(f"Koşum Süresi: {runtime:.2f} saniye")
print(f"Python İşlemi RAM Kullanımı: {proc_mem_usage:.2f} MB")

print("\n=== Sistem Genel RAM Kullanımı ===")
print(f"Toplam RAM:         {end_sys_total:.2f} GB")
print(f"Kullanılan RAM:     {end_sys_used:.2f} GB")
print(f"Boş RAM:            {end_sys_free:.2f} GB")
print(f"RAM Kullanım Oranı: {end_sys_percent:.1f}%")
print(f"Koşum sırasında artan RAM: {sys_used_diff:.2f} GB")

print("\n=== En İyi Model Bilgileri ===")
print(f"En iyi model tipi: {type(best_model).__name__}")
print(f"Model dosyası: {best_model_file}")
print(f"Model boyutu: {model_size_mb:.2f} MB")

# 9. Leaderboard
leaderboard = automl.get_leaderboard()
print("\n=== Leaderboard ===")
print(leaderboard.round(2))

import gc
del automl
gc.collect()
time.sleep(2)
end_proc_mem2 = process.memory_info().rss / (1024 * 1024)
print(f"\n[GC sonrası] Python RSS: {end_proc_mem2:.2f} MB")

"""**HAFİF MLJAR**"""

# -*- coding: utf-8 -*-
"""mljar_heart_light.py"""
import logging
logging.getLogger("matplotlib.font_manager").setLevel(logging.ERROR)

import os
import time
import psutil
import joblib
import pandas as pd
from supervised.automl import AutoML
from sklearn.model_selection import train_test_split
from sklearn.metrics import (
    accuracy_score, roc_auc_score, f1_score,
    precision_score, recall_score, log_loss, confusion_matrix, classification_report
)

# === Sistem RAM ölçüm fonksiyonu ===
def get_system_ram():
    mem = psutil.virtual_memory()
    total = mem.total / (1024**3)
    used = mem.used / (1024**3)
    free = mem.available / (1024**3)
    percent = mem.percent
    return total, used, free, percent

# === Başlangıç ölçümleri ===
process = psutil.Process(os.getpid())
start_proc_mem = process.memory_info().rss / (1024 * 1024)  # MB
start_sys_total, start_sys_used, start_sys_free, start_sys_percent = get_system_ram()
start_time = time.time()

# 1. Veri kümesini yükle
data = pd.read_csv("heart_merged_clean.csv")
y = "target"
X = data.drop(columns=[y])
y_data = data[y]

# 2. Eğitim/Test bölme
X_train, X_test, y_train, y_test = train_test_split(
    X, y_data, test_size=0.2, random_state=42, stratify=y_data
)

# 3. MLJAR AutoML (hafif) ayarları
automl = AutoML(
    mode="Explain",   # hafif ve hızlı
    algorithms=["LightGBM", "Xgboost", "CatBoost", "Random Forest"],
    total_time_limit=60,
    explain_level=1,
    random_state=42,
    start_random_models=2,
    hill_climbing_steps=0,
)

# 4. Eğitim
automl.fit(X_train, y_train)

# 5. Tahmin
y_pred = automl.predict(X_test)
y_pred_proba = automl.predict_proba(X_test)[:, 1]

# 6. Performans ölçümleri
acc = accuracy_score(y_test, y_pred)
auc = roc_auc_score(y_test, y_pred_proba)
f1 = f1_score(y_test, y_pred)
precision = precision_score(y_test, y_pred)
recall = recall_score(y_test, y_pred)
ll = log_loss(y_test, y_pred_proba)
cm = confusion_matrix(y_test, y_pred)

# === Bitiş ölçümleri ===
end_time = time.time()
end_proc_mem = process.memory_info().rss / (1024 * 1024)
end_sys_total, end_sys_used, end_sys_free, end_sys_percent = get_system_ram()

runtime = end_time - start_time
proc_mem_usage = end_proc_mem - start_proc_mem
sys_ram_diff = end_sys_used - start_sys_used

# === En iyi model dosya boyutu (joblib ile) ===
best_model = automl._best_model
best_model_file = "mljar_heart_light_best_model.pkl"
joblib.dump(best_model, best_model_file)
model_size_mb = os.path.getsize(best_model_file) / (1024 * 1024)

# 7. Sonuçlar
print("\n=== MLJAR Hafif Model Performansı (Test) ===")
print(f"Accuracy:  {acc*100:.2f}%")
print(f"AUC:       {auc*100:.2f}%")
print(f"F1 Score:  {f1*100:.2f}%")
print(f"Precision: {precision*100:.2f}%")
print(f"Recall:    {recall*100:.2f}%")
print(f"LogLoss:   {ll:.2f}")

print("\nConfusion Matrix:")
print(cm)

print("\nClassification Report:")
print(classification_report(y_test, y_pred))

print("\n=== Özet Bilgiler ===")
print(f"Koşum Süresi: {runtime:.2f} saniye")
print(f"Python RAM Artışı: {proc_mem_usage:.2f} MB")

print("\n=== Sistem Genel RAM Kullanımı ===")
print(f"Toplam RAM:         {end_sys_total:.2f} GB")
print(f"Kullanılan RAM:     {end_sys_used:.2f} GB")
print(f"Boş RAM:            {end_sys_free:.2f} GB")
print(f"RAM Kullanım Oranı: {end_sys_percent:.1f}%")
print(f"Koşum sırasında artan RAM: {sys_ram_diff:.2f} GB")

print("\n=== En İyi Model Dosyası ===")
print(f"En iyi model tipi: {type(best_model).__name__}")
print(f"Model yolu: {best_model_file}")
print(f"Model boyutu: {model_size_mb:.4f} MB")

# 8. Leaderboard
leaderboard = automl.get_leaderboard()
print("\n=== Leaderboard ===")
print(leaderboard.round(2))

import gc
del automl
gc.collect()
time.sleep(2)
end_proc_mem2 = process.memory_info().rss / (1024 * 1024)
print(f"\n[GC sonrası] Python RSS: {end_proc_mem2:.2f} MB")

"""**FULL AUTOGLUON**"""

!pip install autogluon

# -*- coding: utf-8 -*-
"""autogluon_heart_full.py"""

import os
import time
import psutil
import pandas as pd
import numpy as np
from sklearn.model_selection import train_test_split
from sklearn.metrics import (
    accuracy_score, roc_auc_score, f1_score,
    precision_score, recall_score, log_loss, confusion_matrix
)

from autogluon.tabular import TabularPredictor


def get_system_ram():
    mem = psutil.virtual_memory()
    total = mem.total / (1024**3)
    used = mem.used / (1024**3)
    free = mem.available / (1024**3)
    percent = mem.percent
    return total, used, free, percent


def get_dir_size_mb(path: str) -> float:
    total_bytes = 0
    for root, _, files in os.walk(path):
        for f in files:
            fp = os.path.join(root, f)
            if os.path.isfile(fp):
                total_bytes += os.path.getsize(fp)
    return total_bytes / (1024 * 1024)


def pick_positive_proba(proba_df: pd.DataFrame, y_train: pd.Series) -> np.ndarray:
    # AutoGluon predict_proba => columns are class labels
    labels = sorted(pd.unique(y_train))
    pos_label = labels[-1]  # usually 1
    if pos_label in proba_df.columns:
        return proba_df[pos_label].to_numpy()
    # fallback: take last column
    return proba_df.iloc[:, -1].to_numpy()


# === Başlangıç ölçümleri ===
process = psutil.Process(os.getpid())
start_proc_mem = process.memory_info().rss / (1024 * 1024)  # MB
start_sys_total, start_sys_used, start_sys_free, start_sys_percent = get_system_ram()
start_time = time.time()

# 1) Veri
data = pd.read_csv("heart_merged_clean.csv")
target_col = "target"

X = data.drop(columns=[target_col])
y = data[target_col]

train_df, test_df = train_test_split(
    data, test_size=0.2, random_state=42, stratify=y
)

# 2) AutoGluon ayarları
# - time_limit: toplam süre limiti
# - presets: hızlı eğitim odaklı preset
# - num_bag_folds=5, num_stack_levels=0: stacking kapalı (hızlı)
# Kaynak: AutoGluon TabularPredictor.fit parametreleri. :contentReference[oaicite:3]{index=3}
predictor_path = "autogluon_heart_full_predictor"

predictor = TabularPredictor(
    label=target_col,
    eval_metric="f1",
    path=predictor_path
)

hyperparameters = None

predictor.fit(
    train_data=train_df,
    presets="best_quality",
    time_limit=180,
    num_bag_folds=5,
    num_stack_levels=0,
    hyperparameters=hyperparameters,
)

# 3) Tahmin
y_pred = predictor.predict(test_df)
proba_df = predictor.predict_proba(test_df)
y_pred_proba = pick_positive_proba(proba_df, train_df[target_col])

# 4) Metrikler
y_test = test_df[target_col].to_numpy()
y_pred_np = pd.Series(y_pred).astype(int).to_numpy()

acc = accuracy_score(y_test, y_pred_np)
auc = roc_auc_score(y_test, y_pred_proba)
f1 = f1_score(y_test, y_pred_np)
precision = precision_score(y_test, y_pred_np)
recall = recall_score(y_test, y_pred_np)
ll = log_loss(y_test, y_pred_proba)
cm = confusion_matrix(y_test, y_pred_np)

# === Bitiş ölçümleri ===
end_time = time.time()
end_proc_mem = process.memory_info().rss / (1024 * 1024)
end_sys_total, end_sys_used, end_sys_free, end_sys_percent = get_system_ram()

runtime = end_time - start_time
proc_mem_usage = end_proc_mem - start_proc_mem
sys_used_diff = end_sys_used - start_sys_used

# Predictor boyutu
model_size_mb = get_dir_size_mb(predictor_path)

print("\n=== AutoGluon Full Model Performansı (Test) ===")
print(f"Accuracy:  {acc*100:.2f}%")
print(f"AUC:       {auc*100:.2f}%")
print(f"F1 Score:  {f1*100:.2f}%")
print(f"Precision: {precision*100:.2f}%")
print(f"Recall:    {recall*100:.2f}%")
print(f"LogLoss:   {ll:.2f}")

print("\nConfusion Matrix:")
print(cm)

print("\n=== Özet Bilgiler ===")
print(f"Koşum Süresi: {runtime:.2f} saniye")
print(f"Python İşlemi RAM Kullanımı: {proc_mem_usage:.2f} MB")

print("\n=== Sistem Genel RAM Kullanımı ===")
print(f"Toplam RAM:         {end_sys_total:.2f} GB")
print(f"Kullanılan RAM:     {end_sys_used:.2f} GB")
print(f"Boş RAM:            {end_sys_free:.2f} GB")
print(f"RAM Kullanım Oranı: {end_sys_percent:.1f}%")
print(f"Koşum sırasında artan RAM: {sys_used_diff:.2f} GB")

print("\n=== Model Bilgileri ===")
print(f"Predictor path: {predictor_path}")
print(f"Predictor boyutu: {model_size_mb:.2f} MB")

# Leaderboard
lb = predictor.leaderboard(test_df, silent=True)
print("\n=== Leaderboard ===")
print(lb.round(4))

"""**LIGHT AUTOGLUON**"""

# -*- coding: utf-8 -*-
"""autogluon_heart_light.py"""

import os
import time
import psutil
import pandas as pd
import numpy as np
from sklearn.model_selection import train_test_split
from sklearn.metrics import (
    accuracy_score, roc_auc_score, f1_score,
    precision_score, recall_score, log_loss, confusion_matrix
)

from autogluon.tabular import TabularPredictor


def get_system_ram():
    mem = psutil.virtual_memory()
    total = mem.total / (1024**3)
    used = mem.used / (1024**3)
    free = mem.available / (1024**3)
    percent = mem.percent
    return total, used, free, percent


def get_dir_size_mb(path: str) -> float:
    total_bytes = 0
    for root, _, files in os.walk(path):
        for f in files:
            fp = os.path.join(root, f)
            if os.path.isfile(fp):
                total_bytes += os.path.getsize(fp)
    return total_bytes / (1024 * 1024)


def pick_positive_proba(proba_df: pd.DataFrame, y_train: pd.Series) -> np.ndarray:
    # AutoGluon predict_proba => columns are class labels
    labels = sorted(pd.unique(y_train))
    pos_label = labels[-1]  # usually 1
    if pos_label in proba_df.columns:
        return proba_df[pos_label].to_numpy()
    # fallback: take last column
    return proba_df.iloc[:, -1].to_numpy()


# === Başlangıç ölçümleri ===
process = psutil.Process(os.getpid())
start_proc_mem = process.memory_info().rss / (1024 * 1024)  # MB
start_sys_total, start_sys_used, start_sys_free, start_sys_percent = get_system_ram()
start_time = time.time()

# 1) Veri
data = pd.read_csv("heart_merged_clean.csv")
target_col = "target"

X = data.drop(columns=[target_col])
y = data[target_col]

train_df, test_df = train_test_split(
    data, test_size=0.2, random_state=42, stratify=y
)

# 2) AutoGluon Light ayarları
# - time_limit: toplam süre limiti
# - presets: hızlı eğitim odaklı preset
# - num_bag_folds=0, num_stack_levels=0: bagging/stacking kapalı (hafif)
# Kaynak: AutoGluon TabularPredictor.fit parametreleri. :contentReference[oaicite:3]{index=3}
predictor_path = "autogluon_heart_light_predictor"

predictor = TabularPredictor(
    label=target_col,
    eval_metric="f1",
    path=predictor_path
)

# Model havuzunu da hafif tutmak için sınırlıyoruz.
# (Ortamda bazı opsiyonel bağımlılıklar yoksa AutoGluon zaten ilgili modeli atlayabilir.)
hyperparameters = {
    "GBM": {},   # LightGBM
    "RF": {},    # RandomForest
    "XT": {},    # ExtraTrees
    "LR": {},    # LogisticRegression
}

predictor.fit(
    train_data=train_df,
    presets="medium_quality_faster_train",
    time_limit=60,
    num_bag_folds=0,
    num_stack_levels=0,
    hyperparameters=hyperparameters,
)

# 3) Tahmin
y_pred = predictor.predict(test_df)
proba_df = predictor.predict_proba(test_df)
y_pred_proba = pick_positive_proba(proba_df, train_df[target_col])

# 4) Metrikler
y_test = test_df[target_col].to_numpy()
y_pred_np = pd.Series(y_pred).astype(int).to_numpy()

acc = accuracy_score(y_test, y_pred_np)
auc = roc_auc_score(y_test, y_pred_proba)
f1 = f1_score(y_test, y_pred_np)
precision = precision_score(y_test, y_pred_np)
recall = recall_score(y_test, y_pred_np)
ll = log_loss(y_test, y_pred_proba)
cm = confusion_matrix(y_test, y_pred_np)

# === Bitiş ölçümleri ===
end_time = time.time()
end_proc_mem = process.memory_info().rss / (1024 * 1024)
end_sys_total, end_sys_used, end_sys_free, end_sys_percent = get_system_ram()

runtime = end_time - start_time
proc_mem_usage = end_proc_mem - start_proc_mem
sys_used_diff = end_sys_used - start_sys_used

# Predictor boyutu
model_size_mb = get_dir_size_mb(predictor_path)

print("\n=== AutoGluon Light Model Performansı (Test) ===")
print(f"Accuracy:  {acc*100:.2f}%")
print(f"AUC:       {auc*100:.2f}%")
print(f"F1 Score:  {f1*100:.2f}%")
print(f"Precision: {precision*100:.2f}%")
print(f"Recall:    {recall*100:.2f}%")
print(f"LogLoss:   {ll:.2f}")

print("\nConfusion Matrix:")
print(cm)

print("\n=== Özet Bilgiler ===")
print(f"Koşum Süresi: {runtime:.2f} saniye")
print(f"Python İşlemi RAM Kullanımı: {proc_mem_usage:.2f} MB")

print("\n=== Sistem Genel RAM Kullanımı ===")
print(f"Toplam RAM:         {end_sys_total:.2f} GB")
print(f"Kullanılan RAM:     {end_sys_used:.2f} GB")
print(f"Boş RAM:            {end_sys_free:.2f} GB")
print(f"RAM Kullanım Oranı: {end_sys_percent:.1f}%")
print(f"Koşum sırasında artan RAM: {sys_used_diff:.2f} GB")

print("\n=== Model Bilgileri ===")
print(f"Predictor path: {predictor_path}")
print(f"Predictor boyutu: {model_size_mb:.2f} MB")

# Leaderboard
lb = predictor.leaderboard(test_df, silent=True)
print("\n=== Leaderboard ===")
print(lb.round(4))

"""**H2O AutoML (Full)**"""

!pip install h2o

# -*- coding: utf-8 -*-
"""h2o_heart_full.py"""

import os
import time
import psutil
import pandas as pd
import numpy as np

from sklearn.model_selection import train_test_split
from sklearn.metrics import (
    accuracy_score, roc_auc_score, f1_score,
    precision_score, recall_score, log_loss, confusion_matrix
)

import h2o
from h2o.automl import H2OAutoML


def get_system_ram():
    mem = psutil.virtual_memory()
    total = mem.total / (1024**3)
    used = mem.used / (1024**3)
    free = mem.available / (1024**3)
    percent = mem.percent
    return total, used, free, percent


# === Başlangıç ölçümleri ===
process = psutil.Process(os.getpid())
start_proc_mem = process.memory_info().rss / (1024 * 1024)  # MB
start_sys_total, start_sys_used, start_sys_free, start_sys_percent = get_system_ram()
start_time = time.time()

# 1) Veri
data = pd.read_csv("heart_merged_clean.csv")
target_col = "target"

train_df, test_df = train_test_split(
    data, test_size=0.2, random_state=42, stratify=data[target_col]
)

# 2) H2O başlat
h2o.init(max_mem_size="4G")
h2o.no_progress()

train_h2o = h2o.H2OFrame(train_df)
test_h2o = h2o.H2OFrame(test_df)

# Sınıflandırma için hedefi factor yap
train_h2o[target_col] = train_h2o[target_col].asfactor()
test_h2o[target_col] = test_h2o[target_col].asfactor()

x_cols = [c for c in train_df.columns if c != target_col]
y_col = target_col

# 3) H2O full ayarları
# - max_runtime_secs: toplam süre limiti
# - nfolds=5: CV
# - exclude_algos ile None
# Kaynak: H2OAutoML parametreleri (max_runtime_secs, exclude_algos, StackedEnsemble). :contentReference[oaicite:4]{index=4}
aml = H2OAutoML(
    max_runtime_secs=180,
    nfolds=5,
    seed=42,
    sort_metric="f1",
    exclude_algos=None,
)

aml.train(x=x_cols, y=y_col, training_frame=train_h2o)

best_model = aml.leader

# 4) Tahmin
preds = best_model.predict(test_h2o)  # columns: predict, p0, p1
preds_df = preds.as_data_frame()

y_pred = preds_df["predict"].astype(int).to_numpy()

# p1 sütunu (pozitif sınıf olasılığı) genelde "p1"
# Bazı durumlarda sınıf etiketine göre p0/p1 isimlenmesi değişebilir; bu nedenle fallback ekliyoruz.
proba_col = "p1" if "p1" in preds_df.columns else preds_df.columns[-1]
y_pred_proba = preds_df[proba_col].to_numpy()

y_test = test_df[target_col].astype(int).to_numpy()

# 5) Metrikler
acc = accuracy_score(y_test, y_pred)
auc = roc_auc_score(y_test, y_pred_proba)
f1 = f1_score(y_test, y_pred)
precision = precision_score(y_test, y_pred)
recall = recall_score(y_test, y_pred)
ll = log_loss(y_test, y_pred_proba)
cm = confusion_matrix(y_test, y_pred)

# 6) Modeli kaydet
best_model_file = h2o.save_model(best_model, path=".", force=True)
model_size_mb = os.path.getsize(best_model_file) / (1024 * 1024)

# === Bitiş ölçümleri ===
end_time = time.time()
end_proc_mem = process.memory_info().rss / (1024 * 1024)
end_sys_total, end_sys_used, end_sys_free, end_sys_percent = get_system_ram()

runtime = end_time - start_time
proc_mem_usage = end_proc_mem - start_proc_mem
sys_used_diff = end_sys_used - start_sys_used

print("\n=== H2O AutoML Full Model Performansı (Test) ===")
print(f"Accuracy:  {acc*100:.2f}%")
print(f"AUC:       {auc*100:.2f}%")
print(f"F1 Score:  {f1*100:.2f}%")
print(f"Precision: {precision*100:.2f}%")
print(f"Recall:    {recall*100:.2f}%")
print(f"LogLoss:   {ll:.2f}")

print("\nConfusion Matrix:")
print(cm)

print("\n=== Özet Bilgiler ===")
print(f"Koşum Süresi: {runtime:.2f} saniye")
print(f"Python İşlemi RAM Kullanımı: {proc_mem_usage:.2f} MB")

print("\n=== Sistem Genel RAM Kullanımı ===")
print(f"Toplam RAM:         {end_sys_total:.2f} GB")
print(f"Kullanılan RAM:     {end_sys_used:.2f} GB")
print(f"Boş RAM:            {end_sys_free:.2f} GB")
print(f"RAM Kullanım Oranı: {end_sys_percent:.1f}%")
print(f"Koşum sırasında artan RAM: {sys_used_diff:.2f} GB")

print("\n=== En İyi Model Bilgileri ===")
print(f"En iyi model tipi: {type(best_model).__name__}")
print(f"Model dosyası: {best_model_file}")
print(f"Model boyutu: {model_size_mb:.2f} MB")

print("\n=== Leaderboard ===")
print(aml.leaderboard.as_data_frame().round(4))

# H2O'yu kapat (RAM serbest)
h2o.shutdown(prompt=False)

"""**LIGHT H2O**"""

# -*- coding: utf-8 -*-
"""h2o_heart_light.py"""

import os
import time
import psutil
import pandas as pd
import numpy as np

from sklearn.model_selection import train_test_split
from sklearn.metrics import (
    accuracy_score, roc_auc_score, f1_score,
    precision_score, recall_score, log_loss, confusion_matrix
)

import h2o
from h2o.automl import H2OAutoML


def get_system_ram():
    mem = psutil.virtual_memory()
    total = mem.total / (1024**3)
    used = mem.used / (1024**3)
    free = mem.available / (1024**3)
    percent = mem.percent
    return total, used, free, percent


# === Başlangıç ölçümleri ===
process = psutil.Process(os.getpid())
start_proc_mem = process.memory_info().rss / (1024 * 1024)  # MB
start_sys_total, start_sys_used, start_sys_free, start_sys_percent = get_system_ram()
start_time = time.time()

# 1) Veri
data = pd.read_csv("heart_merged_clean.csv")
target_col = "target"

train_df, test_df = train_test_split(
    data, test_size=0.2, random_state=42, stratify=data[target_col]
)

# 2) H2O başlat
h2o.init(max_mem_size="4G")
h2o.no_progress()

train_h2o = h2o.H2OFrame(train_df)
test_h2o = h2o.H2OFrame(test_df)

# Sınıflandırma için hedefi factor yap
train_h2o[target_col] = train_h2o[target_col].asfactor()
test_h2o[target_col] = test_h2o[target_col].asfactor()

x_cols = [c for c in train_df.columns if c != target_col]
y_col = target_col

# 3) H2O Light ayarları
# - max_runtime_secs: toplam süre limiti
# - nfolds=0: CV yok (hafif)
# - exclude_algos ile StackedEnsemble ve DeepLearning kapalı
# Kaynak: H2OAutoML parametreleri (max_runtime_secs, exclude_algos, StackedEnsemble). :contentReference[oaicite:4]{index=4}
aml = H2OAutoML(
    max_runtime_secs=60,
    nfolds=0,
    seed=42,
    sort_metric="F1",
    exclude_algos=["StackedEnsemble", "DeepLearning"],
)

aml.train(x=x_cols, y=y_col, training_frame=train_h2o)

best_model = aml.leader

# 4) Tahmin
preds = best_model.predict(test_h2o)  # columns: predict, p0, p1
preds_df = preds.as_data_frame()

y_pred = preds_df["predict"].astype(int).to_numpy()

# p1 sütunu (pozitif sınıf olasılığı) genelde "p1"
# Bazı durumlarda sınıf etiketine göre p0/p1 isimlenmesi değişebilir; bu nedenle fallback ekliyoruz.
proba_col = "p1" if "p1" in preds_df.columns else preds_df.columns[-1]
y_pred_proba = preds_df[proba_col].to_numpy()

y_test = test_df[target_col].astype(int).to_numpy()

# 5) Metrikler
acc = accuracy_score(y_test, y_pred)
auc = roc_auc_score(y_test, y_pred_proba)
f1 = f1_score(y_test, y_pred)
precision = precision_score(y_test, y_pred)
recall = recall_score(y_test, y_pred)
ll = log_loss(y_test, y_pred_proba)
cm = confusion_matrix(y_test, y_pred)

# 6) Modeli kaydet
best_model_file = h2o.save_model(best_model, path=".", force=True)
model_size_mb = os.path.getsize(best_model_file) / (1024 * 1024)

# === Bitiş ölçümleri ===
end_time = time.time()
end_proc_mem = process.memory_info().rss / (1024 * 1024)
end_sys_total, end_sys_used, end_sys_free, end_sys_percent = get_system_ram()

runtime = end_time - start_time
proc_mem_usage = end_proc_mem - start_proc_mem
sys_used_diff = end_sys_used - start_sys_used

print("\n=== H2O AutoML Light Model Performansı (Test) ===")
print(f"Accuracy:  {acc*100:.2f}%")
print(f"AUC:       {auc*100:.2f}%")
print(f"F1 Score:  {f1*100:.2f}%")
print(f"Precision: {precision*100:.2f}%")
print(f"Recall:    {recall*100:.2f}%")
print(f"LogLoss:   {ll:.2f}")

print("\nConfusion Matrix:")
print(cm)

print("\n=== Özet Bilgiler ===")
print(f"Koşum Süresi: {runtime:.2f} saniye")
print(f"Python İşlemi RAM Kullanımı: {proc_mem_usage:.2f} MB")

print("\n=== Sistem Genel RAM Kullanımı ===")
print(f"Toplam RAM:         {end_sys_total:.2f} GB")
print(f"Kullanılan RAM:     {end_sys_used:.2f} GB")
print(f"Boş RAM:            {end_sys_free:.2f} GB")
print(f"RAM Kullanım Oranı: {end_sys_percent:.1f}%")
print(f"Koşum sırasında artan RAM: {sys_used_diff:.2f} GB")

print("\n=== En İyi Model Bilgileri ===")
print(f"En iyi model tipi: {type(best_model).__name__}")
print(f"Model dosyası: {best_model_file}")
print(f"Model boyutu: {model_size_mb:.2f} MB")

print("\n=== Leaderboard ===")
print(aml.leaderboard.as_data_frame().round(4))

# H2O'yu kapat (RAM serbest)
h2o.shutdown(prompt=False)

"""**FLAML (Full)**"""

!pip install flaml

# -*- coding: utf-8 -*-
"""flaml_heart_full.py"""

import os
import time
import psutil
import joblib
import pandas as pd
import numpy as np

from sklearn.model_selection import train_test_split
from sklearn.metrics import (
    accuracy_score, roc_auc_score, f1_score,
    precision_score, recall_score, log_loss, confusion_matrix
)

from flaml import AutoML


def get_system_ram():
    mem = psutil.virtual_memory()
    total = mem.total / (1024**3)
    used = mem.used / (1024**3)
    free = mem.available / (1024**3)
    percent = mem.percent
    return total, used, free, percent


def pick_pos_proba(proba: np.ndarray) -> np.ndarray:
    # proba shape: (n,2) beklenir; fallback: son sütun
    if proba.ndim == 2 and proba.shape[1] >= 2:
        return proba[:, 1]
    if proba.ndim == 1:
        return proba
    return proba[:, -1]


# === Başlangıç ölçümleri ===
process = psutil.Process(os.getpid())
start_proc_mem = process.memory_info().rss / (1024 * 1024)  # MB
start_sys_total, start_sys_used, start_sys_free, start_sys_percent = get_system_ram()
start_time = time.time()

# 1) Veri
data = pd.read_csv("heart_merged_clean.csv")
target_col = "target"
X = data.drop(columns=[target_col])
y = data[target_col].astype(int)

X_train, X_test, y_train, y_test = train_test_split(
    X, y, test_size=0.2, random_state=42, stratify=y
)

# 2) FLAML ayarları
# - time_budget: toplam süre limiti (saniye)
# - task/metric ayarları
# - predict_proba ve en iyi estimator'a automl.model.estimator ile erişim
# Kaynak: FLAML sınıflandırma örneği. :contentReference[oaicite:5]{index=5}
automl = AutoML()

automl_settings = {
    "time_budget": 180,
    "task": "classification",
    "metric": "f1",
    "seed": 42,
    "log_file_name": "flaml_heart_full.log",
    # 5-fold CV
    #"eval_method": "cv",
    "n_splits": 5,
    # daha hızlı değerlendirme:
    "eval_method": "holdout",
    "split_ratio": 0.2,
    "n_jobs": -1,
}

automl.fit(X_train=X_train, y_train=y_train, **automl_settings)

# 3) Tahmin
y_pred = automl.predict(X_test).astype(int)
y_pred_proba = pick_pos_proba(automl.predict_proba(X_test))

# 4) Metrikler
acc = accuracy_score(y_test, y_pred)
auc = roc_auc_score(y_test, y_pred_proba)
f1 = f1_score(y_test, y_pred)
precision = precision_score(y_test, y_pred)
recall = recall_score(y_test, y_pred)
ll = log_loss(y_test, y_pred_proba)
cm = confusion_matrix(y_test, y_pred)

# 5) En iyi modeli kaydet
best_model = automl.model
best_model_file = "flaml_heart_full_best_model.pkl"
joblib.dump(best_model, best_model_file)
model_size_mb = os.path.getsize(best_model_file) / (1024 * 1024)

# === Bitiş ölçümleri ===
end_time = time.time()
end_proc_mem = process.memory_info().rss / (1024 * 1024)
end_sys_total, end_sys_used, end_sys_free, end_sys_percent = get_system_ram()

runtime = end_time - start_time
proc_mem_usage = end_proc_mem - start_proc_mem
sys_used_diff = end_sys_used - start_sys_used

print("\n=== FLAML Full Model Performansı (Test) ===")
print(f"Accuracy:  {acc*100:.2f}%")
print(f"AUC:       {auc*100:.2f}%")
print(f"F1 Score:  {f1*100:.2f}%")
print(f"Precision: {precision*100:.2f}%")
print(f"Recall:    {recall*100:.2f}%")
print(f"LogLoss:   {ll:.2f}")

print("\nConfusion Matrix:")
print(cm)

print("\n=== Özet Bilgiler ===")
print(f"Koşum Süresi: {runtime:.2f} saniye")
print(f"Python İşlemi RAM Kullanımı: {proc_mem_usage:.2f} MB")

print("\n=== Sistem Genel RAM Kullanımı ===")
print(f"Toplam RAM:         {end_sys_total:.2f} GB")
print(f"Kullanılan RAM:     {end_sys_used:.2f} GB")
print(f"Boş RAM:            {end_sys_free:.2f} GB")
print(f"RAM Kullanım Oranı: {end_sys_percent:.1f}%")
print(f"Koşum sırasında artan RAM: {sys_used_diff:.2f} GB")

print("\n=== En İyi Model Bilgileri ===")
print(f"En iyi model tipi: {type(best_model).__name__}")
print(f"En iyi estimator:  {getattr(best_model, 'estimator', 'N/A')}")
print(f"Model dosyası: {best_model_file}")
print(f"Model boyutu: {model_size_mb:.2f} MB")

"""**LIGHT FLAML**"""

# -*- coding: utf-8 -*-
"""flaml_heart_light.py"""

import os
import time
import psutil
import joblib
import pandas as pd
import numpy as np

from sklearn.model_selection import train_test_split
from sklearn.metrics import (
    accuracy_score, roc_auc_score, f1_score,
    precision_score, recall_score, log_loss, confusion_matrix
)

from flaml import AutoML


def get_system_ram():
    mem = psutil.virtual_memory()
    total = mem.total / (1024**3)
    used = mem.used / (1024**3)
    free = mem.available / (1024**3)
    percent = mem.percent
    return total, used, free, percent


def pick_pos_proba(proba: np.ndarray) -> np.ndarray:
    # proba shape: (n,2) beklenir; fallback: son sütun
    if proba.ndim == 2 and proba.shape[1] >= 2:
        return proba[:, 1]
    if proba.ndim == 1:
        return proba
    return proba[:, -1]


# === Başlangıç ölçümleri ===
process = psutil.Process(os.getpid())
start_proc_mem = process.memory_info().rss / (1024 * 1024)  # MB
start_sys_total, start_sys_used, start_sys_free, start_sys_percent = get_system_ram()
start_time = time.time()

# 1) Veri
data = pd.read_csv("heart_merged_clean.csv")
target_col = "target"
X = data.drop(columns=[target_col])
y = data[target_col].astype(int)

X_train, X_test, y_train, y_test = train_test_split(
    X, y, test_size=0.2, random_state=42, stratify=y
)

# 2) FLAML Light ayarları
# - time_budget: toplam süre limiti (saniye)
# - task/metric ayarları
# - predict_proba ve en iyi estimator'a automl.model.estimator ile erişim
# Kaynak: FLAML sınıflandırma örneği. :contentReference[oaicite:5]{index=5}
automl = AutoML()

automl_settings = {
    "time_budget": 60,
    "task": "classification",
    "metric": "f1",
    "seed": 42,
    "log_file_name": "flaml_heart_light.log",
    # hafiflik için (opsiyonel) model listesini daralt:
    "estimator_list": ["lgbm", "rf", "extra_tree", "lrl1"],
    # daha hızlı değerlendirme:
    "eval_method": "holdout",
    "split_ratio": 0.2,
    "n_jobs": -1,
}

automl.fit(X_train=X_train, y_train=y_train, **automl_settings)

# 3) Tahmin
y_pred = automl.predict(X_test).astype(int)
y_pred_proba = pick_pos_proba(automl.predict_proba(X_test))

# 4) Metrikler
acc = accuracy_score(y_test, y_pred)
auc = roc_auc_score(y_test, y_pred_proba)
f1 = f1_score(y_test, y_pred)
precision = precision_score(y_test, y_pred)
recall = recall_score(y_test, y_pred)
ll = log_loss(y_test, y_pred_proba)
cm = confusion_matrix(y_test, y_pred)

# 5) En iyi modeli kaydet
best_model = automl.model
best_model_file = "flaml_heart_light_best_model.pkl"
joblib.dump(best_model, best_model_file)
model_size_mb = os.path.getsize(best_model_file) / (1024 * 1024)

# === Bitiş ölçümleri ===
end_time = time.time()
end_proc_mem = process.memory_info().rss / (1024 * 1024)
end_sys_total, end_sys_used, end_sys_free, end_sys_percent = get_system_ram()

runtime = end_time - start_time
proc_mem_usage = end_proc_mem - start_proc_mem
sys_used_diff = end_sys_used - start_sys_used

print("\n=== FLAML Light Model Performansı (Test) ===")
print(f"Accuracy:  {acc*100:.2f}%")
print(f"AUC:       {auc*100:.2f}%")
print(f"F1 Score:  {f1*100:.2f}%")
print(f"Precision: {precision*100:.2f}%")
print(f"Recall:    {recall*100:.2f}%")
print(f"LogLoss:   {ll:.2f}")

print("\nConfusion Matrix:")
print(cm)

print("\n=== Özet Bilgiler ===")
print(f"Koşum Süresi: {runtime:.2f} saniye")
print(f"Python İşlemi RAM Kullanımı: {proc_mem_usage:.2f} MB")

print("\n=== Sistem Genel RAM Kullanımı ===")
print(f"Toplam RAM:         {end_sys_total:.2f} GB")
print(f"Kullanılan RAM:     {end_sys_used:.2f} GB")
print(f"Boş RAM:            {end_sys_free:.2f} GB")
print(f"RAM Kullanım Oranı: {end_sys_percent:.1f}%")
print(f"Koşum sırasında artan RAM: {sys_used_diff:.2f} GB")

print("\n=== En İyi Model Bilgileri ===")
print(f"En iyi model tipi: {type(best_model).__name__}")
print(f"En iyi estimator:  {getattr(best_model, 'estimator', 'N/A')}")
print(f"Model dosyası: {best_model_file}")
print(f"Model boyutu: {model_size_mb:.2f} MB")

"""**1) Klasik ML Full vs Light (Time-budget’lı)**"""

import os, time, psutil, joblib, warnings, random
warnings.filterwarnings("ignore")

import numpy as np
import pandas as pd

from sklearn.model_selection import train_test_split, ParameterSampler
from sklearn.metrics import (
    accuracy_score, roc_auc_score, f1_score, precision_score,
    recall_score, log_loss, confusion_matrix, classification_report
)
from sklearn.preprocessing import StandardScaler
from sklearn.pipeline import Pipeline
from sklearn.base import clone # ADDED THIS LINE

from sklearn.linear_model import LogisticRegression
from sklearn.ensemble import RandomForestClassifier, ExtraTreesClassifier
from sklearn.neural_network import MLPClassifier

# Optional libs
XGB_AVAILABLE = False
LGB_AVAILABLE = False
CAT_AVAILABLE = False

try:
    from xgboost import XGBClassifier
    XGB_AVAILABLE = True
except Exception:
    pass

try:
    import lightgbm as lgb
    LGB_AVAILABLE = True
except Exception:
    pass

try:
    from catboost import CatBoostClassifier
    CAT_AVAILABLE = True
except Exception:
    pass


def set_seed(seed: int = 42):
    random.seed(seed)
    np.random.seed(seed)
    os.environ["PYTHONHASHSEED"] = str(seed)


def get_system_ram():
    mem = psutil.virtual_memory()
    total = mem.total / (1024**3)
    used = mem.used / (1024**3)
    free = mem.available / (1024**3)
    percent = mem.percent
    return total, used, free, percent


def compute_metrics(y_true, y_pred, y_proba):
    acc = accuracy_score(y_true, y_pred)
    auc = roc_auc_score(y_true, y_proba)
    f1 = f1_score(y_true, y_pred)
    precision = precision_score(y_true, y_pred)
    recall = recall_score(y_true, y_pred)
    ll = log_loss(y_true, np.vstack([1 - y_proba, y_proba]).T)
    cm = confusion_matrix(y_true, y_pred)
    return acc, auc, f1, precision, recall, ll, cm


def predict_proba_safe(model, X):
    if hasattr(model, "predict_proba"):
        return model.predict_proba(X)[:, 1]
    if hasattr(model, "decision_function"):
        s = model.decision_function(X)
        return 1 / (1 + np.exp(-s))
    # fallback: hard preds -> prob
    p = model.predict(X)
    return p.astype(float)


def print_block(title, acc, auc, f1, precision, recall, ll, cm, y_true=None, y_pred=None):
    print(f"\n=== {title} (Test) ===")
    print(f"Accuracy:  {acc*100:.2f}%")
    print(f"AUC:       {auc*100:.2f}%")
    print(f"F1 Score:  {f1*100:.2f}%")
    print(f"Precision: {precision*100:.2f}%")
    print(f"Recall:    {recall*100:.2f}%")
    print(f"LogLoss:   {ll:.2f}")
    print("\nConfusion Matrix:")
    print(cm)
    if y_true is not None and y_pred is not None:
        print("\nClassification Report:")
        print(classification_report(y_true, y_pred))


# -------------------------
# Model factory + search spaces
# -------------------------
def build_ml_candidates(seed=42, mode="full"):
    """
    mode="light": sabit hızlı ayarlar
    mode="full" : timed random search ile param arama yapılacak
    """
    candidates = {}

    # Logistic Regression (scale)
    candidates["LogReg"] = {
        "base": Pipeline([("scaler", StandardScaler()),
                          ("clf", LogisticRegression(max_iter=2000, n_jobs=-1))]),
        "space": {
            "clf__C": np.logspace(-2, 2, 30),
            "clf__penalty": ["l2"],  # liblinear + l1 is possible but keep stable
            "clf__solver": ["lbfgs", "liblinear"]
        }
    }

    # RandomForest
    rf_base = RandomForestClassifier(random_state=seed, n_jobs=-1)
    candidates["RF"] = {
        "base": rf_base,
        "space": {
            "n_estimators": [200, 400, 800, 1200] if mode == "full" else [300],
            "max_depth": [None, 3, 5, 7, 10],
            "max_features": ["sqrt", "log2", None],
            "min_samples_split": [2, 5, 10],
            "min_samples_leaf": [1, 2, 4],
            "bootstrap": [True, False]
        }
    }

    # ExtraTrees
    et_base = ExtraTreesClassifier(random_state=seed, n_jobs=-1)
    candidates["ExtraTrees"] = {
        "base": et_base,
        "space": {
            "n_estimators": [300, 600, 1000, 1500] if mode == "full" else [600],
            "max_depth": [None, 3, 5, 7, 10],
            "max_features": ["sqrt", "log2", None],
            "min_samples_split": [2, 5, 10],
            "min_samples_leaf": [1, 2, 4]
        }
    }

    # MLP (scale)
    candidates["MLP_sklearn"] = {
        "base": Pipeline([
            ("scaler", StandardScaler()),
            ("clf", MLPClassifier(
                activation="relu",
                alpha=1e-4,
                max_iter=300 if mode == "full" else 150,
                random_state=seed
            ))
        ]),
        "space": {
            "clf__hidden_layer_sizes": [(32,), (64,), (64, 32), (128, 64)],
            "clf__alpha": [1e-5, 1e-4, 1e-3, 1e-2],
            "clf__learning_rate_init": [1e-4, 3e-4, 1e-3, 3e-3]
        }
    }

    # XGBoost
    if XGB_AVAILABLE:
        candidates["XGBoost"] = {
            "base": XGBClassifier(
                eval_metric="logloss",
                n_jobs=-1,
                random_state=seed,
                tree_method="hist"
            ),
            "space": {
                "n_estimators": [150, 300, 600, 1000] if mode == "full" else [300],
                "max_depth": [2, 3, 4, 5, 6],
                "learning_rate": [0.01, 0.03, 0.05, 0.1, 0.2],
                "subsample": [0.7, 0.85, 1.0],
                "colsample_bytree": [0.7, 0.85, 1.0],
                "min_child_weight": [1, 3, 5, 10],
                "reg_lambda": [0.5, 1.0, 2.0, 5.0]
            }
        }

    # LightGBM
    if LGB_AVAILABLE:
        candidates["LightGBM"] = {
            "base": lgb.LGBMClassifier(
                random_state=seed,
                n_jobs=-1,
                verbose=-1
            ),
            "space": {
                "n_estimators": [200, 500, 900, 1500] if mode == "full" else [400],
                "learning_rate": [0.01, 0.03, 0.05, 0.1, 0.2],
                "num_leaves": [7, 15, 31, 63],
                "min_child_samples": [5, 10, 20, 40],
                "subsample": [0.7, 0.85, 1.0],
                "colsample_bytree": [0.7, 0.85, 1.0],
                "reg_lambda": [0.0, 1.0, 3.0, 10.0]
            }
        }

    # CatBoost
    if CAT_AVAILABLE:
        candidates["CatBoost"] = {
            "base": CatBoostClassifier(
                loss_function="Logloss",
                verbose=False,
                random_seed=seed,
                allow_writing_files=False
            ),
            "space": {
                "iterations": [300, 800, 1500, 2500] if mode == "full" else [600],
                "learning_rate": [0.01, 0.03, 0.05, 0.1, 0.2],
                "depth": [3, 4, 5, 6, 7, 8],
                "l2_leaf_reg": [1, 3, 5, 10]
            }
        }

    return candidates


def timed_random_search_one_model(
    name, base_model, param_space,
    X_train, y_train, X_val, y_val,
    time_left_s, seed=42, max_trials_cap=10_000
):
    """
    Tek bir model için time_left_s kadar random param dene.
    En iyi val F1'ı döndür.
    """
    t_start = time.time()
    best = {"f1": -1, "params": None, "fit_time": 0.0}

    # param örnekleri (çok büyük bir üst limit ile)
    sampler = ParameterSampler(param_space, n_iter=max_trials_cap, random_state=seed)

    tried = 0
    for params in sampler:
        if time.time() - t_start >= time_left_s:
            break

        model = clone(base_model)
        try:
            model.set_params(**params)
        except Exception:
            continue

        t0 = time.time()
        try:
            model.fit(X_train, y_train)
            fit_t = time.time() - t0

            proba = predict_proba_safe(model, X_val)
            pred = (proba >= 0.5).astype(int)
            f1 = f1_score(y_val, pred)

            tried += 1
            if f1 > best["f1"]:
                best = {"f1": f1, "params": params, "fit_time": fit_t}
        except Exception:
            continue

    return best, tried


def run_classic_ml_full_light(
    clean_csv_path="heart_merged_clean.csv",
    target_col="target",
    test_size=0.2,
    seed=42,
    time_budget_full=180,
    time_budget_light=60,
    holdout_frac_for_tuning=0.2
):
    set_seed(seed)

    # load
    df = pd.read_csv(clean_csv_path)
    X = df.drop(columns=[target_col])
    y = df[target_col].astype(int).values

    # outer split (train/test)
    X_train, X_test, y_train, y_test = train_test_split(
        X, y, test_size=test_size, random_state=seed, stratify=y
    )

    # inner split (train/val) for fair tuning, without touching test
    X_tr, X_val, y_tr, y_val = train_test_split(
        X_train, y_train,
        test_size=holdout_frac_for_tuning,
        random_state=seed,
        stratify=y_train
    )

    def run_mode(mode_name, time_budget_s):
        # RAM + time start
        process = psutil.Process(os.getpid())
        start_proc_mem = process.memory_info().rss / (1024 * 1024)
        start_sys_total, start_sys_used, start_sys_free, start_sys_percent = get_system_ram()
        start_time = time.time()

        candidates = build_ml_candidates(seed=seed, mode=mode_name)
        results = []

        # 1) Baseline (her modeli 1 kez default/fast ayarla, val F1 ölç)
        baseline_scores = []
        for name, item in candidates.items():
            model = clone(item["base"])

            # light modda hızlı sabit (az ağaç vs) için base zaten dar;
            # full modda base paramları default kalır, sonra search ile iyileşir.
            t0 = time.time()
            try:
                model.fit(X_tr, y_tr)
                fit_t = time.time() - t0
                proba = predict_proba_safe(model, X_val)
                pred = (proba >= 0.5).astype(int)
                f1v = f1_score(y_val, pred)
            except Exception:
                fit_t = np.nan
                f1v = -1

            baseline_scores.append((name, f1v, fit_t))
            results.append({
                "model": name,
                "phase": "baseline",
                "val_f1": f1v,
                "params": None,
                "fit_time_s": fit_t
            })

        baseline_scores.sort(key=lambda x: x[1], reverse=True)

        # 2) Full mod: timed random search sadece top-K üzerinde (adillik + hız)
        best_global = {"model": None, "val_f1": -1, "params": None}
        for name, f1v, _ in baseline_scores:
            if f1v > best_global["val_f1"]:
                best_global = {"model": name, "val_f1": f1v, "params": None}

        if mode_name == "full":
            # bütçeyi top3 modele dağıt (top3 çoğu zaman yeterli)
            topK = [x[0] for x in baseline_scores[:3]]
            remaining = time_budget_s - (time.time() - start_time)
            remaining = max(0.0, remaining)

            # her modele eşit pay + küçük buffer
            per_model_budget = max(5.0, remaining / max(1, len(topK)))

            for name in topK:
                remaining = time_budget_s - (time.time() - start_time)
                if remaining <= 3:
                    break

                item = candidates[name]
                budget = min(per_model_budget, remaining)

                best, tried = timed_random_search_one_model(
                    name=name,
                    base_model=item["base"],
                    param_space=item["space"],
                    X_train=X_tr, y_train=y_tr,
                    X_val=X_val, y_val=y_val,
                    time_left_s=budget,
                    seed=seed
                )

                results.append({
                    "model": name,
                    "phase": "timed_search",
                    "val_f1": best["f1"],
                    "params": best["params"],
                    "fit_time_s": best["fit_time"],
                    "trials": tried
                })

                if best["f1"] > best_global["val_f1"]:
                    best_global = {"model": name, "val_f1": best["f1"], "params": best["params"]}

        # 3) Best config retrain on full training set (X_train) and evaluate on test
        best_name = best_global["model"]
        best_params = best_global["params"]

        best_model = clone(candidates[best_name]["base"])
        if best_params is not None:
            try:
                best_model.set_params(**best_params)
            except Exception:
                pass

        # retrain on full training
        best_model.fit(X_train, y_train)

        test_proba = predict_proba_safe(best_model, X_test)
        test_pred = (test_proba >= 0.5).astype(int)
        acc, auc, f1, precision, recall, ll, cm = compute_metrics(y_test, test_pred, test_proba)

        # end measures
        end_time = time.time()
        end_proc_mem = process.memory_info().rss / (1024 * 1024)
        end_sys_total, end_sys_used, end_sys_free, end_sys_percent = get_system_ram()

        runtime = end_time - start_time
        proc_mem_usage = end_proc_mem - start_proc_mem
        sys_used_diff = end_sys_used - start_sys_used

        # save model
        out_file = f"classicML_{mode_name}_best_{best_name}.pkl"
        joblib.dump(best_model, out_file)
        model_size_mb = os.path.getsize(out_file) / (1024 * 1024)

        print_block(f"Classic ML {mode_name.upper()} Best = {best_name}",
                    acc, auc, f1, precision, recall, ll, cm,
                    y_true=y_test, y_pred=test_pred)

        print("\n=== Özet Bilgiler ===")
        print(f"Koşum Süresi: {runtime:.2f} saniye (Budget={time_budget_s}s)")
        print(f"Python İşlemi RAM Kullanımı: {proc_mem_usage:.2f} MB")

        print("\n=== Sistem Genel RAM Kullanımı ===")
        print(f"Toplam RAM:         {end_sys_total:.2f} GB")
        print(f"Kullanılan RAM:     {end_sys_used:.2f} GB")
        print(f"Boş RAM:            {end_sys_free:.2f} GB")
        print(f"RAM Kullanım Oranı: {end_sys_percent:.1f}%")
        print(f"Koşum sırasında artan RAM: {sys_used_diff:.2f} GB")

        print("\n=== En İyi Model Dosyası ===")
        print(f"Model: {best_name}")
        print(f"Model yolu: {out_file}")
        print(f"Model boyutu: {model_size_mb:.4f} MB")
        if best_params is not None:
            print(f"Best params: {best_params}")

        results_df = pd.DataFrame(results).sort_values(["phase", "val_f1"], ascending=[True, False])
        return out_file, results_df

    # Run Light then Full
    light_file, light_df = run_mode("light", time_budget_light)
    full_file, full_df = run_mode("full", time_budget_full)

    print("\n=== (Val) Leaderboard Summary (LIGHT) ===")
    display(light_df.head(30))
    print("\n=== (Val) Leaderboard Summary (FULL) ===")
    display(full_df.head(30))

    return {"light_model": light_file, "full_model": full_file, "light_trials": light_df, "full_trials": full_df}

# preprocess -> heart_merged_clean.csv hazır
out = run_classic_ml_full_light(
    clean_csv_path="heart_merged_clean.csv",
    target_col="target",
    seed=42,
    time_budget_full=180,
    time_budget_light=60
)

"""**2) DL Full vs Light ("CNN", "RNN", "GRU", "LSTM", "AE")**"""

# @title
import os, time, random
import numpy as np
import pandas as pd
import psutil

import torch
import torch.nn as nn
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler
from sklearn.metrics import (
    accuracy_score, roc_auc_score, f1_score,
    precision_score, recall_score, log_loss, confusion_matrix, classification_report
)

# -----------------------------
# Utils
# -----------------------------
def set_seed(seed: int = 42):
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)

def get_system_ram():
    mem = psutil.virtual_memory()
    total = mem.total / (1024**3)
    used = mem.used / (1024**3)
    free = mem.available / (1024**3)
    percent = mem.percent
    return total, used, free, percent

def compute_metrics(y_true, y_pred, y_proba):
    acc = accuracy_score(y_true, y_pred)
    auc = roc_auc_score(y_true, y_proba)
    f1  = f1_score(y_true, y_pred)
    pr  = precision_score(y_true, y_pred)
    rc  = recall_score(y_true, y_pred)
    ll  = log_loss(y_true, y_proba)
    cm  = confusion_matrix(y_true, y_pred)
    return acc, auc, f1, pr, rc, ll, cm

def print_block(title, acc, auc, f1, precision, recall, ll, cm, y_true=None, y_pred=None):
    print(f"\n=== {title} (Test) ===")
    print(f"Accuracy:  {acc*100:.2f}%")
    print(f"AUC:       {auc*100:.2f}%")
    print(f"F1 Score:  {f1*100:.2f}%")
    print(f"Precision: {precision*100:.2f}%")
    print(f"Recall:    {recall*100:.2f}%")
    print(f"LogLoss:   {ll:.2f}")
    print("\nConfusion Matrix:")
    print(cm)
    if y_true is not None and y_pred is not None:
        print("\nClassification Report:")
        print(classification_report(y_true, y_pred))

# -----------------------------
# Models
# -----------------------------
class TabularCNN1D(nn.Module):
    """
    Tabular -> 1D-CNN
    Input x: (B, F)  -> (B, 1, F)
    """
    def __init__(self, n_features: int, channels: int = 32, dropout: float = 0.15):
        super().__init__()
        self.conv = nn.Sequential(
            nn.Conv1d(1, channels, kernel_size=3, padding=1),
            nn.ReLU(),
            nn.Conv1d(channels, channels, kernel_size=3, padding=1),
            nn.ReLU(),
            nn.Dropout(dropout),
        )
        self.pool = nn.AdaptiveAvgPool1d(1)  # -> (B, C, 1)
        self.head = nn.Sequential(
            nn.Linear(channels, 32),
            nn.ReLU(),
            nn.Dropout(0.10),
            nn.Linear(32, 1)
        )

    def forward(self, x):
        # x: (B, F)
        x = x.unsqueeze(1)          # (B, 1, F)
        h = self.conv(x)            # (B, C, F)
        h = self.pool(h).squeeze(-1) # (B, C)
        logits = self.head(h).squeeze(1)
        return logits

class TabularRNNBinary(nn.Module):
    """
    Tabular -> sequence: (B, F) -> (B, F, 1)
    rnn_type: 'rnn' | 'gru' | 'lstm'
    """
    def __init__(self, n_features: int, rnn_type="gru", hidden_size=32, num_layers=1, dropout=0.0):
        super().__init__()
        self.n_features = n_features
        self.rnn_type = rnn_type.lower()

        if self.rnn_type == "rnn":
            self.rnn = nn.RNN(input_size=1, hidden_size=hidden_size, num_layers=num_layers,
                              batch_first=True, dropout=dropout if num_layers > 1 else 0.0)
        elif self.rnn_type == "lstm":
            self.rnn = nn.LSTM(input_size=1, hidden_size=hidden_size, num_layers=num_layers,
                               batch_first=True, dropout=dropout if num_layers > 1 else 0.0)
        else:
            self.rnn = nn.GRU(input_size=1, hidden_size=hidden_size, num_layers=num_layers,
                              batch_first=True, dropout=dropout if num_layers > 1 else 0.0)

        self.head = nn.Sequential(
            nn.Linear(hidden_size, 32),
            nn.ReLU(),
            nn.Dropout(0.10),
            nn.Linear(32, 1)
        )

    def forward(self, x):
        # x: (B, F)
        x = x.unsqueeze(-1)  # (B, F, 1)
        out = self.rnn(x)
        if self.rnn_type == "lstm":
            seq_out, (h, c) = out
        else:
            seq_out, h = out
        last_h = seq_out[:, -1, :]  # (B, hidden)
        logits = self.head(last_h).squeeze(1)
        return logits

class AutoEncoder(nn.Module):
    def __init__(self, in_dim: int, latent_dim: int = 8):
        super().__init__()
        self.encoder = nn.Sequential(
            nn.Linear(in_dim, 32),
            nn.ReLU(),
            nn.Linear(32, latent_dim)
        )
        self.decoder = nn.Sequential(
            nn.Linear(latent_dim, 32),
            nn.ReLU(),
            nn.Linear(32, in_dim)
        )
    def forward(self, x):
        z = self.encoder(x)
        xhat = self.decoder(z)
        return xhat

class AEClassifier(nn.Module):
    def __init__(self, encoder: nn.Module, latent_dim: int):
        super().__init__()
        self.encoder = encoder
        self.head = nn.Sequential(
            nn.Linear(latent_dim, 16),
            nn.ReLU(),
            nn.Dropout(0.10),
            nn.Linear(16, 1)
        )
    def forward(self, x):
        z = self.encoder(x)
        return self.head(z).squeeze(1)

# -----------------------------
# Core train (time-budget + early stop)
# -----------------------------
def train_supervised_timebudget(
    model: nn.Module,
    X_tr, y_tr,
    X_val, y_val,
    X_train_full, y_train_full,
    X_test, y_test,
    mode_name: str,
    time_budget_s: int,
    batch_size: int,
    lr: float,
    max_epochs: int,
    patience: int,
    use_scheduler: bool,
    device,
    out_file: str,
):
    process = psutil.Process(os.getpid())
    start_proc_mem = process.memory_info().rss / (1024 * 1024)
    start_sys_total, start_sys_used, start_sys_free, start_sys_percent = get_system_ram()
    start_time = time.time()

    model = model.to(device)
    criterion = nn.BCEWithLogitsLoss()
    optimizer = torch.optim.Adam(model.parameters(), lr=lr)
    scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(optimizer, T_max=max_epochs) if use_scheduler else None

    Xtr_t = torch.tensor(X_tr).to(device)
    ytr_t = torch.tensor(y_tr).float().to(device)
    Xva_t = torch.tensor(X_val).to(device)

    best_val_f1 = -1.0
    best_state = None
    bad = 0
    epochs_done = 0

    # --- train on inner train, select by val F1 ---
    for epoch in range(1, max_epochs + 1):
        if time.time() - start_time >= time_budget_s:
            break

        model.train()
        perm = torch.randperm(Xtr_t.size(0), device=device)
        Xb = Xtr_t[perm]
        yb = ytr_t[perm]

        for i in range(0, Xb.size(0), batch_size):
            if time.time() - start_time >= time_budget_s:
                break
            xb = Xb[i:i+batch_size]
            ybb = yb[i:i+batch_size]
            optimizer.zero_grad()
            logits = model(xb)
            loss = criterion(logits, ybb)
            loss.backward()
            optimizer.step()

        if scheduler is not None:
            scheduler.step()

        model.eval()
        with torch.no_grad():
            va_logits = model(Xva_t)
            va_prob = torch.sigmoid(va_logits).detach().cpu().numpy()
            va_pred = (va_prob >= 0.5).astype(int)
            va_f1 = f1_score(y_val, va_pred)

        epochs_done = epoch
        if va_f1 > best_val_f1 + 1e-4:
            best_val_f1 = va_f1
            best_state = {k: v.detach().cpu().clone() for k, v in model.state_dict().items()}
            bad = 0
        else:
            bad += 1
            if bad >= patience:
                break

    if best_state is not None:
        model.load_state_dict(best_state)

    # --- short finetune on full outer train within remaining time ---
    Xtrain_t = torch.tensor(X_train_full).to(device)
    ytrain_t = torch.tensor(y_train_full).float().to(device)

    finetune_epochs = 20 if mode_name == "full" else 8
    finetuned = 0
    for _ in range(finetune_epochs):
        if time.time() - start_time >= time_budget_s:
            break
        model.train()
        perm = torch.randperm(Xtrain_t.size(0), device=device)
        Xb = Xtrain_t[perm]
        yb = ytrain_t[perm]
        for i in range(0, Xb.size(0), batch_size):
            if time.time() - start_time >= time_budget_s:
                break
            xb = Xb[i:i+batch_size]
            ybb = yb[i:i+batch_size]
            optimizer.zero_grad()
            logits = model(xb)
            loss = criterion(logits, ybb)
            loss.backward()
            optimizer.step()
        finetuned += 1

    # --- test ---
    model.eval()
    with torch.no_grad():
        Xte_t = torch.tensor(X_test).to(device)
        te_logits = model(Xte_t)
        te_prob = torch.sigmoid(te_logits).detach().cpu().numpy()
        te_pred = (te_prob >= 0.5).astype(int)

    acc, auc, f1, precision, recall, ll, cm = compute_metrics(y_test, te_pred, te_prob)

    end_time = time.time()
    end_proc_mem = process.memory_info().rss / (1024 * 1024)
    end_sys_total, end_sys_used, end_sys_free, end_sys_percent = get_system_ram()

    runtime = end_time - start_time
    proc_mem_usage = end_proc_mem - start_proc_mem
    sys_used_diff = end_sys_used - start_sys_used

    torch.save({
        "model_state": model.state_dict(),
        "best_val_f1": best_val_f1,
        "epochs_done": epochs_done,
        "finetuned_epochs": finetuned
    }, out_file)
    model_size_mb = os.path.getsize(out_file) / (1024 * 1024)

    print_block(out_file.replace(".pt","").replace("_", " "),
                acc, auc, f1, precision, recall, ll, cm,
                y_true=y_test, y_pred=te_pred)

    print("\n=== Özet Bilgiler ===")
    print(f"Koşum Süresi: {runtime:.2f} saniye (Budget={time_budget_s}s)")
    print(f"Python İşlemi RAM Kullanımı: {proc_mem_usage:.2f} MB")

    print("\n=== Sistem Genel RAM Kullanımı ===")
    print(f"Toplam RAM:         {end_sys_total:.2f} GB")
    print(f"Kullanılan RAM:     {end_sys_used:.2f} GB")
    print(f"Boş RAM:            {end_sys_free:.2f} GB")
    print(f"RAM Kullanım Oranı: {end_sys_percent:.1f}%")
    print(f"Koşum sırasında artan RAM: {sys_used_diff:.2f} GB")

    print("\n=== Model Dosyası ===")
    print(f"Model yolu: {out_file}")
    print(f"Model boyutu: {model_size_mb:.4f} MB")
    print(f"epochs_done={epochs_done} | finetuned_epochs={finetuned} | best_val_f1={best_val_f1:.4f}")

    return out_file

def train_ae_then_clf_timebudget(
    in_dim,
    X_tr, X_val,
    X_train_full, y_train_full,
    X_test, y_test,
    mode_name: str,
    time_budget_s: int,
    batch_size: int,
    device,
    out_file: str,
):
    """
    AE: reconstruction pretrain (unsupervised) + classifier finetune
    Budget'ı adil tutmak için: light'ta daha az AE pretrain, full'da biraz daha fazla.
    """
    start_time = time.time()
    process = psutil.Process(os.getpid())
    start_proc_mem = process.memory_info().rss / (1024 * 1024)
    start_sys_total, start_sys_used, start_sys_free, start_sys_percent = get_system_ram()

    # configs
    if mode_name == "light":
        ae_epochs = 30
        clf_max_epochs = 60
        patience = 6
        lr_ae = 1e-3
        lr_clf = 1e-3
    else:
        ae_epochs = 80
        clf_max_epochs = 200
        patience = 15
        lr_ae = 8e-4
        lr_clf = 8e-4

    ae = AutoEncoder(in_dim=in_dim, latent_dim=8).to(device)
    opt_ae = torch.optim.Adam(ae.parameters(), lr=lr_ae)
    mse = nn.MSELoss()

    Xtr_t = torch.tensor(X_tr).to(device)
    Xva_t = torch.tensor(X_val).to(device)

    # --- AE pretrain (stop if time over) ---
    ae.train()
    for epoch in range(1, ae_epochs + 1):
        if time.time() - start_time >= time_budget_s * 0.55:  # budget'ın ~%55'i AE'ye
            break
        perm = torch.randperm(Xtr_t.size(0), device=device)
        Xb = Xtr_t[perm]
        for i in range(0, Xb.size(0), batch_size):
            if time.time() - start_time >= time_budget_s * 0.55:
                break
            xb = Xb[i:i+batch_size]
            opt_ae.zero_grad()
            xhat = ae(xb)
            loss = mse(xhat, xb)
            loss.backward()
            opt_ae.step()

    # --- classifier on top of encoder ---
    clf = AEClassifier(encoder=ae.encoder, latent_dim=8).to(device)
    # encoder'ı da fine-tune edelim (genelde iyi olur)
    criterion = nn.BCEWithLogitsLoss()
    opt = torch.optim.Adam(clf.parameters(), lr=lr_clf)

    y_train_full = y_train_full.astype(int)
    y_tr_full_t = torch.tensor(y_train_full).float().to(device)

    # inner split yoksa bile adil olmak için: val üzerinden early-stop
    y_val_dummy = None
    if X_val is not None:
        # y_val yok (AE pretrain'de kullanılmadı), burada supervised val lazım:
        # pratik: full trainin bir kısmını val gibi kullan
        pass

    # burada supervised train için dışarıdan X_tr,y_tr ve X_val,y_val hazırlayacağız:
    # (bu fonksiyon çağrılırken y_tr/y_val de verilse daha iyi, ama sadelik için aşağıda run fonksiyonunda hallediyoruz)
    # -> Bu yüzden AE'yi run fonksiyonunda supervised_train kısmına sokacağız.

    # Bu fonksiyonu basitleştirmek için burada sadece encoder döndürelim:
    return clf, ae, start_time, start_proc_mem, start_sys_used, start_sys_total, start_sys_free, start_sys_percent

# -----------------------------
# Main runner
# -----------------------------
def run_dl_full_light_multi(
    clean_csv_path="heart_merged_clean.csv",
    target_col="target",
    test_size=0.2,
    seed=42,
    time_budget_full=180,
    time_budget_light=60,
    holdout_frac_for_tuning=0.2,
    batch_size=64,
    models_to_run=("CNN", "RNN", "GRU", "LSTM", "AE")
):
    set_seed(seed)
    torch.manual_seed(seed)

    df = pd.read_csv(clean_csv_path)
    X = df.drop(columns=[target_col]).values.astype(np.float32)
    y = df[target_col].astype(int).values

    # outer split
    X_train, X_test, y_train, y_test = train_test_split(
        X, y, test_size=test_size, random_state=seed, stratify=y
    )
    # inner split
    X_tr, X_val, y_tr, y_val = train_test_split(
        X_train, y_train,
        test_size=holdout_frac_for_tuning,
        random_state=seed,
        stratify=y_train
    )

    # ✅ Correct scaling: fit ONLY on outer train
    scaler = StandardScaler()
    X_train_s = scaler.fit_transform(X_train).astype(np.float32)
    X_test_s  = scaler.transform(X_test).astype(np.float32)
    X_tr_s    = scaler.transform(X_tr).astype(np.float32)
    X_val_s   = scaler.transform(X_val).astype(np.float32)

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    out_files = {}

    def get_mode_cfg(mode_name):
        if mode_name == "light":
            return dict(max_epochs=60, patience=6, lr=1e-3, use_scheduler=False)
        else:
            return dict(max_epochs=200, patience=15, lr=8e-4, use_scheduler=True)

    def run_one(model_name: str, mode_name: str, budget_s: int):
        cfg = get_mode_cfg(mode_name)
        in_dim = X_tr_s.shape[1]

        if model_name == "CNN":
            # light daha küçük kanal, full daha büyük
            ch = 24 if mode_name == "light" else 32
            model = TabularCNN1D(n_features=in_dim, channels=ch, dropout=0.15)
            out_file = f"dl_cnn_{mode_name}_best.pt"
            return train_supervised_timebudget(
                model,
                X_tr_s, y_tr, X_val_s, y_val,
                X_train_s, y_train,
                X_test_s, y_test,
                mode_name=mode_name,
                time_budget_s=budget_s,
                batch_size=batch_size,
                lr=cfg["lr"],
                max_epochs=cfg["max_epochs"],
                patience=cfg["patience"],
                use_scheduler=cfg["use_scheduler"],
                device=device,
                out_file=out_file
            )

        elif model_name in ("RNN", "GRU", "LSTM"):
            rnn_type = "rnn" if model_name == "RNN" else model_name.lower()
            # light daha küçük, full biraz daha büyük kapasite
            hidden = 24 if mode_name == "light" else 32
            model = TabularRNNBinary(n_features=in_dim, rnn_type=rnn_type, hidden_size=hidden, num_layers=1, dropout=0.0)
            out_file = f"dl_{model_name.lower()}_{mode_name}_best.pt"
            return train_supervised_timebudget(
                model,
                X_tr_s, y_tr, X_val_s, y_val,
                X_train_s, y_train,
                X_test_s, y_test,
                mode_name=mode_name,
                time_budget_s=budget_s,
                batch_size=batch_size,
                lr=cfg["lr"],
                max_epochs=cfg["max_epochs"],
                patience=cfg["patience"],
                use_scheduler=cfg["use_scheduler"],
                device=device,
                out_file=out_file
            )

        elif model_name == "AE":
            # 1) AE pretrain (unsupervised) budget'ın bir kısmı
            # 2) AEClassifier supervised: kalan sürede train_supervised_timebudget benzeri (ayrı yazalım)
            process = psutil.Process(os.getpid())
            start_proc_mem = process.memory_info().rss / (1024 * 1024)
            start_sys_total, start_sys_used, start_sys_free, start_sys_percent = get_system_ram()
            start_time = time.time()

            # config
            if mode_name == "light":
                ae_epochs = 25
                patience = 6
                lr_ae = 1e-3
                lr_clf = 1e-3
                clf_max_epochs = 60
            else:
                ae_epochs = 70
                patience = 15
                lr_ae = 8e-4
                lr_clf = 8e-4
                clf_max_epochs = 200

            ae = AutoEncoder(in_dim=in_dim, latent_dim=8).to(device)
            opt_ae = torch.optim.Adam(ae.parameters(), lr=lr_ae)
            mse = nn.MSELoss()

            Xtr_t = torch.tensor(X_tr_s).to(device)

            # AE pretrain: budget'ın ~%45'i
            ae.train()
            for epoch in range(1, ae_epochs + 1):
                if time.time() - start_time >= budget_s * 0.45:
                    break
                perm = torch.randperm(Xtr_t.size(0), device=device)
                Xb = Xtr_t[perm]
                for i in range(0, Xb.size(0), batch_size):
                    if time.time() - start_time >= budget_s * 0.45:
                        break
                    xb = Xb[i:i+batch_size]
                    opt_ae.zero_grad()
                    xhat = ae(xb)
                    loss = mse(xhat, xb)
                    loss.backward()
                    opt_ae.step()

            clf = AEClassifier(encoder=ae.encoder, latent_dim=8).to(device)
            criterion = nn.BCEWithLogitsLoss()
            opt = torch.optim.Adam(clf.parameters(), lr=lr_clf)

            Xtr_t = torch.tensor(X_tr_s).to(device)
            ytr_t = torch.tensor(y_tr).float().to(device)
            Xva_t = torch.tensor(X_val_s).to(device)

            best_val_f1 = -1.0
            best_state = None
            bad = 0
            epochs_done = 0

            # supervised train with remaining time
            for epoch in range(1, clf_max_epochs + 1):
                if time.time() - start_time >= budget_s:
                    break
                clf.train()
                perm = torch.randperm(Xtr_t.size(0), device=device)
                Xb = Xtr_t[perm]
                yb = ytr_t[perm]
                for i in range(0, Xb.size(0), batch_size):
                    if time.time() - start_time >= budget_s:
                        break
                    xb = Xb[i:i+batch_size]
                    ybb = yb[i:i+batch_size]
                    opt.zero_grad()
                    logits = clf(xb)
                    loss = criterion(logits, ybb)
                    loss.backward()
                    opt.step()

                clf.eval()
                with torch.no_grad():
                    va_logits = clf(Xva_t)
                    va_prob = torch.sigmoid(va_logits).detach().cpu().numpy()
                    va_pred = (va_prob >= 0.5).astype(int)
                    va_f1 = f1_score(y_val, va_pred)

                epochs_done = epoch
                if va_f1 > best_val_f1 + 1e-4:
                    best_val_f1 = va_f1
                    best_state = {k: v.detach().cpu().clone() for k, v in clf.state_dict().items()}
                    bad = 0
                else:
                    bad += 1
                    if bad >= patience:
                        break

            if best_state is not None:
                clf.load_state_dict(best_state)

            # test
            clf.eval()
            with torch.no_grad():
                Xte_t = torch.tensor(X_test_s).to(device)
                te_logits = clf(Xte_t)
                te_prob = torch.sigmoid(te_logits).detach().cpu().numpy()
                te_pred = (te_prob >= 0.5).astype(int)

            acc, auc, f1, precision, recall, ll, cm = compute_metrics(y_test, te_pred, te_prob)

            end_time = time.time()
            end_proc_mem = process.memory_info().rss / (1024 * 1024)
            end_sys_total, end_sys_used, end_sys_free, end_sys_percent = get_system_ram()

            runtime = end_time - start_time
            proc_mem_usage = end_proc_mem - start_proc_mem
            sys_used_diff = end_sys_used - start_sys_used

            out_file = f"dl_ae_{mode_name}_best.pt"
            torch.save({
                "ae_state": ae.state_dict(),
                "clf_state": clf.state_dict(),
                "best_val_f1": best_val_f1,
                "epochs_done": epochs_done,
                "scaler": scaler,
                "seed": seed,
            }, out_file)
            model_size_mb = os.path.getsize(out_file) / (1024 * 1024)

            print_block(f"DL (PyTorch) AE+Head {mode_name.upper()}",
                        acc, auc, f1, precision, recall, ll, cm,
                        y_true=y_test, y_pred=te_pred)

            print("\n=== Özet Bilgiler ===")
            print(f"Koşum Süresi: {runtime:.2f} saniye (Budget={budget_s}s)")
            print(f"Python İşlemi RAM Kullanımı: {proc_mem_usage:.2f} MB")
            print(f"Koşum sırasında artan RAM (sys): {sys_used_diff:.2f} GB")
            print("\n=== Model Dosyası ===")
            print(f"Model yolu: {out_file}")
            print(f"Model boyutu: {model_size_mb:.4f} MB")
            print(f"epochs_done={epochs_done} | best_val_f1={best_val_f1:.4f}")

            return out_file

        else:
            raise ValueError(f"Unknown model_name: {model_name}")

    # Run LIGHT then FULL for each model
    for m in models_to_run:
        out_files[f"{m.lower()}_light"] = run_one(m, "light", time_budget_light)
        out_files[f"{m.lower()}_full"]  = run_one(m, "full",  time_budget_full)

    return out_files

dl_out = run_dl_full_light_multi(
    clean_csv_path="heart_merged_clean.csv",
    target_col="target",
    seed=42,
    time_budget_full=180,
    time_budget_light=60,
    models_to_run=("CNN", "RNN", "GRU", "LSTM", "AE")
)
print("\nSaved:", dl_out)
