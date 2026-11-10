# =====================================================================
# TUGAS RANDOM FOREST - Dataset: Hotel Booking Demand (hotel_bookings.csv)
# Target : is_canceled  (0 = tidak batal, 1 = batal)
# BAGIAN A : 11 data (10 latih + 1 uji), 5 pohon, 1 fitur/pohon (dibuat dari nol,
#            hasilnya identik dengan perhitungan manual)
# BAGIAN B : seluruh data, Random Forest "seperti biasa" (scikit-learn)
# =====================================================================
import numpy as np
import pandas as pd
from collections import Counter

CSV_PATH = "hotel_bookings.csv"      # ubah jika lokasi file berbeda
df = pd.read_csv(CSV_PATH)
print("Ukuran dataset:", df.shape)

# ---------------------------------------------------------------------
# BAGIAN A  -  RANDOM FOREST DARI NOL PADA 11 DATA
# ---------------------------------------------------------------------
FEATS   = ["lead_time", "adr", "total_of_special_requests", "deposit_type", "market_segment"]
NUMERIC = {"lead_time", "adr", "total_of_special_requests"}
TARGET  = "is_canceled"

# 1) Ambil 10 data latih (5 batal + 5 tidak batal) dan 1 data uji (batal)
rs = np.random.RandomState(3)
c1 = df[df[TARGET] == 1].sample(6, random_state=rs)
c0 = df[df[TARGET] == 0].sample(5, random_state=rs)
train = pd.concat([c1.iloc[:5], c0]).sample(frac=1, random_state=rs)[FEATS + [TARGET]]
test  = c1.iloc[[5]][FEATS + [TARGET]]
train = train.reset_index(drop=True)
train.index = [f"D{i+1}" for i in range(10)]
print("\nDATA LATIH:\n", train.to_string())
print("\nDATA UJI:\n", test.to_string(index=False))

# 2) Fungsi Gini & pencarian split terbaik (CART biner)
def gini(y):
    n = len(y)
    if n == 0:
        return 0.0
    return 1 - sum((v / n) ** 2 for v in Counter(y).values())

def evaluate_splits(x, y, feat):
    """Numerik: kandidat = titik tengah antar nilai berurutan (x <= t).
       Kategorik: kandidat = satu kategori vs sisanya (x == k)."""
    x, y = np.asarray(x), np.asarray(y)
    vals = sorted(set(x))
    cands = [(vals[i] + vals[i + 1]) / 2 for i in range(len(vals) - 1)] if feat in NUMERIC else vals
    g0, rows = gini(y), []
    for c in cands:
        m = (x <= c) if feat in NUMERIC else (x == c)
        yl, yr = y[m], y[~m]
        w = len(yl) / len(y) * gini(yl) + len(yr) / len(y) * gini(yr)
        rows.append(dict(cand=c, nL=len(yl), L0=int((yl == 0).sum()), L1=int((yl == 1).sum()),
                         nR=len(yr), R0=int((yr == 0).sum()), R1=int((yr == 1).sum()),
                         gini_w=w, reduction=g0 - w))
    return g0, rows

def majority(n0, n1, tie=1):
    return tie if n0 == n1 else int(n1 > n0)

# 3) Bangun 5 pohon: tiap pohon = bootstrap 10 data + 1 fitur acak (tanpa pengulangan fitur)
rng  = np.random.RandomState(17)
perm = rng.permutation(len(FEATS))
forest = []
for t in range(5):
    boot = rng.randint(0, 10, 10)                 # bootstrap (dengan pengembalian)
    feat = FEATS[perm[t]]
    sub  = train.iloc[boot]
    g0, rows = evaluate_splits(sub[feat].values, sub[TARGET].values, feat)
    best = max(rows, key=lambda r: r["reduction"])
    forest.append(dict(feat=feat, cand=best["cand"],
                       left=majority(best["L0"], best["L1"]),
                       right=majority(best["R0"], best["R1"])))
    print(f"\nPohon {t+1} | fitur = {feat} | bootstrap = {[f'D{i+1}' for i in boot]}")
    print(f"  Gini awal = {g0:.4f}")
    for r in rows:
        print(f"  kandidat {str(r['cand']):>14} | kiri(0/1)={r['L0']}/{r['L1']} kanan(0/1)={r['R0']}/{r['R1']}"
              f" | Gini bobot={r['gini_w']:.4f} | reduction={r['reduction']:.4f}")
    op = "<=" if feat in NUMERIC else "=="
    print(f"  -> Split terbaik: {feat} {op} {best['cand']}  | kiri -> {forest[-1]['left']}, kanan -> {forest[-1]['right']}")

# 4) Prediksi data uji + voting
def predict_tree(tr, x):
    go_left = (x <= tr["cand"]) if tr["feat"] in NUMERIC else (x == tr["cand"])
    return tr["left"] if go_left else tr["right"]

votes = [predict_tree(tr, test.iloc[0][tr["feat"]]) for tr in forest]
final = Counter(votes).most_common(1)[0][0]
print("\nPrediksi tiap pohon :", votes)
print("Voting              : 0 (tidak batal) =", votes.count(0), "| 1 (batal) =", votes.count(1))
print("Prediksi akhir      :", final, "(Batal)" if final == 1 else "(Tidak batal)")
print("Label sebenarnya    :", int(test.iloc[0][TARGET]))

# ---------------------------------------------------------------------
# BAGIAN B  -  SELURUH DATA (seperti biasa, scikit-learn)
# ---------------------------------------------------------------------
from sklearn.ensemble import RandomForestClassifier
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import OrdinalEncoder
from sklearn.metrics import accuracy_score, precision_score, recall_score, f1_score, confusion_matrix

data = df.copy()
# reservation_status & reservation_status_date = kebocoran target (data leakage) -> dibuang
data = data.drop(columns=["reservation_status", "reservation_status_date"])
# penanganan nilai kosong
data["children"] = data["children"].fillna(0)
data["country"]  = data["country"].fillna("Unknown")
data[["agent", "company"]] = data[["agent", "company"]].fillna(0)

X, y = data.drop(columns=[TARGET]), data[TARGET]
cat_cols = [c for c in X.columns if not pd.api.types.is_numeric_dtype(X[c])]
X[cat_cols] = OrdinalEncoder().fit_transform(X[cat_cols])      # kategorik -> angka

X_tr, X_te, y_tr, y_te = train_test_split(X, y, test_size=0.2, random_state=42, stratify=y)
print("\n\nBAGIAN B - seluruh data | latih:", X_tr.shape, "| uji:", X_te.shape)

def laporan(nama, model):
    model.fit(X_tr, y_tr)
    p = model.predict(X_te)
    print(f"\n[{nama}]")
    print(f"  Accuracy  = {accuracy_score(y_te, p):.4f}")
    print(f"  Precision = {precision_score(y_te, p):.4f}")
    print(f"  Recall    = {recall_score(y_te, p):.4f}")
    print(f"  F1-score  = {f1_score(y_te, p):.4f}")
    print("  Confusion matrix [[TN FP],[FN TP]]:\n", confusion_matrix(y_te, p))
    return model

# (i) model standar
rf = laporan("Random Forest standar: 100 pohon", RandomForestClassifier(n_estimators=100, random_state=42, n_jobs=-1))
# (ii) skema tugas: 5 pohon, 1 fitur per split, kedalaman 1
laporan("Skema tugas: 5 pohon, max_features=1, max_depth=1",
        RandomForestClassifier(n_estimators=5, max_features=1, max_depth=1, random_state=42, n_jobs=-1))

imp = pd.Series(rf.feature_importances_, index=X.columns).sort_values(ascending=False).head(10)
print("\n10 fitur terpenting (model 100 pohon):\n", imp.round(4).to_string())
