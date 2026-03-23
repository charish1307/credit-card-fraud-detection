# 📂 Data Folder

> **The dataset file `creditcard.csv` is NOT included in this repository.**
> GitHub has a 100MB file size limit, and the dataset is ~150MB.

---

## 📥 How to Download the Dataset

### Option 1: Manual Download (Easiest)

1. Go to the Kaggle dataset page:
   👉 [https://www.kaggle.com/datasets/mlg-ulb/creditcardfraud](https://www.kaggle.com/datasets/mlg-ulb/creditcardfraud)
2. Click the **Download** button (you need a free Kaggle account)
3. Unzip the downloaded file
4. Place `creditcard.csv` in this `data/` folder

### Option 2: Kaggle API (Faster for developers)

1. Install the Kaggle CLI:
   ```bash
   pip install kaggle
   ```

2. Set up your Kaggle API token:
   - Go to [https://www.kaggle.com/settings](https://www.kaggle.com/settings)
   - Click **Create New Token** — this downloads `kaggle.json`
   - Place `kaggle.json` in `~/.kaggle/` (Mac/Linux) or `C:\Users\<user>\.kaggle\` (Windows)

3. Download the dataset directly:
   ```bash
   kaggle datasets download -d mlg-ulb/creditcardfraud -p data/ --unzip
   ```

---

## 📋 Dataset Details

| Property | Value |
|----------|-------|
| File name | `creditcard.csv` |
| Size | ~150 MB |
| Rows | 284,807 transactions |
| Columns | 31 (Time, V1-V28, Amount, Class) |
| Fraud cases | 492 (0.172%) |
| Source | European cardholders, September 2013 |
| License | Open Database License (ODbL) |

---

## 📁 Expected Folder Structure After Download

```
data/
├── README.md        ← This file
└── creditcard.csv   ← Download from Kaggle (NOT in git)
```

---

## ⚠️ Note on Git LFS (Optional)

If you want to version-control large files, you can use [Git LFS](https://git-lfs.com/):
```bash
git lfs install
git lfs track "data/*.csv"
git add .gitattributes
```
However, Kaggle API download (Option 2) is recommended for simplicity.
