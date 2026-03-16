# Git LFS 使用说明

本项目使用 Git LFS 管理大文件（向量数据库 `chroma_db/` 和原始数据 `csv/`）。
普通 `git push/pull` **不会**自动同步这些数据，需手动操作。

---

## 一、首次配置（每台机器只需做一次）

### 1. 安装 git-lfs

```bash
# Ubuntu / WSL
sudo apt-get install git-lfs

# macOS
brew install git-lfs
```

### 2. 初始化，并关闭自动下载

```bash
git lfs install --skip-smudge
```

> `--skip-smudge` 让 `git pull` 时不自动拉取大文件，避免每次同步代码都下载几百 MB。

---

## 二、克隆仓库

```bash
git clone git@github.com:unlimitedblad/DigitalTwin.git
cd DigitalTwin
```

克隆后 `chroma_db/` 和 `csv/` 目录为空，按需手动拉取（见下）。

---

## 三、手动拉取数据

```bash
# 拉取全部 LFS 文件
git lfs pull

# 只拉向量数据库
git lfs pull --include="chroma_db/**"

# 只拉原始 CSV
git lfs pull --include="csv/*.csv"
```

---

## 四、手动提交数据更新

`chroma_db/` 和 `csv/` 已在 `.gitignore` 中，需用 `-f` 强制添加：

```bash
git add -f chroma_db/
git add -f csv/
git commit -m "更新向量库"
git push
```

> 代码变更正常 `git add src/` 即可，无需 `-f`。

---

## 五、检查 LFS 状态

```bash
# 查看哪些文件由 LFS 管理
git lfs ls-files

# 查看待上传 / 待提交的 LFS 文件
git lfs status
```

---

## 注意事项

- GitHub LFS 免费额度：存储 **1GB**，月流量 **1GB**，超额需付费
- 不要直接 `git add .` 提交数据，容易超出额度或上传不必要的文件
- `chroma_db/` 由脚本生成，如无需共享可本地重建：`python src/test_csv_final.py`
