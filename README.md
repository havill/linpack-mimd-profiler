# GPU Compute Benchmark (CUDA / OpenCL)

このリポジトリには、GPUの浮動小数点演算性能（TFLOPS）を測定するためのPythonコマンドラインツールと、その結果を視覚化するためのグラフ生成スクリプトが含まれています。

NVIDIAが提供する最適化された `cuSOLVER` を使用する **CUDAバックエンド**（LU分解によるLinpackベンチマーク）、カスタムカーネルを使用する **OpenCLバックエンド**（タイル化行列乗算 / GEMM）、そして最新のスーパーコンピュータの性能指標をシミュレートする **HPL-AIバックエンド**（混合精度 / 反復細分化）を1つのスクリプトで切り替えて実行できます。結果はCSVファイルに自動的に記録され、付属の `generate_charts.py` を使用することで簡単に比較や詳細なグラフ化が可能です。

## ⚙️ システム要件とインストール

このスクリプトを実行するには、Python 3.8以上と以下のGPU環境が必要です。

### 1. Windows ネイティブ環境でのセットアップ

* **NVIDIAディスプレイドライバ**: 最新バージョンにアップデートしてください（OpenCLドライバも自動的にインストールされます）。
* **NVIDIA CUDA Toolkit**: CUDAまたはHPL-AIバックエンドを使用する場合に必須です。公式サイトからダウンロードしてインストールしてください。
* **Pythonライブラリ**: 必要な全ライブラリをインストールします（詳細はセクション3を参照）。

---

### 2. ⚠️ WSL2 (Linux) 環境でのセットアップと制限事項

WSL2環境（Ubuntu 24.04やFedoraなど）で実行する場合、WindowsからGPUが特殊な形でパススルーされるため、**Windowsネイティブとは異なる手順と制約**があります。

#### ✅ CUDA / HPL-AI バックエンドを利用する場合（推奨）

WSL2ではWindows側のNVIDIAドライバが自動的にパススルーされます。**Linux用のNVIDIAディスプレイドライバは絶対にインストールしないでください**（GPUへのアクセスが壊れます）。以下の手順で「ツールキットのみ」を安全にインストールします。

```bash
# Ubuntu環境の例: 事前にコンパイラをインストール
sudo apt update && sudo apt install build-essential -y

# 1. Linux用 CUDA Runfileのダウンロード (バージョンは適宜変更)
wget https://developer.download.nvidia.com/compute/cuda/13.1.0/local_installers/cuda_13.1.0_590.44.01_linux.run

# 2. ツールキットのみをインストール（ドライバのインストールをスキップ）
sudo sh cuda_13.1.0_590.44.01_linux.run --toolkit --silent

# 3. 環境変数の追加（Pythonがライブラリを見つけられるようにする）
echo 'export PATH=/usr/local/cuda-13.1/bin${PATH:+:${PATH}}' >> ~/.bashrc
echo 'export LD_LIBRARY_PATH=/usr/local/cuda-13.1/lib64${LD_LIBRARY_PATH:+:${LD_LIBRARY_PATH}}' >> ~/.bashrc
source ~/.bashrc
```

#### ❌ OpenCLバックエンドを利用する場合（非推奨 / 制限あり）

現在のWSL2仕様およびNVIDIAのドライバ提供方針により、WSL2上でのネイティブなNVIDIA OpenCLの動作は**非常に困難、またはサポート対象外**です。

* Windows側からLinux側へ、OpenCLの必須ライブラリ（`libnvidia-opencl.so.1`）がパススルーされないケースが大半です。
* 代替手段としてMicrosoftのDirectX 12変換レイヤー（`mesa-opencl-icd`）や最新のRusticlエンジンを使用しても、ハードウェアブリッジの問題で `0 devices` と認識されることが確認されています。
* **結論:** OpenCLのベンチマークを実行したい場合は、WSL2ではなく**Windowsネイティブ環境（PowerShellやコマンドプロンプト）でスクリプトを実行することを強く推奨します。**

#### 🐍 Ubuntu 24.04等でのPython環境構築（PEP 668対策）

最新のLinuxディストリビューションではシステムの保護により、グローバルな `pip install` が制限されています。仮想環境（venv）を作成して実行してください。

```bash
# 必要なツールのインストール
sudo apt install python3 python3-pip python3-venv python3-dev -y

# 仮想環境の作成と有効化
python3 -m venv bench_env
source bench_env/bin/activate

# 依存関係のインストール (仮想環境内)
pip install numpy pyopencl cupy-cuda13x pandas matplotlib seaborn plotly pynvml
‘‘‘

#### 🎩 Fedora Remix for WSL (Whitewater Foundry) の場合

Fedora Remixでは、軽量化のためにC/C++コンパイラなどの開発ツールが初期状態では省かれています。Python 3の本体はOSの動作のために内蔵されていますが、`pyopencl` などのビルドに必要な `gcc` や `pip` は `dnf` コマンドで手動で追加する必要があります。

```bash
# 開発ツール群（gcc, makeなど）のインストール
sudo dnf groupinstall "Development Tools" -y
# （最小限のコンパイラのみが必要な場合は `sudo dnf install gcc -y`）

# Pythonのパッケージマネージャー(pip)と開発用ヘッダーのインストール
sudo dnf install python3-pip python3-devel -y
```

### 2. Pythonライブラリのインストール

コマンドプロンプトまたはターミナルを開き、以下のコマンドを実行して必要な依存関係をインストールします。（WSL2の場合は仮想環境内で実行してください）

```bash
# 基本的な数値計算ライブラリとOpenCLバインディングのインストール
pip install numpy pyopencl

# CuPyのインストール (インストールされているCUDAのバージョンに合わせてください)
# 例: CUDA 13.x がインストールされている場合
pip install cupy-cuda13x

# グラフ生成スクリプト用のデータ解析・描画ライブラリ
pip install pandas matplotlib seaborn plotly

# GFLOPS/Watt（電力効率）を計測するための NVML ラッパー
pip install pynvml
```

## 🚀 ベンチマークの使用方法 (`gpu_benchmark.py`)

基本的なコマンドの構文は以下の通りです。

```bash
python gpu_benchmark.py -b <バックエンド> [オプション]
```

### 実行例

**CUDAを使用して、サイズ8192の行列で10回テストを実行する:**

```bash
python gpu_benchmark.py -b cuda -n 8192 -i 10
```

**HPL-AI（混合精度）ベンチマークを実行する（※CUDA必須）:**

```bash
python gpu_benchmark.py -b hpl-ai -n 8192
```

※ hpl-ai バックエンドは、単精度(FP32)で重いLU分解を行い、倍精度(FP64)で反復細分化（Iterative Refinement）を行うことで、Top500のHPL-MxPアルゴリズムをシミュレートします。

**OpenCLを使用してテストを実行し、結果をCSVファイルに保存する:**

```bash
python gpu_benchmark.py -b opencl -n 8192 -i 10 -o benchmark_results.csv
```

**HPC向けの厳しいテスト（倍精度浮動小数点数 FP64 を使用）:**

```bash
python gpu_benchmark.py -b cuda -n 10240 -d float64 -o benchmark_results.csv
```

### 🎛️ ベンチマーク・コマンドラインオプション

| オプション | 省略形 | デフォルト値 | 説明 |
| :--- | :---: | :--- | :--- |
| `--backend` | `-b` | **必須** | 特定のバックエンド（cuda, opencl, hpl-ai）のデータのみにフィルタリングします。 |
| `--size` | `-n` | `8192` | 生成する正方行列のサイズ N （N × N の行列になります）。OpenCLの場合は自動的に16の倍数に丸められます。 |
| `--iterations` | `-i` | `10` | ベンチマークを実行する反復回数です。 |
| `--dtype` | `-d` | `float32` | データの精度を指定します。float32（単精度）または float64（倍精度）。※ hpl-ai の場合は無視され、自動的に混合精度が適用されます。 |
| `--output` | `-o` | `None` | 結果を追記するCSVファイルのパス。指定しない場合、CSV出力は行われずコンソールへの表示のみとなります。 |

## 📊 グラフの自動生成 (`generate_charts.py`)

ベンチマークを実行して出力されたCSVファイルから、パフォーマンスや電力効率を可視化するグラフを自動生成できます。新しく追加された hpl-ai やGPUモデル名の記録にも完全に対応しています。

### 📈 生成されるグラフの種類

1. **Performance Scaling (TFLOPS)**: 行列サイズごとのパフォーマンス推移。GPUがどのサイズで飽和するかを確認できます。
2. **Energy Efficiency (GFLOPS/W)**: ハードウェアの電力効率（GFLOPS / Watt）。どの条件が最も省電力で高パフォーマンスかを示します。
3. **Compute Latency**: 処理時間の推移（対数スケール）。計算量が O(N³) で増加する様子を確認できます。
4. **Power Profile**: 平均消費電力とピーク消費電力の比較グラフ。
5. **Interactive Bubble Chart (オプション)**: すべての実行データをホバーで確認できる対話型HTMLグラフ。

### グラフの生成実行例

**デフォルトのCSVからPNGグラフを生成する:**

```bash
python generate_charts.py -f benchmark_results.csv
```

**特定のバックエンド（CUDA）に絞り、対話型HTMLとPDF画像も出力する:**

```bash
python generate_charts.py -f benchmark_results.csv -b cuda -i -x png,pdf
```

### 🎛️ グラフ生成オプション

| オプション | 省略形 | デフォルト値 | 説明 |
| :--- | :---: | :--- | :--- |
| `--file` | `-f` | `benchmark_results.csv` | 読み込むCSVファイルのパスを指定します。 |
| `--backend` | `-b` | `None` | 特定のバックエンド（`cuda` または `opencl`）のデータのみにフィルタリングして描画します。 |
| `--dtype` | `-d` | `None` | 特定のデータ型（`float32` または `float64`）のデータのみにフィルタリングして描画します。 |
| `--interactive` | `-i` | `False` | Plotlyを使用した対話型のHTMLグラフ（Bubble Chart）を生成します。 |
| `--export-formats` | `-x` | `png` | 出力する画像フォーマットをカンマ区切りで指定します（例: `eps`, `jpeg`, `pdf`, `pgf`, `png`, `ps`, `raw`, `rgba`, `svg`, `svgz`, `tiff`, `webp`）。 |

## 🛠️ トラブルシューティング

**1. `ImportError: DLL load failed while importing cublas` (Windows)**
Windows環境のPython 3.8以降では、セキュリティ上の理由からシステムの `Path` 変数が無視されることがあります。CUDA Toolkitがインストールされていることを確認し、スクリプト内の `hardcoded_bin_path` をご自身の環境（例: `C:\Program Files\NVIDIA GPU Computing Toolkit\CUDA\v13.1\bin`）に合わせて書き換えてください。

**2. `ImportError: libcublas.so.13: cannot open shared object file` (WSL2 / Linux)**
Linux環境でCuPyがCUDAライブラリ（`cuBLAS`など）を見つけられない場合に発生します。CUDA Toolkitが正しくインストールされているか確認し、`~/.bashrc` に `LD_LIBRARY_PATH` が正しく設定されていることを確認してください。

**3. `clGetPlatformIDs failed: PLATFORM_NOT_FOUND_KHR` または `No OpenCL devices found` (WSL2)**
前述の通り、WSL2環境ではNVIDIAのOpenCLドライバが正常にパススルーされません。`clinfo` コマンドで `0 devices` と表示される場合、Linux側からはGPUのOpenCL機能にアクセスできていません。OpenCLのテストはWindowsホスト側で実行してください。

**4. OpenCL実行時の `CompilerWarning: Non-empty compiler output encountered`**
PyOpenCLはカーネルを動的にコンパイルします。NVIDIAのコンパイラが最適化のメモなどを出力しただけでこの警告が表示されますが、ベンチマーク自体が動作しTFLOPSが出力されていれば**完全に無視して問題ありません**。

**5. グラフ生成時に `ModuleNotFoundError: No module named 'plotly'` エラーが出る**
対話型グラフ（`-i` オプション）を生成するために必要なライブラリが不足しています。`pip install plotly` を実行してインストールしてください。
