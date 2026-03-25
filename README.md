# GPU Compute Benchmark (CUDA / OpenCL)

このリポジトリには、GPUの浮動小数点演算性能（TFLOPS）を測定するためのPythonコマンドラインツールが含まれています。

NVIDIAが提供する最適化された `cuSOLVER` を使用する **CUDAバックエンド**（LU分解によるLinpackベンチマーク）と、カスタムカーネルを使用する **OpenCLバックエンド**（タイル化行列乗算 / GEMM）の両方を1つのスクリプトで切り替えて実行できます。結果はCSVファイルに自動的に記録され、比較やグラフ化が容易です。

## ⚙️ システム要件とインストール

このスクリプトを実行するには、Python 3.8以上と以下のGPU環境が必要です。

### 1. Windows ネイティブ環境でのセットアップ
* **NVIDIAディスプレイドライバ**: 最新バージョンにアップデートしてください（OpenCLドライバも自動的にインストールされます）。
* **NVIDIA CUDA Toolkit**: CUDAバックエンドを使用する場合に必須です。公式サイトからダウンロードしてインストールしてください。
* **Pythonライブラリ**: `pip install numpy pyopencl cupy-cuda13x` （CUDAのバージョンに合わせてCuPyをインストールしてください）。

---

### 2. ⚠️ WSL2 (Linux) 環境でのセットアップと制限事項

WSL2環境（Ubuntu 24.04やFedoraなど）で実行する場合、WindowsからGPUが特殊な形でパススルーされるため、**Windowsネイティブとは異なる手順と制約**があります。

#### ✅ CUDAバックエンドを利用する場合（推奨）
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
pip install numpy pyopencl cupy-cuda13x
```

### 2. Pythonライブラリのインストール
コマンドプロンプトまたはターミナルを開き、以下のコマンドを実行して必要な依存関係をインストールします。

```bash
# 基本的な数値計算ライブラリとOpenCLバインディングのインストール
pip install numpy pyopencl

# CuPyのインストール (インストールされているCUDAのバージョンに合わせてください)
# 例: CUDA 13.x がインストールされている場合
pip install cupy-cuda13x

# 例: CUDA 12.x がインストールされている場合
# pip install cupy-cuda12x
```

## 🚀 使用方法

基本的なコマンドの構文は以下の通りです。

```bash
python gpu_benchmark.py -b <バックエンド> [オプション]
```

### 実行例

**CUDAを使用して、サイズ8192の行列で10回テストを実行する:**
```bash
python gpu_benchmark.py -b cuda -n 8192 -i 10
```

**OpenCLを使用してテストを実行し、結果をCSVファイルに保存する:**
```bash
python gpu_benchmark.py -b opencl -n 8192 -i 10 -o my_benchmark_results.csv
```

**HPC向けの厳しいテスト（倍精度浮動小数点数 FP64 を使用）:**
```bash
python gpu_benchmark.py -b cuda -n 10240 -d float64 -o my_benchmark_results.csv
```

## 🎛️ コマンドラインオプション

| オプション | 省略形 | デフォルト値 | 説明 |
| :--- | :---: | :--- | :--- |
| `--backend` | `-b` | **必須** | 実行するバックエンドを指定します。`cuda` または `opencl` のいずれかを選択してください。 |
| `--size` | `-n` | `8192` | 生成する正方行列のサイズ $N$ （$N \times N$ の行列になります）。OpenCLの場合は自動的に16の倍数に丸められます。 |
| `--iterations` | `-i` | `10` | ベンチマークを実行する反復回数です。 |
| `--dtype` | `-d` | `float32` | データの精度を指定します。`float32`（単精度）または `float64`（倍精度）を選択できます。 |
| `--output` | `-o` | `None` | 結果を追記するCSVファイルのパス。指定しない場合、CSV出力は行われずコンソールへの表示のみとなります。 |

## 🛠️ トラブルシューティング

**1. `ImportError: DLL load failed while importing cublas` (Windows)**
Windows環境のPython 3.8以降では、セキュリティ上の理由からシステムの `Path` 変数が無視されることがあります。CUDA Toolkitがインストールされていることを確認し、スクプレ内の `hardcoded_bin_path` をご自身の環境（例: `C:\Program Files\NVIDIA GPU Computing Toolkit\CUDA\v13.1\bin`）に合わせて書き換えてください。

**2. `ImportError: libcublas.so.13: cannot open shared object file` (WSL2 / Linux)**
Linux環境でCuPyがCUDAライブラリ（`cuBLAS`など）を見つけられない場合に発生します。CUDA Toolkitが正しくインストールされているか確認し、`~/.bashrc` に `LD_LIBRARY_PATH` が正しく設定されていることを確認してください。

**3. `clGetPlatformIDs failed: PLATFORM_NOT_FOUND_KHR` または `No OpenCL devices found` (WSL2)**
前述の通り、WSL2環境ではNVIDIAのOpenCLドライバが正常にパススルーされません。`clinfo` コマンドで `0 devices` と表示される場合、Linux側からはGPUのOpenCL機能にアクセスできていません。OpenCLのテストはWindowsホスト側で実行してください。

**4. OpenCL実行時の `CompilerWarning: Non-empty compiler output encountered`**
PyOpenCLはカーネルを動的にコンパイルします。NVIDIAのコンパイラが最適化のメモなどを出力しただけでこの警告が表示されますが、ベンチマーク自体が動作しTFLOPSが出力されていれば**完全に無視して問題ありません**。