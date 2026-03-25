# GPU Compute Benchmark (CUDA / OpenCL)

このリポジトリには、GPUの浮動小数点演算性能（TFLOPS）を測定するためのPythonコマンドラインツールが含まれています。

NVIDIAが提供する最適化された `cuSOLVER` を使用する **CUDAバックエンド**（LU分解によるLinpackベンチマーク）と、カスタムカーネルを使用する **OpenCLバックエンド**（タイル化行列乗算 / GEMM）の両方を1つのスクリプトで切り替えて実行できます。結果はCSVファイルに自動的に記録され、比較やグラフ化が容易です。

## ⚙️ システム要件とインストール

このスクリプトを実行するには、Python 3.8以上と以下のGPU環境が必要です。

### 1. GPUドライバとツールキットの準備
* **最新のNVIDIAディスプレイドライバ**: GPUが最新のCUDA命令を認識できるように、ドライバーを最新バージョンにアップデートしてください。
* **NVIDIA CUDA Toolkit**: CUDAバックエンドを使用する場合に必須です。NVIDIAの公式サイトからダウンロードしてインストールしてください（例: v13.1）。

#### ⚠️ WSL2 (Linux) 環境でのセットアップ注意点
WSL2環境で実行する場合、Windows側のNVIDIAドライバがLinux側に自動的にパススルーされます。そのため、**Linux用のNVIDIAディスプレイドライバをインストールしてはいけません**（GPUへのアクセスが壊れます）。CUDA Toolkit（ライブラリ部分のみ）を以下の手順で安全にインストールしてください。

```bash
# 1. Linux用 CUDA 13.1 Runfileのダウンロード
wget https://developer.download.nvidia.com/compute/cuda/13.1.0/local_installers/cuda_13.1.0_590.44.01_linux.run

# 2. ツールキットのみをインストール（ドライバのインストールをスキップ）
sudo sh cuda_13.1.0_590.44.01_linux.run --toolkit --silent

# 3. 環境変数の追加（Pythonがライブラリを見つけられるようにする）
echo 'export PATH=/usr/local/cuda-13.1/bin${PATH:+:${PATH}}' >> ~/.bashrc
echo 'export LD_LIBRARY_PATH=/usr/local/cuda-13.1/lib64${LD_LIBRARY_PATH:+:${LD_LIBRARY_PATH}}' >> ~/.bashrc
source ~/.bashrc
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