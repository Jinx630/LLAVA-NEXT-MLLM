#### 1. **Clone this repository and navigate to the LLaVA folder:**
```bash
git clone git@github.com:Jinx630/LLAVA-NEXT-MLLM.git
cd LLaVA-NeXT
```

#### 2. **Install the inference package:**
```bash
conda create -n llava-next python=3.10 -y
conda activate llava-next
pip install --upgrade pip  # Enable PEP 660 support.
pip install -e ".[train]"
cd transformers
pip install -e .
```

#### 3. **training:**
```bash
stage1: bash scripts/train/pretrain_clip_moss.sh
stage2: bash scripts/train/finetune_clip_moss.sh
```