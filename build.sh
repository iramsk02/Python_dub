#!/usr/bin/env bash

# Install CPU-only Torch and Whisper manually to avoid heavy CUDA downloads
pip install torch==2.0.1+cpu torchvision==0.15.2+cpu torchaudio==2.0.2+cpu \
  --index-url https://download.pytorch.org/whl/cpu

pip install git+https://github.com/openai/whisper.git
