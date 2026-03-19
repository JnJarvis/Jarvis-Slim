# Jarvis-Slim

A lightweight local Jarvis-style assistant powered by `llama-cpp-python` and `Llama-3.2-8B-Instruct`.

NOTE: This is a demo code is based on a code heavily integrated into my setup (local comuputer, networks, and accounts). Hence, much of the functionality has been removed for ease of setup. RAG, Reminiders, Actions, and all original plugins do not yet exsist for this version.

## Installation
Model link: https://huggingface.co/mradermacher/Llama-3.2-8B-Instruct-GGUF/resolve/main/Llama-3.2-8B-Instruct.Q5_K_M.gguf
1. Install Python 3.10+.
2. Put your model file at:
   - `C:/Users/<your-username>/Llama-3.2-8B-Instruct-Q5_K_M.gguf`
   - Or change `model_path` in `jarvis.py`.
3. In this project folder, run:
   - `python setup_config.py`
4. The setup script will now check/install:
   - `rich`
   - `python-dotenv` (imported as `dotenv`)
   - `llama-cpp-python` (it will prompt for GPU install)

If `llama-cpp-python` was not installed during setup, install it manually:
- `python -m pip install llama-cpp-python`

## Usage

- Start the assistant:
  - `python jarvis.py`
- Chat in the terminal.
- Type `exit` to quit.
