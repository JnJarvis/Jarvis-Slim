import multiprocessing as mp
import argparse

parser = argparse.ArgumentParser()
env = '.env'
parser.add_argument("--color", help="Color of the assistant", default="bold dark_orange3", type=str, required=False)
parser.add_argument("--verbose", help="Verbose output of the code (helpful for debugging)", default=False,required=False)
parser.add_argument("--use_gpu", help="If you want the model to be loaded into the GPU", type=bool, default=True,required=False)
parser.add_argument("--number_gpu_layers", help="How many layers of the llm to offload to the GPU. (If you run out of memory, try lowering this value)", default=-1,required=False)
parser.add_argument("--verbose_llamacpp", help="Verbose output of llama.cpp (Helpful for debugging if its running slow)", default=False,required=False)
parser.add_argument("--context_length", help="Maximum context (conversation) length (higher -> more RAM or VRAM usage)", default=16384,required=False)
parser.add_argument("--batch_size", help="Number of tokens to process concurrently. (higher -> more RAM or VRAM usage)", default=8192,required=False)
parser.add_argument("--number_threads", help="Number of CPU threads to use (If running on the CPU, higher is faster, otherwise not much performance increase)", default=mp.cpu_count(),required=False)
parser.add_argument("--flash_attention", help="Use Flash Attention (faster inference)", default=True,required=False)
args = parser.parse_args()

import sys
import subprocess

try:
    import dotenv
    print('dotenv installed!')
except:
    doteinst = subprocess.Popen([sys.executable, '-m', 'pip', 'install','dotenv'])
    doteinst.wait()

try:
    import rich
    print('rich installed!')
except:
    richinst = subprocess.Popen([sys.executable, '-m', 'pip', 'install','rich'])
    richinst.wait()

dotenv.set_key(env,'assistant_color',str(args.color))
dotenv.set_key(env,'verbose',str(args.verbose))
dotenv.set_key(env,'number_gpu_layers',str(args.number_gpu_layers))
dotenv.set_key(env,'verbose_llamacpp',str(args.verbose_llamacpp))
dotenv.set_key(env,'context_length',str(args.context_length))
dotenv.set_key(env,'batch_size',str(args.batch_size))
dotenv.set_key(env,'number_threads',str(args.number_threads))
dotenv.set_key(env,'flash_attention',str(args.flash_attention))

try:
    import llama_cpp
    print("llama_cpp installed!")
except Exception:
    if args.use_gpu:
        if input("Install llama-cpp-python? y/n:")=='y':
            subprocess.Popen(['set', '-DGGML_CUDA=on'])
            ins = subprocess.Popen([sys.executable, '-m', 'pip', 'install','llama-cpp-python'])
            ins.wait()
    else:
        llamainst = subprocess.Popen([sys.executable, '-m', 'pip', 'install','llama-cpp-python'])
        llamainst.wait()

print("")