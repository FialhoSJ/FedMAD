@echo off
#conda activate pfllib
# Teste rápido sem SLM
python main.py -nmc 30 -nc 100 -jr 1 -atk all -cc 6 -gr 300 -data Cifar10 -t 10 -ls 1 -did 1 -rfake 1 -m VGG

# Teste com SLM aggregator (comentado por default)
# python main.py -nmc 30 -nc 100 -jr 1 -atk all -cc 6 -gr 300 -data Cifar10 -t 10 -ls 1 -did 1 -rfake 1 -m VGG -slm_e True -slm_m microsoft/Phi-3-mini-4k-instruct

###python main.py -nm 0 -nc 100 -jr 1 -atk all -cc 5 -gr 300 -data Cifar10 -t 10 -ls 1 -did 1 -rfake 1
