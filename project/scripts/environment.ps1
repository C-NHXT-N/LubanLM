# 从项目根目录 dot-source：. .\scripts\environment.ps1
$env:CUDA_DEVICE_ORDER = 'PCI_BUS_ID'
$env:CUDA_VISIBLE_DEVICES = 'GPU-c50d3c54-c5d3-f0da-7fa4-a48d195a849f'
$env:HF_HOME = 'D:\LubanLM-Data\cache\huggingface'
$env:HF_DATASETS_CACHE = 'D:\LubanLM-Data\cache\datasets'
$env:PIP_CACHE_DIR = 'D:\LubanLM-Data\cache\pip'
$env:TEMP = 'D:\LubanLM-Data\tmp'
$env:TMP = $env:TEMP
$env:PYTHONUTF8 = '1'
$env:HF_HUB_DISABLE_SYMLINKS_WARNING = '1'
$env:TOKENIZERS_PARALLELISM = 'false'
$env:Path = 'D:\miniconda3\envs\lubanlm-gpu;D:\miniconda3\envs\lubanlm-gpu\Scripts;' + $env:Path
