$ErrorActionPreference='Stop'
$env:LUBAN_DATA='D:\BaiduSyncdisk\知识库\LubanLM\.cache\general-10b-smoke\data'
$env:LUBAN_RUNS='D:\LubanLM-Runs\general-10b-v1'
$env:LUBAN_GPU_BUDGET_MINUTES='6360'
$env:PYTHONIOENCODING='utf-8'
$taskArgs=@('-B','-u','-m','training.train','--config','configs/config_50m.json','--stage','pretrain','--output','D:/LubanLM-Runs/general-10b-v1/checkpoints/startup-smoke','--continue-from','D:/LubanLM-Runs/general-2b-v1/checkpoints/50m-general-base-2b/last.pt','--minutes','8','--steps','1000','--tokens','6144000','--length','256','--micro-batch','24','--accumulation','1','--lr','0.0001','--warmup-steps','334','--execution','optimized','--attention-backend','sdpa','--optimizer-backend','fused','--pin-memory','--validation-micro-batch','2','--pretrain-sampling','full-pass','--prefetch-batches','2','--validation-tokens','10000','--validation-every-tokens','2000000','--log-every','100','--checkpoint-bytes-cap','6442450944')
$taskProc=Start-Process -FilePath 'D:\miniconda3\envs\lubanlm-gpu\python.exe' -ArgumentList $taskArgs -WorkingDirectory 'D:\BaiduSyncdisk\知识库\LubanLM' -WindowStyle Hidden -RedirectStandardOutput 'D:\LubanLM-Runs\general-10b-v1\startup.stdout.log' -RedirectStandardError 'D:\LubanLM-Runs\general-10b-v1\startup.stderr.log' -PassThru
Write-Output $taskProc.Id
