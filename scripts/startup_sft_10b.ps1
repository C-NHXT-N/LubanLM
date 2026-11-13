$ErrorActionPreference='Stop'
$taskRuns='D:\LubanLM-Runs\general-10b-v1'
if (Test-Path -LiteralPath (Join-Path $taskRuns 'reports\gpu_budget.lock')) { throw 'GPU task already active; no parallel smoke test.' }
$env:LUBAN_DATA='D:\LubanLM-Data\general-10b-v1'
$env:LUBAN_RUNS=$taskRuns
$env:LUBAN_GPU_BUDGET_MINUTES='6360'
$env:PYTHONIOENCODING='utf-8'
$taskArgs=@('-B','-u','-m','training.train','--config','configs/config_50m.json','--stage','sft','--output','D:/LubanLM-Runs/general-10b-v1/checkpoints/sft-startup-smoke','--init','D:/LubanLM-Runs/general-2b-v1/checkpoints/50m-general-base-2b/last.pt','--minutes','5','--epochs','1','--micro-batch','16','--accumulation','1','--lr','0.00002','--execution','optimized','--attention-backend','sdpa','--optimizer-backend','fused','--pin-memory','--validation-micro-batch','2','--sft-train','.cache/general-10b-sft-smoke/train.jsonl','--sft-validation','.cache/general-10b-sft-smoke/validation.jsonl','--sft-batching','bucketed','--prefetch-batches','2','--full-sft-validation','--source-validation','--validation-every-steps','4','--log-every','1','--keep-step-checkpoints','2','--disk-free-bytes-floor','21474836480')
$taskProc=Start-Process -FilePath 'D:\miniconda3\envs\lubanlm-gpu\python.exe' -ArgumentList $taskArgs -WorkingDirectory 'D:\BaiduSyncdisk\知识库\LubanLM' -WindowStyle Hidden -RedirectStandardOutput (Join-Path $taskRuns 'sft-startup.stdout.log') -RedirectStandardError (Join-Path $taskRuns 'sft-startup.stderr.log') -PassThru
Write-Output $taskProc.Id
