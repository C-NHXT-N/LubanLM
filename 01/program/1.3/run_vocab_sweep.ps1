# 依次训练 16K、32K、40K，三个候选始终读取同一训练和评估文件。
param(
    [string]$TrainFile = (Join-Path $PSScriptRoot 'sample_train.jsonl'),
    [string]$EvalFile = (Join-Path $PSScriptRoot 'sample_eval.jsonl'),
    [int]$HiddenSize = 1024,
    [string]$OutputRoot = (Join-Path $PSScriptRoot '..\..\..\artifacts')
)

# 任一候选失败就停止，避免把不完整结果当作完整对照。
$ErrorActionPreference = 'Stop'
$OutputRoot = [System.IO.Path]::GetFullPath($OutputRoot)
$TrainScript = Join-Path $PSScriptRoot 'train_tokenizer.py'
$EvalScript = Join-Path $PSScriptRoot 'evaluate_tokenizer.py'
$ModelRoot = Join-Path $OutputRoot 'tokenizers'
$ReportRoot = Join-Path $OutputRoot 'tokenizer-reports'

foreach ($Size in @(16000, 32000, 40000)) {
    # 每个目标词表拥有独立模型和报告目录，不会覆盖前一档结果。
    $Tag = "vocab-$($Size / 1000)k"
    $ModelDir = Join-Path $ModelRoot $Tag
    $ReportDir = Join-Path $ReportRoot $Tag
    Write-Host "Training $Tag ..."
    # 先训练，再只用独立评估集生成报告。
    python $TrainScript --input $TrainFile --vocab-size $Size --output $ModelDir
    if ($LASTEXITCODE -ne 0) { throw "Tokenizer training failed for $Tag (exit $LASTEXITCODE)" }
    python $EvalScript --tokenizer $ModelDir --input $EvalFile --report-dir $ReportDir --hidden-size $HiddenSize
    if ($LASTEXITCODE -ne 0) { throw "Tokenizer evaluation failed for $Tag (exit $LASTEXITCODE)" }
}

Write-Host "Reports: $ReportRoot"
