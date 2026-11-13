$ErrorActionPreference='Stop'
$taskWorkspace='D:\BaiduSyncdisk\知识库\LubanLM'
$taskPython='D:\miniconda3\envs\lubanlm-gpu\pythonw.exe'
$taskPlan=Join-Path $taskWorkspace 'configs\50m-general-pretrain-10b-teaching.json'
$taskName='LubanLM-Teaching-10B-Supervisor'
$taskAccount=[System.Security.Principal.WindowsIdentity]::GetCurrent().Name
$taskLauncher=Join-Path $taskWorkspace 'scripts\run_teaching_supervisor_hidden.pyw'
if (!(Test-Path -LiteralPath $taskPython) -or !(Test-Path -LiteralPath $taskLauncher)) { throw 'Hidden supervisor launcher is missing' }
$taskAction=New-ScheduledTaskAction -Execute $taskPython -Argument ('-B "'+$taskLauncher+'" --plan "'+$taskPlan+'"') -WorkingDirectory $taskWorkspace
$taskTriggers=@((New-ScheduledTaskTrigger -Once -At (Get-Date).AddMinutes(1) -RepetitionInterval (New-TimeSpan -Minutes 5)),(New-ScheduledTaskTrigger -AtLogOn -User $taskAccount))
$taskPrincipal=New-ScheduledTaskPrincipal -UserId $taskAccount -LogonType Interactive -RunLevel Limited
$taskSettings=New-ScheduledTaskSettingsSet -MultipleInstances IgnoreNew -ExecutionTimeLimit (New-TimeSpan -Minutes 2) -StartWhenAvailable -AllowStartIfOnBatteries -DontStopIfGoingOnBatteries
Register-ScheduledTask -TaskName $taskName -Action $taskAction -Trigger $taskTriggers -Principal $taskPrincipal -Settings $taskSettings -Description 'LubanLM 10B independent supervisor; serial preparation; bounded recovery; self-removes on completion/cancel.' -Force | Out-Null
Start-ScheduledTask -TaskName $taskName
Get-ScheduledTask -TaskName $taskName | Select-Object TaskName,State
