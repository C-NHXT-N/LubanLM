"""将真实短测和切换记录写入中文说明，不把短测速度当作能力验收。"""
from scripts.common import RUNS, ROOT, read_json
import shutil

def main():
    report=read_json(RUNS/'reports/performance_comparison.json')
    switch=read_json(RUNS/'reports/performance_switch.json')
    profile=read_json(RUNS/'reports/execution_profile.json')
    lines=['# Win11 训练效率优化实测','',f"短测时间（UTC）：{report['started_utc']} 至 {report.get('completed_utc','尚未完成')}。本报告只说明训练执行效率，聊天能力仍按独立评分验收。",'',f"从第 {report['checkpoint_step']:,} 步、{report['checkpoint_tokens']:,} 个本轮 Token 的同一检查点开始短测；有效批次为 8，序列长度 256。各候选预热 20 步，测量 100 步，重复三次。测试权重没有覆盖正式模型。",'', '|设置|中位有效输入 Token/s|GPU 平均利用率|整卡峰值 MiB|损失绝对差|梯度相对 L2 误差|结果|','|---|---:|---:|---:|---:|---:|---|']
    for row in report['tests']:
        reps=row['repeats'];resources=[r['resources'] for r in reps]
        gpu=sum(r.get('gpu_percent',0) for r in resources)/len(resources) if resources else 0
        memory=max((r.get('card_peak_mib',0) for r in resources),default=0)
        speed=f"{row['median_tokens_per_second']:.0f}" if row.get('passed') else '—'
        status='通过' if row.get('passed') else row.get('rejection','未完成三次测量')
        lines.append(f"|{row['name']}|{speed}|{gpu:.1f}%|{memory:.0f}|{row.get('loss_abs_error',0):.6f}|{row.get('gradient_relative_l2',0):.4%}|{status}|")
    lines+=['',f"选定预训练设置：`{report['selected']}`，同期短测速度为原设置的 **{report['speedup']:.2f} 倍**。SFT 的有效输入 Token 不包含 padding；其独立短测速度比为 {report.get('sft_speedup',1):.2f} 倍。短测排除了验证与保存时间，正式运行的端到端速度以进度页为准。",'', '```json',__import__('json').dumps(dict(pretrain=profile['pretrain'],sft=profile['sft']),ensure_ascii=False,indent=2),'```','', '## 切换与恢复','',f"正式恢复使用保存点 {switch['restored_tokens']:,} Token。保存时间比较最初漏掉秒以下精度，导致没有等到下一次保存；已修正为按秒比较。本次最近完整日志能确认至少 **{switch['discarded_logged_tokens']:,} Token** 的未保存更新需要重跑。原 GPU 时间全部结算保留，没有清空账本或把重跑次数重复计为已保存进度。",'',r'切换前权重保存在 `D:\LubanLM-Runs\chat-v2\checkpoints\performance-switch\source.pt`。迁移保留模型、Adam 状态、随机数、采样位置、步数和学习率；分组变化有浮点误差，不能称为逐位一致。', '', '普通 `--resume` 仍拒绝改变执行配置。只有显式 `--migrate-execution` 才允许有效批次不变的分组和执行后端迁移；验证微批次保持 2，验证使用原手写注意力路径。旧权重没有新字段时使用原设置。', '', '## 正确性、预算与后续监控','', '已核验采样顺序、因果掩码、padding、矩形 KV cache、SFT 监督 Token 加权、相同设置的状态恢复及非有限损失拦截。短测采用损失绝对差≤0.01、梯度相对 L2≤2%的门槛；不通过的候选不会上线。', '',f"实际 attention 运算：`{report.get('attention_operators',[])}`。只有实际调用记录支持时才称为融合后端；平台不支持 CUDA profiler 时保留错误信息。", '', '所有短测计入原预训练阶段和24小时累计GPU预算，合计最多20分钟。固定5000条开发阶段probe保持原执行设置；完整SFT使用独立v3数据及通过短测的SFT设置。固定验证损失比切换前增加超过0.1时保存异常权重并自动回退；用户已取消定时检查，之后仅按用户询问检查显存、速度、进程和能力评分。', '', '0.5B尚未启动正式训练。本次代码可供它复用，但仍需独立测定微批次与显存。Win11本轮没有新增编译依赖或迁移WSL；GTX1650保留用于桌面。', '', '最新正式训练状态见 [50M续训进度](50M续训进度.md)，能力完成条件见 [训练闭环](50M聊天训练闭环.md)。','']
    (ROOT/'docs/Win11训练效率优化.md').write_text('\n'.join(lines),encoding='utf-8')
    rollout_path=RUNS/'reports/performance_rollout.json'
    if rollout_path.exists():
        rollout=read_json(rollout_path)
        with (ROOT/'docs/Win11训练效率优化.md').open('a',encoding='utf-8') as stream:
            stream.write(f"\n上线核验（UTC {rollout['updated_utc']}）：控制器 {rollout['controller_pid']}、训练子进程 {rollout['child_pid']}；真实磁盘预训练/SFT恢复均通过。全部性能与恢复检查实际GPU耗时 **{rollout['performance_check_gpu_seconds']/60:.2f} 分钟**，未超过20分钟。正式训练恢复后的资源原始采样见 `performance_rollout.json`。\n")
    destination=ROOT/'docs/run-results/chat-v2';destination.mkdir(parents=True,exist_ok=True)
    for name in ('performance_comparison.json','execution_profile.json','performance_switch.json','execution_resume_checks.json','performance_rollout.json'):
        if (RUNS/'reports'/name).exists():shutil.copy2(RUNS/'reports'/name,destination/name)
    print('performance report updated')

if __name__=='__main__':main()
