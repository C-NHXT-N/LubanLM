"""Content-review decisions, training-only corrections and a hash-bound training gate.

This is an agent re-review, not an independent human/expert or board-execution claim.
The recorded decisions cover every record; regex checks never grant approval.
"""
from __future__ import annotations
import argparse
import hashlib
import json
from pathlib import Path
import shutil
from scripts.common import WORKSPACE,sha256,write_json,utc
from scripts.linux_assistant.common import jsonl,write_jsonl,fingerprint,verify_data,training_rows,review_digest

DATA=WORKSPACE/'datasets/linux-assistant-unified'
REPORT=WORKSPACE/'reports/linux-assistant-training-review-20261020'
REVIEWER='Codex agent content re-review, 2026-10-20; not independent human review'

# Same operational question/fault case must not occur on both sides of a split.
# Related topics with a different diagnosis/constraint are allowed and documented.
EXCLUDE={
 'environment/storage':('environment/mountinfo','相同目标路径磁盘余量查询，不能因分组名不同而视为独立案例'),
 'environment/path-space':('environment/mountinfo','同一df -h目标路径查询'),
 'environment/memory-and-swap-output':('environment/swap','同一不能将Swap与物理内存简单相加的判断'),
 'linux/permissions':('linux/test-chmod-symbolic','同一所有者增加执行位任务'),
 'linux/tail':('linux/dev-tail-lines','同一持续跟踪追加日志任务'),
 'deployment/hash':('deployment/dev-sha256sum-check','同一传输后用SHA256清单检查模型内容的任务'),
 'deployment/stop-reason':('deployment/eos','同一限长与EOS停止原因区分'),
 'deployment/bundle-manifest':('deployment/dev-sha256sum-check','同一交付包逐文件摘要检查任务'),
 'deployment/staging':('interaction/twooptions','同一新模型独立目录与保留旧模型回退任务'),
 'deployment/context-budget-reason':('troubleshooting/contextoverflow','相同完整历史裁剪处理案例'),
 'deployment/worker-ready':('deployment/loadreturn','相同启动提示不能证明模型初始化成功的判断'),
 'troubleshooting/permissionpath':('interaction/test-ask-minimal-log','相同执行位已存在仍拒绝访问的诊断案例'),
 'troubleshooting/permission-parent':('interaction/test-ask-minimal-log','相同父目录搜索权限与noexec排查'),
 'troubleshooting/no-space-diagnosis':('troubleshooting/diskfull','相同空间不足先核对容量与inode后清理的案例'),
 'troubleshooting/library-symbol-version':('troubleshooting/undefinedsymbol','相同符号缺失与实际库路径/ABI排查'),
 'troubleshooting/npu-core-request':('deployment/twocore','相同请求双核心不等于实际性能提升的案例'),
 'interaction/insufficient':('interaction/dev-ask-for-error','相同仅给模型不能启动时索要诊断资料的案例'),
 'interaction/minimum-diagnostics':('interaction/dev-ask-for-error','同类启动诊断且将板子启动与模型启动混为一谈'),
 'interaction/safe-command-choice':('interaction/dev-explain-risk','同一权限信息不足不递归chmod 777的案例'),
 'interaction/unknown-interface':('interaction/noinventapi','同一未知SDK函数先查看固定版本头文件的案例'),
 'interaction/multiturn-context':('troubleshooting/undefinedsymbol','相同symbol lookup error库路径和ABI排查，且无依据称程序可信'),
}
MERGE={
 'deployment/template-once':'deployment/nativeids',
 'deployment/adapter-base-pair':'deployment/revision',
 'deployment/calibration-split':'deployment/calibration',
 'troubleshooting/first-token-warmup':'troubleshooting/slowfirst',
 'troubleshooting/corrupt-transfer':'troubleshooting/partial',
 'troubleshooting/worker-error-vs-end':'troubleshooting/emptyanswer',
 'interaction/redact-secrets':'interaction/secret',
 'interaction/no-claim-of-execution':'interaction/unseenoutput',
 'interaction/ask-only-missing':'interaction/preservecontext',
 'interaction/uncertain-followup':'interaction/uncertain',
}

# Specific retained legacy-scenario review notes, beyond schema/keyword tests.
NOTES={}
for category,lines in {
 'environment':'''kernel|uname -r是运行内核，不是发行版；十种提示包装没有改变问题条件。
arch|uname -m与file分查系统和制品；仅提供架构证据，不承诺运行库兼容。
distro|核对os-release的ID及VERSION_ID；原来源过于间接，补充os-release手册。
ram|存储容量不能当运行RAM；free与df的检查对象不同。
gpu|Windows NVIDIA CUDA不能证明RK3576 NPU驱动及Runtime可用；与本地设备报告相符。
python|sys.executable是实际解释器路径；提示符名称不构成证据。
driver_runtime|驱动与用户态Runtime分开记录；不编造两者版本相同。
envvar|printf固定格式串并引用变量，只描述当前环境，不推广到其他服务。
findmnt-target|上下文已有/mnt/models，应在命令中沿用它；findmnt用于路径所属挂载点。
swap-areas|swapon --show列活动交换区；区别于仅有持久配置而尚未启用的交换区。
nproc-scope|nproc查询当前进程可用处理单元；环境变量及配额影响依实现，不能当安装总数。
effective-groups|无USER参数的id -G报告调用进程组集合；不冒充任意目标服务进程的组。
effective-uid|id -u报告有效UID；普通文件检查通常与其一致，但Linux fsuid可独立改变，补充限定。
filesystem-type|上下文/data/models必须进入df -T参数；结果是包含该路径的文件系统。
open-files-limit|ulimit -Sn是当前shell软描述符限制，不等于运行服务或系统全局限制。
clock-query|timedatectl status是只读查询，依赖systemd，状态输出不保证实际同步可靠。''',
 'linux':'''pwd|pwd针对当前工作目录；不声称相对路径在其他目录仍指向同一文件。
ls|ls -la包括点文件详细项；只读且没有隐含修改。
mkdir|mkdir -p按需创建父目录；不保证无权限、同名普通文件时也成功。
head|head -n 20读取前20行，没有修改动作。
copy|cp -a复制目录并保留元信息；先确认目标存在性以免层级误解，保留受权限及文件系统约束。
find|find的-name模式已引用，避免shell提前展开；递归仅在当前目录。
umask-query|umask无参数或-S仅查询；最终权限还受程序请求及默认ACL等因素影响。
grep-recursive|grep -R -n -F --分别处理递归、行号、固定字串及选项边界。
safe-find-pipe|NUL分隔配合-print0和-0可保留空格/换行文件名边界；不是可直接执行的占位命令。
tee-copy|2>&1将两路输出交给tee；默认覆盖，-a才追加，涉及日志写入有明确用户目的。
symlink-create|ln -s目标与链接位置顺序正确；相对目标按链接所在目录解释。
graceful-signal|TERM可由应用处理以释放资源但并非保证退出；KILL无法捕获，先确认PID与状态。
resolve-command|command -v可能返回别名或函数；type用于确认解析类型，不谎称总是外部路径。
compare-files|cmp比较内容字节，状态0为一致；非零还可能是访问错误，应保留错误信息。
printf-safe-value|固定%s格式串、变量作为独立引用参数，避免格式串注入与错误转义。
background-jobs|jobs -l只列当前交互shell管理的作业，不代表机器全部进程。''',
 'deployment':'''template|Qwen词嵌入与原生Tokenizer ID匹配，不能替换为50M自训32K映射。
adapter|LoRA仅保存增量需相同基座；本地交付流程在合并后重新验证和转换。
bf16w4|训练BF16与部署W4A16是不同阶段，不能从部署格式推断训练量化方式。
revision|基座revision与权重摘要应和训练时一致；同名不同版本不能随意配对。
calibration|校准参与模型处理，不能抽最终题；独立校准数据与开发验收隔离。
ctxbudget|固定2048减256得到1792；输入包括系统、历史、当前问题与模板特殊Token。
libs|本地原模型报告支持Toolkit/Runtime 1.3.1实测组合，不声称所有驱动通用兼容。
nativeids|模板仅渲染一次；再编码禁自动重复特殊Token以保持PC与板端一致。
runtime-dependencies|模型文件存在不等于能运行，还要检查架构、驱动、运行库与实际生成。
interrupt-history|本地聊天程序只提交成功完整回答；中断/错误文本不进入后续历史。
offline-tokenizer|tokenizers及Jinja2实现离线原生模板，需固定文件并做Token ID对照；不依赖PyTorch。''',
 'troubleshooting':'''wrongarch|Exec format error可由格式/架构等原因产生，file与uname -m对照只是诊断，不直接判权限。
libmissing|仅对可信程序使用ldd；记录实际依赖路径、架构与搜索环境，不先替换系统库。
oom|记录模型/上下文/进程及系统内存；减少上下文不是无条件修复承诺。
hashmismatch|摘要不匹配停止使用并留证，核对传输和源文件，不冒充完整制品。
emptyanswer|空文本须结合EOS、错误和回调状态，不能自动判成功；与本地协议实现一致。
gibberish|先核对模型、Tokenizer、模板、特殊ID与编码；不无依据判权重损坏。
slowfirst|冷启动与预热后延迟分开，多次同条件测量；不以单次结果推稳定性能。
repeat|固定输入查历史重复、模板、停止和解码；保留原始输出再比模型/量化。
partial|保留部分文件可续传，但完成后仍须完整SHA256核验；非空不是完整证明。
service-state|is-failed读取失败状态，journalctl -u -b读取本次系统启动相关服务日志；不声称只含最近一次服务重启。''',
 'interaction':'''unknownboard|不把教程中的RK3576自动作为用户设备；先索取实际板型。
unseenoutput|没有实际访问和输出不能声称检查通过；明确未验证并请求证据。
jsonfields|用户要求仅JSON、字段task=check和status=pending；参考无包装且字段值正确。
onlycommand|用户仅要当前目录命令，参考严格为pwd；包装前缀未改变格式约束。
uncertain|缺证据时明确假设与事实，索取原始日志，不假装确定原因或执行检查。
preservecontext|沿用已有板型、系统和Runtime上下文，仅追问缺项；RK3576是芯片而非具体板型，避免误称。
steps|建议有顺序的检查、预期观察与后续分支；未编造接口或设备状态。
secret|只需相关配置，隐去密码/令牌/私钥等；无需索取完整秘密字段。
noexecute|本地离线助手无工具访问，解释能力边界并给用户检查方法，不假称已访问。
separate-evidence|日志只证实库加载失败，驱动不兼容仍是假设；请求版本/路径/日志以区分。
strict-json|将含糊元问题改为明确仅输出JSON请求，布尔false不写成字符串或数字。'''
}.items():
    for line in lines.splitlines():
        slug,note=line.split('|',1);NOTES[category+'/'+slug]=note

CURATED_NOTES={
 'venv-prefixes':'限定标准venv；prefix/base_prefix状态不依赖提示符，未泛化为conda判定。',
 'block-sector-topology':'LOG-SEC/PHY-SEC是扇区字节大小，SIZE才是设备容量；未把扇区大小推断为可用容量。',
 'boot-id-scope':'boot_id区分启动实例而非日期或硬件序列号；跨日不改变同次启动标识。',
 'process-cgroup-membership':'0::路径是统一层级成员关系，不是CPU或内存指标；PID文件还受存活/权限影响。',
 'argument-environment-budget':'ARG_MAX涉及exec参数和环境长度；单参数限制、环境长度与模型文件大小不能混用。',
 'locale-current-encoding':'locale charmap只描述当前locale，不能保证旧文件编码；查看LC_CTYPE及产生端。',
 'stdout-terminal-check':'-t 1检查stdout，0/2是另外的流；管道输出通常不应加终端颜色序列。',
 'active-bash-versus-login-shell':'SHELL变量不证明当前解释器；显式sh执行不会按shebang切换Bash。',
 'process-working-directory':'程序路径与cwd独立；proc/PID/cwd需权限及进程存在，相对路径依cwd。',
 'python-user-site-enabled':'site路径存在不等于启用；-s禁用用户site，须核对ENABLE_USER_SITE及sys.path。',
 'leading-dash-filename':'--或./避免cat将文件名当选项；第二条严格只有一条命令。',
 'grep-no-match-versus-error':'GNU grep 0匹配、1无匹配、2错误，-q另有提前成功情况；不吞stderr。',
 'read-preserve-backslashes':'IFS=保留首尾空白，read -r保留反斜杠；末行无换行用非空残段分支。',
 'pipeline-failure-propagation':'默认末命令状态；pipefail取最右非零，示例2|0结果2，不等同set -e。',
 'safe-temporary-directory':'mktemp -u只生成名称不创建；-d实际创建目录且须确认成功。',
 'delimited-fields-not-bytes':'冒号第二字段用-d: -f2，-b按字节位置不适于变长字段；示例结果ready。',
 'xargs-empty-input':'GNU -r禁止空输入运行，-0处理NUL边界；明确实现依赖，两个选项不替代。',
 'required-parameter-empty-check':':?拒绝未设置或空值；?只拒绝未设置，尚未检查路径存在；非交互退出限定正确。',
 'command-scoped-environment':'限定普通外部命令；命令前赋值只传环境，MODE=为空而非unset，不修改父shell。',
 'hard-link-identity':'硬链接共享inode不是独立副本；不能跨文件系统，改权限不解决该边界。',
 'installed-dependency-check':'pip check仅验证已装包约束，不能证明实际导入或NPU推理；两例结论一致。',
 'offline-wheel-install':'--no-index配--find-links限制本地索引来源；需全依赖及平台/Python ABI兼容，通用纯Python另判。',
 'dependency-artifact-hashes':'版本不保证制品字节；hash确认字节而非API行为，需全依赖固定与功能测试。',
 'venv-relocation-shebang':'标准venv入口shebang有绝对路径，改PATH不改脚本；新位置重建并保留回退。',
 'stdout-protocol-stderr-logs':'stdout逐行JSON协议与stderr诊断分离，合并后再次破坏JSON记录边界。',
 'utf8-incremental-stream':'中为E4 B8 AD，拆块不是损坏；增量解码保留状态、final严格检查，ignore可能丢字。',
 'resumable-jsonl-commit':'只复用完整提交且输入身份一致记录；中间坏行停止恢复，原子快照不等于磁盘持久化保证。',
 'service-working-directory':'systemd程序路径不决定工作目录；WorkingDirectory和绝对配置之外仍有相对资源。',
 'service-environment-not-login':'系统服务不自动继承SSH终端export；EnvironmentFile不是Bash，不执行export或命令替换。',
 'elf-origin-relative-libraries':'动态加载器ORIGIN是程序/对应共享对象目录；shell单引号保留字面量，路径正确仍须ABI兼容。',
 'crlf-shebang-interpreter':'^M及0D 0A支持先查CRLF，非重装依据；备份再转LF，不删除系统解释器。',
 'python-standard-module-shadow':'本地select.py/json.py可遮蔽标准模块；改自有模块名并确认导入路径，升级pip无助根因。',
 'consumer-closed-pipe':'head可主动关闭管道引发BrokenPipe；须核对消费端和生产者状态，不直接判模型崩溃或无限重试。',
 'sticky-directory-ownership':'1777带sticky限制删除/重命名，文件写权限不同于目录删除权限；未建议去掉sticky。',
 'executable-open-for-writing':'ETXTBSY涉及执行与写句柄冲突；核对使用者并独立制品受控切换，不盲目覆盖重试。',
 'bind-address-in-use':'EADDRINUSE通常是绑定冲突，需地址/端口/实例证据；不直接认定断网或杀所有同名进程。',
 'missing-elf-interpreter':'文件存在仍可缺脚本/ELF解释器而ENOENT；链接同名不能解决架构/ABI不兼容。',
 'readonly-filesystem-error':'EROFS是只读文件系统，不靠chmod修复；只读模型介质应把日志指到允许写的位置。',
 'decode-error-preserve-evidence':'保存原字节核对编码和无效位置；ignore会丢信息，单错不能推硬件损坏。',
 'jsonl-versus-json-document':'两行各合法JSON不构成单文档；按协议逐行或单数组，不忽略Extra data丢后续记录。',
 'measurement-denominator':'24/6=4且无首字时刻；40/8=5与40/10=4口径不同，数值均显式标模拟。',
 'readonly-constraint':'test -d不修改路径；失败需区分不存在/类型/访问，不擅自扩大只读授权。',
 'untrusted-log-instructions':'模拟日志指令只是数据；checksum mismatch才是给定故障，不冒充成功或泄露秘密。',
 'pause-scope-no-restart':'用户限定保留记录/数据检查，答复不启动训练、不附启动命令或假称停了进程。',
 'strict-tsv-no-wrapper':'两行两列制表符，无表头代码块；第二例按最新更正missing/ready而非历史。',
 'do-not-evaluate-log-data':'eval/命令替换可执行字符串；日志当数据解析，模型不执行也不编造结果。',
 'latest-user-correction':'Ubuntu改Debian与x86改aarch64均采最新明确条件，旧程序不直接通用。',
 'unknown-json-null':'未知Runtime用JSON null，已知版本用字符串；verified布尔false不凭版本变true。',
 'explain-only-user-defined-terms':'遵循题内定义，一句自然语言，无命令，第二例不用第一例的术语含义。',
 'provided-usage-controls-syntax':'仅照题给虚构help选项，首例model/tokens，次例limit，m.bin明确是示例路径。',
}

def prepare():
    raise RuntimeError('Legacy review preparation is retired because it snapshots the removed corpus. Review new imported records with the independent review workflow.')
    if (DATA/'curation/quality-contract.json').exists():
        raise ValueError('Historical review cannot overwrite the rebuilt dataset; use the new data-quality audit')
    REPORT.mkdir(parents=True,exist_ok=True)
    if (REPORT/'before/train.jsonl').exists():
        raise FileExistsError('Review already prepared; use --stage finalize after checking evidence')
    before=REPORT/'before';before.mkdir()
    for name in ('train.jsonl','manifest.json','source-review.jsonl','source-review-lock.json'):
        shutil.copy2(DATA/name,before/name)
    groups={g:{'exclude':True,'overlaps':v[0],'reason':v[1]} for g,v in EXCLUDE.items()}
    for g,target in MERGE.items():groups[g]={'merge_into':target,'reason':'同一训练案例家族，合并计数而不冒充新场景'}
    def revise(group,reference,rubric,sources=None):
        groups.setdefault(group,{}).update(reference=reference,rubric=rubric)
        if sources:groups[group]['sources']=sources
    revise('environment/findmnt-target','运行 `findmnt --target /mnt/models`，查看覆盖该路径的文件系统和挂载点；不要只凭目录名推断挂载位置。',r'findmnt;--target;/mnt/models')
    revise('environment/filesystem-type','运行 `df -T /data/models` 查看该路径所在文件系统的类型；输出针对包含该路径的文件系统。',r'df;-T;/data/models')
    revise('environment/effective-uid','在当前shell运行 `id -u` 查看调用进程的有效用户ID，不代表其他服务的身份。Linux文件权限检查通常与它一致，但实际还涉及fsuid、组和访问控制，不能仅凭UID断定可访问。',r'id -u;有效;fsuid',[
        'https://man7.org/linux/man-pages/man1/id.1.html','https://man7.org/linux/man-pages/man7/path_resolution.7.html'])
    revise('environment/distro','查看 /etc/os-release 中的 ID、VERSION_ID 等字段；不要用内核版本代替发行版版本。',r'os-release;VERSION_ID',[
        'https://man7.org/linux/man-pages/man5/os-release.5.html'])
    revise('interaction/ask-only-missing','已知芯片是RK3576、系统是Ubuntu 22.04。请补充具体Runtime版本、驱动版本和完整错误输出；不重复索取已有芯片和系统信息。',r'Runtime;驱动;不重复')
    records={
      'v2-interaction-strict-json':{'messages':[{'role':'user','content':'只输出JSON对象，字段ok为布尔false，不要解释或代码块。'},
          {'role':'assistant','content':'{"ok":false}'}]},
      'troubleshooting-decode-error-preserve-evidence-02':{'messages':[{'role':'user','content':'已确认日志预期编码为UTF-8，文件已经写完，但其中一个字节序列无效，能仅凭报错断言设备内存损坏吗？'},
          {'role':'assistant','content':'不能。应记录无效序列位置和原始字节，检查写入、传输或拼接过程；单个解码错误不足以证明硬件内存损坏。'}]},
    }
    write_json(DATA/'curation/content-corrections.json',{'reviewer':REVIEWER,'groups':groups,'records':records,
        'policy':'Exclude same held-out cases; merge same training families; retain useful phrasing variants without counting as independent cases.'})
    from scripts.linux_assistant.build_unified_dataset import main
    main()
    rows=jsonl(DATA/'train.jsonl');old=jsonl(before/'train.jsonl');index={r['id']:r for r in rows}
    reviews=[];decisions=[]
    for r in old:
        g=r['group'];active=index.get(r['id'])
        if active is None:
            decisions.append({'id':r['id'],'before_sha256':fingerprint(r),'decision':'excluded',**groups[g]});continue
        target=active['group'];slug=target.split('/curated-',1)[-1]
        note=CURATED_NOTES.get(slug) if '/curated-' in target else NOTES.get(target)
        if not note:raise ValueError('Missing explicit content decision: '+target)
        verdict={'id':active['id'],'group':target,'record_sha256':fingerprint(active),'accepted':True,'reviewer':REVIEWER,
            'reason':note,'source_urls':active['sources'],'checks':{'question_and_history':True,'reference_facts':True,
                'format_and_constraints':True,'no_fabricated_execution':True,'split_case_identity':True},
            'independent_human_review':False,'linux_board_execution':False}
        reviews.append(verdict)
        decisions.append({'id':r['id'],'before_sha256':fingerprint(r),'after_sha256':fingerprint(active),
            'decision':'retained' if r==active else 'corrected_or_regrouped','reason':note})
    write_jsonl(REPORT/'record-decisions.jsonl',decisions)
    write_jsonl(REPORT/'approved-training-records.jsonl',reviews)
    write_json(REPORT/'content-review.json',{'completed':True,'reviewer':REVIEWER,'before_records':len(old),
        'approved_records':len(reviews),'excluded_records':len(old)-len(rows),'training_scenarios':len({r['group'] for r in rows}),
        'unique_references':len({r['reference'] for r in rows}),'train_sha256':sha256(DATA/'train.jsonl'),
        'corrected_or_regrouped':sum(d['decision']=='corrected_or_regrouped' for d in decisions),
        'distinct_review_notes':len({r['reason'] for r in reviews}),'external_expert_review':False,
        'linux_runtime_test':'not performed; Linux/Bash cases checked against documented semantics, not actual board execution',
        'related_topics_allowed':'Same commands/knowledge may occur with different requested observations; same fault/task case is excluded.',
        'retained_variants':'Legacy prompt-prefix variants retained as augmentation; counts are records, not independent knowledge points.'})

def finalize():
    raise RuntimeError('Automatic approval from the legacy review ledger is retired. New content approval requires the independent-review schema.')
    audit=json.loads((WORKSPACE/'reports/linux-assistant-dataset/native-token-audit.json').read_text('utf8'))
    tests=json.loads((REPORT/'tests.json').read_text('utf8'))
    if not audit['passed'] or audit['train_sha256']!=sha256(DATA/'train.jsonl') or not tests['passed']:
        raise ValueError('Native mask audit and regression tests must pass before approval')
    result={'passed':True,'completed_utc':utc(),'reviewer':REVIEWER,'review_scope':'all active training records only',
        'train_sha256':sha256(DATA/'train.jsonl'),'manifest_sha256':sha256(DATA/'manifest.json'),
        'source_review_sha256':sha256(DATA/'source-review.jsonl'),
        'ledger_sha256':sha256(REPORT/'approved-training-records.jsonl'),
        'evidence':{str(p.relative_to(WORKSPACE)).replace('\\','/'):sha256(p) for p in [
            REPORT/'content-review.json',REPORT/'record-decisions.jsonl',REPORT/'tests.json',
            WORKSPACE/'reports/linux-assistant-dataset/native-token-audit.json']},
        'external_expert_review':False,'linux_board_execution':False}
    write_json(REPORT/'training-approval.json',result)
    validate_training_review({'data':DATA,'training_review':str((REPORT/'training-approval.json').relative_to(WORKSPACE))})
    print(json.dumps(result,ensure_ascii=False))

def validate_training_review(c):
    path=(WORKSPACE/c['training_review']).resolve();approval=json.loads(path.read_text('utf8'))
    if approval.get('approval_schema')!='lubanlm.dataset-content-review.v2' or approval.get('review_mode')!='independent_agent_cross_review':
        raise ValueError('Legacy or non-independent content approval schema rejected')
    if approval.get('passed') is not True:raise ValueError('Training data not approved')
    if c.get('experiment_scope')=='reviewed_small_scale_exploration':
        if (approval.get('scope')!='reviewed_small_scale_exploration' or
                approval.get('user_choice')!=2 or approval.get('coverage_incomplete') is not True):
            raise ValueError('Explicit small-experiment content approval scope is missing')
    if approval.get('system_prompt_sha256'):
        if hashlib.sha256(c.get('system_prompt','').encode()).hexdigest()!=approval['system_prompt_sha256']:
            raise ValueError('System prompt differs from the approved data contract')
    for name,key in [('train.jsonl','train_sha256'),('manifest.json','manifest_sha256'),('source-review.jsonl','source_review_sha256')]:
        if sha256(c['data']/name)!=approval[key]:raise ValueError('Approved data changed: '+name)
    manifest=json.loads((c['data']/'manifest.json').read_text(encoding='utf8'))
    if 'general-train.jsonl' in manifest.get('data_splits',[]):
        if sha256(c['data']/'general-train.jsonl')!=approval.get('general_train_sha256'):
            raise ValueError('Approved data changed or missing general training approval: general-train.jsonl')
    ledger=path.parent/'approved-training-records.jsonl'
    if sha256(ledger)!=approval['ledger_sha256']:raise ValueError('Content-review ledger changed')
    for name,digest in approval['evidence'].items():
        if sha256(WORKSPACE/name)!=digest:raise ValueError('Review evidence changed: '+name)
    rows=training_rows(c);reviews=jsonl(ledger)
    by_id={r['id']:r for r in reviews}
    if len(by_id)!=len(reviews) or set(by_id)!={r['id'] for r in rows}:raise ValueError('Incomplete per-record content review')
    for row in rows:
        decision=by_id[row['id']]
        author=row.get('created_by') or row.get('evidence',{}).get('created_by')
        if decision.get('decision')!='accept' or not decision.get('reviewer_id') or decision.get('reviewer_id')==author or not decision.get('reason') or fingerprint(row)!=review_digest(decision):
            raise ValueError('Unapproved content: '+row['id'])
    verify_data(c)
    return {'approval_sha256':sha256(path),'approved_records':len(rows),'reviewer':approval['reviewer']}

if __name__=='__main__':
    parser=argparse.ArgumentParser(description=__doc__);parser.add_argument('--stage',choices=['prepare','finalize'],required=True)
    args=parser.parse_args();prepare() if args.stage=='prepare' else finalize()
