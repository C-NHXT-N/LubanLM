"""Build a separately versioned, non-frozen Linux-assistant data candidate."""
from __future__ import annotations
import hashlib, json
from pathlib import Path
from scripts.common import ROOT, WORKSPACE
from scripts.linux_assistant.common import jsonl, write_jsonl, screen

BASE = WORKSPACE / 'datasets/linux-assistant-v1-reviewed'
OUT = WORKSPACE / 'datasets/linux-assistant-v2-candidate'

# One independently authored task per group. References use primary manuals or
# project evidence; variants are deliberately not fabricated to inflate counts.
ITEMS = [
('environment','findmnt-target','模型目录容量与预期不符，怎样确认该路径实际落在哪个文件系统？','运行 findmnt --target /模型目录，查看覆盖该路径的文件系统和挂载点；不要只凭目录名推断挂载位置。','findmnt;--target',['https://man7.org/linux/man-pages/man8/findmnt.8.html']),
('environment','swap-areas','如何列出系统当前配置的交换区？','运行 swapon --show 列出交换设备或文件；交换区不能当作等速的物理内存来估算模型容量。','swapon;--show',['https://man7.org/linux/man-pages/man8/swapon.8.html']),
('environment','nproc-scope','nproc 的结果是否一定等于机器安装的CPU总数？','不一定。nproc 默认报告当前进程可用的处理单元，可能受CPU配额或环境变量影响；它不必等于在线处理器总数。','nproc;当前进程',['https://www.gnu.org/software/coreutils/manual/html_node/nproc-invocation.html']),
('environment','effective-groups','怎样查看当前进程所属的所有组ID？','运行 id -G 查看当前进程的组ID列表；如需组名可使用 id -Gn。','id -G;id -Gn',['https://man7.org/linux/man-pages/man1/id.1.html']),
('environment','effective-uid','怎样查看当前进程用于权限检查的有效用户ID？','运行 id -u 查看当前进程的有效用户ID；它不等于用户名输出，也不能说明其他服务进程的身份。','id -u;有效',['https://man7.org/linux/man-pages/man1/id.1.html','https://man7.org/linux/man-pages/man7/credentials.7.html']),
('environment','filesystem-type','怎样查看模型目录所在文件系统的类型？','运行 df -T /模型目录查看该路径所在文件系统的类型；输出针对包含该路径的文件系统。','df -T;文件系统',['https://www.man7.org/linux/man-pages/man1/df.1.html']),
('environment','path-space','检查模型目录可用磁盘空间时，命令应该针对哪个路径？','运行 df -h /模型目录，传入目标目录以查看其所在文件系统的可用空间；不要用其他挂载点的余量代替。','df -h;目标目录',['https://www.man7.org/linux/man-pages/man1/df.1.html']),
('environment','memory-and-swap-output','free -h 的输出能把可用内存和交换空间直接相加作为模型可用内存吗？','不能。free -h 分别报告内存和交换空间统计；物理内存与交换空间的性能和用途不同，需分别记录。','free -h;不能',['https://man7.org/linux/man-pages/man1/free.1.html','https://man7.org/linux/man-pages/man5/proc_meminfo.5.html']),
('environment','open-files-limit','程序提示打开文件数受限，怎样查看当前shell的软限制？','运行 ulimit -Sn 查看当前shell的软文件描述符限制；这不是整台机器的全局上限。','ulimit -Sn;软',['https://man7.org/linux/man-pages/man1/bash.1.html']),
('environment','clock-query','怎样只读查看当前系统时间和时区信息？','可运行 timedatectl status 查看系统时钟和时区状态；命令用于读取状态，不代表时间同步服务已可靠工作。','timedatectl;时区',['https://man7.org/linux/man-pages/man1/timedatectl.1.html']),

('linux','umask-query','怎样查看当前shell的新文件创建掩码，而不修改它？','运行 umask 或 umask -S 只读查看当前创建掩码；它会与程序请求的权限共同影响新建文件或目录权限。','umask;只读',['https://www.gnu.org/software/bash/manual/html_node/Bourne-Shell-Builtins.html']),
('linux','grep-recursive','怎样递归查找日志目录中包含固定文本的行，并显示行号？','使用 grep -R -n -F -- "固定文本" /日志目录；-R递归查找，-n显示行号，-F按固定字符串匹配。','grep -R;-n;-F',['https://man7.org/linux/man-pages/man1/grep.1.html']),
('linux','safe-find-pipe','文件名可能含空格或换行时，如何安全地把 find 的结果交给后续命令？','使用 find ... -print0 配合 xargs -0，以NUL分隔文件名；普通换行分隔可能误拆特殊文件名。','-print0;-0',['https://www.man7.org/linux/man-pages/man1/find.1.html']),
('linux','tee-copy','怎样在查看命令输出的同时把它写入新日志文件？','使用 command 2>&1 | tee run.log；tee 会把标准输入复制到标准输出和文件，默认会覆盖目标文件，追加时用 -a。','tee;run.log',['https://man7.org/linux/man-pages/man1/tee.1p.html']),
('linux','symlink-create','怎样创建一个指向现有模型文件的符号链接？','使用 ln -s /models/model.rkllm /current/model.rkllm 创建符号链接；相对目标按链接所在目录解析，创建前确认目标名不存在。','ln -s;符号链接',['https://man7.org/linux/man-pages/man1/ln.1.html']),
('linux','graceful-signal','停止一个卡住的自有进程时，为什么通常先发TERM而不是KILL？','先用 kill -TERM PID 请求进程正常退出并清理资源；只有确认它仍未退出时才考虑 KILL，因为KILL不能被捕获或清理。','TERM;KILL',['https://man7.org/linux/man-pages/man1/kill.1.html']),
('linux','resolve-command','怎样确认shell会解析到哪个外部命令文件？','运行 command -v 程序名查看shell可找到的命令；结果还可能是别名或函数，需结合 type 程序名判断。','command -v;type',['https://www.gnu.org/software/bash/manual/html_node/Bash-Builtins.html']),
('linux','compare-files','怎样判断两个文件内容是否逐字节相同？','运行 cmp 文件A 文件B 逐字节比较；无差异时通常不输出内容且返回状态0，有差异时会报告位置或返回非零状态。','cmp;逐字节',['https://www.gnu.org/software/diffutils/manual/html_node/Invoking-cmp.html']),
('linux','printf-safe-value','shell变量中含反斜杠或格式字符时，如何按原样输出变量内容？','使用 printf "%s\\n" "$value"；把变量作为%s参数传入，避免把变量内容当成printf格式串。','printf;%s',['https://www.gnu.org/software/bash/manual/html_node/Bash-Builtins.html']),
('linux','background-jobs','怎样查看当前交互shell启动的后台作业及其进程ID？','运行 jobs -l 查看当前shell的作业和进程ID；它只列出该shell的作业，不是系统所有进程清单。','jobs -l;当前shell',['https://www.gnu.org/software/bash/manual/html_node/Job-Control-Builtins.html']),

('deployment','context-budget-reason','多轮聊天构造模型输入时，系统提示和旧对话是否占上下文额度？','是。系统提示、保留的历史、当前问题和预留输出都占上下文预算；按原生Tokenizer计数后再决定删除最早完整轮次。','系统提示;历史;Tokenizer',['LubanLM/docs/Linux助手微调与交付.md']),
('deployment','template-once','同一轮聊天已经用原生聊天模板渲染后，还应再手工包一遍消息特殊标记吗？','不应。每轮只应用一次模型的原生聊天模板；重复包特殊标记会改变实际输入。','一次;模板',['LubanLM/deploy/linux_assistant/frontend.py','LubanLM/deploy/linux_assistant/README.md']),
('deployment','stop-reason','生成恰好达到输出Token上限时，能否记录成模型自然回答完成？','不能。应区分模型EOS、运行时完成、错误和长度上限；达到长度上限应记录为截断或限长停止。','EOS;错误;长度',['LubanLM/deploy/linux_assistant/board_chat.cpp','LubanLM/deploy/linux_assistant/acceptance.py']),
('deployment','bundle-manifest','脱机交付聊天程序时，怎样让接收端核对文件是否完整？','随包提供逐文件SHA256清单；接收端校验模型、Tokenizer、配置和程序摘要，任一摘要不符就停止使用该包。','SHA256;清单',['LubanLM/scripts/linux_assistant/bundle.py','LubanLM/scripts/linux_assistant/transfer.py']),
('deployment','runtime-dependencies','模型文件复制完成就足以保证板端能够启动推理吗？','不够。还需匹配并记录板端架构、Runtime和驱动，检查程序依赖，并通过实际加载和生成验证。','Runtime;驱动;加载',['LubanLM/docs/Linux助手微调与交付.md']),
('deployment','adapter-base-pair','LoRA适配器与另一个版本的同名基础模型可以随意组合吗？','不可以。适配器依赖训练时的基础权重；固定模型revision和文件摘要，合并前核对适配器与基础模型配对。','适配器;revision',['https://huggingface.co/docs/peft/main/en/developer_guides/checkpoint']),
('deployment','calibration-split','量化校准输入可以从最终验收集抽取吗？','不可以。校准数据须与开发和最终验收数据隔离，不能用最终题选择或调整量化参数。','不可以;隔离',['LubanLM/docs/Linux助手微调与交付.md']),
('deployment','worker-ready','C++推理进程返回了启动提示，就能证明模型已经加载成功吗？','不能。应检查模型加载接口的返回状态及错误信息，再发送请求验证生成；启动提示本身不足以证明加载成功。','返回状态;加载',['LubanLM/deploy/linux_assistant/board_chat.cpp']),
('deployment','interrupt-history','用户中断生成后，部分回答应写进下一轮聊天历史吗？','不应。只把完整成功的助手回答加入历史；中断或错误回答要丢弃，避免污染下一轮上下文。','中断;历史',['LubanLM/deploy/linux_assistant/frontend.py','LubanLM/deploy/linux_assistant/acceptance.py']),
('deployment','offline-tokenizer','不安装PyTorch的板端聊天前端如何保持和PC一致的分词？','使用包内固定版本的Tokenizer文件与轻量tokenizers实现，按同一聊天模板生成Token ID，并用固定文本对照校验。','Tokenizer;Token ID',['LubanLM/deploy/linux_assistant/frontend.py','LubanLM/scripts/linux_assistant/parity.py']),

('troubleshooting','ldd-untrusted','能对来历不明的Linux可执行文件直接运行ldd吗？','不建议。某些实现可能执行目标文件；对不可信文件可用 objdump -p 查看直接NEEDED依赖，但它不显示完整依赖树。','不建议;objdump',['https://www.man7.org/linux/man-pages/man1/ldd.1.html']),
('troubleshooting','permission-parent','脚本本身有执行位仍报Permission denied，应检查哪些路径条件？','检查路径各级父目录是否有搜索权限，并检查文件系统是否以 noexec 挂载；不要直接对整棵目录递归开放权限。','父目录;noexec',['https://man7.org/linux/man-pages/man7/path_resolution.7.html','https://man7.org/linux/man-pages/man8/mount.8.html']),
('troubleshooting','exit-137','进程退出码137是否能单独证明发生了内存不足？','不能。137常见于进程被信号9终止，但仅凭退出码不能确定终止原因；还要核对系统日志、进程日志和内存记录。','不能;信号',['LubanLM/docs/Linux助手微调与交付.md']),
('troubleshooting','service-state','systemd服务启动失败，怎样先查状态再看本次启动日志？','先运行 systemctl is-failed 服务名 查看失败状态，再用 journalctl -u 服务名 -b 检查本次启动相关日志。','systemctl is-failed;journalctl;-b',['https://man7.org/linux/man-pages/man1/systemctl.1.html','https://man7.org/linux/man-pages/man1/journalctl.1%40%40systemd.html']),
('troubleshooting','no-space-diagnosis','写文件报No space left，先检查什么再决定是否清理？','先用 df -h 目标路径检查所在文件系统块空间，并用 df -i 检查inode余量；确认归属和备份后再决定清理对象。','df -h;df -i',['https://www.man7.org/linux/man-pages/man1/df.1.html']),
('troubleshooting','library-symbol-version','undefined symbol出现时，怎样避免只凭库文件名判定兼容？','检查实际加载路径、符号和ABI/版本；对可信程序用 ldd 核对依赖路径，文件同名不能证明ABI兼容。','实际加载;ABI',['https://www.man7.org/linux/man-pages/man1/ldd.1.html','LubanLM/docs/Linux助手微调与交付.md']),
('troubleshooting','npu-core-request','配置请求两个NPU核心但速度没变化，应先确认什么？','先记录实际加载日志、设备运行状态和测量条件；配置请求不等于运行时实际使用或性能提升。','实际;测量',['LubanLM/docs/Linux助手微调与交付.md']),
('troubleshooting','corrupt-transfer','传输中断后重传完成，怎样确认目标模型可以继续部署？','先对照完整文件SHA256清单；所有文件摘要匹配后再尝试加载，部分文件或仅有非零大小都不能证明传输完整。','SHA256;加载',['LubanLM/scripts/linux_assistant/transfer.py']),
('troubleshooting','first-token-warmup','只测一次首次请求的首字延迟，能代表预热后的稳定性能吗？','不能。分别记录冷启动、预热和多次稳定请求的延迟，并固定输入长度与输出长度再比较。','不能;预热',['LubanLM/docs/Linux助手微调与交付.md']),
('troubleshooting','worker-error-vs-end','推理回调没有返回文本，怎样区分正常结束和异常？','同时检查回调状态码、停止原因和错误通道；没有文本本身不能判作成功EOS，错误回答也不要写入历史。','状态码;错误;EOS',['LubanLM/deploy/linux_assistant/board_chat.cpp','LubanLM/deploy/linux_assistant/frontend.py']),

('interaction','minimum-diagnostics','用户说“板子启动不了”，应先要哪些信息以便有效排查？','请提供板卡型号、系统与架构、模型和Runtime版本、启动命令及完整错误输出；凭这句话还不能确定故障位置。','板卡;Runtime;错误',['LubanLM/docs/Linux助手微调与交付.md']),
('interaction','redact-secrets','排查依赖时用户准备贴完整环境配置，应提醒什么？','请先删除密码、访问令牌、私钥和订阅地址，再提供与故障相关的库路径、版本及错误日志。','删除;令牌',['LubanLM/docs/Linux助手微调与交付.md']),
('interaction','separate-evidence','日志只显示库加载失败，能直接断言驱动不兼容吗？','不能。先说明“库加载失败”是已观察事实；驱动不兼容只是候选原因，需要版本、依赖路径和完整日志验证。','事实;候选原因',['LubanLM/docs/Linux助手微调与交付.md']),
('interaction','no-claim-of-execution','没有终端访问能力时，用户问“你刚才检查通过了吗”，应怎样回答？','明确说明我没有访问设备或执行检查；提供可运行的命令，并请用户贴回真实输出再判断。','没有;执行',['LubanLM/docs/Linux助手微调与交付.md']),
('interaction','ask-only-missing','用户已给出板型和系统，但没给运行库版本；下一步应怎样问？','沿用已提供的板型和系统，只询问具体Runtime与驱动版本以及完整错误，不重复索取已有信息。','只询问;Runtime',['LubanLM/docs/Linux助手微调与交付.md']),
('interaction','safe-command-choice','用户要修复权限但没说明文件归属，能直接建议chmod -R 777吗？','不能。先确认文件归属、用途和需要的权限，再给最小范围的权限修改建议；不建议递归开放所有用户权限。','不能;最小',['https://man7.org/linux/man-pages/man1/chmod.1.html']),
('interaction','unknown-interface','文档没有某SDK函数的签名时，如何避免编造API？','说明目前未核实该接口，请用户提供固定版本头文件或官方示例；不猜函数名、参数或返回值。','未核实;头文件',['LubanLM/docs/Linux助手微调与交付.md']),
('interaction','strict-json','用户要求只输出JSON字段ok=false，能在JSON前加一句解释吗？','{"ok":false}','EXACT:{"ok":false}',['LubanLM/docs/Linux助手微调与交付.md']),
('interaction','uncertain-followup','缺少日志时，怎样给出有用的下一步而不把猜测当结论？','列出一到两个可区分原因的只读检查，并请用户提供原始输出；明确当前原因尚未确认。','尚未确认;输出',['LubanLM/docs/Linux助手微调与交付.md']),
('interaction','multiturn-context','用户上一轮已贴出Runtime版本和错误日志，本轮问“下一步呢”，应如何回答？','引用已有版本和日志，只提出下一项能区分原因的检查；不要重复索取已提供信息，也不要声称检查已执行。','已有;不要重复',['LubanLM/docs/Linux助手微调与交付.md']),
]

# New held-out scenarios are authored separately from training. They are added
# to the candidate development/test files but remain sealed until dataset lock.
HOLDOUT_ITEMS = [
('dev','environment','getconf-online','怎样查看当前系统可用的在线处理器数量？','运行 getconf _NPROCESSORS_ONLN 查询在线处理器数；它不保证等于当前进程实际可用的CPU配额。','getconf;_NPROCESSORS_ONLN',['https://man7.org/linux/man-pages/man1/getconf.1.html']),
('dev','environment','lsblk-filesystems','怎样查看块设备及其文件系统和挂载点？','运行 lsblk -f 查看块设备的文件系统标识和挂载点；设备名本身不足以判断实际挂载情况。','lsblk;-f',['https://man7.org/linux/man-pages/man8/lsblk.8.html']),
('dev','environment','memavailable','Linux上查看可供新程序使用的内存，应优先看哪个/proc/meminfo字段？','查看 /proc/meminfo 中的 MemAvailable；MemFree 不包含可回收缓存估算，不能直接等同于可用内存。','MemAvailable;MemFree',['https://man7.org/linux/man-pages/man5/proc_meminfo.5.html']),
('dev','environment','df-inodes','磁盘还有空闲字节但创建文件失败，怎样检查inode是否耗尽？','运行 df -i 目标路径检查该文件系统的inode使用情况；df -h显示的字节空间不能替代inode余量。','df -i;inode',['https://man7.org/linux/man-pages/man1/df.1.html']),
('dev','linux','stat-permissions','怎样只查看一个文件当前的传统八进制权限位？','运行 stat -c %a 文件路径查看权限位；结果不包含ACL等额外访问控制信息。','stat;-c;%a',['https://man7.org/linux/man-pages/man1/stat.1.html']),
('dev','linux','resolve-symlink','怎样把符号链接解析为最终规范路径？','运行 readlink -f 链接路径解析其规范化目标路径；目标不存在或路径不可解析时命令可能失败。','readlink;-f',['https://man7.org/linux/man-pages/man1/readlink.1.html']),
('dev','linux','directory-size','怎样粗略查看目录树占用的磁盘空间并使用易读单位？','运行 du -sh 目录路径查看目录树汇总占用；结果是文件系统分配统计的近似使用视图，不是模型文件逻辑大小的校验。','du;-sh',['https://man7.org/linux/man-pages/man1/du.1.html']),
('dev','linux','tail-lines','怎样读取日志文件最后50行并持续观察后续新增内容？','运行 tail -n 50 -f 日志路径先显示末尾50行，再跟踪新增内容；退出跟踪可按Ctrl-C。','tail;-n 50;-f',['https://man7.org/linux/man-pages/man1/tail.1.html']),
('dev','deployment','sha256sum-check','收到SHA256SUMS清单后，怎样核对其中列出的文件？','在清单和目标文件所在目录运行 sha256sum -c SHA256SUMS；任何摘要不匹配或文件缺失都应视为校验失败。','sha256sum;-c',['https://www.gnu.org/software/coreutils/manual/html_node/sha2-utilities.html']),
('dev','deployment','inspect-elf','怎样先确认一个Linux程序是何种ELF架构，再尝试部署？','运行 file 程序路径查看文件类型和架构信息；仍需与板端架构及运行库匹配，并实际启动验证。','file;架构',['https://man7.org/linux/man-pages/man1/file.1.html']),
('dev','deployment','bounded-smoke','怎样给一个短时部署冒烟测试设置最长运行时间？','使用 timeout 30s 命令及其参数限制测试最多运行30秒；超时返回状态与被测程序自身的失败状态应分别记录。','timeout;30s',['https://man7.org/linux/man-pages/man1/timeout.1.html']),
('dev','deployment','avoid-env-dump','排查动态库路径时，怎样避免把环境中的密钥一并打印出来？','只检查相关变量，例如 printf %s "$LD_LIBRARY_PATH"；不要为了查看单个设置而完整导出或粘贴所有环境变量。','LD_LIBRARY_PATH;不要',['https://man7.org/linux/man-pages/man1/printenv.1.html','https://www.gnu.org/software/bash/manual/html_node/Bash-Builtins.html']),
('dev','troubleshooting','kernel-errors','怎样查看本次启动以来内核日志中的错误级别消息？','运行 journalctl -k -b -p err 查看本次启动的内核日志错误级别记录；没有记录不等于故障不存在。','journalctl;-k;-b;-p err',['https://man7.org/linux/man-pages/man1/journalctl.1%40%40systemd.html']),
('dev','troubleshooting','trace-openat','程序报找不到模型文件，怎样观察它尝试打开哪些路径？','可在受控测试中使用 strace -f -e trace=openat 程序及参数观察文件打开调用；跟踪结果可能含敏感路径，应限制分享范围。','strace;-e;openat',['https://man7.org/linux/man-pages/man1/strace.1.html']),
('dev','troubleshooting','journal-time','怎样只查看最近10分钟的某个服务日志？','运行 journalctl -u 服务名 --since "10 minutes ago" 限定服务和时间范围；保留完整错误上下文以便进一步判断。','journalctl;-u;--since',['https://man7.org/linux/man-pages/man1/journalctl.1%40%40systemd.html']),
('dev','troubleshooting','shell-status','脚本执行失败后，怎样立即查看上一条命令的退出状态？','紧接失败命令运行 printf "%s\\n" "$?" 查看其退出状态；之后再执行其他命令会改变这个值。','printf;\\$\\?',['https://www.gnu.org/software/bash/manual/html_node/Bash-Variables.html']),
('dev','interaction','ask-for-error','用户只说“模型启动失败”，还没给日志。助手先应索取哪些信息？','先询问完整启动命令、模型文件路径与摘要、板端型号和系统版本，以及从启动到失败的完整错误输出；不要仅凭“启动失败”猜测根因。','启动命令;型号;错误',['LubanLM/docs/Linux助手微调与交付.md']),
('dev','interaction','safe-command-clarify','用户要求“给我修复命令”，但没有说明故障路径和报错。怎样回应？','先明确指出缺少具体路径和完整报错，并请用户提供命令、标准错误及相关版本；得到信息前不应编造会修改系统的命令。','缺少;报错;不应',['LubanLM/docs/Linux助手微调与交付.md']),
('dev','interaction','explain-risk','用户要求递归修改整个模型目录权限，怎样给出更稳妥的下一步？','先询问程序实际运行用户、目标文件和具体拒绝信息；优先检查文件与父目录权限，不直接建议对整棵目录递归开放权限。','运行用户;父目录;不直接',['https://man7.org/linux/man-pages/man7/path_resolution.7.html']),
('dev','interaction','separate-fact-guess','日志显示内存分配失败但没有系统OOM记录，回答应怎样区分结论和假设？','说明日志能确认分配失败，但现有信息不能证明系统OOM杀进程；把内存压力列为待验证假设，并索取进程峰值内存、系统日志和运行时错误。','不能证明;假设;索取',['LubanLM/docs/Linux助手微调与交付.md']),
('test','environment','getconf-bits','怎样查看用户空间ABI的数据位宽？','运行 getconf LONG_BIT 查看当前环境的数据位宽；它描述当前用户空间配置，不能代替file对某个二进制文件的检查。','getconf;LONG_BIT',['https://man7.org/linux/man-pages/man1/getconf.1.html']),
('test','environment','thermal-zone','在Linux热区接口中，thermal_zone*/temp常见的数值单位是什么？','读取 /sys/class/thermal/thermal_zone*/temp；常见内核接口使用毫摄氏度，但应结合具体驱动和内核文档确认。','thermal_zone;毫摄氏度',['https://www.kernel.org/doc/Documentation/ABI/testing/sysfs-class-thermal']),
('test','environment','process-status','怎样从/proc查看指定进程的常驻内存和历史峰值字段？','读取 /proc/PID/status 中的 VmRSS 和 VmHWM；它们不能覆盖所有驱动或设备侧内存。','VmRSS;VmHWM',['https://man7.org/linux/man-pages/man5/proc_pid_status.5.html']),
('test','environment','mount-target-options','怎样检查一个路径所在挂载点是否带有noexec选项？','运行 findmnt --target 路径 -o TARGET,OPTIONS 查看覆盖路径的挂载点和选项，再确认选项中是否包含 noexec。','findmnt;OPTIONS;noexec',['https://man7.org/linux/man-pages/man8/findmnt.8.html']),
('test','linux','chmod-symbolic','只给当前文件所有者增加执行权限，怎样避免改动其他权限位？','运行 chmod u+x 文件路径，仅为所有者增加执行位；修改前后可用 stat -c %a 核对。','chmod;u\\+x',['https://man7.org/linux/man-pages/man1/chmod.1.html']),
('test','linux','install-mode','复制一个配置文件并明确设为所有者可写、其他用户只读，怎样设置常见0644权限？','可运行 install -m 0644 源文件 目标文件复制并设置权限；目标属主仍需按部署身份另行核对。','install;-m;0644',['https://man7.org/linux/man-pages/man1/install.1.html']),
('test','linux','sort-unique','怎样按行排序文本并去掉重复行？','运行 sort -u 文件；若原始顺序有意义，应先保存原文件，因为排序会改变行顺序。','sort;-u',['https://man7.org/linux/man-pages/man1/sort.1.html']),
('test','linux','diff-context','怎样比较两个配置文件并显示统一格式的上下文差异？','运行 diff -u 文件A 文件B 查看统一格式差异；diff返回非零也可能只是表示文件内容不同。','diff;-u',['https://man7.org/linux/man-pages/man1/diff.1.html']),
('test','deployment','inspect-archive','怎样先列出tar归档中的文件名而不解压？','运行 tar -tf 归档路径查看归档成员；检查名称是否含绝对路径或向上跳转后，再决定是否解压。','tar;-tf',['https://www.gnu.org/software/tar/manual/html_node/list.html']),
('test','deployment','elf-needed','怎样查看ELF文件声明的直接动态库依赖？','运行 readelf -d 程序路径并查看 NEEDED 项；它只列直接依赖，不证明板端能找到或加载这些库。','readelf;-d;NEEDED',['https://man7.org/linux/man-pages/man1/readelf.1.html']),
('test','deployment','copy-preserve-mode','部署配置文件时，怎样复制并保留源文件的权限和时间戳？','使用 cp -p 源文件 目标路径保留权限、属主可保留项及时间戳；目标系统上的属主保留可能受权限限制。','cp;-p',['https://man7.org/linux/man-pages/man1/cp.1.html']),
('test','deployment','service-unit-verify','修改systemd单元文件后，怎样检查语法而不启动服务？','运行 systemd-analyze verify 单元文件路径检查单元定义；验证通过仍需在目标环境进行实际启动测试。','systemd-analyze;verify',['https://man7.org/linux/man-pages/man1/systemd-analyze.1.html']),
('test','troubleshooting','signal-termination','退出状态143通常对应什么情况，能否单独说明根因？','143常对应128加SIGTERM编号15，表示进程可能因TERM结束；单凭该状态不能确定是谁或为何发送信号。','143;SIGTERM;不能',['https://man7.org/linux/man-pages/man7/signal.7.html']),
('test','troubleshooting','dmesg-recent','怎样查看内核环形缓冲区最近的消息并带人类可读时间？','运行 dmesg -T 查看带可读时间格式的内核缓冲区消息；时间换算可能不精确，应结合journal和事件时间核对。','dmesg;-T',['https://man7.org/linux/man-pages/man1/dmesg.1.html']),
('test','troubleshooting','systemctl-cat','怎样查看systemd实际加载的服务单元内容及其覆盖配置？','运行 systemctl cat 服务名 查看主单元及相关drop-in内容；同时记录服务状态与日志，不能只看配置推定运行结果。','systemctl;cat;drop-in',['https://man7.org/linux/man-pages/man1/systemctl.1.html']),
('test','troubleshooting','check-process-identity','服务能读模型而当前用户不能读，怎样确认服务实际使用的身份？','检查 systemctl show 服务名 -p User -p Group，并结合进程凭据和文件父目录权限核实；空User字段不必然代表普通用户。','systemctl show;User;Group',['https://man7.org/linux/man-pages/man1/systemctl.1.html']),
('test','interaction','request-repro','用户只提供“昨天还能用”，排查部署回归时还应追问什么？','请用户提供最近一次成功与首次失败的时间、期间改动、完整复现命令和错误日志，以及模型、运行库和驱动版本。','成功;失败;版本',['LubanLM/docs/Linux助手微调与交付.md']),
('test','interaction','distinguish-host-board','用户把PC上的CUDA显存错误当成板端NPU内存不足，怎样澄清？','先确认报错来自PC还是板端，并分别询问设备型号、进程和内存指标；CUDA显存与板端NPU/系统内存不能混作同一指标。','PC;板端;不能',['LubanLM/docs/Linux助手微调与交付.md']),
('test','interaction','ask-minimal-log','用户说“Permission denied”，但没给操作对象。应先索取哪些最小信息？','询问完整命令、目标文件或目录、运行用户、文件和父目录权限，以及挂载选项；不要直接建议递归chmod。','完整命令;运行用户;父目录',['https://man7.org/linux/man-pages/man7/path_resolution.7.html']),
('test','interaction','format-steps','用户要求按“检查、判断、下一步”三项回答，助手怎样组织建议？','严格按“检查、判断、下一步”三项分别回答；每项只写对应内容，信息不足时在下一步中提出明确追问。','检查;判断;下一步',['LubanLM/docs/Linux助手微调与交付.md']),
]

# Thirty naturally contextual follow-ups (30% of all training scenarios).
# These vary the dialogue state and evidence instead of repeating a generic
# greeting before the same standalone question.
MULTITURN = {
('environment','findmnt-target'):[('user','我把模型放在 /mnt/models，但机器上有多个挂载盘。'),('assistant','明白。你想确认这个目录实际使用哪个文件系统。')],
('environment','swap-areas'):[('user','我用 free -h 看到有Swap，想确认具体启用了哪些交换区。'),('assistant','可以查询当前活动的交换设备或文件。')],
('environment','nproc-scope'):[('user','这台机器显示16个处理器，但容器内的任务可能有限制。'),('assistant','容器中的进程可用处理单元数可能受配额影响。')],
('environment','open-files-limit'):[('user','服务日志出现 Too many open files。'),('assistant','先确认运行该服务的用户和进程环境；shell中的限制可能不同。')],
('environment','filesystem-type'):[('user','板端目录 /data/models 看起来容量和预期不一样。'),('assistant','容量信息需要针对这个路径所在的文件系统查询。')],
('environment','effective-groups'):[('user','当前程序能读某目录，但我不知道它属于哪些组。'),('assistant','应查看当前进程的组身份，而不是只看登录用户名。')],
('environment','effective-uid'):[('user','我在shell里看到一个用户名，想确认进程权限检查用的用户ID。'),('assistant','查询当前进程的有效用户ID可以与用户名输出区分开。')],
('environment','memory-and-swap-output'):[('user','设备有4GB RAM和8GB Swap，模型却仍然加载失败。'),('assistant','两者用途和性能不同，先分别确认内存统计。')],
('environment','path-space'):[('user','我准备向 /data/models 传一个大文件。'),('assistant','需要核对这个目标路径所在文件系统的可用空间。')],
('environment','clock-query'):[('user','日志时间和电脑时间差了几个小时，我想先只读查看设备时钟状态。'),('assistant','可以查询系统时间、时区和同步状态，不要先修改时钟。')],
('linux','umask-query'):[('user','新建文件权限和我预想的不一样，我还不想改系统设置。'),('assistant','可以先读取当前shell的创建掩码，再结合程序请求的权限判断。')],
('linux','safe-find-pipe'):[('user','目录中有带空格和换行的文件名，我想把查找结果传给另一个命令。'),('assistant','这种场景不能依赖普通换行来分隔路径。')],
('linux','grep-recursive'):[('user','日志目录里有多个文件，我想定位包含某条固定错误文本的行。'),('assistant','可以递归搜索，并同时显示匹配行号。')],
('linux','tee-copy'):[('user','我在终端运行安装检查，希望边看输出边保存日志。'),('assistant','可以把标准输入同时复制到屏幕和文件。')],
('linux','symlink-create'):[('user','多个测试程序都要访问同一份模型，我想用一个链接路径。'),('assistant','符号链接可以指向目标文件，但创建前要确认链接名。')],
('linux','graceful-signal'):[('user','我需要停掉自己的推理进程，但希望它先释放资源。'),('assistant','应先请求正常退出，并确认进程状态。')],
('linux','resolve-command'):[('user','安装了两个版本的工具，我不确定shell会找到哪一个。'),('assistant','先查询当前shell能解析到的命令位置或类型。')],
('linux','printf-safe-value'):[('user','变量内容中有反斜杠，echo输出看起来被解释了。'),('assistant','应让变量作为普通字符串参数，而不是格式模板。')],
('linux','background-jobs'):[('user','我刚在这个终端启动了几个后台任务。'),('assistant','当前shell的作业列表可以显示相关作业及进程号。')],
('deployment','context-budget-reason'):[('user','对话已经有几轮，最近开始提示上下文超限。'),('assistant','检查实际Token总量，并保留系统约束和当前问题。')],
('deployment','stop-reason'):[('user','日志显示每次回答都刚好在256个Token附近停止。'),('assistant','这可能是输出限长；还需区分自然结束与运行时状态。')],
('deployment','interrupt-history'):[('user','用户按了中断，屏幕上已经显示一部分回答。'),('assistant','部分文本不一定是完整且可靠的助手消息。')],
('deployment','template-once'):[('user','我已经把多轮消息渲染成模板文本。'),('assistant','接下来应避免重复添加同一套聊天标记。')],
('deployment','bundle-manifest'):[('user','我准备把模型包复制到没有网络的设备。'),('assistant','接收端需要能核验包内文件是否与源端一致。')],
('deployment','runtime-dependencies'):[('user','模型权重已经复制到板端，启动程序仍报错。'),('assistant','权重文件只是依赖之一，还要检查运行环境和实际加载。')],
('deployment','adapter-base-pair'):[('user','我找到一个同系列模型的LoRA适配器，想直接试着加载。'),('assistant','加载前要先核对它训练时对应的基础模型revision。')],
('deployment','calibration-split'):[('user','量化时需要校准文本，手边有一份最终验收题。'),('assistant','最终验收数据必须与模型转换过程隔离。')],
('deployment','worker-ready'):[('user','C++服务打印了“worker started”，但我还没发过推理请求。'),('assistant','启动日志需要和模型加载状态区分。')],
('deployment','offline-tokenizer'):[('user','板端不能装PyTorch，但需要与PC逐字一致地分词。'),('assistant','可以检查包内轻量Tokenizer和模板渲染是否一致。')],
('troubleshooting','service-state'):[('user','服务重启后又退出了，我可以查看systemd记录。'),('assistant','先读服务当前状态，再查看本次启动的日志。')],
('troubleshooting','library-symbol-version'):[('user','同一个.so文件名在两个目录下都有，程序报undefined symbol。'),('assistant','要先找出实际加载路径，再核对ABI和符号版本。')],
('troubleshooting','permission-parent'):[('user','我已给脚本加了执行位，但运行仍报Permission denied。'),('assistant','执行权限只是一个条件，路径和挂载选项也会影响执行。')],
('troubleshooting','ldd-untrusted'):[('user','我下载了一个陌生二进制文件，想看它缺什么动态库。'),('assistant','查看依赖的方法需要考虑运行不可信文件的风险。')],
('troubleshooting','exit-137'):[('user','程序退出码是137，但没有其他日志。'),('assistant','这个数字不足以单独判断进程为何终止。')],
('troubleshooting','no-space-diagnosis'):[('user','写模型文件时出现 No space left on device。'),('assistant','除了普通磁盘块，还要检查文件系统inode。')],
('troubleshooting','corrupt-transfer'):[('user','断点续传已经结束，目标端文件也不是空文件。'),('assistant','文件大小或非空状态不能替代内容摘要验证。')],
('troubleshooting','first-token-warmup'):[('user','第一次生成比后续请求慢很多。'),('assistant','首次加载和预热会影响性能测量。')],
('troubleshooting','worker-error-vs-end'):[('user','推理界面没有显示文字，但进程还在运行。'),('assistant','需要区分空输出、正常结束和回调错误。')],
('interaction','minimum-diagnostics'):[('user','我只知道“板子启动不了”，还没整理日志。'),('assistant','先收集设备、软件版本、启动命令和原始错误。')],
('interaction','redact-secrets'):[('user','我准备把配置文件和日志发出来让你看。'),('assistant','可以先检查与故障相关的字段，并删除敏感凭据。')],
('interaction','separate-evidence'):[('user','我看到一次库加载失败，怀疑是驱动。'),('assistant','目前确认的是加载失败；驱动只是一个待验证的可能原因。')],
('interaction','no-claim-of-execution'):[('user','请告诉我你刚刚是否登录板端检查过。'),('assistant','只有真实执行并取得输出后才能声称检查过。')],
('interaction','multiturn-context'):[('user','Runtime 是 1.3.1，日志里的错误是 symbol lookup error。'),('assistant','这两项信息已记录；下一步应核对实际加载库及ABI。')],
('interaction','ask-only-missing'):[('user','我的设备是 RK3576，系统是 Ubuntu 22.04。'),('assistant','我已记录板型和系统；排查时只需补充还缺的版本与错误。')],
}
MULTITURN_SELECTED = {
('environment','findmnt-target'),('environment','open-files-limit'),('environment','filesystem-type'),
('environment','swap-areas'),('environment','effective-groups'),('environment','path-space'),
('linux','umask-query'),('linux','safe-find-pipe'),('linux','graceful-signal'),
('linux','grep-recursive'),('linux','tee-copy'),('linux','resolve-command'),
('deployment','context-budget-reason'),('deployment','stop-reason'),('deployment','interrupt-history'),
('deployment','bundle-manifest'),('deployment','adapter-base-pair'),('deployment','calibration-split'),
('troubleshooting','service-state'),('troubleshooting','permission-parent'),('troubleshooting','ldd-untrusted'),
('troubleshooting','exit-137'),('troubleshooting','corrupt-transfer'),('troubleshooting','worker-error-vs-end'),
('interaction','minimum-diagnostics'),('interaction','redact-secrets'),('interaction','multiturn-context'),
('interaction','separate-evidence'),('interaction','no-claim-of-execution'),('interaction','ask-only-missing'),
}

def digest(path: Path) -> str:
    return hashlib.sha256(path.read_bytes()).hexdigest()

def main():
    raise RuntimeError('This legacy candidate builder is retired because it reads the removed v1 corpus.')
    OUT.mkdir(parents=True, exist_ok=True)
    original=json.loads((BASE/'manifest.json').read_text(encoding='utf8'))
    old=jsonl(BASE/'train.jsonl')
    # Keep one simple canonical record per independent scenario; discard only
    # the mechanical prompt variants in this candidate, not in frozen v1.
    canonical={}
    for row in old:
        if row['group'] not in canonical or row.get('variant')==3:
            canonical[row['group']]=row
    base_train=[]
    for group,row in canonical.items():
        row=dict(row);row['id']='v2-base-'+row['id'].rsplit('-',1)[0]
        row['variant']=0
        row['source_review']='inherited-v1-author-review'
        base_train.append(row)
    new=[];source_reviews=[]
    for category,slug,question,answer,rubric,sources in ITEMS:
        group=f'{category}/v2-{slug}';row_id=f'v2-{category}-{slug}'
        context=[{'role':role,'content':content} for role,content in
                 MULTITURN.get((category,slug),[]) if (category,slug) in MULTITURN_SELECTED]
        row={'id':row_id,'group':group,'kind':'task','category':category,
             'messages':context+[{'role':'user','content':question}],
             'reference':answer,'rubric':rubric,'facts':[answer],
             'sources':sources,'variant':0,'source_review':'candidate-author-review'}
        new.append(row)
        source_reviews.append({'group':group,'split':'train','question':question,
            'reference':answer,'source_urls':sources,'review_status':'candidate_author_review',
            'reviewer':'implementation_agent','evidence':{'reference_sha256':hashlib.sha256(answer.encode()).hexdigest(),
            'note':'核对命令用途、限制条件和回答是否支持评分规则；仍需独立复核。',
            'review_type':'author review only; independent review required'}})
    train=base_train+new
    dev=jsonl(BASE/'dev.jsonl');test=jsonl(BASE/'test.jsonl')
    heldout_reviews=[]
    for split,category,slug,question,answer,rubric,sources in HOLDOUT_ITEMS:
        group=f'{category}/v2-{split}-{slug}'
        row_id=f'v2-{split}-{category}-{slug}'
        row={'id':row_id,'group':group,'kind':'task','category':category,
             'messages':[{'role':'user','content':question}], 'reference':answer,
             'rubric':rubric,'facts':[answer],'sources':sources,'variant':0,
             'source_review':'candidate-author-review'}
        if not screen(row,answer)['screen_pass']:
            raise ValueError(f'Held-out reference fails its necessary-condition screen: {row_id}')
        (dev if split=='dev' else test).append(row)
        heldout_reviews.append({'group':group,'split':split,'question':question,
            'reference':answer,'source_urls':sources,'review_status':'candidate_author_review',
            'reviewer':'implementation_agent','evidence':{'reference_sha256':hashlib.sha256(answer.encode()).hexdigest(),
            'note':'核对命令用途、限制条件和回答是否支持评分规则；仍需独立复核。',
            'review_type':'author review only; independent review required'}})
    if len({r['group'] for r in train})!=len(train):raise ValueError('Training groups must be independent')
    if {r['group'] for r in train}&({r['group'] for r in dev}|{r['group'] for r in test}):
        raise ValueError('A scenario group crosses a data split')
    if any(not r['sources'] or not screen(r,r['reference'])['screen_pass'] for r in new):
        raise ValueError('A new reference is missing a source or required rubric condition')
    if len(MULTITURN_SELECTED)!=30 or any(k not in MULTITURN for k in MULTITURN_SELECTED):
        raise ValueError('Expected 30 explicitly reviewed multi-turn contexts')
    if sum(len(r['messages'])>2 for r in new)!=30:raise ValueError('Expected 30 varied multi-turn training contexts')
    for name,rows in [('train.jsonl',train),('dev.jsonl',dev),('test.jsonl',test),
                      ('general-dev.jsonl',jsonl(BASE/'general-dev.jsonl')),
                      ('general-test.jsonl',jsonl(BASE/'general-test.jsonl'))]:
        write_jsonl(OUT/name,rows)
    # Keep every original split's source checks plus every new candidate review.
    old_reviews=jsonl(BASE/'source-review.jsonl')
    inherited=old_reviews
    write_jsonl(OUT/'source-review.jsonl',inherited+source_reviews+heldout_reviews)
    categories={}
    for split,rows in [('train',train),('dev',dev),('test',test)]:
        categories[split]={cat:sum(r['category']==cat for r in rows) for cat in sorted({r['category'] for r in rows})}
    manifest={'version':'linux-assistant-v2-candidate','status':'candidate_not_frozen',
      'derived_from':'linux-assistant-v1-reviewed; original remains unchanged',
      'files':{p.name:digest(p) for p in OUT.glob('*.jsonl')},
      'row_counts':{k:sum(1 for _ in (OUT/k).open(encoding='utf8')) for k in ('train.jsonl','dev.jsonl','test.jsonl','general-dev.jsonl','general-test.jsonl')},
      'independent_scenarios':{'train':len({r['group'] for r in train}),'dev':len({r['group'] for r in dev}),'test':len({r['group'] for r in test})},
      'category_counts':categories,'new_independent_train_scenarios':len(new),
      'training_policy':'One canonical record per independent scenario; no synthetic paraphrase rows; 30 of 50 new records use non-template multi-turn context, giving 30% multi-turn records overall.',
      'final_test_policy':'Never evaluate before a candidate model and rules are locked.',
      'review_status':'New source checks are author review only; independent review required before training.',
      'limitations':['Development and final test each combine v1 with 20 new candidate scenarios; independent review remains required.',
        'General development and final test files are carried forward unchanged.',
        'This candidate is not wired into the training configuration and is not frozen.']}
    (OUT/'manifest.json').write_text(json.dumps(manifest,ensure_ascii=False,indent=2)+'\n',encoding='utf8')
    (OUT/'source-review-lock.json').write_text(json.dumps({'sha256':digest(OUT/'source-review.jsonl'),'review_status':'candidate only; independent review required'},ensure_ascii=False,indent=2)+'\n',encoding='utf8')
    print(json.dumps({'train_records':len(train),'new_independent_train_scenarios':len(new),
        'independent_scenarios':manifest['independent_scenarios'],'manifest':str(OUT/'manifest.json')},ensure_ascii=False))

if __name__=='__main__':main()
