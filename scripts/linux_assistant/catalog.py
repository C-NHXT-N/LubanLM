"""Authored factual scenario seeds. Variants are NOT independent factual cases.

Each line: split/category/topic/question/reference answer/required concept regex groups/source.
References paraphrase documentation, never observed outputs. Five variants of each held-out
scenario remain together; ten training variants of each seed remain together.
"""
SOURCES = {
    "coreutils": "https://www.gnu.org/software/coreutils/manual/html_node/",
    "proc": "https://man7.org/linux/man-pages/man5/proc.5.html",
    "ldd": "https://man7.org/linux/man-pages/man1/ldd.1.html",
    "ldso": "https://man7.org/linux/man-pages/man8/ld.so.8.html",
    "file": "https://man7.org/linux/man-pages/man1/file.1.html",
    "systemd": "https://www.freedesktop.org/software/systemd/man/latest/journalctl.html",
    "bash": "https://www.gnu.org/software/bash/manual/bash.html",
    "ssh": "https://man.openbsd.org/ssh",
    "sdk": "https://github.com/airockchip/rknn-llm/blob/f7390530443bf84f0394255a449d7cbe81e69d1c/rkllm-runtime/Linux/librkllm_api/include/rkllm.h",
    "local": "LubanLM/PROJECT_STATUS.md",
    "sha": "https://www.gnu.org/software/coreutils/manual/html_node/sha2-utilities.html",
    "find": "https://www.gnu.org/software/findutils/manual/html_mono/find.html",
    "python": "https://docs.python.org/3.11/library/venv.html",
    "peft": "https://huggingface.co/docs/peft/developer_guides/checkpoint",
}

ROWS = r"""
train|environment|kernel|板端怎样查看正在运行的内核版本？|运行 uname -r 查看运行中的内核版本；它不是发行版版本。|uname\s+-r;内核|coreutils
train|environment|arch|怎样确认板端程序应使用哪种CPU架构？|先运行 uname -m 查看架构，再用 file 检查程序；aarch64 与 x86-64 的程序不能直接互换。|uname\s+-m;file|coreutils,file
train|environment|distro|怎样确认Ubuntu发行版信息？|查看 /etc/os-release 中的 ID、VERSION_ID 等字段；不要用内核版本代替发行版版本。|/etc/os-release;VERSION_ID|local
train|environment|ram|能把64GB存储当作模型运行内存吗？|不能。存储保存模型文件，内存用于进程运行；分别检查 free -h 与 df -h。|不能;free\s+-h;df\s+-h|local
train|environment|memavailable|判断Linux还有多少内存可用，应关注什么？|查看 /proc/meminfo 的 MemAvailable；它估算不交换时可供新应用使用的内存，不只看 MemFree。|MemAvailable;MemFree|proc
train|environment|gpu|Windows里CUDA可用，能证明板端NPU可用吗？|不能。CUDA验证的是PC上的NVIDIA GPU；RK3576的NPU需要分别检查驱动、RKLLM Runtime和实际加载。|不能;CUDA;NPU|local
train|environment|python|怎样确认当前运行的是项目Python环境？|在目标终端运行 python -c "import sys; print(sys.executable)"，核对解释器路径；环境名提示不足以证明实际解释器。|sys.executable;路径|python
train|environment|driver_runtime|NPU驱动版本和RKLLM Runtime版本是同一项吗？|不是。驱动属于内核侧，Runtime是用户态运行库；分别记录并按SDK兼容要求核对，不能只记一个版本。|驱动;Runtime;分别|local,sdk
train|environment|storage|模型传输前怎样检查目标目录的存储余量？|运行 df -h /目标目录 查看对应文件系统的可用空间，并考虑临时文件；不能用另一挂载点的空闲空间代替。|df\s+-h;文件系统|coreutils
train|environment|envvar|怎样检查当前动态库搜索环境变量？|运行 printf '%s\n' "$LD_LIBRARY_PATH"；它只反映当前进程环境，不代表所有终端或服务配置。|LD_LIBRARY_PATH;当前|bash,ldso
dev|environment|cpuinfo|怎样查看Linux内核提供的CPU信息？|读取 /proc/cpuinfo；字段随架构和内核变化，不应从单个字段推断NPU功能。|/proc/cpuinfo;架构|proc
dev|environment|swap|交换空间能等同于物理内存吗？|不能。交换空间使用存储承接部分内存页，性能和用途不同；记录Swap和物理内存，不能把两者简单相加保证模型可运行。|不能;交换|proc
dev|environment|uptime|如何查看板端运行时长和系统负载？|查看 /proc/uptime 和 /proc/loadavg；负载不是NPU利用率。|/proc/uptime;/proc/loadavg|proc
dev|environment|boardidentity|已在LubanCat-3上跑通，能写成鲁班猫4已经验收吗？|不能。应记录实际板型LubanCat-3与RK3576，鲁班猫4需要另行实测；芯片相同不等于整机条件相同。|不能;LubanCat-3;鲁班猫4|local
test|environment|processrss|怎样查看某个Linux进程的常驻内存与历史峰值？|读取 /proc/进程号/status，关注 VmRSS 和 VmHWM；这不是所有NPU驱动内存的完整统计。|VmRSS;VmHWM|proc
test|environment|threadcount|怎样确认Linux进程当前线程数？|读取 /proc/进程号/status 中的 Threads；它反映该进程线程数，不能代表CPU核心数。|/proc/;Threads|proc
test|environment|mountinfo|同名目录在不同机器上容量不同，先检查什么？|检查目录对应的挂载点和文件系统，使用 df -h /该目录；不能只根据路径名称判断容量。|df\s+-h;挂载|coreutils
test|environment|hostname|怎样查看系统网络节点名称？|运行 uname -n；主机名用于标识节点，不是CPU架构或芯片型号。|uname\s+-n;主机|coreutils
train|linux|pwd|相对路径找不到文件，首先怎样确认当前位置？|运行 pwd 查看当前工作目录，再核对相对路径；同一个相对路径在不同目录可能指向不同位置。|pwd;相对路径|coreutils
train|linux|ls|怎样查看目录内包括隐藏文件在内的详细列表？|运行 ls -la /目标目录；以点开头的隐藏文件通常不会由普通 ls 列出。|ls\s+-la;隐藏|coreutils
train|linux|mkdir|怎样创建可能尚不存在的多级实验目录？|使用 mkdir -p /目标/实验目录；它会按需创建父目录，已有目录不作为普通的已存在错误。|mkdir\s+-p;目录|coreutils
train|linux|permissions|运行自有脚本提示无执行权限，应怎样检查和处理？|先用 ls -l 查看权限与文件归属；确认文件可信且归自己管理后用 chmod u+x 添加所有者执行权限，不要直接给所有人所有权限。|ls\s+-l;chmod\s+u\+x|coreutils
train|linux|tail|怎样持续观察应用追加的日志？|使用 tail -f 日志路径 查看新增内容；按 Ctrl+C 结束观察，不会终止被观察的应用。|tail\s+-f;Ctrl\+C|coreutils
train|linux|head|怎样只查看文本文件前20行？|运行 head -n 20 文件路径；这只是读取前20行，不会修改文件。|head\s+-n\s+20;不.*修改|coreutils
train|linux|du|怎样查看一个模型目录自身占用的磁盘空间？|运行 du -sh /模型目录；du统计该目录占用，df查看整个文件系统，两者用途不同。|du\s+-sh;df|coreutils
train|linux|readlink|怎样确认符号链接最终指向哪里？|运行 readlink -f /链接路径 解析路径；检查目标存在及权限，不要只凭链接文件名判断实际运行库。|readlink\s+-f;目标|coreutils
train|linux|copy|怎样保留元信息复制自己的实验目录？|使用 cp -a 源目录 目标目录，并先检查目标是否存在，避免误解复制后的目录层级。|cp\s+-a;目标|coreutils
train|linux|find|怎样在当前实验目录递归查找.rkllm文件？|使用 find . -type f -name '*.rkllm'；给通配符加引号，使它由find匹配而不是先由shell展开。|find\s+\.;-type\s+f;-name|find
dev|linux|journal|怎样查看某systemd服务本次开机的日志？|使用 journalctl -u 服务名 -b；核对服务名，权限不足时按系统授权规则读取日志。|journalctl;-u;-b|systemd
dev|linux|whichpython|如何在Python环境里调用对应的pip？|使用 python -m pip，确保由当前解释器调用pip；需要时先打印 sys.executable 核对路径。|python\s+-m\s+pip;sys.executable|python
dev|linux|quotepath|目录名包含空格时怎样避免命令参数被拆开？|给完整路径加引号，例如 cd "我的 实验"；空格在未引用时会用于分隔shell参数。|引号;空格|bash
dev|linux|exitcode|怎样立即查看上一条shell命令的退出状态？|在紧接着的命令中读取 $?；后续命令会覆盖这个状态，0通常表示成功。|\$\?;0|bash
test|linux|stdinredirect|命令中的大于号重定向对已有文件有什么影响？|普通 > 会创建或截断目标文件；只想追加时用 >>，操作前确认目标路径。|截断|覆盖;>>|bash
test|linux|diskinode|磁盘还有容量却提示无法创建文件，除了容量还查什么？|用 df -i 检查inode是否耗尽，同时检查目录权限与挂载状态；空闲字节不代表一定有可用inode。|df\s+-i;inode|coreutils
test|linux|sortnumeric|怎样把文本中的数字按数值排序而非字典顺序？|使用 sort -n 文件路径进行数值排序；默认文本顺序可能把10排在2前面。|sort\s+-n;数值|coreutils
test|linux|sshport|SSH使用非默认端口时命令怎么写？|使用 ssh -p 端口 用户@主机；端口写在-p之后，并核对目标主机身份。|ssh\s+-p;端口|ssh
train|deployment|hash|模型复制后文件名相同，怎样确认内容一致？|在源端记录SHA256，在目标端用 sha256sum -c 校验清单；文件名和大小相同都不足以证明内容相同。|SHA256;sha256sum\s+-c|sha
train|deployment|template|Qwen能直接使用50M的Tokenizer吗？|不能。Qwen必须保留原生Tokenizer、特殊ID和聊天模板；50M的32K Token ID与其词嵌入不对应。|不能;Tokenizer;模板|local
train|deployment|adapter|LoRA适配器目录可以直接当完整模型部署吗？|不能。适配器保存增量，需配套相同版本的原模型；本流程先合并为完整模型，再重新加载验证和转换。|不能;原模型;合并|peft
train|deployment|bf16w4|板端使用W4A16能说明PC训练也是4位吗？|不能。板端量化格式与PC训练精度是不同配置；本轮训练使用BF16 LoRA，部署另做W4A16量化。|不能;BF16;W4A16|local
train|deployment|revision|为什么LoRA合并要固定原模型revision？|适配器是在特定原权重上训练的；固定revision及文件哈希才能确保合并对象一致和结果可复现。|revision;哈希|peft,local
train|deployment|calibration|量化校准可以使用最终验收题吗？|不可以。校准属于模型处理过程，应使用训练侧独立样本；最终验收题需隔离，避免评价泄漏。|不可以|不能;校准;训练|local
train|deployment|ctxbudget|2048上下文、最多生成256Token时输入最多多少Token？|最多1792 Token，因为输入与预留输出总计不能超过2048；多轮历史和系统提示也属于输入。|1792;历史;系统|sdk
train|deployment|libs|板端运行库应与哪一部分核对版本？|核对RKLLM Toolkit、Runtime以及NPU驱动的兼容关系，记录实际版本；本项目已验证的Toolkit和Runtime均为1.3.1。|Toolkit;Runtime;1\.3\.1|local
train|deployment|staging|新模型应直接覆盖已跑通的模型吗？|不应。给新实验独立目录，核验摘要并完成加载与质量检查后再选择使用，保留已验证产物以便回退。|独立目录;保留|local
train|deployment|nativeids|把已经渲染好的聊天文本再加特殊Token会怎样？|可能重复BOS或其他特殊Token，造成输入不一致；模板只应用一次，再编码时关闭重复添加特殊Token。|重复;一次|local
dev|deployment|eos|停止生成只需要检查是否达到最大长度吗？|不够。还要处理模型EOS、运行时完成和错误状态；达到长度上限应与自然结束区分记录。|EOS;错误;长度|sdk
dev|deployment|vision|现有Qwen3.5板端测试能证明看图功能可用吗？|不能。本轮仅部署语言分支，没有部署视觉编码器，图像功能需要另行转换和验证。|不能;语言;视觉|local
dev|deployment|twocore|RKLLM配置两个NPU核心就能证明程序实际加速吗？|不能。配置只是请求；需要记录实际加载、运行结果和测量条件，通过可比较实测判断性能。|不能;实测|sdk,local
dev|deployment|contextmodel|原模型支持长上下文，板端就自动支持相同长度吗？|不会。板端受导出配置、Runtime和内存限制；本轮导出上下文为2048，不能套用上游最大长度。|2048;内存|sdk,local
test|deployment|quantcache|微调后量化可以直接复用原模型的GRQ缓存吗？|不能直接复用。权重已经变化，校准与量化参数需对应当前模型；重新生成并记录模型和校准数据哈希。|不能;权重;校准|local
test|deployment|reconvert|原模型能在板端运行，合并模型还需要重新验证吗？|需要。合并改变权重，必须重新加载、量化和板端检查，原模型成功不能代替新产物验收。|需要;量化;板端|local
test|deployment|manifest|模型交付清单除了权重还应包括什么？|包括配置、Tokenizer、聊天模板、适配器或合并来源、依赖版本、文件SHA256及复现说明。|Tokenizer;模板;SHA256|local,peft
test|deployment|loadreturn|C++加载RKLLM时怎样确认初始化成功？|检查 rkllm_init 的返回值及有效句柄，失败时记录错误并停止推理；文件存在不代表初始化成功。|rkllm_init;返回|sdk
train|troubleshooting|wrongarch|Linux运行文件报Exec format error应先查什么？|先用 file 检查文件格式并与 uname -m 对照；架构不匹配或文件格式有误都是候选原因，不能仅凭这一行断言权限问题。|file;uname\s+-m|file,coreutils
train|troubleshooting|libmissing|程序提示librkllmrt.so找不到，怎样排查？|先确认库文件存在，用 ldd 检查可信程序的依赖，再核对架构与 LD_LIBRARY_PATH；不要先替换系统库。|ldd;LD_LIBRARY_PATH;架构|ldd,ldso
train|troubleshooting|oom|模型加载时内存不足，应先记录哪些信息？|记录模型文件、上下文配置、进程与系统可用内存，并检查是否有其他大进程；减少上下文等调整需重新实测，不能保证一定解决。|上下文;内存;进程|local
train|troubleshooting|hashmismatch|板端SHA256与源端不一致，还能继续运行吗？|不应继续。保留异常文件和校验记录，检查传输是否完整以及是否使用了正确源文件，再重新传输并核验。|不应|不能;传输;核验|sha
train|troubleshooting|permissionpath|有执行权限仍报Permission denied，接着查什么？|检查父目录的搜索权限、文件归属和挂载选项；权限问题不一定只在文件自身，不要直接递归放开全部权限。|父目录;挂载|coreutils
train|troubleshooting|emptyanswer|模型输出空字符串时应怎样定位？|保存输入Token、模板、停止原因和回调状态；检查是否立即生成EOS或发生错误，不能把空输出直接记为正常回答。|Token;EOS;回调|sdk,local
train|troubleshooting|gibberish|模型输出乱码或无关内容，首先核对哪些输入条件？|核对Tokenizer、模型revision、聊天模板和特殊ID是否匹配，再检查编码和生成设置，不先假定模型权重已损坏。|Tokenizer;模板;ID|local
train|troubleshooting|slowfirst|第一题特别慢，能直接当作稳定首Token延迟吗？|不能。首次加载、初始化和预热可能影响结果；分别记录冷启动和预热后指标，使用多次测量。|不能;预热;冷启动|local
train|troubleshooting|repeat|回答出现重复，应怎样排查而不直接归因于模型大小？|固定输入并检查模板、历史是否重复、停止条件和解码设置；保留原始输出，再比较模型及量化差异。|模板;历史;停止|local
train|troubleshooting|partial|传输中断只留下部分模型文件，该怎样处理？|保留部分文件用于受校验的续传；恢复后核对完整文件SHA256，不能把非空文件直接当作完整模型。|续传;SHA256;完整|sha,local
dev|troubleshooting|undefinedsymbol|运行时报undefined symbol，应先排查什么？|检查程序实际加载的动态库、版本和ABI是否匹配，用ldd核对可信程序的依赖路径；文件名相同不保证符号兼容。|ldd;版本;ABI|ldd,ldso
dev|troubleshooting|contextoverflow|多轮对话突然超出上下文限制，应该怎样处理历史？|先按原生Tokenizer计数，保留系统提示与当前问题，删除最早完整对话轮次；当前问题仍过长时提示缩短，不截断任意Token。|Tokenizer;系统;完整|local
dev|troubleshooting|sshfail|SSH连不上就能断定开发板关机了吗？|不能。先区分地址解析、连通性、端口和认证错误，记录原始错误；不同失败阶段需要不同处理。|不能;认证;端口|ssh
dev|troubleshooting|speedcompare|两个模型输出长度不同，能只按总耗时比较生成速度吗？|不能。分别记录输入输出Token数、prefill和decode，并在相同条件下比较吞吐；总耗时混合了不同工作量。|不能;Token;decode|sdk,local
test|troubleshooting|diskfull|出现No space left on device应直接删除旧模型吗？|不应。先检查目标文件系统容量、inode和临时文件，确认可清理对象及用途，保留已验收模型和必要记录。|inode;文件系统;保留|coreutils
test|troubleshooting|exit137|进程退出码137能直接证明CUDA显存不足吗？|不能。137通常对应被SIGKILL终止，还需结合系统日志和进程记录判断，可能是OOM或人为终止，不能直接归因CUDA。|不能;SIGKILL;日志|bash,proc
test|troubleshooting|ldduntrusted|可以对陌生下载的程序直接运行ldd吗？|不应。某些情况下ldd可能执行目标代码；对不可信文件先用 objdump -p 查看NEEDED等静态信息。|不应|不能;objdump\s+-p;NEEDED|ldd
test|troubleshooting|frequency|一次速度较快能代表持续运行性能吗？|不能。记录温度、频率、预热与负载条件，进行持续测量并报告波动；不要把单次峰值当作稳定性能。|不能;频率;持续|local
train|interaction|insufficient|用户只说“模型跑不了”，第一步怎么回答？|请提供板卡型号、系统和架构、模型名称、Runtime与驱动版本、启动命令及完整错误日志；目前信息不足以确定原因。|请;日志;版本|local
train|interaction|unknownboard|用户没有给出板卡型号，能直接写“你的板卡是RK3576”吗？|不能。应先询问板卡型号或请用户提供实际设备信息，避免把教程示例当作用户设备事实。|不能;型号|local
train|interaction|unseenoutput|没有命令输出时，可以回答“已经检查通过”吗？|不可以。应提供待执行的检查步骤并请求实际输出，明确当前尚未验证，不能编造执行结果。|不可以|不能;输出;验证|local
train|interaction|jsonfields|只输出JSON，字段task为check、status为pending。|{"task":"check","status":"pending"}|JSON:task=check,status=pending|local
train|interaction|onlycommand|只输出查看当前工作目录的命令，不加解释。|pwd|EXACT:pwd|coreutils
train|interaction|uncertain|日志不足时如何表达故障原因？|明确目前只能提出候选原因，列出需要补充的日志或检查结果；不要把推测写成已确认事实。|候选|可能;日志|local
train|interaction|preservecontext|用户前面已说明系统和板型，追问时应如何利用上下文？|沿用用户已提供的系统和板型，只追问仍缺失的版本、命令或错误输出，避免重复索取相同信息。|已提供;缺失|local
train|interaction|steps|用户要求按步骤说明排错，应怎样组织回答？|先确认现象和环境，再给有顺序的检查步骤、预期观察与下一步分支，区分检查命令和可能修改状态的操作。|步骤;观察|local
train|interaction|secret|用户愿意贴完整配置排查问题，应该提醒提供什么内容？|请提供与故障有关的配置和日志，并先去除密码、令牌及订阅地址；通常无需这些凭据就能核对版本与路径。|配置;日志;密码|local
train|interaction|noexecute|用户让聊天模型检查本机文件，但模型没有工具访问，如何回答？|说明目前无法直接读取文件，给出用户可执行的检查方法并请求相关输出，不声称已经访问或修改。|无法|不能;输出|local
dev|interaction|shortanswer|只用一句话说明内存和存储的区别。|内存用于运行中的数据，存储用于保存文件。|内存;存储;MAXLINES:1|local
dev|interaction|jsonboolean|只输出JSON对象，字段verified为false。|{"verified":false}|JSON:verified=false|local
dev|interaction|noinventapi|没有在固定版本头文件里找到某RKLLM函数，应怎么回答？|说明该接口尚未核实，查看固定版本头文件与官方示例；不编造函数名、参数或成功结果。|头文件;核实|sdk
dev|interaction|clarifyversion|用户只给出“SDK最新版”，如何继续询问？|请提供具体Toolkit、Runtime和驱动版本，以及模型标识；“最新版”无法确定实际兼容组合。|Toolkit;Runtime;驱动|local
test|interaction|onlyusername|只输出显示当前有效用户名的命令，不加解释。|whoami|EXACT:whoami|coreutils
test|interaction|jsoncount|只输出JSON，字段context为2048、output为256。|{"context":2048,"output":256}|JSON:context=2048,output=256|sdk
test|interaction|unknownperformance|没有实测数据时，可以承诺某模型达到每秒10Token吗？|不能。应说明速度待测，给出测量条件和方法，不能把目标值写成已有实测结果。|不能;测|local
test|interaction|twooptions|用户没说要保留旧模型还是替换旧模型，部署前先确认什么？|先确认是否需要保留旧模型及回退方式，默认将新模型放入独立目录，不擅自覆盖已有产物。|保留;独立目录|local
"""


def cases():
    result = []
    for line in ROWS.strip().splitlines():
        # '|' inside rubric denotes alternatives; split first five and final source.
        fields = line.split('|', 5)
        rubric, source = fields.pop().rsplit('|', 1)
        split, category, topic, question, answer = fields
        result.append(dict(split=split, category=category, group=f'{category}/{topic}',
                           question=question, answer=answer, rubric=rubric,
                           source_keys=source.split(',')))
    return result
