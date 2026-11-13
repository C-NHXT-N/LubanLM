"""Apply explicitly authored per-answer development judgments; never grade by regex alone.

Scope is the immutable 2026-10-20 clean baseline and final checkpoint-134 only.
This records agent semantic review, not independent human or sealed acceptance scores.
"""
from __future__ import annotations
import hashlib
import json
from collections import Counter
from pathlib import Path
from scripts.common import WORKSPACE,sha256,write_json,utc
from scripts.linux_assistant.common import jsonl,write_jsonl,screen,reviewed_scores,config,gate

ROOT=WORKSPACE/'reports/linux-assistant-qwen35-2b-reviewed-20261020'
REVIEW=ROOT/'semantic-review'
BASE=WORKSPACE/'reports/linux-assistant-baseline-qwen35-20261020/answers.jsonl'
ADAPTER=WORKSPACE/'runs/linux-assistant-qwen35-2b-reviewed-20261020/lr-1e-4/dev-134.jsonl'

# + means accepted after reading the entire actual answer. Otherwise rejected.
# Rejections distinguish false claims from missing locked requirements.
TASK_JUDGMENTS='''0|uname -p被当架构字段，后续CPUID管道无合理依据，且答案未完成。|有cat /proc/cpuinfo，但未说明架构相关限制，cpu-get-features旁支无助于必要事实。
1|把lscpu输出写成/proc/cpuinfo格式并混用示例CPU核心信息，未给架构差异说明。|有读取命令，但将CPU Core/Family/Model当通用字段，未覆盖架构相关限制。
2|称lscpu由内核提供并列不实频率/字段，示例不可靠。|+给出cat /proc/cpuinfo，并明确查看核心/架构/指令集；未断言目标设备支持AVX。
3|把x86 flags/model字段泛化为所有Linux并称model为型号，跨架构说明不准确。|泛称所有CPU详细信息，未覆盖架构相关限制。
4|未说明字段随架构变化，且末尾温度检查被截断，不能视为完整正确回答。|读取路径正确，但没有架构相关限制或其等价说明。
5|声称仅RAM耗尽后才使用Swap，交换策略被绝对化；末尾表格截断。|+正确区分Swap页面交换与RAM；free -h已同时显示两者，附加-m冗余但procps实现不因此报错。
6|把free -h列名写成MemAvailable并把Swap行只当已用值，说明不准确。|+正确区分Swap与RAM，检查命令能显示两者；-h -m冗余而非两个相互独立的内存来源。
7|声称RAM访问为毫秒级、交换为秒级，且说交换通常远小于RAM却举相反例子。|+正确说明交换数据与磁盘、物理内存运行程序；free输出可分别读取Mem和Swap行。
8|把RAM访问写成毫秒级且断言仅内存不足才交换，包含事实错误。|free --swap并非该procps接口支持的选项，命令不能按建议执行。
9|RAM毫秒级、Swap通常远小于RAM等过度概括，结论说明被截断。|+Swap与RAM区分正确，free -h显示两类数据；提醒检查余量没有承诺能装下模型。
10|虚构uptime输出为Last login，要求手动算时长，工具解释错误。|+uptime可同时读取系统运行时长与负载，top/htop为补充观察；程序预热旁注不作为系统时长扣除步骤。
11|uptime输出格式解释错误并编造FreeRTOS接口，跨系统建议不可靠。|虽有uptime，却绝对声称time不会改变系统负载，测量开销与系统状态不可这样保证。
12|把uptime的输出当Last login，且把CPU/MEM百分比当系统负载，错误明显。|将系统运行时长转为程序time耗时，未完整回答板端自启动时长。
13|把Last login说成uptime输出，再用ps -ef解释进程时长，说明不完整准确。|把uptime解释为进程运行时间，混淆系统uptime和进程耗时。
14|将Last login/Elapsed since boot当uptime字段并要求错误的时间相减。|+uptime同时覆盖系统自启动时长和负载，top/htop补充进程负载；未要求从系统uptime实际扣除启动时间。
15|+明确LubanCat-3跑通不能证明鲁班猫4验收，并按硬件/软件差异列核对项；已答关键结论，后续截断不单独判错。|把板型3/4改成3.1.0与4.0软件版本，虚构版本与验证记录。
16|+明确不能用LubanCat-3覆盖鲁班猫4，要求后者独立验证；使用版本一词但没有编造具体测试。|虚构4.1.0及以上条件与签名检查，未正确识别板型验收范围。
17|+区分两种设备和证据范围，并要求鲁班猫4专项测试/报告，没有把旧结果当新实测。|只谈软件版本/SDK，未识别LubanCat-3与鲁班猫4的硬件验收对象。
18|虚构用户实际正在运行鲁班猫4的v4.x软件，并改变了题目给定设备。|只谈版本、安装源，混淆硬件板型与软件版本。
19|+明确不能跨硬件型号宣称验收，并要求目标设备配置和验证输出。|+明确现有证据只覆盖LubanCat-3，目标设备即问题中的鲁班猫4，需另行验证与记录。
20|没有本次boot筛选且将systemd笼统当服务名，后续说明未完成。|-b后追加启动时间占位不是本次boot查询用法，命令带无效/歧义参数。
21|未用-b限定本次开机，把dmesg当服务整体启动日志来源，缺必要条件。|在-b后加根，且说历史实例再加-b，误解boot参数与服务实例。
22|缺-b限定启动，且把默认日志级别说为info，说明不准确。|在-b后加根分区，命令参数错误，命中关键词仍不通过。
23|编造journalctl --start且未给服务过滤，不能完成指定任务。|在-b后加根分区，额外无效参数导致命令不可用。
24|编造--since systemd-start时间表达式，不是本次boot可靠用法。|把-b解释为指定根目录并追加根参数，错误。
25|python -m pip方向正确，但没有固定规则要求的解释器路径核验。|python -m pip正确，但漏掉固定规则要求的sys.executable路径核验；属于漏答，不把主命令判成错误。
26|缺解释器路径核验，并将python -m pip说成Python2别名，术语错误。|主命令正确，缺固定规则要求的实际解释器路径确认。
27|缺sys.executable核验，建议pip3不确保同一虚拟环境。|主命令正确，未补解释器路径核验，未满足全部固定条件。
28|声称python -m pip避免python本身路径问题不准确，且未核对解释器路径。|主命令正确，但未核对当前解释器路径。
29|错误声称pip属于标准库且可用缺失的pip安装自身，未核对解释器路径。|主命令正确，但缺sys.executable/解释器路径核验；通常不是标准入口的表述也过强。
30|同一有引号命令同时被判错误和正确，并不实贬低单引号兼容性。|文字说加单引号，但实际例子ls /home 用户没有引号，操作与解释矛盾。
31|编造shopt选项IFS并重复nullglob，未给正确解决方法。|示例没有为含空格的路径加引号，和解释矛盾。
32|shopt -s IFS无此选项，后续eval建议也不能解决引用问题。|+明确单引号包裹参数、不要拆成多个参数，示例参数确有引用；没有空格字样但语义覆盖保留路径边界。
33|把set -o posix当禁止空格拆分，命令和解释错误。|示例ls /home 用户未引用，不能保留一个含空格路径。
34|把extglob/extglob2当路径截断开关，编造选项且错误归因。|加引号说明与未引用的实际例子不一致。
35|编造history -p和last能查看上一命令退出状态。|echo $?执行后会更新$?；声称不会修改状态错误。
36|声称没有上一命令时$?通常127，属于错误附加解释。|echo $?正确，但漏固定规则要求的0成功含义。
37|无上一命令时默认1以及last/history可核验的说法错误。|命令正确，但漏0通常成功的固定条件。
38|把history -p输出说成含退出状态，错误。|命令正确，但缺0成功的必要条件。
39|把无上一命令的$?固定说成1，重复错误。|命令正确，但漏0成功的预定说明。
40|没有EOS与运行时错误处理，并把资源耗尽必然说成OOM终止。|有EOS，但未包括错误/运行时异常停止，必要条件不足。
41|编造温度阈值停止等机制，未说明EOS与运行时错误处理。|只泛称停止原因/特殊令牌，漏运行时错误区分。
42|既称同时又写或，且没有EOS/错误分类，不完整。|改为内容偏题/重复判断，漏原生EOS和错误处理。
43|把停止生成只解释为用户主动操作，漏EOS和运行时异常。|只讲长度/重复/已停止，缺EOS与错误。
44|编造90%/95%内存阈值触发OOM等通用规则，未给EOS错误分类。|只谈完整性和有害内容，未回答运行时EOS与错误状态。
45|没有说明当前仅语言分支；泛列图像验证步骤不能证明现有部署边界。|虚构当前已测试部分图像，与仅语言分支事实不符。
46|假定摄像头是看图必要条件，并未说明当前仅部署语言分支。|虚构已覆盖部分图像与渲染任务。
47|明确虚构板端已具备视觉/OCR测试证据，严重事实错误。|虚构已做部分图像测试。
48|没有当前语言分支边界，泛化板端测试不包含功能测试，未回答项目事实。|虚构部分图像任务结果，缺语言分支边界。
49|猜测未核实qwen-vl命令并要求CUDA，误配RK3576板端环境。|宣称部分图像任务格式已可用，没有来源证据。
50|把算法复杂度与线性/指数加速混为一谈，含基本原理错误。|比较单/双核方向正确，但把配置请求写成本例实际只用了两核心，未取得运行证据。
51|把TensorRT/OpenVINO当NPU架构，建议不匹配RKLLM，逻辑不可靠。|+记录单/双核实测结果并算加速比，固定模型量化配置、验证核心是否启用；满足实测语义。
52|为RKLLM NPU强加TensorRT与libnpu.so等未经核实的接口/依赖。|+要求记录总/单核结果、固定模型量化进程温度并重复测量；实测而非仅配置判断。
53|把NNPACK虚构接口当NPU支持算子例子，混淆CPU库与板端NPU。|+记录单/双核输入输出与实际吞吐，固定模型量化并查通信调度，明确不能凭核心数推断。
54|把算法复杂度与多核加速作错误对应且断言满载加核心必显著提升。|把RK3576多个核心说成每个NPU，未清楚给单/双核心可比测试。
55|将INT4/INT8泛称不支持长上下文，无2048配置及准确边界。|未给本项目2048限制及内存条件，不能当完整回答。
56|将CNN/Transformer与板端引擎支持作错误泛化，无2048与内存必要事实。|编造轮播器配置，且缺2048及内存条件。
57|断言INT4/INT8通常不支持长上下文，错误且缺2048。|缺本项目2048与内存条件。
58|将原模型混同云端服务，泛称本地显存压缩规则且缺2048。|擅自说按系统默认再应用户请求扩展，缺2048与内存约束。
59|缺2048，并把上下文通常mmap到内核空间等说法混在一起。|缺2048与内存条件。
60|把运行时错误转为.o/.a诊断，nm -D用于普通.o不适当，缺实际库路径/ABI。|缺ldd实际依赖路径核对、版本与ABI，LD_RUN_PATH不等于运行时搜索变量。
61|围绕静态链接和升级链接器，未查运行时实际加载路径/ABI。|版本/符号方向合理，但缺ldd实际路径与ABI核对。
62|说符号必须在目标文件定义过度，未给运行时版本/ABI诊断。|缺ldd、版本和ABI。
63|错误声称.o通常没有符号表，选项解释不实。|缺ldd/版本/ABI，还把LD_RUN_PATH当运行时变量。
64|把-D宏定义当全局链接缺符号的依据，偏离运行时问题。|只有版本重建比较，缺实际依赖路径和ABI。
65|缺保留系统提示/当前问题及完整轮次裁剪算法和原生Tokenizer计数。|只说留一定历史去重，缺Tokenizer/系统提示/完整轮次。
66|gzip/base64压缩对话减少Token建议不成立，缺完整轮次裁剪。|只留近期历史，缺计数和系统提示/完整轮次。
67|说已满上下文再合并历史可恢复，未给必要裁剪方法。|有查长度，但缺原生Tokenizer和完整轮次/系统提示保留。
68|把系统负载导致截断混同长度超限，缺历史处理规则。|缺原生Tokenizer与完整轮次裁剪。
69|无依据断言当前版本无续写能力，未完整给系统提示/当前输入保留规则。|没给原生计数/完整轮次和系统提示保留规则。
70|有网络/服务/认证，缺固定条件的端口核对。|区分连接和权限方向合理，但缺端口检查。
71|无依据称关机最常见并建议关闭主机密钥检查，缺完整认证/端口核对。|把无密码登录和远程登录列作失败阶段，分类不当。
72|称关机通常不关闭SSH及网卡DOWN通常就是关机，明显错误。|+核对网络/服务/密钥和22或自定义端口；密钥核对属于认证检查。
73|有网络/服务/端口，但缺认证，完整步骤被截断。|没有认证阶段。
74|把无root权限当SSH失败通用原因，并混淆网络与串口。|缺端口检查，分类也未准确区分认证阶段。
75|只用Token除总耗时，缺prefill/decode分离；同为10秒又说仅比较时间会判B快，推理错误。|没有Token计数口径和独立decode测量完整说明。
76|100/1和1000/10均100 Token/s，却说前者远高，算术错误。|只有泛称长度和耗时，缺Token计数与decode吞吐。
77|没给输出Token与decode计时方法，停留于上下文和并行度。|未核对输出Token数量，固定输入不能补足生成吞吐。
78|虽提Token但没给独立decode计时，示例中途截断。|未给Token计数口径及独立decode计时。
79|必要输出Token/decode测量方法缺失，input_context_length检查未完成。|只记录输入/模板/策略，没有输出Token与decode耗时。
80|+一句话正确区分RAM临时工作与持久存储。|+一句话正确区分程序运行与持久数据，free/df检查对象正确。
81|+临时工作区与持久存储区别正确，单行。|+一句话区分运行内存与持久文件。
82|+单句分号结构区分运行临时数据与长期保存。|+一句话区分运行程序与持久数据/系统文件。
83|+工作台/仓库比喻符合通常RAM和持久存储用途，单句。|+一句话正确区分运行与持久保存。
84|+通常RAM直接访问与I/O持久存储区别正确，单句。|+运行数据和文件区分正确，free/df对象也正确且单句。
85|Markdown代码块违反仅JSON对象。|+仅JSON对象，verified为布尔false，无额外字段。
86|多了message字段，违反字段集。|+仅指定JSON对象和布尔false。
87|含代码块包装，严格JSON解析失败。|+仅指定JSON对象。
88|额外message字段，不符合固定字段集合。|+字段集合和布尔类型正确。
89|含代码块，违反仅JSON。|+无包装或多余字段。
90|猜rkllm_func.h路径并转向已确认该函数的模板，缺尚未核实边界。|未找到却要求提供已确认函数/签名，依据不足。
91|编造rkllm_v1.h/rkllm_v2.h具体接口，过度推断版本移除。|无依据称模板依赖标准库，且缺固定头文件核实。
92|把未声明函数归为动态加载，编译/链接说明混乱。|未找到却要求给已验证函数名和位置，证据不足。
93|把用户态RKLLM API当内核驱动/DRM问题。|没有固定版本头文件核实步骤。
94|泛化硬件版本决定API并猜路径/命名空间，缺承认尚未核实。|有无法确认边界，但缺固定版本头文件核实。
95|只泛问SDK，缺Toolkit/Runtime/驱动三项版本。|只问SDK/平台/错误，漏Runtime/驱动。
96|询问版本需求而非已装组合，缺三项实际版本。|缺Toolkit/Runtime/驱动具体版本。
97|只问用途/名称，缺三项版本。|只问SDK路径/编译器，缺Runtime/驱动。
98|举STM32为Linux板型等不当，并缺具体运行库/驱动版本。|SDK版本/平台不足以补齐Runtime与驱动。
99|编造ESP-IDF 24.0.0示例要求，没索取三项版本。|泛称SDK/路径，未区分Toolkit/Runtime/驱动。
100|把nproc当在线数并说可能返回0，解释错误。|nproc是进程可用单元，不等于系统在线处理器数。
101|把lsblk TYPE列说成文件系统类型，把df Filesystem当挂载点，错误。|+df -h给文件系统/挂载映射，blkid给文件系统标识/类型，lsblk给块设备拓扑，组合完成同一查询。
102|优先MemFree并把MemAvailable公式说错，概念相反。|把MemAvailable说成当前进程立即可用而非系统估计，漏MemFree区别。
103|编造IUsed%列并错列顺序/含义，判定依据不可靠。|+df -i查看inode，明确空闲字节不等于可用inode。
104|说八进制权限不含执行位，并把644映射rwxr-xr-x，错误。|+stat -c %a正确；不解释特殊权限指未展开语义，而非数字排除特殊位。
105|+realpath是规范化符号链接的有效替代；正常替代命令不判错。|+realpath解析符号链接并给规范路径。
106|把du默认单位说成字节、-l说成详细列表，含错误。|du -sh正确却将易读结果说成固定字节再换算，单位说明错误。
107|缺-f，且用head -n100避免tail卡顿无依据。|tail -n +51从第51行开始，不是最后50行。
108|给清单自身算摘要而非核验列出的文件，并编造/etc常规位置。|sha256sum -r不是GNU支持选项，命令无效。
109|用ELF Class区分位宽当ISA鉴别，未核对Machine。|+file检查程序格式/架构，拒绝凭文件名判断。
110|用了300秒，不符合固定30s条件；题目未写30秒，题规过严单独登记。|给timeout 300，未满足固定30s；问题未指定30秒，题规缺陷单独登记。
111|把LD_LIBRARY_PATH当最常见密钥变量，并无依据建议unset。|没有明确仅查相关变量而不完整导出环境，dladdr旁支不补足具体步骤。
112|用error文本和日志文件代替本次启动内核严重级别过滤，缺限定。|grep文本不等于优先级，dmesg -w也不保证持久文件。
113|只泛看日志/代码，没有strace观察openat实际路径，答案未完成。|torch.get_default_dtype不能查看加载模型或文件打开路径。
114|用最后10行代替10分钟时间筛选。|tail -n 10m不是10分钟，命令错误。
115|编造last和通用LAST_STATUS可查看上一命令退出状态。|echo $?本可替代printf，却又否认脚本标准退出状态，附加解释错误。
116|有启动命令/环境，但缺板型和完整原始错误日志必要项。|缺启动命令与板型，信息不完整。
117|+明确未给路径/报错不能给针对性修复，要求补证据，仅给只读检查而无编造修改命令。|没有明确索取路径和完整报错，只有泛化确认现象。
118|称chmod/chown不可逆、chmod -R覆盖文件内容，基本错误。|编造find -R选项，漏运行用户与父目录权限核对。
119|从无OOM记录推断内核未触发OOM，把E2BIG当内存分配错误，结论越界。|正确说不能断言OOM，但漏待验证假设与索取峰值内存/系统日志等必要证据。
'''

EQUIVALENTS={
 ('baseline',15):{'鲁班猫4':'回答鲁班猫 4/LubanCat-4为同一硬件型号，中间空格不改变对象。'},
 ('baseline',16):{'鲁班猫4':'回答鲁班猫 4即题目中的鲁班猫4，只有排版空格差异。'},
 ('baseline',17):{'鲁班猫4':'鲁班猫 4为同一设备型号，不因空格扣语义分。'},
 ('baseline',19):{'鲁班猫4':'回答的鲁班猫 4/LubanCat-4是题中目标硬件。'},
 ('adapter',10):{'/proc/uptime':'uptime同时显示系统自启动时长。','/proc/loadavg':'uptime同时显示1/5/15分钟平均负载。'},
 ('adapter',14):{'/proc/uptime':'uptime是自启动时长的有效查询工具。','/proc/loadavg':'uptime同时提供系统平均负载。'},
 ('adapter',19):{'鲁班猫4':'目标设备明确指问题中与LubanCat-3对照的鲁班猫4，并要求独立验证。'},
 ('adapter',32):{'空格':'明确单引号包裹参数，不会拆成多个参数，语义即保留题中空格路径边界。'},
 ('adapter',51):{'实测':'明确记录单/双核结果计算实际加速比并固定条件、验证启用。'},
 ('adapter',52):{'实测':'明确固定条件记录结果并重复测量。'},
 ('adapter',53):{'实测':'记录单/双核输入输出与实际吞吐并固定模型量化。'},
 ('adapter',72):{'认证':'检查SSH密钥即核对认证，另有服务/网络/端口核对。'},
 ('adapter',101):{'-f':'df -h、blkid、lsblk组合覆盖挂载映射、文件系统标识和设备拓扑，为同目标有效替代。'},
 ('baseline',105):{'readlink':'realpath同样解析到规范路径。','-f':'GNU realpath默认规范化与readlink -f相同路径存在性边界。'},
 ('adapter',105):{'readlink':'realpath是同目标规范化工具。','-f':'GNU realpath默认物理规范化等价于readlink -f。'},
 ('baseline',117):{'缺少':'未提供故障路径和具体报错等价指出缺资料。','不应':'明确无法直接给针对性修复并要求补证据，没有给修改命令。'},
 ('baseline',146):{'熔化':'融化/融解在实际回答中明确为固态冰吸热变液态，语义等价。'},
}

GENERAL_BASE_STRICT={0,1,2,3,4,10,11,12,13,14,16,26,35,36,37,39}
GENERAL_BASE_CONTENT=GENERAL_BASE_STRICT|{15,17,18,19,24,30,31,32,33,34}
GENERAL_ADAPTER_STRICT={1,3,4,*range(10,27),28,29,*range(30,38),39}
GENERAL_ADAPTER_CONTENT=set(GENERAL_ADAPTER_STRICT)
GENERAL_TYPE_LABELS={f'general/dev/{i}':label for i,label in enumerate(
    ('加法','乘法','指定文字输出','JSON格式','关系排序','常识','应用题计算','减法'))}

GENERAL_NOTES={
 0:('23+18=41正确且仅数字。','23+18应41，实际31。'),
 1:('26+18=44，正确。','26+18=44，正确。'),
 2:('29+18=47，正确。','29+18应47，实际37。'),
 3:('32+18=50，正确。','32+18=50，正确。'),
 4:('35+18=53，正确。','35+18=53，正确。'),
 5:('只复述8×7，没有56。','将8×7拼成87，没有乘法。'),
 6:('对8×8误称缺板型信息，没给64。','8×8应64，实际8。'),
 7:('对8×9无故追问，未给72。','8×9应72，实际8 9。'),
 8:('对8×10拒答，未给80。','复述8×10，未给80。'),
 9:('对8×11拒答，未给88。','8×11应88，实际811。'),
 10:('仅河流，正确。','仅河流，正确。'),
 11:('仅橘子，正确。','仅橘子，正确。'),
 12:('仅雨伞，正确。','仅雨伞，正确。'),
 13:('仅灯塔，正确。','仅灯塔，正确。'),
 14:('仅茶杯，正确。','仅茶杯，正确。'),
 15:('字段值正确，但数组和代码块违反仅JSON对象；独立内容分只核字段值。','JSON对象count=10及布尔ready=true，正确。'),
 16:('JSON对象count=11且ready=true，正确。','字段和类型正确。'),
 17:('字段值正确但有代码块，格式不通过。','JSON对象count=12且ready=true，正确。'),
 18:('字段值正确但有代码块，格式不通过。','JSON对象count=13且ready=true，正确。'),
 19:('字段值正确但有代码块，格式不通过。','JSON对象count=14且ready=true，正确。'),
 20:('最高小林，实际小周。','小林>小周>小吴，正确。'),
 21:('最高小林，实际吴。','小林，正确且仅姓名。'),
 22:('最高小林，实际小周。','小林，正确且仅姓名。'),
 23:('先说无法确定，又作矛盾推理，内容也不正确。','小林，正确。'),
 24:('小林结论正确，但有最高者前缀，严格格式失败。','仅小林，正确。'),
 25:('把结冰混称凝华并误解过冷，虽命中凝固仍含事实错误。','凝固，液态变固态，正确。'),
 26:('融化为熔化的日常同义表达，实际解释为固态吸热变液态，正确。','熔化和吸热解释正确。'),
 27:('说沸腾只在表面且需高于沸点，混淆过程。','将一般液态到气态统称蒸发，漏汽化总称并混淆进一步变水蒸气。'),
 28:('凝结总称正确，但把冷瓶外水珠说成瓶内水蒸气形成，附加事实错误。','液化，气态变液态，正确。'),
 29:('二氧化碳正确，但类囊体腔和氧气光解说明错误。','二氧化碳为原料、氧气为产物，正确。'),
 30:('6×4=24内容正确，但多了公式和单位。','24，计算和格式正确。'),
 31:('7×4=28正确，但附公式和单位。','28，正确。'),
 32:('8×4=32正确，但附说明违反仅结果。','32，正确。'),
 33:('9×4=36正确，但附式子和单位。','36，正确。'),
 34:('10×4=40正确，但附式子和单位。','40，正确。'),
 35:('40-17=23，正确。','40-17=23，正确。'),
 36:('41-17=24，正确。','41-17=24，正确。'),
 37:('42-17=25，正确。','42-17=25，正确。'),
 38:('43-17应26，实际16。','43-17应26，实际16。'),
 39:('44-17=27，正确。','44-17=27，正确。'),
}

def main():
    baseline=jsonl(BASE);adapter=jsonl(ADAPTER)
    generation=json.loads((ROOT/'development-comparison.json').read_text('utf8'))
    if sha256(BASE)!=generation['baseline']['answers_sha256'] or sha256(ADAPTER)!=generation['candidate_answers_sha256']:
        raise ValueError('Raw outputs changed after generation')
    if len(baseline)!=160 or len(adapter)!=160:raise ValueError('Incomplete coverage')
    catalog={}
    for line in TASK_JUDGMENTS.strip().splitlines():
        i,b,a=line.split('|',2);catalog[int(i)]=(b,a)
    if set(catalog)!=set(range(120)) or set(GENERAL_NOTES)!=set(range(40)):
        raise ValueError('Explicit judgments must cover every answer')
    REVIEW.mkdir(parents=True,exist_ok=True)
    outputs={};scores={}
    for name,source in [('baseline',baseline),('adapter',adapter)]:
        reviewed=[];decisions=[]
        for i,row in enumerate(source):
            other=adapter[i] if name=='baseline' else baseline[i]
            if any(row[k]!=other[k] for k in ('id','messages','prompt','input_ids','reference','rubric')):
                raise ValueError('Input/rule mismatch: '+row['id'])
            if i<120:
                note=catalog[i][0 if name=='baseline' else 1];accepted=note.startswith('+');note=note.lstrip('+');content=None
            else:
                j=i-120;note=GENERAL_NOTES[j][0 if name=='baseline' else 1]
                accepted=j in (GENERAL_BASE_STRICT if name=='baseline' else GENERAL_ADAPTER_STRICT)
                content=j in (GENERAL_BASE_CONTENT if name=='baseline' else GENERAL_ADAPTER_CONTENT)
            d={'id':row['id'],'answer_sha256':hashlib.sha256(row['answer'].encode()).hexdigest(),
                'accepted':accepted,'reason':note,'reviewer':'Codex per-answer semantic review, 2026-10-20; not independent human review',
                'equivalents':EQUIVALENTS.get((name,i),{})}
            checked={**row,**screen(row,row['answer']),'accepted':accepted,'review_reason':note,
                'reviewer':d['reviewer'],'equivalents':d['equivalents'],'semantic_review':'complete'}
            if content is not None:d['content_correct']=content;checked['content_correct']=content
            decisions.append(d);reviewed.append(checked)
        rates=reviewed_scores(reviewed)
        def count(selected):
            return {'correct':sum(r['accepted'] for r in selected),'total':len(selected),
                'accuracy':sum(r['accepted'] for r in selected)/len(selected)}
        s={kind:count([r for r in reviewed if r['kind']==kind]) for kind in ('task','general')}
        s['overall']=count(reviewed)
        s['general_content']={'correct':sum(r.get('content_correct') is True for r in reviewed if r['kind']=='general'),
            'total':40,'accuracy':rates['general_content']}
        s['task_categories']={cat:count([r for r in reviewed if r['category']==cat])
            for cat in sorted({r['category'] for r in reviewed if r['kind']=='task'})}
        s['general_types']={g:count([r for r in reviewed if r['group']==g])
            for g in sorted({r['group'] for r in reviewed if r['kind']=='general'})}
        write_jsonl(REVIEW/f'{name}-decisions.jsonl',decisions)
        write_jsonl(REVIEW/f'{name}-answers.reviewed.jsonl',reviewed)
        write_json(REVIEW/f'{name}-scores.json',s);scores[name]=s;outputs[name]=reviewed
    c=config(WORKSPACE/'LubanLM/configs/linux-assistant-qwen35-2b-reviewed.json')
    b,a=scores['baseline'],scores['adapter']
    passed=gate(b['task']['accuracy'],a['task']['accuracy'],b['general']['accuracy'],a['general']['accuracy'],
        c['acceptance'],b['general_content']['accuracy'],a['general_content']['accuracy'])
    result={'completed':True,'reviewed_utc':utc(),'scope':'fixed160 development records, final checkpoint-134 only',
        'reviewed_answers':320,'review_type':'agent semantic review, not independent human review',
        'scores':scores,'engineering_quality_gate_passed':passed,'final_acceptance_run':False,
        'all_checkpoints_reviewed':False,'baseline_answers_sha256':sha256(BASE),'adapter_answers_sha256':sha256(ADAPTER),
        'task_scenarios':40,'general_type_groups':8,
        'frozen_rule_limitations':['pip题附加sys.executable核验，主命令正确但漏该项仍严格扣分。',
            '退出状态题附加0成功含义，仅正确命令未必通过。','超时题未在问题中指定30秒，但固定规则要求30s；保留原规则并登记题规缺陷。'],
        'truncation_policy':'Length limit alone does not imply failure; reject actual errors/missing applicable facts.',
        'general_content_policy':'Ignore wrappers/extra explanation only; correct named JSON fields inside an array still count as content but not format success. Contradictory/false extra claims fail content.',
        'judgments_sha256':{n:sha256(REVIEW/f'{n}-decisions.jsonl') for n in ('baseline','adapter')}}
    write_json(REVIEW/'summary.json',result)
    lines=['# Qwen3.5-2B完整开发集正确率','',
        '逐题复核原模型与最终LoRA checkpoint-134共320条实际回答。正确指满足适用固定条件、无实质事实或操作错误且格式符合要求；有效同义命令须记录等价依据。没有用关键词筛查代替语义评分。','',
        '| 范围 | 原模型 | LoRA | 变化（百分点） |','|---|---:|---:|---:|']
    for kind,label in [('task','任务题'),('general','通用题（含格式）'),('overall','全部开发题'),('general_content','通用内容正确（忽略格式）')]:
        x,y=b[kind],a[kind]
        lines.append(f"| {label} | {x['correct']}/{x['total']}（{x['accuracy']:.2%}） | {y['correct']}/{y['total']}（{y['accuracy']:.2%}） | {(y['accuracy']-x['accuracy'])*100:+.2f} |")
    labels={'environment':'环境检查','linux':'Linux操作','deployment':'部署流程','troubleshooting':'故障排查','interaction':'交互规范'}
    lines+=['','## 五类任务','', '| 类别 | 原模型 | LoRA |','|---|---:|---:|']
    for cat,label in labels.items():
        x,y=b['task_categories'][cat],a['task_categories'][cat]
        lines.append(f"| {label} | {x['correct']}/{x['total']}（{x['accuracy']:.2%}） | {y['correct']}/{y['total']}（{y['accuracy']:.2%}） |")
    lines+=['','## 通用题型','', '| 题型组 | 原模型 | LoRA |','|---|---:|---:|']
    for g,x in b['general_types'].items():
        y=a['general_types'][g];lines.append(f"| {GENERAL_TYPE_LABELS[g]} | {x['correct']}/{x['total']} | {y['correct']}/{y['total']} |")
    lines+=['','## 判定与限制','',
        '任务能力未达到80%工程门槛，不能交付。本次只评最终检查点，不声称所有中间检查点已完成评审或已选出最优候选；不自动启动新训练、合并、量化或封存验收。',
        'LoRA主要改善简洁输出、JSON格式、关系排序；23+18/29+18仍错、8乘法五题全错、43-17仍答16，且会虚构journalctl/sha256sum参数或设备版本。数据通过复核不等于模型已学会，增加数量不是唯一优化方向。',
        '固定规则存在额外限制：pip要求sys.executable、退出码要求解释0；超时题问题没有要求30秒而规则要求30s。本次仍按预先规则判漏答，并不把正确主命令说成错误。后续应先修订题规再用新的独立题集评价，不能看完回答后放宽规则抬分。',
        '120任务题来自40场景，40通用题来自8题型组，包含同案例问法变化，不能当160个独立场景统计保证。结果为代理语义复核，不是独立人工专家评分；封存验收未运行。',
        '边界核对：[uptime手册](https://man7.org/linux/man-pages/man1/uptime.1.html)、[realpath手册](https://man7.org/linux/man-pages/man1/realpath.1.html)、[readlink手册](https://man7.org/linux/man-pages/man1/readlink.1.html)、[free v4.0.4源码](https://gitlab.com/procps-ng/procps/-/raw/v4.0.4/src/free.c)、[sha256sum手册](https://man7.org/linux/man-pages/man1/sha256sum.1.html)。free的-h为独立易读标志，-h -m不据此误判单位冲突；--swap没有定义。',
        '全部实际答案与逐题理由：[逐题复核明细](逐题复核明细.md)。机器证据：[summary.json](summary.json)和两组decisions/answers.reviewed.jsonl。']
    (REVIEW/'REPORT.md').write_text('\n'.join(lines)+'\n',encoding='utf8')
    detail=['# 160题逐题复核明细','','保留实际生成文本，没有替模型改答案。','']
    for i,(left,right) in enumerate(zip(outputs['baseline'],outputs['adapter'])):
        detail.extend([f"## {i+1}. {left['id']}",'','问题：'+next(m['content'] for m in reversed(left['messages']) if m['role']=='user'),
            '', '固定规则：`'+left['rubric']+'`','', '参考：'+left['reference'],''])
        for label,row in [('原模型',left),('LoRA',right)]:
            detail.extend([f"{label}：{'通过' if row['accepted'] else '不通过'}。{row['review_reason']}",'','~~~~text',row['answer'],'~~~~',''])
    (REVIEW/'逐题复核明细.md').write_text('\n'.join(detail),encoding='utf8')
    write_json(REVIEW/'integrity-check.json',{'passed':True,'raw_outputs_unchanged':sha256(BASE)==result['baseline_answers_sha256'] and sha256(ADAPTER)==result['adapter_answers_sha256'],
        'review_coverage_each':160,'identical_inputs_and_rules':True,'scores_recomputed_from_all_decisions':True,
        'all_accepted_conditions_or_explicit_equivalents_validated':True})
    print(json.dumps(result,ensure_ascii=False,indent=2))

if __name__=='__main__':main()
