// Persistent RKLLM 1.3.1 worker. stdin: request-id TAB whitespace-separated token IDs.
// stdout: JSON lines only; SDK stdout and diagnostic logs go to stderr.
#include <cstddef>
#include "rkllm.h"
#include <unistd.h>
#include <csignal>
#include <chrono>
#include <fstream>
#include <iomanip>
#include <iostream>
#include <sstream>
#include <string>
#include <vector>
#include <limits>
#include <algorithm>

using Clock=std::chrono::steady_clock;
static LLMHandle handle=nullptr;
static volatile sig_atomic_t interrupted=0;
static FILE* protocol=nullptr;
static std::vector<int32_t> stop_ids;
struct Response {std::string id;bool failed=false,finished=false,native_end=false;int tokens=0;Clock::time_point start;double first=-1,last=-1;RKLLMPerfStat perf={};};
std::string quote(const std::string& s){
    std::ostringstream o;o<<'"';
    for(unsigned char c:s){switch(c){case '"':o<<"\\\"";break;case '\\':o<<"\\\\";break;case '\n':o<<"\\n";break;case '\r':o<<"\\r";break;case '\t':o<<"\\t";break;default:if(c<32)o<<"\\u00"<<std::hex<<std::setw(2)<<std::setfill('0')<<(int)c<<std::dec;else o<<c;}}
    return o.str()+'"';
}
void emit(const std::string& text){fputs((text+"\n").c_str(),protocol);fflush(protocol);}
std::string hexbytes(const char* p){std::ostringstream s;for(;*p;++p)s<<std::hex<<std::setw(2)<<std::setfill('0')<<static_cast<unsigned int>(static_cast<unsigned char>(*p));return s.str();}
long status(const std::string& key,const std::string& path="/proc/self/status"){
    std::ifstream f(path);std::string line;
    while(std::getline(f,line)){if(line.rfind(key+":",0)==0){std::istringstream s(line.substr(key.size()+1));long n=0;s>>n;return n;}}return -1;
}
void signal_handler(int){interrupted=1;}
int callback(RKLLMResult* result,void* data,LLMCallState state){
    auto& r=*static_cast<Response*>(data);
    if(interrupted)return 1; // Pause; worker aborts and clears cache after rkllm_run returns.
    if(state==RKLLM_RUN_ERROR)r.failed=true;
    if(state==RKLLM_RUN_NORMAL && result){
        if(std::find(stop_ids.begin(),stop_ids.end(),result->token_id)!=stop_ids.end()){r.native_end=true;return 1;}
        if(r.native_end)return 1;
        double now=std::chrono::duration<double>(Clock::now()-r.start).count();
        if(r.first<0)r.first=now;r.last=now;r.tokens++;
        if(result->text)emit("{\"event\":\"chunk\",\"id\":"+quote(r.id)+",\"text_hex\":"+quote(hexbytes(result->text))+"}");
    }
    if(state==RKLLM_RUN_FINISH){r.finished=true;if(result)r.perf=result->perf;}
    return 0;
}
int main(int argc,char** argv){
    // Save original stdout for protocol before SDK initialization can print anything.
    protocol=fdopen(dup(STDOUT_FILENO),"w");if(!protocol)return 2;
    if(dup2(STDERR_FILENO,STDOUT_FILENO)<0)return 2;
    if(argc!=5){std::cerr<<"Usage: board_chat model.rkllm output-limit context-limit comma-separated-eos-ids\n";return 2;}
    std::signal(SIGINT,signal_handler);
    int output=std::stoi(argv[2]),context=std::stoi(argv[3]);
    if(output<1||context<=output||context>2048)return 2;
    std::string ends=argv[4];std::replace(ends.begin(),ends.end(),',',' ');std::istringstream end_input(ends);int32_t end;
    while(end_input>>end){if(end<0)return 2;stop_ids.push_back(end);}
    if(stop_ids.empty()||!end_input.eof())return 2;
    std::ifstream model(argv[1],std::ios::binary);if(!model){std::cerr<<"Model does not exist\n";return 3;}
    RKLLMParam p=rkllm_createDefaultParam();p.model_path=argv[1];p.max_new_tokens=output;p.max_context_len=context;
    p.top_k=1;p.top_p=1;p.temperature=1;p.repeat_penalty=1;p.frequency_penalty=0;p.presence_penalty=0;
    p.skip_special_token=true;p.ignore_eos_token=false;p.is_async=false;p.extend_param.embed_flash=1;
    RKLLMCallback cb={};cb.result_callback=callback;
    if(rkllm_init(&handle,&p,&cb)!=0||handle==nullptr)return 4;
    emit("{\"event\":\"ready\"}");
    std::string line;
    while(std::getline(std::cin,line)){
        if(line=="QUIT")break;
        auto tab=line.find('\t');std::string id=tab==std::string::npos?"invalid":line.substr(0,tab);
        std::vector<int32_t> ids;bool valid=tab!=std::string::npos&&!id.empty()&&id.size()<80;
        if(valid){std::istringstream in(line.substr(tab+1));long long n;while(in>>n){if(n<0||n>std::numeric_limits<int32_t>::max()){valid=false;break;}ids.push_back(static_cast<int32_t>(n));}if(!in.eof())valid=false;}
        if(!valid||ids.empty()||ids.size()>static_cast<size_t>(context-output)){
            emit("{\"event\":\"error\",\"id\":"+quote(id)+",\"reason\":\"invalid_or_overlong_input\"}");continue;
        }
        if(rkllm_clear_kv_cache(handle,0,nullptr,nullptr)!=0){emit("{\"event\":\"error\",\"id\":"+quote(id)+",\"reason\":\"clear_cache_failed\"}");continue;}
        RKLLMInput input={};input.role="user";input.enable_thinking=false;input.input_type=RKLLM_INPUT_TOKEN;
        input.token_input.input_ids=ids.data();input.token_input.n_tokens=ids.size();
        RKLLMInferParam infer={};infer.mode=RKLLM_INFER_GENERATE;infer.keep_history=0;
        interrupted=0;Response response;response.id=id;response.start=Clock::now();
        int ret=rkllm_run(handle,&input,&infer,&response);
        int abort_status=(interrupted||(response.native_end&&!response.finished))?rkllm_abort(handle):0;
        double seconds=std::chrono::duration<double>(Clock::now()-response.start).count();
        bool success=abort_status==0&&!interrupted&&(response.native_end||(ret==0&&!response.failed&&response.finished));
        double wall_tps=response.tokens>1&&response.last>response.first?(response.tokens-1)/(response.last-response.first):0;
        std::ostringstream out;
        out<<"{\"event\":"<<quote(success||interrupted?"done":"error")<<",\"id\":"<<quote(id)
           <<",\"return_code\":"<<ret<<",\"interrupted\":"<<(interrupted?"true":"false")
           <<",\"seconds\":"<<seconds<<",\"first_token_seconds\":"<<response.first
           <<",\"native_eos\":"<<(response.native_end?"true":"false")<<",\"abort_status\":"<<abort_status
           <<",\"generated_token_count\":"<<response.tokens<<",\"decode_tokens_per_second_wall\":"<<wall_tps
           <<",\"generate_tokens\":"<<response.perf.generate_tokens<<",\"generate_ms\":"<<response.perf.generate_time_ms
           <<",\"prefill_tokens\":"<<response.perf.prefill_tokens<<",\"prefill_ms\":"<<response.perf.prefill_time_ms
           <<",\"rss_kib\":"<<status("VmRSS")<<",\"hwm_kib\":"<<status("VmHWM")
           <<",\"system_available_kib\":"<<status("MemAvailable","/proc/meminfo")
           <<",\"stop_reason\":"<<quote(interrupted?"interrupted":response.native_end?"native_eos":response.perf.generate_tokens>=output?"length_limit":"runtime_finish")<<"}";
        emit(out.str());
    }
    int ret=rkllm_destroy(handle);handle=nullptr;fclose(protocol);return ret==0?0:5;
}
