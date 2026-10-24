// Bounded text-only RKLLM 1.3.1 probe. Prompts are native-tokenized on the PC.
#include <cstddef>
#include "rkllm.h"
#include <chrono>
#include <fstream>
#include <iostream>
#include <sstream>
#include <string>
#include <vector>
#include <iomanip>
#include <stdexcept>
using Clock=std::chrono::steady_clock;
struct Row {std::string id;std::vector<int32_t> tokens;};
struct Result {std::string text;std::vector<int32_t> ids;bool error=false,finished=false;double first=-1;Clock::time_point start;RKLLMPerfStat perf={};};
double elapsed(Clock::time_point a){return std::chrono::duration<double>(Clock::now()-a).count();}
std::string quote(const std::string& s){
    std::ostringstream o;o<<'"';
    for(unsigned char c:s){switch(c){case '"':o<<"\\\"";break;case '\\':o<<"\\\\";break;case '\n':o<<"\\n";break;case '\r':o<<"\\r";break;case '\t':o<<"\\t";break;default:if(c<32)o<<"\\u00"<<std::hex<<std::setw(2)<<std::setfill('0')<<(int)c<<std::dec;else o<<c;}}
    return o.str()+'"';
}
long status_kib(const std::string& key){std::ifstream f("/proc/self/status");std::string line;while(std::getline(f,line)){if(line.rfind(key+":",0)==0){std::istringstream s(line.substr(key.size()+1));long n=0;s>>n;return n;}}return -1;}
int callback(RKLLMResult* r,void* data,LLMCallState state){
    auto& x=*static_cast<Result*>(data);
    if((state==RKLLM_RUN_NORMAL||state==RKLLM_RUN_WAITING)&&x.first<0)x.first=elapsed(x.start);
    if(state==RKLLM_RUN_ERROR)x.error=true;
    if(state==RKLLM_RUN_NORMAL && r){
        if(r->text)x.text+=r->text;
        x.ids.push_back(r->token_id);
    }
    if(state==RKLLM_RUN_FINISH){x.finished=true;if(r)x.perf=r->perf;}
    return 0;
}
int main(int argc,char** argv){
    if(argc!=6){std::cerr<<"Usage: board_probe model.rkllm native-tokens.tsv results.jsonl max_new_tokens max_context\n";return 2;}
    std::vector<Row> rows;std::ifstream inputs(argv[2]);std::string line;
    while(std::getline(inputs,line)){
        auto tab=line.find('\t');if(tab==std::string::npos)throw std::runtime_error("Invalid token row");
        Row r;r.id=line.substr(0,tab);std::istringstream s(line.substr(tab+1));int32_t n;while(s>>n)r.tokens.push_back(n);
        if(r.tokens.empty())throw std::runtime_error("Empty prompt");rows.push_back(r);
    }
    if(rows.size()!=8)throw std::runtime_error("Expected eight questions");
    RKLLMParam p=rkllm_createDefaultParam();p.model_path=argv[1];p.max_new_tokens=std::stoi(argv[4]);p.max_context_len=std::stoi(argv[5]);
    p.top_k=1;p.top_p=1;p.temperature=1;p.repeat_penalty=1;p.frequency_penalty=0;p.presence_penalty=0;
    p.skip_special_token=true;p.ignore_eos_token=false;p.is_async=false;p.extend_param.embed_flash=1;
    RKLLMCallback cb={};cb.result_callback=callback;LLMHandle handle=nullptr;
    auto start=Clock::now();int init=rkllm_init(&handle,&p,&cb);
    std::cout<<"INIT status="<<init<<" seconds="<<elapsed(start)<<std::endl;
    if(init!=0)return 3;
    std::ofstream out(argv[3]);if(!out){rkllm_destroy(handle);return 4;}
    for(int round=0;round<3;round++){
        size_t count=round==0?1:rows.size();
        for(size_t i=0;i<count;i++){
            if(rkllm_clear_kv_cache(handle,1,nullptr,nullptr)!=0){rkllm_destroy(handle);return 5;}
            RKLLMInput in={};in.role="user";in.enable_thinking=false;in.input_type=RKLLM_INPUT_TOKEN;
            in.token_input.input_ids=rows[i].tokens.data();in.token_input.n_tokens=rows[i].tokens.size();
            RKLLMInferParam param={};param.mode=RKLLM_INFER_GENERATE;param.keep_history=0;
            Result result;result.start=Clock::now();int ret=rkllm_run(handle,&in,&param,&result);double seconds=elapsed(result.start);
            out<<"{\"round\":"<<round<<",\"warmup\":"<<(round==0?"true":"false")<<",\"question_id\":"<<quote(rows[i].id)
               <<",\"output\":"<<quote(result.text)<<",\"return_code\":"<<ret<<",\"callback_error\":"<<(result.error?"true":"false")
               <<",\"finished\":"<<(result.finished?"true":"false")<<",\"seconds\":"<<seconds<<",\"first_token_seconds\":"<<result.first
               <<",\"stop_reason\":"<<quote(result.perf.generate_tokens>=p.max_new_tokens||result.ids.size()>=(size_t)p.max_new_tokens?"runtime_finish_at_limit":"runtime_finish")
               <<",\"rss_kib\":"<<status_kib("VmRSS")<<",\"hwm_kib\":"<<status_kib("VmHWM")
               <<",\"prefill_ms\":"<<result.perf.prefill_time_ms<<",\"prefill_tokens\":"<<result.perf.prefill_tokens
               <<",\"generate_ms\":"<<result.perf.generate_time_ms<<",\"generate_tokens\":"<<result.perf.generate_tokens
               <<",\"runtime_memory_mb\":"<<result.perf.memory_usage_mb<<",\"callback_token_ids\":[";
            for(size_t k=0;k<result.ids.size();k++){if(k)out<<',';out<<result.ids[k];}out<<"]}\n";out.flush();
            std::cout<<"DONE round="<<round<<" question="<<rows[i].id<<" seconds="<<seconds<<" output="<<result.text<<std::endl;
            if(ret!=0||result.error||!result.finished){rkllm_destroy(handle);return 6;}
        }
    }
    return rkllm_destroy(handle)==0?0:7;
}
