// Host-only protocol test double. Never included in board bundles or reported as NPU evidence.
#include <cstddef>
#include "rkllm.h"
#include <cstdio>
static RKLLMCallback callbacks={};
static bool paused=false;
RKLLMParam rkllm_createDefaultParam(){return RKLLMParam{};}
int rkllm_init(LLMHandle* handle,RKLLMParam*,RKLLMCallback* cb){callbacks=*cb;*handle=reinterpret_cast<LLMHandle>(1);puts("SDK_INIT_NOISE");return 0;}
int rkllm_destroy(LLMHandle){return 0;}
int rkllm_clear_kv_cache(LLMHandle,int,int*,int*){return paused?1:0;}
int rkllm_abort(LLMHandle){if(!paused)return 9;paused=false;return 0;}
int rkllm_run(LLMHandle,RKLLMInput* input,RKLLMInferParam*,void* data){
    puts("SDK_RUN_NOISE");
    RKLLMResult r={};r.text=const_cast<char*>("\xe4\xb8");r.token_id=100;
    callbacks.result_callback(&r,data,RKLLM_RUN_NORMAL);
    r.text=const_cast<char*>("\xad");r.token_id=101;callbacks.result_callback(&r,data,RKLLM_RUN_NORMAL);
    r.text=const_cast<char*>("<|im_end|>");r.token_id=248046;
    int code=callbacks.result_callback(&r,data,RKLLM_RUN_NORMAL);
    if(input->token_input.input_ids[0]==2){callbacks.result_callback(&r,data,RKLLM_RUN_FINISH);paused=false;return 0;}
    if(code==1){paused=true;return 0;}
    r.text=const_cast<char*>("SHOULD_NOT_APPEAR");r.token_id=102;
    callbacks.result_callback(&r,data,RKLLM_RUN_NORMAL);callbacks.result_callback(&r,data,RKLLM_RUN_FINISH);
    return 0;
}
