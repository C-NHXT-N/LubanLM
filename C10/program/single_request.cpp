// Teaching example for RKLLM Runtime 1.3.1, Linux ARM64. One native-token request.
#include <cstddef>
#include "rkllm.h"
#include <cstdint>
#include <fstream>
#include <iostream>
#include <sstream>
#include <string>
#include <vector>

struct Reply { std::string text; bool failed=false, finished=false; };
int callback(RKLLMResult* result, void* data, LLMCallState state) {
    auto& reply = *static_cast<Reply*>(data);
    if (state == RKLLM_RUN_NORMAL && result && result->text) reply.text += result->text;
    if (state == RKLLM_RUN_ERROR) reply.failed = true;
    if (state == RKLLM_RUN_FINISH) reply.finished = true;
    return 0;
}

int main(int argc, char** argv) {
    if (argc != 3) { std::cerr << "Usage: single_request model.rkllm native-tokens.tsv\n"; return 2; }
    std::ifstream file(argv[2]);
    std::string line;
    if (!std::getline(file, line)) return 2;
    auto tab = line.find('\t');
    if (tab == std::string::npos) return 2;
    std::istringstream stream(line.substr(tab+1));
    std::vector<int32_t> ids;
    int64_t value;
    while (stream >> value) {
        if (value < 0 || value > INT32_MAX) return 2;
        ids.push_back(static_cast<int32_t>(value));
    }
    if (!stream.eof() || ids.empty() || ids.size() > 1792) return 2;

    RKLLMParam p = rkllm_createDefaultParam();
    p.model_path = argv[1]; p.max_new_tokens = 256; p.max_context_len = 2048;
    p.top_k = 1; p.top_p = 1; p.temperature = 1; p.repeat_penalty = 1;
    p.frequency_penalty = 0; p.presence_penalty = 0;
    p.skip_special_token = true; p.ignore_eos_token = false; p.is_async = false;
    p.extend_param.embed_flash = 1;
    RKLLMCallback cb = {}; cb.result_callback = callback;
    LLMHandle handle = nullptr;
    if (rkllm_init(&handle, &p, &cb) != 0 || !handle) return 3;

    RKLLMInput input = {}; input.role = "user"; input.enable_thinking = false;
    input.input_type = RKLLM_INPUT_TOKEN;
    input.token_input.input_ids = ids.data(); input.token_input.n_tokens = ids.size();
    RKLLMInferParam infer = {}; infer.mode = RKLLM_INFER_GENERATE; infer.keep_history = 0;
    Reply reply;
    int ret = rkllm_run(handle, &input, &infer, &reply);
    // p/input/ids/reply remain alive through the synchronous call and its callbacks.
    int released = rkllm_destroy(handle);
    if (ret != 0 || reply.failed || !reply.finished || released != 0) return 4;
    std::cout << reply.text << '\n';
    return 0;
}
