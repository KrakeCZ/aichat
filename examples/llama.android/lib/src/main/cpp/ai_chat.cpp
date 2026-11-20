#include <android/log.h>
#include <dlfcn.h>
#include <jni.h>
#include <sys/stat.h>
#include <unistd.h>

#include <algorithm>
#include <atomic>
#include <cctype>
#include <cmath>
#include <cstdlib>
#include <cstring>
#include <deque>
#include <iomanip>
#include <mutex>
#include <sstream>
#include <string>
#include <unordered_set>
#include <vector>

// FIX: Include ordering and usage of quotes to ensure local headers are found
#include "build-info.h"
#include "chat.h"
#include "common.h"
#include "ggml-backend.h"
#include "llama.h"
#include "logging.h"
#include "mtmd-helper.h"
#include "mtmd.h"
#include "sampling.h"

static std::string g_native_lib_dir      = "";
static std::string g_active_backend_info = "CPU (ARM64)";

template <class T> static std::string join(const std::vector<T> & values, const std::string & delim) {
    std::ostringstream str;
    for (size_t i = 0; i < values.size(); i++) {
        str << values[i];
        if (i < values.size() - 1) {
            str << delim;
        }
    }
    return str.str();
}

/**
 * LLama resources: context, model, batch and sampler
 */
constexpr int N_THREADS_MIN      = 2;
constexpr int N_THREADS_MAX      = 4;
constexpr int N_THREADS_HEADROOM = 2;

constexpr int   DEFAULT_CONTEXT_SIZE = 4096;
constexpr int   OVERFLOW_HEADROOM    = 4;
constexpr int   BATCH_SIZE           = 512;
constexpr float DEFAULT_SAMPLER_TEMP = 0.7f;

static llama_model *             g_model;
static llama_context *           g_context;
static llama_batch               g_batch{};
static bool                      g_batch_initialized = false;
static std::recursive_mutex      g_config_mutex;
static common_chat_templates_ptr g_chat_templates;
static bool                      g_use_jinja = false;  // Resolved at prepare() based on model template
static bool g_template_has_reasoning         = false;  // Resolved at prepare(): true if template uses reasoning_content
static common_sampler * g_sampler;
static mtmd_context *   g_mtmd_ctx = nullptr;

// Global variable to store KV Cache Mode between load() and prepare()/init_context()
// 0=F16, 1=Q8_0, 2=Q4_0, 3=Q4_1, 4=Q5_0, 5=Q5_1, 6=IQ1_S
static int g_kv_cache_mode = 1;  // Default to Q8_0 (Balanced)

// Globals for new parameters
static int     g_n_ctx                        = 0;
static int     g_n_batch                      = BATCH_SIZE;
static int     g_n_ubatch                     = BATCH_SIZE;
static bool    g_flash_attn                   = true;
static int     g_hexagon_va_limit_mb          = 3500;
static int     g_hexagon_ndev                 = 1;
static int     g_kv_replay_on_ctx_resize_mode = 0;  // 0=AUTO, 1=ALWAYS, 2=NEVER
static size_t  g_loaded_model_file_bytes      = 0;
static int     s_hexagon_initialized_ndev     = 0;
// New parameters globals
static bool    g_ignore_eos                   = false;
static bool    g_no_perf                      = false;
static bool    g_timing_per_token             = false;
static int32_t g_tool_markup_format =
    0;  // 0=JINJA_NATIVE, 1=XML_TOOL_CALL, 2=MISTRAL_TOOL_CALLS, 3=GEMMA, 4=HARMONY, 5=API
static bool g_allow_native_tools_hint = true;
static bool g_tools_active_hint       = false;
static bool g_custom_cot_rules_set    = false;

// Global sampling params — persistent between setGrammar/configureSampling calls
static common_params_sampling g_sparams;

// Globals for Thinking / Reasoning control
static bool        g_enable_thinking          = true;
static int32_t     g_reasoning_budget_tokens  = 0;  // 0 = unlimited / no constraint
static std::string g_thinking_effort          = "medium";
static std::string g_model_thinking_start_tag = "<think>";
static std::string g_model_thinking_end_tag   = "</think>";

enum class FinishReason : int32_t { EOG = 0, LENGTH = 1, CONTEXT_FULL = 2, CANCELLED = 3, ERROR_STATE = 4 };

static const char * finish_reason_to_string(FinishReason reason) {
    switch (reason) {
        case FinishReason::EOG:
            return "EOG";
        case FinishReason::LENGTH:
            return "LENGTH";
        case FinishReason::CONTEXT_FULL:
            return "CONTEXT_FULL";
        case FinishReason::CANCELLED:
            return "CANCELLED";
        case FinishReason::ERROR_STATE:
            return "ERROR";
    }
    return "UNKNOWN";
}

enum class SemanticEventType : int32_t {
    REASONING_DELTA = 0,
    CONTENT_DELTA   = 1,
    TOOL_CALL_DELTA = 2,
    FINISHED        = 3,
    ERROR_EVENT     = 4
};

struct NativeSemanticEvent {
    int64_t           generation_id = 0;
    int64_t           sequence      = 0;
    SemanticEventType type          = SemanticEventType::CONTENT_DELTA;
    std::string       text;
    // Tool call fields
    int32_t           tool_index = -1;
    std::string       tool_id;
    std::string       tool_name;
    std::string       tool_args_delta;
    // Finished fields
    FinishReason      finish_reason    = FinishReason::EOG;
    int32_t           prompt_tokens    = 0;
    int32_t           generated_tokens = 0;
};

// --- Phase 1: Native Tool Protocol & Turn State ---
enum class NativeToolChoice : int32_t { AUTO = 0, REQUIRED = 1, NONE = 2 };

struct PendingToolCallState {
    std::string id;
    std::string name;
    std::string arguments;
    bool        completed          = false;
    bool        waiting_for_result = false;
};

struct NativeTurnFormat {
    common_chat_format                  format = COMMON_CHAT_FORMAT_CONTENT_ONLY;
    std::string                         prompt;
    std::string                         generation_prompt;
    std::string                         parser_source;
    std::string                         grammar;
    bool                                grammar_lazy = false;
    std::vector<common_grammar_trigger> grammar_triggers;
    std::vector<std::string>            preserved_tokens;
    std::string                         thinking_start_tag;
    std::vector<std::string>            thinking_end_tags;
    std::vector<std::string>            additional_stops;
};

static std::vector<common_chat_tool>     g_configured_tools;
static NativeToolChoice                  g_tool_policy = NativeToolChoice::AUTO;
static PendingToolCallState              g_pending_tool_call;
static std::vector<PendingToolCallState> g_pending_tool_calls;
static std::vector<common_chat_msg>      g_staged_tool_msgs;
static NativeTurnFormat                  g_current_turn_format;
static std::string                       g_last_tool_formatted_delta;
static bool                              g_sampler_has_turn_constraints = false;
static bool                              g_enable_native_tool_grammar   = false;

struct CoTTagDef {
    std::string              open_tag;
    std::vector<std::string> close_tags;
};

static uint64_t                        g_tools_config_version = 0;
static std::string                     g_last_tools_json_str;
static std::unordered_set<llama_token> g_preserved_token_ids;
static std::unordered_set<llama_token> g_additional_stop_token_ids;
static std::vector<std::string>        g_template_stop_strings;
static std::vector<CoTTagDef>          g_active_cot_rules;
static std::string                     g_cot_lead_chars                   = "<[";
static bool                            g_template_honors_disable_thinking = false;
static bool                            g_vocab_and_template_probed        = false;

struct CachedModelCapabilities {
    bool        ready                     = false;
    bool        has_chat_template         = false;
    bool        supports_jinja            = false;
    bool        supports_thinking         = false;
    bool        supports_reasoning_effort = false;
    bool        supports_tools            = false;
    bool        supports_tool_calls       = false;
    int         detected_format           = static_cast<int>(COMMON_CHAT_FORMAT_CONTENT_ONLY);
    std::string handler_name              = "default";
    int32_t     context_length            = 0;
    std::string chat_template;
    bool        supports_system_role         = true;
    bool        supports_parallel_tool_calls = false;
};

struct CachedRenderedHistory {
    bool             valid     = false;
    size_t           msg_count = 0;
    std::string      last_role;
    std::string      last_content;
    uint64_t         tools_version    = 0;
    NativeToolChoice tool_policy      = NativeToolChoice::AUTO;
    bool             use_native_tools = false;
    bool             enable_thinking  = true;
    std::string      thinking_effort;
    std::string      rendered_prompt;
};

struct CachedAnchorBase {
    bool             valid            = false;
    uint64_t         tools_version    = 0;
    NativeToolChoice tool_policy      = NativeToolChoice::AUTO;
    bool             use_native_tools = false;
    bool             enable_thinking  = true;
    std::string      thinking_effort;
    std::string      base_u;
    std::string      base_ua;
};

struct CachedNativeParser {
    bool                                valid           = false;
    uint64_t                            tools_version   = 0;
    NativeToolChoice                    tool_policy     = NativeToolChoice::AUTO;
    bool                                enable_thinking = true;
    std::string                         thinking_effort;
    common_chat_format                  format = COMMON_CHAT_FORMAT_CONTENT_ONLY;
    std::string                         parser_source;
    std::string                         generation_prompt;
    std::string                         grammar;
    bool                                grammar_lazy = false;
    std::vector<common_grammar_trigger> grammar_triggers;
    std::vector<std::string>            preserved_tokens;
    std::string                         thinking_start_tag;
    std::vector<std::string>            thinking_end_tags;
    std::vector<std::string>            additional_stops;
};

struct CachedSamplingState {
    bool                valid           = false;
    const llama_model * model_ptr       = nullptr;
    float               temp            = 0.0f;
    int32_t             top_k           = 0;
    float               top_p           = 0.0f;
    float               min_p           = 0.0f;
    float               xtc_prob        = 0.0f;
    float               xtc_thresh      = 0.0f;
    float               typical_p       = 0.0f;
    float               dyn_temp_range  = 0.0f;
    float               dyn_temp_exp    = 0.0f;
    float               penalty_repeat  = 0.0f;
    float               penalty_freq    = 0.0f;
    float               penalty_present = 0.0f;
    int32_t             penalty_last_n  = 0;
    int32_t             mirostat        = 0;
    float               mirostat_tau    = 0.0f;
    float               mirostat_eta    = 0.0f;
    std::string         samplers_seq;
    bool                use_backend_sampling = false;
    bool                ignore_eos           = false;
    bool                no_perf              = false;
    bool                timing_per_token     = false;
};

static CachedModelCapabilities g_caps_cache;
static CachedRenderedHistory   g_prev_prompt_cache;
static CachedAnchorBase        g_anchor_base_cache;
static CachedNativeParser      g_native_parser_cache;
static CachedSamplingState     g_sampling_cache;

static void                           probe_model_template_and_vocab_once();
static const std::vector<CoTTagDef> & get_universal_cot_catalog();
static CachedModelCapabilities        get_or_compute_model_capabilities();
static int                            decode_tokens_in_batches(llama_context *      context,
                                                               llama_batch &        batch,
                                                               const llama_tokens & tokens,
                                                               const llama_pos      start_pos,
                                                               const bool           compute_last_logit = false);
static void                           reset_long_term_states(bool clear_kv_cache = true, bool keep_bitmap = false);
static void                           reset_short_term_states();
static bool                           should_add_bos_for_prompt(const std::string & formatted_text,
                                                                bool                has_chat_template,
                                                                bool                assume_start_of_context = false);
static jstring                        safe_new_string_utf(JNIEnv * env, const std::string & str);
static jstring                        safe_new_string_utf_nonnull(JNIEnv * env, const std::string & str);

static std::string cesu8_to_utf8(const char * raw) {
    if (!raw || !*raw) {
        return "";
    }
    const auto * bytes = reinterpret_cast<const unsigned char *>(raw);
    const size_t len   = std::strlen(raw);
    std::string  out;
    out.reserve(len);
    for (size_t i = 0; i < len;) {
        // Android ART Modified UTF-8 (CESU-8) encodes U+10000..U+10FFFF as two 3-byte UTF-8 surrogates:
        // high surrogate: ED [A0..AF] [80..BF], low surrogate: ED [B0..BF] [80..BF]
        if (i + 5 < len && bytes[i] == 0xED && (bytes[i + 1] & 0xF0) == 0xA0 && (bytes[i + 2] & 0xC0) == 0x80 &&
            bytes[i + 3] == 0xED && (bytes[i + 4] & 0xF0) == 0xB0 && (bytes[i + 5] & 0xC0) == 0x80) {
            const uint32_t hi = 0xD800u | (static_cast<uint32_t>(bytes[i + 1] & 0x0Fu) << 6) |
                                static_cast<uint32_t>(bytes[i + 2] & 0x3Fu);
            const uint32_t lo = 0xDC00u | (static_cast<uint32_t>(bytes[i + 4] & 0x0Fu) << 6) |
                                static_cast<uint32_t>(bytes[i + 5] & 0x3Fu);
            const uint32_t cp = 0x10000u + (((hi - 0xD800u) << 10) | (lo - 0xDC00u));
            out.push_back(static_cast<char>(0xF0u | ((cp >> 18) & 0x07u)));
            out.push_back(static_cast<char>(0x80u | ((cp >> 12) & 0x3Fu)));
            out.push_back(static_cast<char>(0x80u | ((cp >> 6) & 0x3Fu)));
            out.push_back(static_cast<char>(0x80u | (cp & 0x3Fu)));
            i += 6;
            continue;
        }
        // Modified UTF-8 encodes NUL (U+0000) as C0 80
        if (i + 1 < len && bytes[i] == 0xC0 && bytes[i + 1] == 0x80) {
            i += 2;
            continue;
        }
        out.push_back(static_cast<char>(bytes[i]));
        ++i;
    }
    return out;
}

static std::string jstring_to_utf8(JNIEnv * env, jstring jstr) {
    if (!env || !jstr) {
        return "";
    }
    const char * raw = env->GetStringUTFChars(jstr, nullptr);
    if (!raw) {
        if (env->ExceptionCheck()) {
            env->ExceptionClear();
        }
        return "";
    }
    std::string utf8 = cesu8_to_utf8(raw);
    env->ReleaseStringUTFChars(jstr, raw);
    return utf8;
}

static bool template_source_has_reasoning_tags(const std::string & src) {
    if (src.empty()) {
        return false;
    }
    static const char * const k_markers[] = {
        "reasoning_content",
        "enable_thinking",
        "reasoning_effort",
        "<|channel>thought",
        "<|channel|>thought",
        "<|channel|>analysis<|message|>",
        "<|think|>",
        "[THINK]",
        "[THOUGHT]",
        "[REASONING]",
        "<think>",
        "<thought>",
        "<thinking>",
        "<reasoning>",
        "<reason>",
        "<reflection>",
        "<analysis>",
        "<scratchpad>",
        "<inner_monologue>",
        "<|thought|>",
        "<|reasoning|>",
        "<start_of_thought>",
        "<|begin_of_thought|>",
        "<｜begin▁of▁thought｜>",
        "<|im_start|>thought",
        "<seed:think>",
        "<channel>thought",
    };
    for (const char * marker : k_markers) {
        if (src.find(marker) != std::string::npos) {
            return true;
        }
    }
    return false;
}

static std::string extract_template_default_reasoning_system_prompt() {
    if (!g_chat_templates || !g_chat_templates.get()) {
        return "";
    }
    const std::string src = common_chat_templates_source(g_chat_templates.get());
    if (src.empty()) {
        return "";
    }
    const size_t key_pos = src.find("default_system_message");
    if (key_pos == std::string::npos) {
        return "";
    }
    const size_t eq_pos = src.find('=', key_pos + 22);
    if (eq_pos == std::string::npos || eq_pos - key_pos > 40) {
        return "";
    }
    size_t pos = eq_pos + 1;
    while (pos < src.size() && (src[pos] == ' ' || src[pos] == '\t' || src[pos] == '\r' || src[pos] == '\n')) {
        ++pos;
    }
    if (pos >= src.size() || (src[pos] != '\'' && src[pos] != '"')) {
        return "";
    }
    const char  quote = src[pos++];
    std::string out;
    out.reserve(512);
    while (pos < src.size()) {
        const char c = src[pos++];
        if (c == '\\' && pos < src.size()) {
            const char esc = src[pos++];
            switch (esc) {
                case 'n':
                    out.push_back('\n');
                    break;
                case 'r':
                    out.push_back('\r');
                    break;
                case 't':
                    out.push_back('\t');
                    break;
                case '\\':
                    out.push_back('\\');
                    break;
                case '\'':
                    out.push_back('\'');
                    break;
                case '"':
                    out.push_back('"');
                    break;
                default:
                    out.push_back(esc);
                    break;
            }
        } else if (c == quote) {
            break;
        } else {
            out.push_back(c);
        }
    }
    if (out.empty() || !template_source_has_reasoning_tags(out)) {
        return "";
    }
    return out;
}

static std::string merge_template_default_reasoning_system_prompt(const std::string & user_system_prompt) {
    if (!g_enable_thinking) {
        return user_system_prompt;
    }
    const std::string default_sys = extract_template_default_reasoning_system_prompt();
    if (default_sys.empty()) {
        return user_system_prompt;
    }
    if (!user_system_prompt.empty() &&
        user_system_prompt.find(default_sys.substr(0, std::min<size_t>(40, default_sys.size()))) != std::string::npos) {
        return user_system_prompt;
    }
    if (user_system_prompt.empty()) {
        return default_sys;
    }
    return default_sys + "\n\n" + user_system_prompt;
}

static void detach_backend_sampler_if_attached() {
    if (g_context) {
        llama_set_sampler(g_context, 0, nullptr);
    }
}

static void attach_backend_sampler_if_enabled(bool turn_has_constraints = false) {
    const bool has_constraints = turn_has_constraints || g_sampler_has_turn_constraints || !g_sparams.grammar.empty() ||
                                 g_sparams.reasoning_budget_tokens >= 0;
    if (g_context && g_sampler && g_sparams.backend_sampling && !has_constraints) {
        struct llama_sampler * raw_smpl = common_sampler_get(g_sampler);
        if (raw_smpl) {
            llama_set_sampler(g_context, 0, raw_smpl);
            return;
        }
    }
    if (g_context) {
        llama_set_sampler(g_context, 0, nullptr);
    }
}

static void apply_ignore_eos_logit_bias(common_params_sampling & sparams) {
    if (!sparams.ignore_eos || !g_model) {
        return;
    }
    const llama_vocab * vocab = llama_model_get_vocab(g_model);
    if (!vocab) {
        return;
    }
    const int32_t n_vocab = llama_vocab_n_tokens(vocab);
    for (int32_t id = 0; id < n_vocab; ++id) {
        if (llama_vocab_is_eog(vocab, id)) {
            bool already_biased = false;
            for (const auto & lb : sparams.logit_bias) {
                if (lb.token == id) {
                    already_biased = true;
                    break;
                }
            }
            if (!already_biased) {
                sparams.logit_bias.push_back({ id, -INFINITY });
            }
        }
    }
}

static void invalidate_turn_prompt_caches() {
    g_prev_prompt_cache   = CachedRenderedHistory{};
    g_anchor_base_cache   = CachedAnchorBase{};
    g_native_parser_cache = CachedNativeParser{};
}

static void invalidate_model_template_caches() {
    g_caps_cache                       = CachedModelCapabilities{};
    g_prev_prompt_cache                = CachedRenderedHistory{};
    g_anchor_base_cache                = CachedAnchorBase{};
    g_native_parser_cache              = CachedNativeParser{};
    g_sampling_cache                   = CachedSamplingState{};
    g_sampler_has_turn_constraints     = false;
    g_vocab_and_template_probed        = false;
    g_template_honors_disable_thinking = false;
    g_preserved_token_ids.clear();
    g_additional_stop_token_ids.clear();
    g_template_stop_strings.clear();
    g_active_cot_rules.clear();
    g_cot_lead_chars = "<[";
}

static std::mutex                      g_event_queue_mutex;
static std::deque<NativeSemanticEvent> g_native_event_queue;
static std::atomic<int64_t>            g_current_generation_id{ 0 };
static std::atomic<int64_t>            g_event_sequence_counter{ 0 };
static std::atomic<bool>               g_cancel_requested{ false };

// Returns the length of the valid UTF-8 prefix of str that ends on a complete character boundary.
// Any trailing incomplete multi-byte sequence (1 to 3 bytes) is preserved in pending buffer.
static size_t utf8_valid_complete_prefix_len(const std::string & str) {
    if (str.empty()) {
        return 0;
    }
    size_t len        = str.size();
    size_t i          = 0;
    size_t last_valid = 0;
    while (i < len) {
        unsigned char c       = static_cast<unsigned char>(str[i]);
        size_t        seq_len = 0;
        if ((c & 0x80) == 0x00) {
            seq_len = 1;
        } else if ((c & 0xE0) == 0xC0) {
            seq_len = 2;
        } else if ((c & 0xF0) == 0xE0) {
            seq_len = 3;
        } else if ((c & 0xF8) == 0xF0) {
            seq_len = 4;
        } else {
            // Stray continuation byte or unexpected prefix: skip 1 byte to keep stream moving
            i++;
            last_valid = i;
            continue;
        }

        if (i + seq_len <= len) {
            bool ok = true;
            for (size_t j = 1; j < seq_len; ++j) {
                if ((static_cast<unsigned char>(str[i + j]) & 0xC0) != 0x80) {
                    ok = false;
                    break;
                }
            }
            if (ok) {
                i += seq_len;
                last_valid = i;
            } else {
                i++;
                last_valid = i;
            }
        } else {
            // Incomplete trailing sequence: stop here and wait for next token piece
            break;
        }
    }
    return last_valid;
}

static std::string g_pending_content_utf8;
static std::string g_pending_reasoning_utf8;
static int32_t     get_current_generated_tokens_count();

static void push_semantic_event(NativeSemanticEvent ev) {
    std::lock_guard<std::mutex> lock(g_event_queue_mutex);

    if (ev.type == SemanticEventType::CONTENT_DELTA) {
        if (!g_pending_content_utf8.empty()) {
            ev.text = g_pending_content_utf8 + ev.text;
            g_pending_content_utf8.clear();
        }
        if (ev.text.empty()) {
            return;
        }
        size_t valid_len = utf8_valid_complete_prefix_len(ev.text);
        if (valid_len < ev.text.size()) {
            g_pending_content_utf8 = ev.text.substr(valid_len);
            ev.text                = ev.text.substr(0, valid_len);
        }
        if (ev.text.empty()) {
            return;
        }
    } else if (ev.type == SemanticEventType::REASONING_DELTA) {
        if (!g_pending_reasoning_utf8.empty()) {
            ev.text = g_pending_reasoning_utf8 + ev.text;
            g_pending_reasoning_utf8.clear();
        }
        if (ev.text.empty()) {
            return;
        }
        size_t valid_len = utf8_valid_complete_prefix_len(ev.text);
        if (valid_len < ev.text.size()) {
            g_pending_reasoning_utf8 = ev.text.substr(valid_len);
            ev.text                  = ev.text.substr(0, valid_len);
        }
        if (ev.text.empty()) {
            return;
        }
    }

    ev.generation_id = g_current_generation_id.load();
    ev.sequence      = g_event_sequence_counter.fetch_add(1);
    if (ev.type == SemanticEventType::CONTENT_DELTA || ev.type == SemanticEventType::REASONING_DELTA) {
        ev.generated_tokens = get_current_generated_tokens_count();
    }

    if (!g_native_event_queue.empty()) {
        auto & last = g_native_event_queue.back();
        if (last.generation_id == ev.generation_id) {
            if (last.type == SemanticEventType::REASONING_DELTA && ev.type == SemanticEventType::REASONING_DELTA) {
                last.text += ev.text;
                last.generated_tokens = ev.generated_tokens;
                return;
            }
            if (last.type == SemanticEventType::CONTENT_DELTA && ev.type == SemanticEventType::CONTENT_DELTA) {
                last.text += ev.text;
                last.generated_tokens = ev.generated_tokens;
                return;
            }
        }
    }
    if (g_native_event_queue.size() >= 8192) {
        // Protect critical events from being dropped: only drop oldest text deltas
        bool dropped = false;
        for (auto it = g_native_event_queue.begin(); it != g_native_event_queue.end(); ++it) {
            if (it->type == SemanticEventType::REASONING_DELTA || it->type == SemanticEventType::CONTENT_DELTA) {
                g_native_event_queue.erase(it);
                dropped = true;
                break;
            }
        }
        if (!dropped) {
            g_native_event_queue.pop_front();
        }
    }
    g_native_event_queue.push_back(std::move(ev));
}

static void clear_native_event_queue() {
    std::lock_guard<std::mutex> lock(g_event_queue_mutex);
    g_native_event_queue.clear();
}

static std::string generate_tool_call_id() {
    static std::atomic<uint64_t> counter{ 1 };
    char                         buf[32];
    snprintf(buf, sizeof(buf), "call_%08llx", (unsigned long long) counter.fetch_add(1));
    return std::string(buf);
}

static void erase_all_occurrences(std::string & text, const std::string & tag) {
    if (tag.empty()) {
        return;
    }
    size_t pos = text.find(tag);
    while (pos != std::string::npos) {
        text.erase(pos, tag.length());
        pos = text.find(tag, pos);
    }
}

// Per-generation parse state for incremental, model-aware parsing
struct GenerationParseState {
    std::string               generated_raw_text;
    common_chat_msg           last_parsed_msg;
    common_chat_parser_params parser_params;
    bool                      has_parser   = false;
    bool                      in_reasoning = false;
    std::vector<std::string>  active_cot_close_tags;
    size_t                    emitted_raw_pos = 0;

    // Semantic channels accumulated for this generation
    std::string                        accumulated_reasoning;
    std::string                        accumulated_content;
    std::vector<common_chat_tool_call> accumulated_tool_calls;
    std::vector<std::string>           tool_call_ids_cache;

    // Metrics & finalization guard
    int32_t      prompt_tokens_count    = 0;
    int32_t      generated_tokens_count = 0;
    bool         is_active              = false;
    bool         is_finalized           = false;
    FinishReason finish_reason          = FinishReason::EOG;

    // Fast-path parsing optimizations
    int32_t tokens_since_last_parse = 0;
    bool    in_structural_sequence  = false;

    void apply_diff(const common_chat_msg_diff & diff) {
        if (!g_enable_thinking) {
            // When thinking is disabled (OFF), reasoning cannot exist. Treat any parsed text strictly as content.
            std::string text = diff.reasoning_content_delta + diff.content_delta;
            if (!text.empty()) {
                const auto & rules = !g_active_cot_rules.empty() ? g_active_cot_rules : get_universal_cot_catalog();
                for (const auto & rule : rules) {
                    erase_all_occurrences(text, rule.open_tag);
                    for (const auto & close_tag : rule.close_tags) {
                        erase_all_occurrences(text, close_tag);
                    }
                }
                if (!text.empty()) {
                    accumulated_content += text;
                    NativeSemanticEvent ev;
                    ev.type = SemanticEventType::CONTENT_DELTA;
                    ev.text = text;
                    push_semantic_event(std::move(ev));
                    LOGd("[SEMANTIC CONTENT DELTA (OFF)]: `%s`", text.c_str());
                }
            }
            if (diff.tool_call_index != std::string::npos) {
                if (accumulated_tool_calls.size() <= diff.tool_call_index) {
                    accumulated_tool_calls.resize(diff.tool_call_index + 1);
                }
                auto & tc = accumulated_tool_calls[diff.tool_call_index];
                if (!diff.tool_call_delta.id.empty()) {
                    tc.id = diff.tool_call_delta.id;
                }
                if (!diff.tool_call_delta.name.empty()) {
                    tc.name = diff.tool_call_delta.name;
                }
                tc.arguments += diff.tool_call_delta.arguments;

                NativeSemanticEvent ev;
                ev.type            = SemanticEventType::TOOL_CALL_DELTA;
                ev.tool_index      = static_cast<int32_t>(diff.tool_call_index);
                ev.tool_id         = diff.tool_call_delta.id;
                ev.tool_name       = diff.tool_call_delta.name;
                ev.tool_args_delta = diff.tool_call_delta.arguments;
                push_semantic_event(std::move(ev));
            }
            return;
        }

        if (!diff.reasoning_content_delta.empty()) {
            in_reasoning = true;
            accumulated_reasoning += diff.reasoning_content_delta;
            NativeSemanticEvent ev;
            ev.type = SemanticEventType::REASONING_DELTA;
            ev.text = diff.reasoning_content_delta;
            push_semantic_event(std::move(ev));
            LOGd("[SEMANTIC REASONING DELTA]: `%s`", diff.reasoning_content_delta.c_str());
        }
        if (!diff.content_delta.empty()) {
            in_reasoning     = false;
            std::string text = diff.content_delta;
            // Guard: If a closing think tag leaked into content delta at transition boundary, strip it
            if (!active_cot_close_tags.empty()) {
                for (const auto & close_tag : active_cot_close_tags) {
                    erase_all_occurrences(text, close_tag);
                }
            } else if (!g_active_cot_rules.empty()) {
                for (const auto & rule : g_active_cot_rules) {
                    for (const auto & close_tag : rule.close_tags) {
                        erase_all_occurrences(text, close_tag);
                    }
                }
            } else {
                erase_all_occurrences(text, "</think>");
            }
            if (!text.empty()) {
                accumulated_content += text;
                NativeSemanticEvent ev;
                ev.type = SemanticEventType::CONTENT_DELTA;
                ev.text = text;
                push_semantic_event(std::move(ev));
                LOGd("[SEMANTIC CONTENT DELTA]: `%s`", text.c_str());
            }
        }
        if (diff.tool_call_index != std::string::npos) {
            if (accumulated_tool_calls.size() <= diff.tool_call_index) {
                accumulated_tool_calls.resize(diff.tool_call_index + 1);
            }
            auto & tc = accumulated_tool_calls[diff.tool_call_index];
            if (!diff.tool_call_delta.id.empty()) {
                tc.id = diff.tool_call_delta.id;
            }
            if (!diff.tool_call_delta.name.empty()) {
                tc.name = diff.tool_call_delta.name;
            }
            tc.arguments += diff.tool_call_delta.arguments;

            NativeSemanticEvent ev;
            ev.type            = SemanticEventType::TOOL_CALL_DELTA;
            ev.tool_index      = static_cast<int32_t>(diff.tool_call_index);
            ev.tool_id         = diff.tool_call_delta.id;
            ev.tool_name       = diff.tool_call_delta.name;
            ev.tool_args_delta = diff.tool_call_delta.arguments;
            push_semantic_event(std::move(ev));
            LOGi("[SEMANTIC TOOL_CALL DELTA]: index=%zu, id='%s', name='%s', args_delta='%s'", diff.tool_call_index,
                 diff.tool_call_delta.id.c_str(), diff.tool_call_delta.name.c_str(),
                 diff.tool_call_delta.arguments.c_str());
        }
    }

    void reset() {
        generated_raw_text.clear();
        last_parsed_msg = common_chat_msg();
        parser_params   = common_chat_parser_params();
        has_parser      = false;
        in_reasoning    = false;
        active_cot_close_tags.clear();
        emitted_raw_pos = 0;
        accumulated_reasoning.clear();
        accumulated_content.clear();
        accumulated_tool_calls.clear();
        tool_call_ids_cache.clear();
        prompt_tokens_count     = 0;
        generated_tokens_count  = 0;
        is_active               = false;
        is_finalized            = false;
        finish_reason           = FinishReason::EOG;
        tokens_since_last_parse = 0;
        in_structural_sequence  = false;
    }
};

static GenerationParseState g_gen_state;

static int32_t get_current_generated_tokens_count() {
    return g_gen_state.generated_tokens_count;
}

// Tracking of conversational messages and tokens position
constexpr const char * ROLE_SYSTEM    = "system";
constexpr const char * ROLE_USER      = "user";
constexpr const char * ROLE_ASSISTANT = "assistant";
constexpr const char * ROLE_TOOL      = "tool";

static std::vector<common_chat_msg> chat_msgs;
static llama_pos                    system_prompt_position;
static llama_pos                    current_position;
static llama_pos     stop_generation_position = 0;  // Declared early so KV shift/rebuild adjust remaining budget
static std::string   g_deferred_system_prompt;
static size_t        g_kv_cells_used = 0;           // Skutečné obsazení KV cache buněk (nezávislé na M-RoPE pozicích)
static mtmd_bitmap * g_cached_bmp    = nullptr;
static std::vector<llama_token> g_cached_tokens;    // Prefix Cache token tracking

extern "C" JNIEXPORT void JNICALL Java_com_arm_aichat_internal_InferenceEngineImpl_init(JNIEnv * env,
                                                                                        jobject  thiz,
                                                                                        jstring  nativeLibDir) {
    (void) thiz;  // Unused
    // Set llama log handler to Android
    llama_log_set(aichat_android_log_callback, nullptr);

    if (nativeLibDir != nullptr) {
        std::string lib_dir = jstring_to_utf8(env, nativeLibDir);
        if (!lib_dir.empty()) {
            LOGi("Loading backends from %s", lib_dir.c_str());
            g_native_lib_dir = lib_dir;

            // Set FastRPC path for DSP skeleton (libggml-htp-v75.so / libggml-htp-v73.so) while preserving vendor/system RFSA paths
            // Note: Do NOT set GGML_BACKEND_PATH to a directory — ggml expects a single .so file path in GGML_BACKEND_PATH.
            const std::string adsp_path = lib_dir + ";/vendor/lib/rfsa/adsp;/system/lib/rfsa/adsp;/dsp";
            setenv("ADSP_LIBRARY_PATH", adsp_path.c_str(), 1);

            // Adreno 750 Vulkan stability & numerical accuracy defaults (must be set before libggml-vulkan.so is loaded):
            // - GGML_VK_DISABLE_COOPMAT=1 avoids Adreno 0762 cooperative_matrix GPU hang -> VK_ERROR_DEVICE_LOST
            // - GGML_VK_DISABLE_INTEGER_DOT_PRODUCT=1 avoids Adreno 0762 SPIR-V dot4_i8 miscompilation -> garbled Q4_0 output
            // - GGML_VK_DISABLE_F16=1 avoids Adreno 0762 FP16 accumulator overflow/NaN in quantized matmul shaders
            // - GGML_VK_DISABLE_FUSION=1 / GGML_VK_DISABLE_MMVQ=1 avoids Adreno 0762 subgroup fusion/MMVQ miscompilation
            // - GGML_VK_DISABLE_ASYNC=1 / GGML_VK_SERIALIZE_SUBMISSIONS=1 ensures barrier coherency across Adreno command buffers
            // - GGML_VK_MAX_NODES_PER_SUBMIT=64 splits large prefill submits below Android KGSL 5s TDR watchdog
            setenv("GGML_VK_DISABLE_COOPMAT", "1", 0);
            setenv("GGML_VK_MAX_NODES_PER_SUBMIT", "64", 0);

            ggml_backend_load_all_from_path(lib_dir.c_str());
        }
    } else {
        LOGw("init called with null nativeLibDir; skipping ggml_backend_load_all_from_path.");
    }

    // Initialize backends
    llama_backend_init();
    LOGi("Backend initiated; Build: %s; Log handler set.", llama_build_info());
}

extern "C" JNIEXPORT jstring JNICALL
Java_com_arm_aichat_internal_InferenceEngineImpl_getLoadedBackendInfoNative(JNIEnv * env, jobject thiz) {
    (void) thiz;
    std::lock_guard<std::recursive_mutex> cfg_lock(g_config_mutex);
    return safe_new_string_utf_nonnull(env, g_active_backend_info);
}

static bool model_has_embedded_mm_metadata(const llama_model * model) {
    if (!model) {
        return false;
    }
    const int32_t meta_count = llama_model_meta_count(model);
    char          key_buf[256];
    for (int32_t i = 0; i < meta_count; ++i) {
        const int32_t len = llama_model_meta_key_by_index(model, i, key_buf, sizeof(key_buf));
        if (len > 0) {
            if (strncmp(key_buf, "clip.", 5) == 0 || strncmp(key_buf, "vision.", 7) == 0 ||
                strstr(key_buf, ".vision") != nullptr || strstr(key_buf, ".mmproj") != nullptr ||
                strstr(key_buf, "has_vision_encoder") != nullptr || strstr(key_buf, "has_audio_encoder") != nullptr) {
                return true;
            }
        }
    }
    return false;
}

extern "C" JNIEXPORT jlong JNICALL
Java_com_arm_aichat_internal_InferenceEngineImpl_loadModelNative(JNIEnv * env,
                                                                 jobject  thiz,
                                                                 jstring  jmodel_path,
                                                                 jstring  jprojector_path,
                                                                 jint     kvCacheMode,
                                                                 jboolean useMmap,
                                                                 jint     nCtx,
                                                                 jint     nBatch,
                                                                 jint     nUBatch,
                                                                 jboolean flashAttn,
                                                                 jstring  jbackend,
                                                                 jboolean strictBackend,
                                                                 jint     gpuLayers,
                                                                 jint     hexagonVaLimitMb,
                                                                 jint     hexagonNdev,
                                                                 jint     kvReplayOnCtxResizeMode) {
    (void) thiz;
    std::lock_guard<std::recursive_mutex> cfg_lock(g_config_mutex);
    // Guard against double-load leaks if caller skipped unload()
    g_cancel_requested.store(false, std::memory_order_release);
    reset_long_term_states(false, false);
    reset_short_term_states();
    clear_native_event_queue();
    if (g_sampler) {
        detach_backend_sampler_if_attached();
        common_sampler_free(g_sampler);
        g_sampler = nullptr;
    }
    g_sparams.grammar              = common_grammar();
    g_sampler_has_turn_constraints = false;
    g_sampling_cache.valid         = false;
    if (g_batch_initialized) {
        llama_batch_free(g_batch);
        g_batch             = {};
        g_batch_initialized = false;
    }
    if (g_cached_bmp) {
        mtmd_bitmap_free(g_cached_bmp);
        g_cached_bmp = nullptr;
    }
    if (g_mtmd_ctx) {
        mtmd_free(g_mtmd_ctx);
        g_mtmd_ctx = nullptr;
    }
    if (g_context) {
        llama_free(g_context);
        g_context = nullptr;
    }
    if (g_model) {
        llama_model_free(g_model);
        g_model = nullptr;
    }
    g_chat_templates.reset();
    g_configured_tools.clear();
    g_last_tools_json_str.clear();
    g_tool_policy = NativeToolChoice::AUTO;
    invalidate_model_template_caches();
    g_custom_cot_rules_set          = false;
    g_allow_native_tools_hint       = true;
    g_tools_active_hint             = false;
    g_tool_markup_format            = 0;
    llama_model_params model_params = llama_model_default_params();

    // Dynamické nastavení MMAP dle volby uživatele
    // true = OS spravuje paměť (Doporučeno)
    // false = Načíst celý model do RAM (Riziko OOM)
    model_params.load_mode = useMmap ? LLAMA_LOAD_MODE_MMAP : LLAMA_LOAD_MODE_NONE;

    // Store KV Cache mode for context initialization
    g_kv_cache_mode                = kvCacheMode;
    g_n_ctx                        = nCtx;
    g_n_batch                      = nBatch;
    g_n_ubatch                     = nUBatch;
    g_flash_attn                   = (bool) flashAttn;
    g_hexagon_va_limit_mb          = static_cast<int>(hexagonVaLimitMb);
    g_hexagon_ndev                 = std::clamp(static_cast<int>(hexagonNdev), 1, 4);
    g_kv_replay_on_ctx_resize_mode = std::clamp(static_cast<int>(kvReplayOnCtxResizeMode), 0, 2);
    g_loaded_model_file_bytes      = 0;

    std::string backend_str = jstring_to_utf8(env, jbackend);
    if (backend_str.empty()) {
        backend_str = "cpu";
    }

    // Static vector to hold null-terminated devices pointer list for llama_model_params
    static std::vector<ggml_backend_dev_t> s_configured_devices;
    s_configured_devices.clear();

    static std::vector<void *> s_dlopen_backend_handles;
    auto                       load_backend_with_reg_fallback = [&](const std::string & so_path,
                                              const char *        reg_symbol) -> ggml_backend_reg_t {
        ggml_backend_reg_t reg = ggml_backend_load(so_path.c_str());
        if (reg) {
            return reg;
        }
        // Fallback for prebuilt backend .so compiled with GGML_BACKEND_DL=OFF (exports ggml_backend_*_reg instead of ggml_backend_init)
        void * handle = dlopen(so_path.c_str(), RTLD_NOW | RTLD_LOCAL);
        if (!handle) {
            LOGe("dlopen(%s) failed: %s", so_path.c_str(), dlerror());
            return nullptr;
        }
        using reg_fn_t = ggml_backend_reg_t (*)();
        auto reg_fn    = reinterpret_cast<reg_fn_t>(dlsym(handle, reg_symbol));
        if (!reg_fn) {
            LOGe("dlsym(%s, %s) failed: %s", so_path.c_str(), reg_symbol, dlerror());
            dlclose(handle);
            return nullptr;
        }
        reg = reg_fn();
        if (!reg) {
            LOGe("%s() returned nullptr for %s", reg_symbol, so_path.c_str());
            dlclose(handle);
            return nullptr;
        }
        ggml_backend_register(reg);
        s_dlopen_backend_handles.push_back(handle);
        LOGi("✅ Registered prebuilt backend via %s from %s (devs=%zu)", reg_symbol, so_path.c_str(),
                                   ggml_backend_reg_dev_count(reg));
        return reg;
    };

    constexpr jint GPU_LAYERS_OFF = -2;

    auto try_acquire_hexagon_reg = [&]() -> ggml_backend_reg_t {
        ggml_backend_reg_t reg = ggml_backend_reg_by_name("HTP");
        if (!reg) {
            reg = ggml_backend_reg_by_name("Hexagon");
        }
        if (!reg && !g_native_lib_dir.empty()) {
            const std::string hex_path = g_native_lib_dir + "/libggml-hexagon.so";
            if (access(hex_path.c_str(), F_OK) == 0) {
                LOGi("Attempting explicit load of %s", hex_path.c_str());
                reg = load_backend_with_reg_fallback(hex_path, "ggml_backend_hexagon_reg");
            }
        }
        return reg;
    };

    auto select_cpu_backend = [&](const char * suffix) -> bool {
        model_params.n_gpu_layers  = 0;
        ggml_backend_dev_t cpu_dev = ggml_backend_dev_by_type(GGML_BACKEND_DEVICE_TYPE_CPU);
        if (cpu_dev) {
            s_configured_devices.push_back(cpu_dev);
            s_configured_devices.push_back(nullptr);
            model_params.devices = s_configured_devices.data();
        }
        g_active_backend_info = std::string("CPU (ARM64 Neon)") + (suffix ? suffix : "");
        return cpu_dev != nullptr;
    };

    LOGi("🎯 [Backend Engine] Selected backend: '%s' (strict: %d, gpuLayers: %d)", backend_str.c_str(),
         (int) strictBackend, gpuLayers);

    if (backend_str == "opencl" || backend_str == "vulkan") {
        if (gpuLayers == GPU_LAYERS_OFF) {
            // Explicit "off" (-2): disable GPU offload completely; try NPU (Hexagon/HTP) first, then CPU
            LOGi("ℹ️ gpuLayers=OFF (-2) requested with '%s': bypassing GPU, probing NPU (Hexagon/HTP)...",
                 backend_str.c_str());
            ggml_backend_reg_t reg_hex = try_acquire_hexagon_reg();
            if (reg_hex && ggml_backend_reg_dev_count(reg_hex) > 0) {
                ggml_backend_dev_t hex_dev = ggml_backend_reg_dev_get(reg_hex, 0);
                s_configured_devices.push_back(hex_dev);
                s_configured_devices.push_back(nullptr);
                model_params.devices      = s_configured_devices.data();
                model_params.n_gpu_layers = -1;
                g_active_backend_info =
                    std::string("Hexagon/HTP (") + ggml_backend_dev_name(hex_dev) + ") [NPU Fallback (GPU off)]";
                LOGi("✅ NPU fallback selected (GPU off): %s (%s), offloading %d layers",
                     ggml_backend_dev_name(hex_dev), ggml_backend_dev_description(hex_dev), model_params.n_gpu_layers);
            } else {
                LOGw("NPU unavailable while gpuLayers=OFF (-2); falling back to CPU with n_gpu_layers=0");
                const bool has_cpu = select_cpu_backend(" [GPU off]");
                if (!has_cpu && strictBackend) {
                    return (backend_str == "vulkan") ? 103 : 101;
                }
            }
        } else {
            const int resolved_layers = (gpuLayers == 0) ? -1 : static_cast<int>(gpuLayers);
            if (backend_str == "opencl") {
                ggml_backend_reg_t reg_ocl = ggml_backend_reg_by_name("OpenCL");
                if (!reg_ocl && !g_native_lib_dir.empty()) {
                    const std::string ocl_path = g_native_lib_dir + "/libggml-opencl.so";
                    if (access(ocl_path.c_str(), F_OK) == 0) {
                        LOGi("Attempting explicit load of %s", ocl_path.c_str());
                        reg_ocl = load_backend_with_reg_fallback(ocl_path, "ggml_backend_opencl_reg");
                    }
                }
                if (!reg_ocl || ggml_backend_reg_dev_count(reg_ocl) == 0) {
                    LOGe("❌ OpenCL backend unavailable! (reg_ocl=%p)", reg_ocl);
                    if (strictBackend) {
                        return 101;  // ERR_OPENCL_UNAVAILABLE
                    }
                    LOGw("Strict mode disabled: falling back to CPU");
                    select_cpu_backend(" [Fallback]");
                } else {
                    ggml_backend_dev_t ocl_dev = ggml_backend_reg_dev_get(reg_ocl, 0);
                    s_configured_devices.push_back(ocl_dev);
                    s_configured_devices.push_back(nullptr);
                    model_params.devices      = s_configured_devices.data();
                    model_params.n_gpu_layers = resolved_layers;
                    g_active_backend_info     = std::string("OpenCL (") + ggml_backend_dev_name(ocl_dev) + ")";
                    LOGi("✅ OpenCL device selected: %s (%s), offloading %d layers", ggml_backend_dev_name(ocl_dev),
                         ggml_backend_dev_description(ocl_dev), model_params.n_gpu_layers);
                }
            } else {
                ggml_backend_reg_t reg_vk = ggml_backend_reg_by_name("Vulkan");
                if (!reg_vk && !g_native_lib_dir.empty()) {
                    const std::string vk_path = g_native_lib_dir + "/libggml-vulkan.so";
                    if (access(vk_path.c_str(), F_OK) == 0) {
                        LOGi("Attempting explicit load of %s", vk_path.c_str());
                        reg_vk = load_backend_with_reg_fallback(vk_path, "ggml_backend_vk_reg");
                    }
                }
                if (!reg_vk || ggml_backend_reg_dev_count(reg_vk) == 0) {
                    LOGe("❌ Vulkan backend unavailable! (reg_vk=%p)", reg_vk);
                    if (strictBackend) {
                        return 103;  // ERR_VULKAN_UNAVAILABLE
                    }
                    LOGw("Strict mode disabled: falling back to CPU");
                    select_cpu_backend(" [Fallback]");
                } else {
                    ggml_backend_dev_t vk_dev = nullptr;
                    const size_t       n_devs = ggml_backend_reg_dev_count(reg_vk);
                    for (size_t i = 0; i < n_devs; ++i) {
                        ggml_backend_dev_t cand = ggml_backend_reg_dev_get(reg_vk, i);
                        if (cand && (ggml_backend_dev_type(cand) == GGML_BACKEND_DEVICE_TYPE_GPU ||
                                     ggml_backend_dev_type(cand) == GGML_BACKEND_DEVICE_TYPE_IGPU)) {
                            vk_dev = cand;
                            break;
                        }
                    }
                    if (!vk_dev) {
                        vk_dev = ggml_backend_reg_dev_get(reg_vk, 0);
                    }
                    s_configured_devices.push_back(vk_dev);
                    s_configured_devices.push_back(nullptr);
                    model_params.devices      = s_configured_devices.data();
                    model_params.n_gpu_layers = resolved_layers;
                    g_active_backend_info     = std::string("Vulkan (") + ggml_backend_dev_name(vk_dev) + ")";
                    LOGi("✅ Vulkan device selected: %s (%s), offloading %d layers", ggml_backend_dev_name(vk_dev),
                         ggml_backend_dev_description(vk_dev), model_params.n_gpu_layers);
                }
            }
        }
    } else if (backend_str == "hexagon" || backend_str == "htp_v75" || backend_str == "htp_v73") {
        if (gpuLayers == GPU_LAYERS_OFF) {
            LOGi("ℹ️ gpuLayers=OFF (-2) requested with '%s': using CPU with n_gpu_layers=0", backend_str.c_str());
            const bool has_cpu = select_cpu_backend(" [Offload off]");
            if (!has_cpu && strictBackend) {
                return 102;
            }
        } else {
            static std::string s_hexagon_initialized_arch;
            const std::string  requested_hex_arch = (backend_str == "htp_v75") ? "v75" :
                                                    (backend_str == "htp_v73") ? "v73" :
                                                                                 "auto";
            const int          requested_hex_ndev = std::clamp(g_hexagon_ndev, 1, 4);
            const bool         hex_already_registered =
                (ggml_backend_reg_by_name("HTP") != nullptr || ggml_backend_reg_by_name("Hexagon") != nullptr);

            if (hex_already_registered) {
                if (!s_hexagon_initialized_arch.empty() && s_hexagon_initialized_arch != requested_hex_arch) {
                    LOGw(
                        "⚠️ Hexagon HTP singleton already initialized with arch=%s; switching to %s requires process "
                        "restart (ggml_hexagon_init runs once per process).",
                        s_hexagon_initialized_arch.c_str(), requested_hex_arch.c_str());
                }
                if (s_hexagon_initialized_ndev > 0 && s_hexagon_initialized_ndev != requested_hex_ndev) {
                    LOGw(
                        "⚠️ Hexagon HTP singleton already initialized with ndev=%d; switching to ndev=%d requires "
                        "process restart (ggml_hexagon_init runs once per process).",
                        s_hexagon_initialized_ndev, requested_hex_ndev);
                }
            } else {
                setenv("GGML_HEXAGON_NDEV", std::to_string(requested_hex_ndev).c_str(), 1);
                LOGi("Configured GGML_HEXAGON_NDEV=%d", requested_hex_ndev);
                if (backend_str == "htp_v75") {
                    setenv("GGML_HEXAGON_ARCH", "75", 1);
                    LOGi("Configured GGML_HEXAGON_ARCH=75 (Snapdragon 8 Gen 3)");
                } else if (backend_str == "htp_v73") {
                    setenv("GGML_HEXAGON_ARCH", "73", 1);
                    LOGi("Configured GGML_HEXAGON_ARCH=73 (Snapdragon 8 Gen 1/2)");
                } else {
                    unsetenv("GGML_HEXAGON_ARCH");
                    LOGi("Cleared GGML_HEXAGON_ARCH -> using hardware autodetection (htpdrv_get_arch)");
                }
            }

            ggml_backend_reg_t reg_hex = try_acquire_hexagon_reg();
            if (!reg_hex || ggml_backend_reg_dev_count(reg_hex) == 0) {
                LOGe("❌ Hexagon/HTP backend unavailable! (reg_hex=%p)", reg_hex);
                if (strictBackend) {
                    return 102;  // ERR_HEXAGON_UNAVAILABLE
                }
                LOGw("Strict mode disabled: falling back to CPU");
                select_cpu_backend(" [Fallback]");
            } else {
                if (s_hexagon_initialized_arch.empty()) {
                    s_hexagon_initialized_arch = requested_hex_arch;
                }
                const size_t avail_hex_devs = ggml_backend_reg_dev_count(reg_hex);
                const size_t use_hex_devs   = std::min<size_t>(avail_hex_devs, static_cast<size_t>(requested_hex_ndev));
                if (s_hexagon_initialized_ndev <= 0) {
                    s_hexagon_initialized_ndev = static_cast<int>(use_hex_devs);
                }
                for (size_t i = 0; i < use_hex_devs; ++i) {
                    s_configured_devices.push_back(ggml_backend_reg_dev_get(reg_hex, i));
                }
                s_configured_devices.push_back(nullptr);
                model_params.devices         = s_configured_devices.data();
                model_params.n_gpu_layers    = (gpuLayers == 0) ? -1 : static_cast<int>(gpuLayers);
                ggml_backend_dev_t first_dev = ggml_backend_reg_dev_get(reg_hex, 0);
                std::string        dev_label = ggml_backend_dev_name(first_dev);
                if (use_hex_devs > 1) {
                    ggml_backend_dev_t last_dev = ggml_backend_reg_dev_get(reg_hex, use_hex_devs - 1);
                    dev_label += "..";
                    dev_label += ggml_backend_dev_name(last_dev);
                }
                g_active_backend_info =
                    std::string("Hexagon/HTP (") + dev_label + ", arch=" + s_hexagon_initialized_arch + ")";
                LOGi("✅ Hexagon/HTP device(s) selected: %s (count=%zu/%zu, arch=%s), offloading %d layers",
                     dev_label.c_str(), use_hex_devs, avail_hex_devs, s_hexagon_initialized_arch.c_str(),
                     model_params.n_gpu_layers);
            }
        }
    } else {
        // CPU
        select_cpu_backend(nullptr);
        LOGi("✅ CPU backend selected explicitly. n_gpu_layers = 0");
    }

    const std::string model_path_str = jstring_to_utf8(env, jmodel_path);
    if (model_path_str.empty()) {
        LOGe("❌ loadModelNative called with null/empty model_path!");
        return 1;
    }
    struct stat model_st{};
    if (stat(model_path_str.c_str(), &model_st) == 0 && model_st.st_size > 0) {
        g_loaded_model_file_bytes = static_cast<size_t>(model_st.st_size);
    }
    LOGd("%s: Loading model from: \n%s (MMAP: %d, KV-Mode: %d, ctx: %d, batch: %d, ubatch: %d, FA: %d)\n", __func__,
         model_path_str.c_str(), (int) useMmap, kvCacheMode, nCtx, nBatch, nUBatch, (int) flashAttn);

    auto * model = llama_model_load_from_file(model_path_str.c_str(), model_params);
    if (!model) {
        LOGe("❌ llama_model_load_from_file failed on backend '%s'!", backend_str.c_str());
        return 1;
    }
    g_model = model;

    invalidate_model_template_caches();
    g_chat_templates = common_chat_templates_init(g_model, "");
    if (g_chat_templates && g_chat_templates.get()) {
        const bool        has_explicit = common_chat_templates_was_explicit(g_chat_templates.get());
        const std::string tmpl_src     = common_chat_templates_source(g_chat_templates.get());
        std::string       tool_src;
        try {
            tool_src = common_chat_templates_source(g_chat_templates.get(), "tool_use");
        } catch (...) {
        }
        const std::string & effective_src = !tool_src.empty() ? tool_src : tmpl_src;
        g_use_jinja                       = has_explicit &&
                      (effective_src.find("{%") != std::string::npos || effective_src.find("{{") != std::string::npos);
        bool native_thinking_api = false;
        try {
            native_thinking_api = has_explicit && common_chat_templates_support_enable_thinking(g_chat_templates.get());
        } catch (...) {
            native_thinking_api = false;
        }
        g_template_has_reasoning =
            has_explicit && (native_thinking_api || template_source_has_reasoning_tags(effective_src));
    }
    probe_model_template_and_vocab_once();
    LOGi(
        "Chat template initialized in loadModelNative. Jinja2: %s, reasoning: %s, cot_rules: %zu, stops: %zu, "
        "preserved: %zu",
        g_use_jinja ? "YES" : "NO", g_template_has_reasoning ? "YES" : "NO", g_active_cot_rules.size(),
        g_template_stop_strings.size(), g_preserved_token_ids.size());

    if (jprojector_path != nullptr) {
        const std::string projector_path = jstring_to_utf8(env, jprojector_path);
        if (!projector_path.empty()) {
            const bool is_self_path = (!model_path_str.empty() && model_path_str == projector_path);
            if (is_self_path && !model_has_embedded_mm_metadata(g_model)) {
                LOGd(
                    "Explicit projector path matches main model without embedded clip/vision metadata -> skipping "
                    "text-only mtmd init.");
            } else {
                LOGi("Loading mtmd multimodal projector from explicit path: %s", projector_path.c_str());
                struct mtmd_context_params mparams = mtmd_context_params_default();
                mparams.use_gpu                    = (backend_str == "opencl" || backend_str == "vulkan");
                g_mtmd_ctx                         = mtmd_init_from_file(projector_path.c_str(), g_model, mparams);
                if (!g_mtmd_ctx) {
                    LOGe("Failed to initialize mtmd_context projector from explicit path!");
                } else {
                    LOGi("mtmd_context vision projector initialized successfully.");
                }
            }
        }
    }

    if (!g_mtmd_ctx && !model_path_str.empty()) {
        if (model_has_embedded_mm_metadata(g_model)) {
            LOGi("Embedded multimodal metadata detected in GGUF -> initializing mtmd from model file: %s",
                 model_path_str.c_str());
            struct mtmd_context_params mparams = mtmd_context_params_default();
            mparams.use_gpu                    = (backend_str == "opencl" || backend_str == "vulkan");
            g_mtmd_ctx                         = mtmd_init_from_file(model_path_str.c_str(), g_model, mparams);
            if (!g_mtmd_ctx) {
                LOGw("Embedded mtmd probe did not initialize projector context.");
            } else {
                LOGi("mtmd_context embedded vision/audio projector initialized successfully.");
            }
        } else {
            LOGd(
                "No multimodal projector path or embedded clip/vision metadata -> skipping redundant mtmd probe "
                "(text-only fast start).");
        }
    }
    return 0;
}

extern "C" JNIEXPORT jboolean JNICALL
Java_com_arm_aichat_internal_InferenceEngineImpl_processImageNative(JNIEnv *   env,
                                                                    jobject    thiz,
                                                                    jbyteArray image_bytes) {
    (void) thiz;
    std::lock_guard<std::recursive_mutex> cfg_lock(g_config_mutex);

    if (!g_mtmd_ctx || !g_context || !image_bytes) {
        LOGe("MTMD Projector, LLM context or image_bytes invalid. Cannot process image.");
        return JNI_FALSE;
    }

    jsize length = env->GetArrayLength(image_bytes);
    if (length <= 0) {
        LOGe("processImageNative received empty image byte array (%d).", length);
        return JNI_FALSE;
    }
    jbyte * bytes = env->GetByteArrayElements(image_bytes, nullptr);
    if (!bytes) {
        if (env->ExceptionCheck()) {
            env->ExceptionClear();
        }
        LOGe("GetByteArrayElements returned null in processImageNative.");
        return JNI_FALSE;
    }

    LOGi("Creating vision embeddings from %d bytes...", length);

    auto bmp_wrapper  = mtmd_helper_bitmap_init_from_buf(g_mtmd_ctx, reinterpret_cast<const unsigned char *>(bytes),
                                                         length, false, mtmd_helper_init_opt_default());
    mtmd_bitmap * bmp = bmp_wrapper.bitmap;
    env->ReleaseByteArrayElements(image_bytes, bytes, JNI_ABORT);

    if (!bmp) {
        LOGe("Failed to extract image bitmap via mtmd_helper_bitmap_init_from_buf.");
        return JNI_FALSE;
    }

    // Cache the bitmap for Delayed Evaluation inside processUserPrompt
    if (g_cached_bmp) {
        mtmd_bitmap_free(g_cached_bmp);
    }
    g_cached_bmp = bmp;
    LOGi("Image bitmap cached successfully for the next user prompt.");

    return JNI_TRUE;
}

extern "C" JNIEXPORT jboolean JNICALL
Java_com_arm_aichat_internal_InferenceEngineImpl_hasProjectorNative(JNIEnv * env, jobject thiz) {
    (void) env;
    (void) thiz;
    std::lock_guard<std::recursive_mutex> cfg_lock(g_config_mutex);
    return (g_mtmd_ctx != nullptr) ? JNI_TRUE : JNI_FALSE;
}

static llama_context * init_context(llama_model * model, const int n_ctx_arg = 0, const int n_batch_arg = 0) {
    if (!model) {
        LOGe("%s: model cannot be null", __func__);
        return nullptr;
    }

    // Multi-threading setup
    const int n_threads =
        std::max(N_THREADS_MIN, std::min(N_THREADS_MAX, (int) sysconf(_SC_NPROCESSORS_ONLN) - N_THREADS_HEADROOM));
    LOGi("%s: Using %d threads", __func__, n_threads);

    // Context parameters setup
    llama_context_params ctx_params = llama_context_default_params();

    // Resolve context size
    // Priority: Argument > Global Setting > Default Constant
    int n_ctx_to_use = DEFAULT_CONTEXT_SIZE;
    if (n_ctx_arg > 0) {
        n_ctx_to_use = n_ctx_arg;
    } else if (g_n_ctx > 0) {
        n_ctx_to_use = g_n_ctx;
    }

    const int trained_context_size = llama_model_n_ctx_train(model);
    if (n_ctx_to_use > trained_context_size) {
        LOGw("%s: Model was trained with only %d context size! Enforcing %d context size...", __func__,
             trained_context_size, n_ctx_to_use);
    }
    ctx_params.n_ctx = n_ctx_to_use;

    // Use global batch size if set (or caller benchmark override), otherwise default
    int n_batch_to_use = (n_batch_arg > 0) ? n_batch_arg : ((g_n_batch > 0) ? g_n_batch : BATCH_SIZE);
    ctx_params.n_batch = n_batch_to_use;

    // Use global physical batch size (ubatch), clamped to <= n_batch to prevent llama-context assert & OpenCL prealloc_act_trans overflow
    int n_ubatch_to_use = (g_n_ubatch > 0) ? g_n_ubatch : BATCH_SIZE;
    if (n_ubatch_to_use > n_batch_to_use) {
        n_ubatch_to_use = n_batch_to_use;
    }
    const bool is_opencl_active = (g_active_backend_info.rfind("OpenCL", 0) == 0);
    const bool is_vulkan_active = (g_active_backend_info.rfind("Vulkan", 0) == 0);
    if (is_opencl_active && n_ubatch_to_use > 256) {
        // Keep physical ubatch conservative on Adreno OpenCL to avoid prealloc_act_trans (43 MiB) sub-buffer CL_CHECK abort
        n_ubatch_to_use = 256;
    } else if (is_vulkan_active && n_ubatch_to_use > 128) {
        // Keep each Vulkan vkQueueSubmit well below Android KGSL 5s GPU TDR watchdog (prevents VK_ERROR_DEVICE_LOST on Adreno 750)
        n_ubatch_to_use = 128;
    }
    ctx_params.n_ubatch = n_ubatch_to_use;

    ctx_params.n_threads       = n_threads;
    ctx_params.n_threads_batch = n_threads;
    ctx_params.no_perf         = g_no_perf;

    // Flash Attention:
    // - OpenCL and Vulkan use AUTO when user enabled FA so llama.cpp dynamically picks supported FA kernels without hardcoding DISABLED/F16.
    if (is_opencl_active || is_vulkan_active) {
        ctx_params.flash_attn_type = g_flash_attn ? LLAMA_FLASH_ATTN_TYPE_AUTO : LLAMA_FLASH_ATTN_TYPE_DISABLED;
    } else {
        ctx_params.flash_attn_type = g_flash_attn ? LLAMA_FLASH_ATTN_TYPE_ENABLED : LLAMA_FLASH_ATTN_TYPE_DISABLED;
    }

    // Mapování int z Kotlinu (arrays.xml index 0-14) na ggml_type
    // Upraveno pro sekvenční ID od uživatele
    switch (g_kv_cache_mode) {
        // --- HIGH QUALITY ---
        case 0:  // F16
            ctx_params.type_k = GGML_TYPE_F16;
            ctx_params.type_v = GGML_TYPE_F16;
            break;
        case 1:  // Q8_0
            ctx_params.type_k = GGML_TYPE_Q8_0;
            ctx_params.type_v = GGML_TYPE_Q8_0;
            break;
        case 2:  // Q6_K
            ctx_params.type_k = GGML_TYPE_Q6_K;
            ctx_params.type_v = GGML_TYPE_Q6_K;
            break;
        case 3:  // Q5_K
            ctx_params.type_k = GGML_TYPE_Q5_K;
            ctx_params.type_v = GGML_TYPE_Q5_K;
            break;
        case 4:  // Q4_K
            ctx_params.type_k = GGML_TYPE_Q4_K;
            ctx_params.type_v = GGML_TYPE_Q4_K;
            break;

        // --- LEGACY ---
        case 5:  // Q5_1
            ctx_params.type_k = GGML_TYPE_Q5_1;
            ctx_params.type_v = GGML_TYPE_Q5_1;
            break;
        case 6:  // Q5_0
            ctx_params.type_k = GGML_TYPE_Q5_0;
            ctx_params.type_v = GGML_TYPE_Q5_0;
            break;
        case 7:  // Q4_1
            ctx_params.type_k = GGML_TYPE_Q4_1;
            ctx_params.type_v = GGML_TYPE_Q4_1;
            break;
        case 8:  // Q4_0
            ctx_params.type_k = GGML_TYPE_Q4_0;
            ctx_params.type_v = GGML_TYPE_Q4_0;
            break;

        // --- IQ SERIES ---
        case 9:  // IQ4_NL
            ctx_params.type_k = GGML_TYPE_IQ4_NL;
            ctx_params.type_v = GGML_TYPE_IQ4_NL;
            break;
        case 10:  // IQ3_M - FALLBACK to IQ3_S (IQ3_M not defined in ggml.h)
            ctx_params.type_k = GGML_TYPE_IQ3_S;
            ctx_params.type_v = GGML_TYPE_IQ3_S;
            __android_log_print(ANDROID_LOG_WARN, "LlamaJNI",
                                "IQ3_M requested but not available in ggml.h. Falling back to IQ3_S.");
            break;
        case 11:  // IQ3_S
            ctx_params.type_k = GGML_TYPE_IQ3_S;
            ctx_params.type_v = GGML_TYPE_IQ3_S;
            break;
        case 12:  // IQ2_XS
            ctx_params.type_k = GGML_TYPE_IQ2_XS;
            ctx_params.type_v = GGML_TYPE_IQ2_XS;
            break;
        case 13:  // IQ2_XXS
            ctx_params.type_k = GGML_TYPE_IQ2_XXS;
            ctx_params.type_v = GGML_TYPE_IQ2_XXS;
            break;
        case 14:  // IQ1_S
            ctx_params.type_k = GGML_TYPE_IQ1_S;
            ctx_params.type_v = GGML_TYPE_IQ1_S;
            break;

        default:
            __android_log_print(ANDROID_LOG_WARN, "LlamaJNI", "Unknown cache mode %d, defaulting to Q8_0",
                                g_kv_cache_mode);
            ctx_params.type_k = GGML_TYPE_Q8_0;
            ctx_params.type_v = GGML_TYPE_Q8_0;
            break;
    }
    LOGi("KV Cache quantization set to mode: %d", g_kv_cache_mode);

    // Guard against quantized V cache when Flash Attention is disabled (llama_init_from_model rejects quantized V without FA)
    if (ctx_params.flash_attn_type == LLAMA_FLASH_ATTN_TYPE_DISABLED && ggml_is_quantized(ctx_params.type_v)) {
        LOGw("%s: Flash Attention disabled with quantized V cache (%d) -> falling back type_v to F16.", __func__,
             (int) ctx_params.type_v);
        ctx_params.type_v = GGML_TYPE_F16;
    }
    // Guard against incompatible head_dim % blck_size (e.g. Q4_K/IQ2 with blck_size=256 on head_dim=64/128)
    const int64_t n_head_q = llama_model_n_head(g_model);
    const int64_t head_dim = (n_head_q > 0) ? (llama_model_n_embd(g_model) / n_head_q) : 0;
    if (head_dim > 0 && (head_dim % ggml_blck_size(ctx_params.type_k)) != 0) {
        LOGw("%s: head_dim=%lld incompatible with type_k blck_size=%lld -> falling back type_k to F16.", __func__,
             (long long) head_dim, (long long) ggml_blck_size(ctx_params.type_k));
        ctx_params.type_k = GGML_TYPE_F16;
    }
    if (head_dim > 0 && (head_dim % ggml_blck_size(ctx_params.type_v)) != 0) {
        LOGw("%s: head_dim=%lld incompatible with type_v blck_size=%lld -> falling back type_v to F16.", __func__,
             (long long) head_dim, (long long) ggml_blck_size(ctx_params.type_v));
        ctx_params.type_v = GGML_TYPE_F16;
    }

    auto * context = llama_init_from_model(g_model, ctx_params);
    if (context == nullptr && (ctx_params.type_k != GGML_TYPE_F16 || ctx_params.type_v != GGML_TYPE_F16)) {
        LOGw("%s: llama_init_from_model failed with quantized KV cache -> retrying with F16 KV cache.", __func__);
        ctx_params.type_k = GGML_TYPE_F16;
        ctx_params.type_v = GGML_TYPE_F16;
        context           = llama_init_from_model(g_model, ctx_params);
    }
    if (context == nullptr) {
        LOGe("%s: llama_new_context_with_model() returned null)", __func__);
    }
    return context;
}

static common_sampler * new_sampler(float temp) {
    common_params_sampling sparams;
    sparams.temp = temp;
    return common_sampler_init(g_model, sparams);
}

// --- NEW FUNCTION: Configure Sampling (RENAMED TO MATCH KOTLIN) ---
extern "C" JNIEXPORT void JNICALL
Java_com_arm_aichat_internal_InferenceEngineImpl_configureSamplingNative(JNIEnv * env,
                                                                         jobject  thiz,
                                                                         jfloat   temp,
                                                                         jint     top_k,
                                                                         jfloat   top_p,
                                                                         jfloat   min_p,
                                                                         jfloat   xtc_prob,
                                                                         jfloat   xtc_thresh,
                                                                         jfloat   typical_p,
                                                                         jfloat   dyn_temp_range,
                                                                         jfloat   dyn_temp_exp,
                                                                         jfloat   penalty_repeat,
                                                                         jfloat   penalty_freq,
                                                                         jfloat   penalty_present,
                                                                         jint     penalty_last_n,
                                                                         jint     mirostat,
                                                                         jfloat   mirostat_tau,
                                                                         jfloat   mirostat_eta,
                                                                         jstring  samplers_sequence,
                                                                         jboolean use_backend_sampling,
                                                                         // New parameters
                                                                         jboolean ignore_eos,
                                                                         jboolean no_perf,
                                                                         jboolean timing_per_token) {
    (void) env;
    (void) thiz;

    std::lock_guard<std::recursive_mutex> cfg_lock(g_config_mutex);
    const std::string                     seq_str = jstring_to_utf8(env, samplers_sequence);

    const int32_t safe_mirostat    = std::clamp<int32_t>(mirostat, 0, 2);
    const float   safe_penalty_rep = (std::isfinite(penalty_repeat) && penalty_repeat > 0.0f) ? penalty_repeat : 1.0f;
    const bool    prev_no_perf     = g_no_perf;

    if (g_model != nullptr && g_sampler != nullptr && !g_sampler_has_turn_constraints && g_sampling_cache.valid &&
        g_sampling_cache.model_ptr == g_model && g_sampling_cache.temp == temp && g_sampling_cache.top_k == top_k &&
        g_sampling_cache.top_p == top_p && g_sampling_cache.min_p == min_p && g_sampling_cache.xtc_prob == xtc_prob &&
        g_sampling_cache.xtc_thresh == xtc_thresh && g_sampling_cache.typical_p == typical_p &&
        g_sampling_cache.dyn_temp_range == dyn_temp_range && g_sampling_cache.dyn_temp_exp == dyn_temp_exp &&
        g_sampling_cache.penalty_repeat == safe_penalty_rep && g_sampling_cache.penalty_freq == penalty_freq &&
        g_sampling_cache.penalty_present == penalty_present && g_sampling_cache.penalty_last_n == penalty_last_n &&
        g_sampling_cache.mirostat == safe_mirostat && g_sampling_cache.mirostat_tau == mirostat_tau &&
        g_sampling_cache.mirostat_eta == mirostat_eta && g_sampling_cache.samplers_seq == seq_str &&
        g_sampling_cache.use_backend_sampling == (bool) use_backend_sampling &&
        g_sampling_cache.ignore_eos == (bool) ignore_eos && g_sampling_cache.no_perf == (bool) no_perf &&
        g_sampling_cache.timing_per_token == (bool) timing_per_token) {
        g_ignore_eos       = (bool) ignore_eos;
        g_no_perf          = (bool) no_perf;
        g_timing_per_token = (bool) timing_per_token;
        if (g_context && prev_no_perf != g_no_perf) {
            llama_perf_context_reset(g_context);
        }
        attach_backend_sampler_if_enabled();
        return;
    }

    LOGi("Configuring sampling: T=%.2f, K=%d, P=%.2f, MinP=%.2f, PenRep=%.2f, IgnoreEOS=%d, BackendSmpl=%d, NoPerf=%d",
         temp, top_k, top_p, min_p, safe_penalty_rep, ignore_eos, use_backend_sampling, no_perf);

    // Update globals
    g_ignore_eos       = (bool) ignore_eos;
    g_no_perf          = (bool) no_perf;
    g_timing_per_token = (bool) timing_per_token;
    if (g_context && prev_no_perf != g_no_perf) {
        llama_perf_context_reset(g_context);
        LOGi("configureSamplingNative: no_perf changed (%d -> %d); perf context counters reset.", prev_no_perf ? 1 : 0,
             g_no_perf ? 1 : 0);
    }

    if (g_sampler) {
        detach_backend_sampler_if_attached();
        common_sampler_free(g_sampler);
        g_sampler = nullptr;
    }
    g_sampling_cache.valid = false;

    common_params_sampling sparams;
    sparams.temp              = temp;
    sparams.top_k             = top_k;
    sparams.top_p             = top_p;
    sparams.min_p             = min_p;
    sparams.xtc_probability   = xtc_prob;
    sparams.xtc_threshold     = xtc_thresh;
    sparams.typ_p             = typical_p;
    sparams.dynatemp_range    = dyn_temp_range;
    sparams.dynatemp_exponent = dyn_temp_exp;
    sparams.penalty_repeat    = safe_penalty_rep;
    sparams.penalty_freq      = penalty_freq;
    sparams.penalty_present   = penalty_present;
    sparams.penalty_last_n    = penalty_last_n;
    sparams.mirostat          = safe_mirostat;
    sparams.mirostat_tau      = mirostat_tau;
    sparams.mirostat_eta      = mirostat_eta;

    sparams.backend_sampling = (bool) use_backend_sampling;
    sparams.no_perf          = (bool) no_perf;
    sparams.ignore_eos       = (bool) ignore_eos;
    sparams.timing_per_token = (bool) timing_per_token;

    // Parse samplers sequence string from Kotlin:
    //   1. Keyword format (semicolon-separated or exact single keyword name)
    //   2. Legacy single-char format (e.g. "kpytxdrn", plus upstream aliases 'e'/'s'/'a'/'i')
    if (!seq_str.empty()) {
        sparams.samplers.clear();

        auto is_known_keyword = [](const std::string & tok) -> bool {
            return tok == "top_k" || tok == "top_p" || tok == "min_p" || tok == "typ_p" || tok == "temperature" ||
                   tok == "temp" || tok == "xtc" || tok == "dry" || tok == "top_n_sigma" || tok == "penalties" ||
                   tok == "repetition_penalties" || tok == "mirostat" || tok == "mirostat_v2" || tok == "adaptive_p" ||
                   tok == "infill";
        };
        std::string trimmed_seq = seq_str;
        {
            const size_t fs = trimmed_seq.find_first_not_of(" \t\r\n");
            const size_t ls = trimmed_seq.find_last_not_of(" \t\r\n");
            trimmed_seq     = (fs != std::string::npos) ? trimmed_seq.substr(fs, ls - fs + 1) : "";
        }
        const bool keyword_mode = (trimmed_seq.find(';') != std::string::npos) ||
                                  (trimmed_seq.find('_') != std::string::npos) || is_known_keyword(trimmed_seq);

        if (keyword_mode) {
            std::istringstream stream(trimmed_seq);
            std::string        token;
            while (std::getline(stream, token, ';')) {
                const size_t first_non_ws = token.find_first_not_of(" \t\r\n");
                if (first_non_ws == std::string::npos) {
                    continue;
                }
                const size_t last_non_ws = token.find_last_not_of(" \t\r\n");
                token                    = token.substr(first_non_ws, last_non_ws - first_non_ws + 1);

                if (token == "top_k") {
                    sparams.samplers.push_back(COMMON_SAMPLER_TYPE_TOP_K);
                } else if (token == "top_p") {
                    sparams.samplers.push_back(COMMON_SAMPLER_TYPE_TOP_P);
                } else if (token == "min_p") {
                    sparams.samplers.push_back(COMMON_SAMPLER_TYPE_MIN_P);
                } else if (token == "typ_p") {
                    sparams.samplers.push_back(COMMON_SAMPLER_TYPE_TYPICAL_P);
                } else if (token == "temperature" || token == "temp") {
                    sparams.samplers.push_back(COMMON_SAMPLER_TYPE_TEMPERATURE);
                } else if (token == "xtc") {
                    sparams.samplers.push_back(COMMON_SAMPLER_TYPE_XTC);
                } else if (token == "dry") {
                    sparams.samplers.push_back(COMMON_SAMPLER_TYPE_DRY);
                } else if (token == "top_n_sigma") {
                    sparams.samplers.push_back(COMMON_SAMPLER_TYPE_TOP_N_SIGMA);
                } else if (token == "penalties" || token == "repetition_penalties") {
                    sparams.samplers.push_back(COMMON_SAMPLER_TYPE_PENALTIES);
                } else if (token == "adaptive_p") {
                    sparams.samplers.push_back(COMMON_SAMPLER_TYPE_ADAPTIVE_P);
                } else if (token == "infill") {
                    sparams.samplers.push_back(COMMON_SAMPLER_TYPE_INFILL);
                } else if (token == "mirostat") {
                    sparams.mirostat = 1;
                    LOGi("Mirostat V1 activated from keyword sequence.");
                } else if (token == "mirostat_v2") {
                    sparams.mirostat = 2;
                    LOGi("Mirostat V2 activated from keyword sequence.");
                } else {
                    LOGw("Unknown sampler keyword '%s' in sequence, ignoring.", token.c_str());
                }
            }
            LOGi("Applied keyword sampler sequence: %s", seq_str.c_str());
        } else {
            // Legacy single-char parser (preserving kpytxdrn + upstream common_sampler_type_to_chr aliases)
            for (char c : seq_str) {
                switch (c) {
                    case 'k':
                        sparams.samplers.push_back(COMMON_SAMPLER_TYPE_TOP_K);
                        break;
                    case 'p':
                        sparams.samplers.push_back(COMMON_SAMPLER_TYPE_TOP_P);
                        break;
                    case 'm':
                        sparams.samplers.push_back(COMMON_SAMPLER_TYPE_MIN_P);
                        break;
                    case 'y':
                        sparams.samplers.push_back(COMMON_SAMPLER_TYPE_TYPICAL_P);
                        break;
                    case 't':
                        sparams.samplers.push_back(COMMON_SAMPLER_TYPE_TEMPERATURE);
                        break;
                    case 'x':
                        sparams.samplers.push_back(COMMON_SAMPLER_TYPE_XTC);
                        break;
                    case 'd':
                        sparams.samplers.push_back(COMMON_SAMPLER_TYPE_DRY);
                        break;
                    case 'n':
                    case 's':
                        sparams.samplers.push_back(COMMON_SAMPLER_TYPE_TOP_N_SIGMA);
                        break;
                    case 'r':
                    case 'e':
                        sparams.samplers.push_back(COMMON_SAMPLER_TYPE_PENALTIES);
                        break;
                    case 'a':
                        sparams.samplers.push_back(COMMON_SAMPLER_TYPE_ADAPTIVE_P);
                        break;
                    case 'i':
                        sparams.samplers.push_back(COMMON_SAMPLER_TYPE_INFILL);
                        break;
                    case 'M':
                        sparams.mirostat = 1;
                        LOGi("Mirostat V1 sampler activated from sequence.");
                        break;
                    case 'v':
                        sparams.mirostat = 2;
                        LOGi("Mirostat V2 sampler activated from sequence.");
                        break;
                    default:
                        LOGw("Unknown sampler char '%c' in sequence, ignoring.", c);
                        break;
                }
            }
            LOGi("Applied single-char sampler sequence: %s", seq_str.c_str());
        }
    }

    // Preserve grammar if already set (from setGrammarNative)
    sparams.grammar = g_sparams.grammar;

    // Add -INFINITY logit bias on all EOG tokens when ignore_eos=true
    apply_ignore_eos_logit_bias(sparams);

    // Store clean persistent globals (never allow common_sampler_init to clear g_sparams.backend_sampling)
    g_sparams                      = sparams;
    g_sampler_has_turn_constraints = !g_sparams.grammar.empty();

    if (g_model == nullptr) {
        LOGw("configureSamplingNative: g_model is null, saved params for next prepare().");
        return;
    }

    common_params_sampling init_sparams        = g_sparams;
    auto                   init_clean_fallback = [&](const common_params_sampling & base) -> common_sampler * {
        common_params_sampling clean = base;
        clean.grammar                = common_grammar();
        clean.grammar_lazy           = false;
        clean.grammar_triggers.clear();
        clean.preserved_tokens.clear();
        clean.reasoning_budget_tokens = -1;
        clean.reasoning_budget_start.clear();
        clean.reasoning_budget_end.clear();
        clean.reasoning_budget_forced.clear();
        try {
            return common_sampler_init(g_model, clean);
        } catch (...) {
            common_params_sampling bare;
            bare.temp = (clean.temp >= 0.0f) ? clean.temp : DEFAULT_SAMPLER_TEMP;
            try {
                return common_sampler_init(g_model, bare);
            } catch (...) {
                return nullptr;
            }
        }
    };
    try {
        g_sampler                      = common_sampler_init(g_model, init_sparams);
        g_sampler_has_turn_constraints = !g_sparams.grammar.empty();
    } catch (const std::exception & e) {
        LOGe("configureSamplingNative: common_sampler_init failed (%s), retrying clean fallback.", e.what());
        g_sparams.grammar              = common_grammar();
        g_sampler_has_turn_constraints = false;
        g_sampler                      = init_clean_fallback(g_sparams);
    } catch (...) {
        LOGe("configureSamplingNative: unknown sampler init error, retrying clean fallback.");
        g_sparams.grammar              = common_grammar();
        g_sampler_has_turn_constraints = false;
        g_sampler                      = init_clean_fallback(g_sparams);
    }
    if (g_sampler != nullptr && g_model != nullptr) {
        attach_backend_sampler_if_enabled();
        g_sampling_cache.valid                = true;
        g_sampling_cache.model_ptr            = g_model;
        g_sampling_cache.temp                 = temp;
        g_sampling_cache.top_k                = top_k;
        g_sampling_cache.top_p                = top_p;
        g_sampling_cache.min_p                = min_p;
        g_sampling_cache.xtc_prob             = xtc_prob;
        g_sampling_cache.xtc_thresh           = xtc_thresh;
        g_sampling_cache.typical_p            = typical_p;
        g_sampling_cache.dyn_temp_range       = dyn_temp_range;
        g_sampling_cache.dyn_temp_exp         = dyn_temp_exp;
        g_sampling_cache.penalty_repeat       = safe_penalty_rep;
        g_sampling_cache.penalty_freq         = penalty_freq;
        g_sampling_cache.penalty_present      = penalty_present;
        g_sampling_cache.penalty_last_n       = penalty_last_n;
        g_sampling_cache.mirostat             = safe_mirostat;
        g_sampling_cache.mirostat_tau         = mirostat_tau;
        g_sampling_cache.mirostat_eta         = mirostat_eta;
        g_sampling_cache.samplers_seq         = seq_str;
        g_sampling_cache.use_backend_sampling = (bool) use_backend_sampling;
        g_sampling_cache.ignore_eos           = (bool) ignore_eos;
        g_sampling_cache.no_perf              = (bool) no_perf;
        g_sampling_cache.timing_per_token     = (bool) timing_per_token;
    }
    LOGi("Sampler re-initialized successfully.");
}

extern "C" JNIEXPORT void JNICALL
Java_com_arm_aichat_internal_InferenceEngineImpl_configureThinkingNative(JNIEnv * env,
                                                                         jobject  thiz,
                                                                         jboolean enabled,
                                                                         jint     reasoningBudgetTokens,
                                                                         jstring  effort) {
    (void) thiz;
    const bool                            new_enabled = (bool) enabled;
    const int32_t                         new_budget  = (int32_t) reasoningBudgetTokens;
    std::string                           new_effort  = jstring_to_utf8(env, effort);
    std::lock_guard<std::recursive_mutex> cfg_lock(g_config_mutex);
    const bool prompt_affecting_change = (new_enabled != g_enable_thinking) || (new_effort != g_thinking_effort);
    g_enable_thinking                  = new_enabled;
    g_reasoning_budget_tokens          = new_budget;
    g_thinking_effort                  = std::move(new_effort);
    if (prompt_affecting_change) {
        invalidate_turn_prompt_caches();
    }
    LOGi("Thinking configured: enabled=%d, budgetTokens=%d (0=unlimited), effort=%s", g_enable_thinking,
         g_reasoning_budget_tokens, g_thinking_effort.c_str());
}

extern "C" JNIEXPORT jboolean JNICALL
Java_com_arm_aichat_internal_InferenceEngineImpl_supportsThinkingNative(JNIEnv * env, jobject thiz) {
    (void) env;
    (void) thiz;
    std::lock_guard<std::recursive_mutex> cfg_lock(g_config_mutex);
    const CachedModelCapabilities         caps = get_or_compute_model_capabilities();
    return caps.supports_thinking ? JNI_TRUE : JNI_FALSE;
}

// ----------------------------------------

// --- GBNF Grammar Constraint for Agent Mode ---
extern "C" JNIEXPORT void JNICALL Java_com_arm_aichat_internal_InferenceEngineImpl_setGrammarNative(JNIEnv * env,
                                                                                                    jobject  thiz,
                                                                                                    jstring  jgrammar) {
    (void) thiz;
    std::lock_guard<std::recursive_mutex> cfg_lock(g_config_mutex);
    g_sampling_cache.valid        = false;
    const std::string grammar_str = jstring_to_utf8(env, jgrammar);
    if (jgrammar == nullptr || grammar_str.empty()) {
        // Clear grammar
        g_sparams.grammar = common_grammar();
        LOGi("Grammar constraint cleared.");
    } else {
        g_sparams.grammar = common_grammar(COMMON_GRAMMAR_TYPE_USER, grammar_str);
        LOGi("Grammar constraint set (%d chars).", (int) g_sparams.grammar.grammar.size());
    }

    if (g_model == nullptr) {
        g_sampler_has_turn_constraints = !g_sparams.grammar.empty();
        LOGw("setGrammarNative: g_model is null, saved grammar constraint for next prepare().");
        return;
    }

    // Reinitialize sampler with updated grammar (use local copy so backend_sampling stays intact in g_sparams)
    if (g_sampler) {
        detach_backend_sampler_if_attached();
        common_sampler_free(g_sampler);
        g_sampler = nullptr;
    }
    g_sampler_has_turn_constraints      = !g_sparams.grammar.empty();
    common_params_sampling init_sparams = g_sparams;
    try {
        g_sampler = common_sampler_init(g_model, init_sparams);
        attach_backend_sampler_if_enabled();
        LOGi("Sampler re-initialized with grammar.");
    } catch (const std::exception & e) {
        LOGe("setGrammarNative: failed to apply grammar (%s). Reverting to clean sampler.", e.what());
        g_sparams.grammar              = common_grammar();
        g_sampler_has_turn_constraints = false;
        init_sparams                   = g_sparams;
        try {
            g_sampler = common_sampler_init(g_model, init_sparams);
        } catch (...) {
            g_sampler = new_sampler(DEFAULT_SAMPLER_TEMP);
        }
        attach_backend_sampler_if_enabled();
    } catch (...) {
        LOGe("setGrammarNative: unknown failure applying grammar. Reverting to clean sampler.");
        g_sparams.grammar              = common_grammar();
        g_sampler_has_turn_constraints = false;
        init_sparams                   = g_sparams;
        try {
            g_sampler = common_sampler_init(g_model, init_sparams);
        } catch (...) {
            g_sampler = new_sampler(DEFAULT_SAMPLER_TEMP);
        }
        attach_backend_sampler_if_enabled();
    }
}

// --- Phase 1 JNI APIs for Native Tool Protocol ---

static std::vector<common_chat_tool> parse_tools_json(const std::string & tools_str) {
    std::vector<common_chat_tool> tools;
    if (tools_str.empty()) {
        return tools;
    }
    try {
        auto parsed = common_json::parse_no_throw(tools_str);
        if (parsed.is_discarded()) {
            LOGe("parse_tools_json: JSON parse failed (discarded)");
            return tools;
        }
        // Try OpenAI format first: [{"type": "function", "function": {...}}]
        try {
            tools = common_chat_tools_parse_oaicompat(parsed);
            if (!tools.empty()) {
                return tools;
            }
        } catch (...) {
            tools.clear();
        }
        // Fallback: direct array [{"name": "...", "description": "...", "parameters": {...}}]
        if (parsed.is_array()) {
            for (size_t i = 0; i < parsed.size(); i++) {
                const auto & item = parsed[i];
                if (item.is_object() && item.contains("name")) {
                    common_chat_tool tool;
                    tool.name        = item.value("name", "");
                    tool.description = item.value("description", "");
                    if (item.contains("parameters")) {
                        tool.parameters = item.at("parameters").dump();
                    } else {
                        tool.parameters = "{}";
                    }
                    tools.push_back(std::move(tool));
                }
            }
        }
    } catch (const std::exception & e) {
        LOGe("parse_tools_json: exception: %s", e.what());
    }
    return tools;
}

extern "C" JNIEXPORT void JNICALL
Java_com_arm_aichat_internal_InferenceEngineImpl_setToolDefinitionsNative(JNIEnv * env,
                                                                          jobject  thiz,
                                                                          jstring  jtools_json) {
    (void) thiz;
    const std::string                     tools_str = jstring_to_utf8(env, jtools_json);
    std::lock_guard<std::recursive_mutex> cfg_lock(g_config_mutex);
    if (tools_str == g_last_tools_json_str) {
        return;
    }
    g_last_tools_json_str = tools_str;
    ++g_tools_config_version;
    invalidate_turn_prompt_caches();
    if (tools_str.empty()) {
        g_configured_tools.clear();
        LOGi("setToolDefinitionsNative: Tools cleared (0 tools configured)");
        return;
    }
    g_configured_tools = parse_tools_json(tools_str);
    LOGi("setToolDefinitionsNative: %zu tools successfully configured", g_configured_tools.size());
    for (const auto & t : g_configured_tools) {
        LOGd("  Configured tool: '%s' (desc: %zu chars, params: %zu chars)", t.name.c_str(), t.description.size(),
             t.parameters.size());
    }
}

extern "C" JNIEXPORT void JNICALL Java_com_arm_aichat_internal_InferenceEngineImpl_setToolPolicyNative(JNIEnv * env,
                                                                                                       jobject  thiz,
                                                                                                       jint jpolicy) {
    (void) env;
    (void) thiz;
    std::lock_guard<std::recursive_mutex> cfg_lock(g_config_mutex);
    const NativeToolChoice                prev_policy = g_tool_policy;
    switch (jpolicy) {
        case 0:
            g_tool_policy = NativeToolChoice::AUTO;
            LOGi("setToolPolicyNative: AUTO (0)");
            break;
        case 1:
            g_tool_policy = NativeToolChoice::REQUIRED;
            LOGi("setToolPolicyNative: REQUIRED (1)");
            break;
        case 2:
            g_tool_policy = NativeToolChoice::NONE;
            LOGi("setToolPolicyNative: NONE (2)");
            break;
        default:
            LOGw("setToolPolicyNative: unknown policy %d, defaulting to AUTO", jpolicy);
            g_tool_policy = NativeToolChoice::AUTO;
            break;
    }
    if (g_tool_policy != prev_policy) {
        invalidate_turn_prompt_caches();
    }
}

extern "C" JNIEXPORT void JNICALL
Java_com_arm_aichat_internal_InferenceEngineImpl_setToolMarkupModeNative(JNIEnv * env,
                                                                         jobject  thiz,
                                                                         jint     jformat_id,
                                                                         jboolean jallow_native) {
    (void) env;
    (void) thiz;
    std::lock_guard<std::recursive_mutex> cfg_lock(g_config_mutex);
    const bool                            prev_allow  = g_allow_native_tools_hint;
    const int32_t                         prev_format = g_tool_markup_format;
    const bool                            prev_active = g_tools_active_hint;
    g_tools_active_hint                               = (jformat_id >= 0);
    g_tool_markup_format                              = (jformat_id >= 0) ? static_cast<int32_t>(jformat_id) : 0;
    g_allow_native_tools_hint                         = (bool) jallow_native;
    if (prev_allow != g_allow_native_tools_hint || prev_format != g_tool_markup_format ||
        prev_active != g_tools_active_hint) {
        invalidate_turn_prompt_caches();
        LOGi("setToolMarkupModeNative: format=%d, allowNative=%d, toolsActive=%d", g_tool_markup_format,
             g_allow_native_tools_hint ? 1 : 0, g_tools_active_hint ? 1 : 0);
    }
}

extern "C" JNIEXPORT void JNICALL
Java_com_arm_aichat_internal_InferenceEngineImpl_configureCotRulesNative(JNIEnv *     env,
                                                                         jobject      thiz,
                                                                         jobjectArray jopen_tags,
                                                                         jobjectArray jclose_packed,
                                                                         jboolean     include_universal_fallback) {
    (void) thiz;
    std::lock_guard<std::recursive_mutex> cfg_lock(g_config_mutex);
    std::vector<CoTTagDef>                next_rules;

    auto add_rule_unique = [&](const CoTTagDef & rule) {
        if (rule.open_tag.empty()) {
            return;
        }
        for (const auto & existing : next_rules) {
            if (existing.open_tag == rule.open_tag) {
                return;
            }
        }
        next_rules.push_back(rule);
    };

    if (env && jopen_tags != nullptr && jclose_packed != nullptr) {
        const jsize n_open  = env->GetArrayLength(jopen_tags);
        const jsize n_close = env->GetArrayLength(jclose_packed);
        const jsize count   = std::min(n_open, n_close);
        for (jsize i = 0; i < count; ++i) {
            auto        jopen      = static_cast<jstring>(env->GetObjectArrayElement(jopen_tags, i));
            auto        jpacked    = static_cast<jstring>(env->GetObjectArrayElement(jclose_packed, i));
            std::string open_str   = jstring_to_utf8(env, jopen);
            std::string packed_str = jstring_to_utf8(env, jpacked);
            if (jopen) {
                env->DeleteLocalRef(jopen);
            }
            if (jpacked) {
                env->DeleteLocalRef(jpacked);
            }
            if (open_str.empty()) {
                continue;
            }
            std::vector<std::string> closes;
            std::istringstream       ss(packed_str);
            std::string              line;
            while (std::getline(ss, line, '\n')) {
                if (!line.empty()) {
                    closes.push_back(line);
                }
            }
            if (closes.empty()) {
                closes.push_back("</think>");
            }
            add_rule_unique({ std::move(open_str), std::move(closes) });
        }
    }

    if (include_universal_fallback == JNI_TRUE || next_rules.empty()) {
        // Universal catalog declared below; resolve via helper after load or fallback list
        static const CoTTagDef k_fallback[] = {
            { "<|channel>thought",              { "<channel|>", "<|channel>", "<|turn>", "</channel>" }                           },
            { "<|channel|>thought",             { "<channel|>", "<|channel|>", "</channel>", "<|channel>", "<|turn>", "<|end|>" } },
            { "<|channel|>analysis<|message|>", { "<|end|>", "<|channel|>final<|message|>" }                                      },
            { "<|think|>",                      { "<|/think|>", "</think>", "<|think|>" }                                         },
            { "[THINK]",                        { "[/THINK]" }                                                                    },
            { "[THOUGHT]",                      { "[/THOUGHT]" }                                                                  },
            { "[REASONING]",                    { "[/REASONING]" }                                                                },
            { "<think>",                        { "</think>" }                                                                    },
            { "<thinking>",                     { "</thinking>" }                                                                 },
            { "<thought>",                      { "</thought>" }                                                                  },
            { "<reasoning>",                    { "</reasoning>" }                                                                },
            { "<reason>",                       { "</reason>" }                                                                   },
            { "<reflection>",                   { "</reflection>" }                                                               },
            { "<analysis>",                     { "</analysis>" }                                                                 },
            { "<scratchpad>",                   { "</scratchpad>" }                                                               },
            { "<inner_monologue>",              { "</inner_monologue>" }                                                          },
            { "<seed:think>",                   { "</seed:think>" }                                                               },
            { "<start_of_thought>",             { "<end_of_thought>" }                                                            },
            { "<|thought|>",                    { "<|thought|>", "</thought>" }                                                   },
            { "<|reasoning|>",                  { "<|/reasoning|>", "</reasoning>" }                                              },
            { "<|begin_of_thought|>",           { "<|end_of_thought|>" }                                                          },
            { "<｜begin▁of▁thought｜>",         { "<｜end▁of▁thought｜>" }                                                        },
            { "<|im_start|>thought",            { "<|im_end|>", "</thought>", "</think>" }                                        },
            { "<channel>thought",               { "<channel|>", "</channel>" }                                                    }
        };
        for (const auto & item : k_fallback) {
            add_rule_unique(item);
        }
    }

    g_active_cot_rules     = std::move(next_rules);
    g_custom_cot_rules_set = true;

    if (!g_active_cot_rules.empty()) {
        g_model_thinking_start_tag = g_active_cot_rules.front().open_tag;
        if (!g_active_cot_rules.front().close_tags.empty()) {
            g_model_thinking_end_tag = g_active_cot_rules.front().close_tags.front();
        }
    }

    g_cot_lead_chars.clear();
    for (const auto & rule : g_active_cot_rules) {
        if (!rule.open_tag.empty()) {
            const char lead = rule.open_tag.front();
            if (g_cot_lead_chars.find(lead) == std::string::npos) {
                g_cot_lead_chars.push_back(lead);
            }
        }
    }
    if (g_cot_lead_chars.empty()) {
        g_cot_lead_chars = "<[";
    }
    if (g_model) {
        const llama_vocab * vocab = llama_model_get_vocab(g_model);
        if (vocab) {
            for (const auto & rule : g_active_cot_rules) {
                auto ids = common_tokenize(vocab, rule.open_tag, false, true);
                if (ids.size() == 1) {
                    g_preserved_token_ids.insert(ids[0]);
                }
                for (const auto & ct : rule.close_tags) {
                    auto cids = common_tokenize(vocab, ct, false, true);
                    if (cids.size() == 1) {
                        g_preserved_token_ids.insert(cids[0]);
                    }
                }
            }
        }
    }
    LOGi("configureCotRulesNative: configured %zu CoT rules (lead='%s').", g_active_cot_rules.size(),
         g_cot_lead_chars.c_str());
}

static CachedModelCapabilities get_or_compute_model_capabilities() {
    if (g_caps_cache.ready) {
        return g_caps_cache;
    }
    if (!g_vocab_and_template_probed && g_model) {
        probe_model_template_and_vocab_once();
    }
    CachedModelCapabilities caps_out;
    caps_out.supports_jinja = g_use_jinja;
    if (g_model) {
        caps_out.context_length = static_cast<int32_t>(llama_model_n_ctx_train(g_model));
    }
    if (g_chat_templates && g_chat_templates.get()) {
        caps_out.has_chat_template = common_chat_templates_was_explicit(g_chat_templates.get());
        try {
            caps_out.supports_thinking = common_chat_templates_support_enable_thinking(g_chat_templates.get());
        } catch (const std::exception & e) {
            LOGw("get_or_compute_model_capabilities: support_enable_thinking warning: %s", e.what());
        } catch (...) {
        }

        std::string tmpl_src;
        std::string tool_src;
        std::string effective_src;
        try {
            tmpl_src = common_chat_templates_source(g_chat_templates.get());
            try {
                tool_src = common_chat_templates_source(g_chat_templates.get(), "tool_use");
            } catch (...) {
            }
            effective_src          = !tool_src.empty() ? tool_src : tmpl_src;
            caps_out.chat_template = effective_src;
        } catch (...) {
        }

        if (!caps_out.supports_thinking && caps_out.has_chat_template) {
            caps_out.supports_thinking = g_template_has_reasoning || g_template_honors_disable_thinking ||
                                         template_source_has_reasoning_tags(effective_src) ||
                                         template_source_has_reasoning_tags(tmpl_src);
        } else if (!caps_out.supports_thinking && g_template_has_reasoning) {
            caps_out.supports_thinking = true;
        }

        if (caps_out.supports_jinja) {
            try {
                auto caps = common_chat_templates_get_caps(g_chat_templates.get());
                if (caps.count("supports_tools")) {
                    caps_out.supports_tools = caps["supports_tools"];
                }
                if (caps.count("supports_tool_calls")) {
                    caps_out.supports_tool_calls = caps["supports_tool_calls"];
                }
                if (caps.count("supports_system_role")) {
                    caps_out.supports_system_role = caps["supports_system_role"];
                }
                if (caps.count("supports_parallel_tool_calls")) {
                    caps_out.supports_parallel_tool_calls = caps["supports_parallel_tool_calls"];
                }
                if (caps.count("supports_reasoning_effort") && caps["supports_reasoning_effort"]) {
                    caps_out.supports_thinking         = true;
                    caps_out.supports_reasoning_effort = true;
                }
            } catch (const std::exception & e) {
                LOGw("get_or_compute_model_capabilities: get_caps warning: %s", e.what());
            } catch (...) {
            }
        }
        if (!caps_out.supports_reasoning_effort && caps_out.has_chat_template &&
            (effective_src.find("reasoning_effort") != std::string::npos ||
             tmpl_src.find("reasoning_effort") != std::string::npos)) {
            caps_out.supports_reasoning_effort = true;
            caps_out.supports_thinking         = true;
        }

        try {
            if (caps_out.supports_jinja && caps_out.supports_tools && caps_out.supports_tool_calls) {
                common_chat_templates_inputs probe_in;
                probe_in.use_jinja             = true;
                probe_in.force_pure_content    = false;
                probe_in.add_generation_prompt = true;
                common_chat_msg pm;
                pm.role           = ROLE_USER;
                pm.content        = "ping";
                probe_in.messages = { pm };
                common_chat_tool pt;
                pt.name        = "probe_fn";
                pt.description = "probe";
                pt.parameters  = "{\"type\":\"object\",\"properties\":{}}";
                probe_in.tools = { pt };
                caps_out.detected_format =
                    static_cast<int>(common_chat_templates_apply(g_chat_templates.get(), probe_in).format);
            } else if (caps_out.supports_jinja) {
                common_chat_templates_inputs probe_in;
                probe_in.use_jinja             = true;
                probe_in.force_pure_content    = false;
                probe_in.add_generation_prompt = true;
                common_chat_msg pm;
                pm.role           = ROLE_USER;
                pm.content        = "ping";
                probe_in.messages = { pm };
                caps_out.detected_format =
                    static_cast<int>(common_chat_templates_apply(g_chat_templates.get(), probe_in).format);
            } else {
                caps_out.detected_format = static_cast<int>(COMMON_CHAT_FORMAT_CONTENT_ONLY);
            }
        } catch (const std::exception & e) {
            LOGw("get_or_compute_model_capabilities: format probe warning: %s", e.what());
            caps_out.detected_format = static_cast<int>(COMMON_CHAT_FORMAT_CONTENT_ONLY);
        } catch (...) {
            caps_out.detected_format = static_cast<int>(COMMON_CHAT_FORMAT_CONTENT_ONLY);
        }

        caps_out.handler_name =
            (caps_out.has_chat_template && caps_out.supports_jinja) ?
                ((caps_out.supports_tools && caps_out.supports_tool_calls) ? "jinja2_native_tools" : "jinja2") :
                "default";
    }
    if (g_model != nullptr) {
        caps_out.ready = true;
        g_caps_cache   = caps_out;
    }
    return caps_out;
}

extern "C" JNIEXPORT jobject JNICALL
Java_com_arm_aichat_internal_InferenceEngineImpl_getModelCapabilitiesNative(JNIEnv * env, jobject thiz) {
    (void) thiz;
    std::lock_guard<std::recursive_mutex> cfg_lock(g_config_mutex);
    const CachedModelCapabilities         caps_out = get_or_compute_model_capabilities();

    jclass cls = env->FindClass("com/arm/aichat/NativeModelCapabilities");
    if (!cls) {
        LOGe("getModelCapabilitiesNative: Failed to find NativeModelCapabilities class");
        return nullptr;
    }
    jmethodID ctor = env->GetMethodID(cls, "<init>", "(ZZZZZILjava/lang/String;ZILjava/lang/String;ZZ)V");
    if (!ctor) {
        LOGe("getModelCapabilitiesNative: Failed to find NativeModelCapabilities <init>");
        env->DeleteLocalRef(cls);
        return nullptr;
    }
    jstring jhandler = safe_new_string_utf_nonnull(env, caps_out.handler_name);
    jstring jtmpl    = safe_new_string_utf_nonnull(env, caps_out.chat_template);
    jobject obj =
        env->NewObject(cls, ctor, (jboolean) caps_out.has_chat_template, (jboolean) caps_out.supports_jinja,
                       (jboolean) caps_out.supports_thinking, (jboolean) caps_out.supports_tools,
                       (jboolean) caps_out.supports_tool_calls, (jint) caps_out.detected_format, jhandler,
                       (jboolean) caps_out.supports_reasoning_effort, (jint) caps_out.context_length, jtmpl,
                       (jboolean) caps_out.supports_system_role, (jboolean) caps_out.supports_parallel_tool_calls);
    if (jhandler) {
        env->DeleteLocalRef(jhandler);
    }
    if (jtmpl) {
        env->DeleteLocalRef(jtmpl);
    }
    env->DeleteLocalRef(cls);
    return obj;
}

extern "C" JNIEXPORT jint JNICALL Java_com_arm_aichat_internal_InferenceEngineImpl_prepare(JNIEnv * env, jobject thiz) {
    (void) env;
    (void) thiz;
    std::lock_guard<std::recursive_mutex> cfg_lock(g_config_mutex);
    if (!g_model) {
        LOGe("prepare called with null g_model!");
        return 1;
    }
    if (g_sampler) {
        detach_backend_sampler_if_attached();
        common_sampler_free(g_sampler);
        g_sampler = nullptr;
    }
    if (g_batch_initialized) {
        llama_batch_free(g_batch);
        g_batch             = {};
        g_batch_initialized = false;
    }
    if (g_context) {
        llama_free(g_context);
        g_context = nullptr;
    }

    auto * context = init_context(g_model);
    if (!context) {
        return 1;
    }
    g_context = context;

    int n_batch_to_use  = (g_n_batch > 0) ? g_n_batch : BATCH_SIZE;
    g_batch             = llama_batch_init(n_batch_to_use, 0, 1);
    g_batch_initialized = true;

    if (!g_chat_templates && g_model) {
        g_chat_templates = common_chat_templates_init(g_model, "");
        if (g_chat_templates && g_chat_templates.get()) {
            const bool        has_explicit = common_chat_templates_was_explicit(g_chat_templates.get());
            const std::string tmpl_src     = common_chat_templates_source(g_chat_templates.get());
            std::string       tool_src;
            try {
                tool_src = common_chat_templates_source(g_chat_templates.get(), "tool_use");
            } catch (...) {
            }
            const std::string & effective_src = !tool_src.empty() ? tool_src : tmpl_src;
            g_use_jinja                       = has_explicit && (effective_src.find("{%") != std::string::npos ||
                                           effective_src.find("{{") != std::string::npos);
            bool native_thinking_api          = false;
            try {
                native_thinking_api =
                    has_explicit && common_chat_templates_support_enable_thinking(g_chat_templates.get());
            } catch (...) {
            }
            g_template_has_reasoning =
                has_explicit && (native_thinking_api || template_source_has_reasoning_tags(effective_src) ||
                                 template_source_has_reasoning_tags(tmpl_src));
        }
    }
    LOGi("prepare: Jinja2 mode=%s, native reasoning=%s", g_use_jinja ? "ENABLED" : "DISABLED",
         g_template_has_reasoning ? "YES" : "NO");
    if (!g_vocab_and_template_probed && g_model) {
        probe_model_template_and_vocab_once();
    }

    if (g_sampling_cache.valid || !g_sparams.grammar.empty() || g_sparams.ignore_eos) {
        common_params_sampling prep_sparams = g_sparams;
        apply_ignore_eos_logit_bias(prep_sparams);
        try {
            g_sampler                      = common_sampler_init(g_model, prep_sparams);
            g_sampler_has_turn_constraints = !g_sparams.grammar.empty();
        } catch (...) {
            g_sparams.grammar              = common_grammar();
            g_sampler_has_turn_constraints = false;
            g_sampler                      = new_sampler(DEFAULT_SAMPLER_TEMP);
        }
    } else {
        g_sampler_has_turn_constraints = false;
        g_sampler                      = new_sampler(DEFAULT_SAMPLER_TEMP);
    }
    attach_backend_sampler_if_enabled();
    return 0;
}

static std::string get_backend() {
    std::vector<std::string> backends;
    for (size_t i = 0; i < ggml_backend_reg_count(); i++) {
        auto *      reg  = ggml_backend_reg_get(i);
        std::string name = ggml_backend_reg_name(reg);
        if (name != "CPU") {
            backends.emplace_back(ggml_backend_reg_name(reg));
        }
    }
    return backends.empty() ? "CPU" : join(backends, ",");
}

extern "C" JNIEXPORT jstring JNICALL Java_com_arm_aichat_internal_InferenceEngineImpl_systemInfo(JNIEnv * env,
                                                                                                 jobject  thiz) {
    (void) thiz;
    const char * info = llama_print_system_info();
    return safe_new_string_utf_nonnull(env, info ? info : "");
}

extern "C" JNIEXPORT jstring JNICALL Java_com_arm_aichat_internal_InferenceEngineImpl_benchModel(JNIEnv * env,
                                                                                                 jobject  thiz,
                                                                                                 jint     pp,
                                                                                                 jint     tg,
                                                                                                 jint     pl,
                                                                                                 jint     nr) {
    (void) thiz;
    std::lock_guard<std::recursive_mutex> cfg_lock(g_config_mutex);
    if (!g_model) {
        const auto * const err_msg = "Model is not loaded! Bench aborted.";
        LOGe("%s", err_msg);
        return safe_new_string_utf_nonnull(env, err_msg);
    }
    if (pp <= 0 || tg <= 0 || pl <= 0 || nr <= 0) {
        const auto * const err_msg = "Invalid benchmark parameters! Bench aborted.";
        LOGe("%s", err_msg);
        return safe_new_string_utf_nonnull(env, err_msg);
    }
    const int64_t tg_pl_64 = static_cast<int64_t>(tg) * static_cast<int64_t>(pl);
    if (tg_pl_64 > INT32_MAX) {
        const auto * const err_msg = "Benchmark context size overflow! Bench aborted.";
        LOGe("%s", err_msg);
        return safe_new_string_utf_nonnull(env, err_msg);
    }
    const int bench_batch_cap = std::max<int>(1, std::max<int>(pp, pl));
    const int bench_n_ctx     = std::max<int>(pp, std::max<int>(bench_batch_cap, static_cast<int>(tg_pl_64)));

    bool                 temporarily_freed_g_ctx = false;
    bool                 had_non_text_media      = (g_kv_cells_used != g_cached_tokens.size());
    std::vector<uint8_t> saved_state_buf;
    size_t               saved_state_bytes = 0;
    const int            saved_g_n_ctx     = g_context ? static_cast<int>(llama_n_ctx(g_context)) : g_n_ctx;

    auto release_active_context_for_bench = [&]() {
        if (!g_context) {
            return;
        }
        if (g_kv_cells_used > 0) {
            try {
                llama_synchronize(g_context);
                const size_t est = llama_state_seq_get_size(g_context, 0);
                if (est > 0) {
                    saved_state_buf.resize(est);
                    saved_state_bytes =
                        llama_state_seq_get_data(g_context, saved_state_buf.data(), saved_state_buf.size(), 0);
                    if (saved_state_bytes == 0) {
                        saved_state_buf.clear();
                    }
                }
            } catch (...) {
                saved_state_buf.clear();
                saved_state_bytes = 0;
            }
        }
        detach_backend_sampler_if_attached();
        llama_free(g_context);
        g_context               = nullptr;
        temporarily_freed_g_ctx = true;
    };

    auto restore_active_context_after_bench = [&]() {
        if (!temporarily_freed_g_ctx) {
            return;
        }
        g_context = init_context(g_model, saved_g_n_ctx);
        if (!g_context) {
            LOGe("benchModel: failed to restore g_context after benchmark (OOM). Clearing long-term state.");
            reset_long_term_states(false, false);
            reset_short_term_states();
            return;
        }
        if (g_kv_cells_used > 0) {
            size_t restored = 0;
            if (saved_state_bytes > 0) {
                try {
                    restored = llama_state_seq_set_data(g_context, saved_state_buf.data(), saved_state_bytes, 0);
                } catch (...) {
                    restored = 0;
                }
            }
            if (restored == 0) {
                if (!had_non_text_media && !g_cached_tokens.empty()) {
                    llama_memory_clear(llama_get_memory(g_context), false);
                    current_position = 0;
                    g_kv_cells_used  = 0;
                    if (!g_batch_initialized ||
                        decode_tokens_in_batches(g_context, g_batch, g_cached_tokens, 0, false) != 0) {
                        reset_long_term_states(true, false);
                    }
                } else {
                    reset_long_term_states(true, false);
                }
            }
        }
        attach_backend_sampler_if_enabled();
    };

    // Free empty g_context up front to avoid double KV cache RAM spike on mobile devices
    if (g_context && g_kv_cells_used == 0) {
        release_active_context_for_bench();
    }

    auto * context = init_context(g_model, bench_n_ctx, bench_batch_cap);
    if (!context && g_context) {
        LOGw("benchModel: OOM with active g_context in RAM; temporarily unloading active KV context and retrying...");
        release_active_context_for_bench();
        context = init_context(g_model, bench_n_ctx, bench_batch_cap);
    }
    if (!context) {
        restore_active_context_after_bench();
        const auto * const err_msg = "Fail to init_context! Bench aborted.";
        LOGe("%s", err_msg);
        return safe_new_string_utf_nonnull(env, err_msg);
    }

    llama_batch bench_batch = llama_batch_init(bench_batch_cap, 0, 1);

    auto pp_avg = 0.0;
    auto tg_avg = 0.0;
    auto pp_std = 0.0;
    auto tg_std = 0.0;

    const uint32_t n_ctx = llama_n_ctx(context);
    LOGi("n_ctx = %d", n_ctx);

    int i;
    int j;
    int nri;
    for (nri = 0; nri < nr; nri++) {
        LOGi("Benchmark prompt processing (pp = %d)", pp);

        common_batch_clear(bench_batch);

        const int n_tokens = pp;
        for (i = 0; i < n_tokens; i++) {
            common_batch_add(bench_batch, 0, i, { 0 }, false);
        }

        bench_batch.logits[bench_batch.n_tokens - 1] = true;
        llama_memory_clear(llama_get_memory(context), false);

        const auto t_pp_start = ggml_time_us();
        if (llama_decode(context, bench_batch) != 0) {
            LOGe("llama_decode() failed during prompt processing");
        }
        const auto t_pp_end = ggml_time_us();

        // bench text generation

        LOGi("Benchmark text generation (tg = %d)", tg);

        llama_memory_clear(llama_get_memory(context), false);
        const auto t_tg_start = ggml_time_us();
        for (i = 0; i < tg; i++) {
            common_batch_clear(bench_batch);
            for (j = 0; j < pl; j++) {
                common_batch_add(bench_batch, 0, i, { j }, true);
            }

            if (llama_decode(context, bench_batch) != 0) {
                LOGe("llama_decode() failed during text generation");
            }
        }
        const auto t_tg_end = ggml_time_us();

        llama_memory_clear(llama_get_memory(context), false);

        const auto t_pp = std::max(1e-6, double(t_pp_end - t_pp_start) / 1000000.0);
        const auto t_tg = std::max(1e-6, double(t_tg_end - t_tg_start) / 1000000.0);

        const auto speed_pp = double(pp) / t_pp;
        const auto speed_tg = double(pl * tg) / t_tg;

        pp_avg += speed_pp;
        tg_avg += speed_tg;

        pp_std += speed_pp * speed_pp;
        tg_std += speed_tg * speed_tg;

        LOGi("pp %f t/s, tg %f t/s", speed_pp, speed_tg);
    }

    llama_batch_free(bench_batch);
    llama_free(context);
    restore_active_context_after_bench();

    pp_avg /= double(nr);
    tg_avg /= double(nr);

    if (nr > 1) {
        pp_std = sqrt(std::max(0.0, pp_std / double(nr - 1) - pp_avg * pp_avg * double(nr) / double(nr - 1)));
        tg_std = sqrt(std::max(0.0, tg_std / double(nr - 1) - tg_avg * tg_avg * double(nr) / double(nr - 1)));
    } else {
        pp_std = 0;
        tg_std = 0;
    }

    char model_desc[128];
    llama_model_desc(g_model, model_desc, sizeof(model_desc));

    const auto model_size     = double(llama_model_size(g_model)) / 1024.0 / 1024.0 / 1024.0;
    const auto model_n_params = double(llama_model_n_params(g_model)) / 1e9;

    const auto        backend = get_backend();
    std::stringstream result;
    result << std::setprecision(3);
    result << "| model | size | params | backend | test | t/s |\n";
    result << "| --- | --- | --- | --- | --- | --- |\n";
    result << "| " << model_desc << " | " << model_size << "GiB | " << model_n_params << "B | " << backend << " | pp "
           << pp << " | " << pp_avg << " ± " << pp_std << " |\n";
    result << "| " << model_desc << " | " << model_size << "GiB | " << model_n_params << "B | " << backend << " | tg "
           << tg << " | " << tg_avg << " ± " << tg_std << " |\n";
    return safe_new_string_utf_nonnull(env, result.str());
}

/**
 * Completion loop's long-term states:
 * - chat management
 * - position tracking
 */
static void reset_long_term_states(const bool clear_kv_cache, const bool keep_bitmap) {
    chat_msgs.clear();
    system_prompt_position = 0;
    current_position       = 0;
    g_kv_cells_used        = 0;
    g_cached_tokens.clear();
    g_deferred_system_prompt.clear();

    g_pending_tool_call = PendingToolCallState{};
    g_pending_tool_calls.clear();
    g_staged_tool_msgs.clear();
    g_current_turn_format.prompt.clear();
    g_last_tool_formatted_delta.clear();
    invalidate_turn_prompt_caches();

    if (!keep_bitmap && g_cached_bmp) {
        mtmd_bitmap_free(g_cached_bmp);
        g_cached_bmp = nullptr;
    }

    if (clear_kv_cache && g_context) {
        llama_memory_clear(llama_get_memory(g_context), false);
    }
}

// Preserves optional leading ROLE_SYSTEM entry, trims oldest non-system history to max_msgs,
// and drops exposed non-user turns so chat templates never receive orphaned assistant/tool turns.
static void prune_chat_history_keeping_recent(const size_t max_msgs) {
    const size_t start_idx = (!chat_msgs.empty() && chat_msgs.front().role == ROLE_SYSTEM) ? 1 : 0;
    while (chat_msgs.size() > max_msgs && chat_msgs.size() > start_idx) {
        chat_msgs.erase(chat_msgs.begin() + static_cast<ptrdiff_t>(start_idx));
    }
    while (chat_msgs.size() > start_idx && chat_msgs[start_idx].role != ROLE_USER) {
        chat_msgs.erase(chat_msgs.begin() + static_cast<ptrdiff_t>(start_idx));
    }
}

/**
 * Context shifting by discarding the older half of the tokens appended after system prompt:
 * - take the [system_prompt_position] first tokens from the original prompt
 * - take half of the last (system_prompt_position - system_prompt_position) tokens
 * - recompute the logits in batches
 *
 * Returns true if shift succeeded, false if in-place shifting is unsupported or impossible.
 * CRITICAL: Never calls reset_long_term_states() internally to prevent decoding orphaned delta prompts!
 */
static bool shift_context_ex(llama_pos & inout_pos, size_t & inout_kv_cells) {
    llama_memory_t mem                = g_context ? llama_get_memory(g_context) : nullptr;
    const bool     has_non_text_media = (g_kv_cells_used != g_cached_tokens.size());
    const bool     is_hexagon_active  = (g_active_backend_info.rfind("Hexagon", 0) == 0);
    if (!mem || has_non_text_media || is_hexagon_active || !llama_memory_can_shift(mem)) {
        LOGw(
            "%s: Model/memory/backend does not support safe in-place K-shift (Hexagon/SWA/ISWA/M-RoPE/Multimodal). "
            "Skipping shift.",
            __func__);
        return false;
    }

    if (system_prompt_position < 0 || inout_pos <= system_prompt_position) {
        LOGw("%s: Insufficient shift range (sys_pos=%d, pos=%d). Skipping shift.", __func__, system_prompt_position,
             inout_pos);
        return false;
    }

    // Parciální kontext shift pro běžné textové modely s podporou posunu (Llama, Mistral, standardní KV)
    const int available = inout_pos - system_prompt_position;
    int       n_discard = available / 2;
    if (available > 1 && available < 240) {
        const int keep_tail = std::min(available - 1, 120);
        n_discard           = available - keep_tail;
    }
    if (n_discard <= 0) {
        LOGw("%s: n_discard <= 0 (%d). Cannot discard tokens via shift.", __func__, n_discard);
        return false;
    }

    LOGi("%s: Context full (shift supported). Discarding %d tokens via llama_memory_seq_rm/add (boundary=%d)", __func__,
         n_discard, inout_pos);
    if (!llama_memory_seq_rm(llama_get_memory(g_context), 0, system_prompt_position,
                             system_prompt_position + n_discard)) {
        LOGe("%s: llama_memory_seq_rm failed during shift_context!", __func__);
        return false;
    }
    llama_memory_seq_add(llama_get_memory(g_context), 0, system_prompt_position + n_discard, inout_pos, -n_discard);
    inout_pos -= n_discard;
    if (g_gen_state.is_active && stop_generation_position != INT32_MAX && stop_generation_position > inout_pos) {
        stop_generation_position = std::max(inout_pos + 1, stop_generation_position - n_discard);
    }
    if (inout_kv_cells >= (size_t) n_discard) {
        inout_kv_cells -= n_discard;
    } else {
        inout_kv_cells = inout_pos;
    }
    if (g_cached_tokens.size() >= (size_t) (system_prompt_position + n_discard)) {
        g_cached_tokens.erase(g_cached_tokens.begin() + system_prompt_position,
                              g_cached_tokens.begin() + system_prompt_position + n_discard);
    } else {
        g_cached_tokens.clear();
    }
    LOGi("%s: Context shifting done! Active position: %d, kv_cells_used: %zu, cached_tokens: %zu", __func__, inout_pos,
         inout_kv_cells, g_cached_tokens.size());
    return true;
}

static bool shift_context(const bool keep_bitmap = false) {
    (void) keep_bitmap;
    return shift_context_ex(current_position, g_kv_cells_used);
}

static int decode_tokens_in_batches(llama_context *      context,
                                    llama_batch &        batch,
                                    const llama_tokens & tokens,
                                    const llama_pos      start_pos,
                                    const bool           compute_last_logit);

static bool rebuild_context_preserving_tail(llama_pos & inout_pos,
                                            size_t &    inout_kv_cells,
                                            const int   reserve_tokens,
                                            const bool  compute_last_logit) {
    if (!g_context || !g_model || g_cached_tokens.empty()) {
        return false;
    }
    llama_memory_t mem = llama_get_memory(g_context);
    if (!mem) {
        return false;
    }

    const int context_size = static_cast<int>(llama_n_ctx(g_context));
    const int max_usable   = std::max(1, context_size - OVERFLOW_HEADROOM);
    const int needed       = std::max(1, reserve_tokens);
    if (needed >= max_usable) {
        return false;
    }

    const int total_cached = static_cast<int>(g_cached_tokens.size());
    const int old_pos      = inout_pos;
    const int budget       = max_usable - needed;
    const int raw_sys      = (system_prompt_position > 0 && system_prompt_position <= total_cached) ?
                                 static_cast<int>(system_prompt_position) :
                                 0;

    int keep_sys = std::min(raw_sys, budget);
    if (keep_sys == 0 && total_cached > 1 && budget > 1) {
        const llama_vocab * vocab = llama_model_get_vocab(g_model);
        const llama_token   bos   = vocab ? llama_vocab_bos(vocab) : LLAMA_TOKEN_NULL;
        if (bos != LLAMA_TOKEN_NULL && g_cached_tokens.front() == bos) {
            keep_sys = 1;
        }
    }

    const int avail_for_tail  = std::max(0, budget - keep_sys);
    const int hist_len        = std::max(0, total_cached - keep_sys);
    const int max_tail_budget = std::max(500, (avail_for_tail * 3) / 5);
    const int desired_tail    = std::clamp((hist_len * 2) / 3, 120, max_tail_budget);
    int       keep_tail       = std::min({ hist_len, avail_for_tail, desired_tail });
    if (keep_sys + keep_tail >= total_cached && hist_len > 1) {
        const int min_discard = std::max(1, std::min(hist_len / 3, 128));
        keep_tail             = std::max(0, std::min(keep_tail, hist_len - min_discard));
    }

    if (keep_sys + keep_tail <= 0) {
        return false;
    }

    llama_tokens rebuilt_tokens;
    rebuilt_tokens.reserve(static_cast<size_t>(keep_sys) + static_cast<size_t>(keep_tail));
    if (keep_sys > 0) {
        rebuilt_tokens.insert(rebuilt_tokens.end(), g_cached_tokens.begin(), g_cached_tokens.begin() + keep_sys);
    }
    if (keep_tail > 0) {
        rebuilt_tokens.insert(rebuilt_tokens.end(), g_cached_tokens.end() - keep_tail, g_cached_tokens.end());
    }

    llama_memory_clear(mem, false);
    current_position       = 0;
    g_kv_cells_used        = 0;
    system_prompt_position = keep_sys;

    const int dec_res = decode_tokens_in_batches(g_context, g_batch, rebuilt_tokens, 0, compute_last_logit);
    if (dec_res != 0) {
        LOGe("%s: decode_tokens_in_batches failed with %d while rebuilding context tail.", __func__, dec_res);
        reset_long_term_states(true, false);
        inout_pos      = 0;
        inout_kv_cells = 0;
        return false;
    }

    g_cached_tokens   = std::move(rebuilt_tokens);
    inout_pos         = current_position;
    inout_kv_cells    = g_kv_cells_used;
    const int dropped = std::max(0, old_pos - inout_pos);
    if (g_gen_state.is_active && stop_generation_position != INT32_MAX && dropped > 0) {
        stop_generation_position = std::max(inout_pos + 1, stop_generation_position - dropped);
    }
    LOGi("%s: Rebuilt KV cache preserving prefix=%d + tail=%d tokens (new pos=%d, dropped=%d).", __func__, keep_sys,
         keep_tail, inout_pos, dropped);
    return true;
}

static const std::vector<CoTTagDef> & get_universal_cot_catalog() {
    static const std::vector<CoTTagDef> catalog = {
        { "<|channel>thought",              { "<channel|>", "<|channel>", "<|turn>", "</channel>" }                           },
        { "<|channel|>thought",             { "<channel|>", "<|channel|>", "</channel>", "<|channel>", "<|turn>", "<|end|>" } },
        { "<|channel|>analysis<|message|>", { "<|end|>", "<|channel|>final<|message|>" }                                      },
        { "<|think|>",                      { "<|/think|>", "</think>", "<|think|>" }                                         },
        { "[THINK]",                        { "[/THINK]" }                                                                    },
        { "[THOUGHT]",                      { "[/THOUGHT]" }                                                                  },
        { "[REASONING]",                    { "[/REASONING]" }                                                                },
        { "<think>",                        { "</think>" }                                                                    },
        { "<thinking>",                     { "</thinking>" }                                                                 },
        { "<thought>",                      { "</thought>" }                                                                  },
        { "<reasoning>",                    { "</reasoning>" }                                                                },
        { "<reason>",                       { "</reason>" }                                                                   },
        { "<reflection>",                   { "</reflection>" }                                                               },
        { "<analysis>",                     { "</analysis>" }                                                                 },
        { "<scratchpad>",                   { "</scratchpad>" }                                                               },
        { "<inner_monologue>",              { "</inner_monologue>" }                                                          },
        { "<seed:think>",                   { "</seed:think>" }                                                               },
        { "<start_of_thought>",             { "<end_of_thought>" }                                                            },
        { "<|thought|>",                    { "<|thought|>", "</thought>" }                                                   },
        { "<|reasoning|>",                  { "<|/reasoning|>", "</reasoning>" }                                              },
        { "<|begin_of_thought|>",           { "<|end_of_thought|>" }                                                          },
        { "<｜begin▁of▁thought｜>",         { "<｜end▁of▁thought｜>" }                                                        },
        { "<|im_start|>thought",            { "<|im_end|>", "</thought>", "</think>" }                                        },
        { "<channel>thought",               { "<channel|>", "</channel>" }                                                    }
    };
    return catalog;
}

static void add_unique_cot_rule(const CoTTagDef & rule) {
    for (const auto & existing : g_active_cot_rules) {
        if (existing.open_tag == rule.open_tag) {
            return;
        }
    }
    g_active_cot_rules.push_back(rule);
}

static bool strip_trailing_think_start(std::string & text, const char * caller) {
    if (g_template_honors_disable_thinking) {
        // The template itself changes output when enable_thinking=false (e.g. emits empty thought block
        // or omits the start tag); never alter template output blindly.
        return false;
    }
    size_t non_ws_end = text.size();
    while (non_ws_end > 0 &&
           (text[non_ws_end - 1] == '\n' || text[non_ws_end - 1] == '\r' || text[non_ws_end - 1] == ' ')) {
        --non_ws_end;
    }
    for (const auto & rule : g_active_cot_rules) {
        if (non_ws_end >= rule.open_tag.size() &&
            text.compare(non_ws_end - rule.open_tag.size(), rule.open_tag.size(), rule.open_tag) == 0) {
            text.erase(non_ws_end - rule.open_tag.size());
            LOGi("%s: Stripped unclosed prefilled thinking tag '%s' (thinking is OFF).", caller, rule.open_tag.c_str());
            return true;
        }
    }
    return false;
}

static bool has_native_tool_template_caps() {
    if (!g_chat_templates || !g_chat_templates.get() || !g_use_jinja || g_configured_tools.empty()) {
        return false;
    }
    const auto caps     = common_chat_templates_get_caps(g_chat_templates.get());
    auto       it_tools = caps.find("supports_tools");
    auto       it_calls = caps.find("supports_tool_calls");
    return (it_tools != caps.end() && it_tools->second) && (it_calls != caps.end() && it_calls->second);
}

static bool supports_native_tool_templates() {
    if (g_tool_policy == NativeToolChoice::NONE || !g_allow_native_tools_hint) {
        return false;
    }
    return has_native_tool_template_caps();
}

static void probe_model_template_and_vocab_once() {
    g_preserved_token_ids.clear();
    g_additional_stop_token_ids.clear();
    g_template_stop_strings.clear();
    if (!g_custom_cot_rules_set) {
        g_active_cot_rules.clear();
    }
    g_template_honors_disable_thinking = false;
    g_vocab_and_template_probed        = true;

    if (!g_model) {
        return;
    }
    const llama_vocab * vocab = llama_model_get_vocab(g_model);
    if (!vocab) {
        return;
    }

    std::string tmpl_src;
    if (g_chat_templates && g_chat_templates.get()) {
        tmpl_src = common_chat_templates_source(g_chat_templates.get());
        std::string tool_src;
        try {
            tool_src = common_chat_templates_source(g_chat_templates.get(), "tool_use");
        } catch (...) {
        }
        if (!tool_src.empty()) {
            tmpl_src = tool_src;
        }
    }

    // 1. Single O(n_vocab) pass: preserve all model-native control/tag tokens without hardcoding token lists.
    const llama_token bos_id  = llama_vocab_bos(vocab);
    const llama_token pad_id  = llama_vocab_pad(vocab);
    const llama_token sep_id  = llama_vocab_sep(vocab);
    const llama_token mask_id = llama_vocab_mask(vocab);
    const int32_t     n_vocab = llama_vocab_n_tokens(vocab);

    for (int32_t id = 0; id < n_vocab; ++id) {
        if (id == bos_id || id == pad_id || id == sep_id || id == mask_id) {
            continue;
        }
        const llama_token_attr attr = llama_vocab_get_attr(vocab, id);
        if ((attr & (LLAMA_TOKEN_ATTR_CONTROL | LLAMA_TOKEN_ATTR_USER_DEFINED)) == 0) {
            continue;
        }
        if (llama_vocab_is_eog(vocab, id)) {
            continue;
        }
        const char * raw_txt = llama_vocab_get_text(vocab, id);
        if (!raw_txt || !*raw_txt) {
            continue;
        }
        if ((attr & LLAMA_TOKEN_ATTR_CONTROL) != 0) {
            g_preserved_token_ids.insert(id);
        } else {
            const std::string_view sv(raw_txt);
            const bool             looks_like_tag =
                (sv.find('<') != std::string_view::npos && sv.find('>') != std::string_view::npos) ||
                (sv.find('[') != std::string_view::npos && sv.find(']') != std::string_view::npos);
            const bool in_template = (!tmpl_src.empty() && tmpl_src.find(sv) != std::string::npos);
            if (looks_like_tag || in_template) {
                g_preserved_token_ids.insert(id);
            }
        }
    }

    // 2. One-time lightweight Jinja probe (force_pure_content=true, zero autoparser/PEG)
    std::string probe_on_prompt;
    std::string probe_off_prompt;
    if (g_chat_templates && g_chat_templates.get() && g_use_jinja) {
        try {
            common_chat_msg u_msg;
            u_msg.role    = ROLE_USER;
            u_msg.content = "U";

            common_chat_msg a_msg;
            a_msg.role    = ROLE_ASSISTANT;
            a_msg.content = "ASST_END_MARKER_X";

            common_chat_templates_inputs stop_inputs;
            stop_inputs.use_jinja             = true;
            stop_inputs.force_pure_content    = false;
            stop_inputs.enable_thinking       = true;
            stop_inputs.reasoning_format      = COMMON_REASONING_FORMAT_DEEPSEEK;
            stop_inputs.add_eos               = false;
            stop_inputs.add_generation_prompt = false;
            stop_inputs.messages              = { u_msg, a_msg };

            std::string hist_rendered;
            try {
                hist_rendered = common_chat_templates_apply(g_chat_templates.get(), stop_inputs).prompt;
            } catch (...) {
                stop_inputs.force_pure_content = true;
                stop_inputs.reasoning_format   = COMMON_REASONING_FORMAT_NONE;
                hist_rendered = common_chat_templates_apply(g_chat_templates.get(), stop_inputs).prompt;
            }
            const size_t marker_len = sizeof("ASST_END_MARKER_X") - 1;
            const size_t marker_pos = hist_rendered.rfind("ASST_END_MARKER_X");
            if (marker_pos != std::string::npos) {
                std::string term = hist_rendered.substr(marker_pos + marker_len);
                while (!term.empty() && (term.back() == '\n' || term.back() == '\r' || term.back() == ' ')) {
                    term.pop_back();
                }
                while (!term.empty() && (term.front() == '\n' || term.front() == '\r' || term.front() == ' ')) {
                    term.erase(term.begin());
                }
                if (!term.empty() && term.size() <= 40 && term.find("U") == std::string::npos) {
                    g_template_stop_strings.push_back(term);
                    auto ids = common_tokenize(vocab, term, false, true);
                    if (ids.size() == 1 && !llama_vocab_is_eog(vocab, ids[0])) {
                        g_additional_stop_token_ids.insert(ids[0]);
                    }
                }
            }

            common_chat_templates_inputs think_inputs;
            think_inputs.use_jinja             = true;
            think_inputs.force_pure_content    = true;
            think_inputs.reasoning_format      = COMMON_REASONING_FORMAT_NONE;
            think_inputs.add_eos               = false;
            think_inputs.add_generation_prompt = true;
            think_inputs.messages              = { u_msg };

            think_inputs.enable_thinking = true;
            probe_on_prompt              = common_chat_templates_apply(g_chat_templates.get(), think_inputs).prompt;

            think_inputs.enable_thinking = false;
            probe_off_prompt             = common_chat_templates_apply(g_chat_templates.get(), think_inputs).prompt;

            g_template_honors_disable_thinking = (probe_on_prompt != probe_off_prompt);
        } catch (const std::exception & e) {
            LOGw("%s: lightweight template probe warning: %s", __func__, e.what());
        } catch (...) {
            LOGw("%s: lightweight template probe warning (unknown error)", __func__);
        }
    }

    if (!g_custom_cot_rules_set) {
        // 3. Prioritize CoT rules detected in this model's template/probe outputs, then keep full universal catalog active
        for (const auto & rule : get_universal_cot_catalog()) {
            const bool in_src = (!tmpl_src.empty() && tmpl_src.find(rule.open_tag) != std::string::npos);
            const bool in_on  = (!probe_on_prompt.empty() && probe_on_prompt.find(rule.open_tag) != std::string::npos);
            const bool in_off =
                (!probe_off_prompt.empty() && probe_off_prompt.find(rule.open_tag) != std::string::npos);
            if (in_src || in_on || in_off) {
                add_unique_cot_rule(rule);
            }
        }
        // Fallback for LoRA/fine-tuned/template-less GGUF models that emit CoT delimiters absent from template source
        for (const auto & rule : get_universal_cot_catalog()) {
            add_unique_cot_rule(rule);
        }

        if (!g_active_cot_rules.empty()) {
            g_model_thinking_start_tag = g_active_cot_rules.front().open_tag;
            if (!g_active_cot_rules.front().close_tags.empty()) {
                g_model_thinking_end_tag = g_active_cot_rules.front().close_tags.front();
            }
        }

        g_cot_lead_chars.clear();
        for (const auto & rule : g_active_cot_rules) {
            if (!rule.open_tag.empty()) {
                const char lead = rule.open_tag.front();
                if (g_cot_lead_chars.find(lead) == std::string::npos) {
                    g_cot_lead_chars.push_back(lead);
                }
            }
        }
        if (g_cot_lead_chars.empty()) {
            g_cot_lead_chars = "<[";
        }
    }
}

static bool should_add_bos_for_prompt(const std::string & formatted_text,
                                      bool                has_chat_template,
                                      bool                assume_start_of_context) {
    if (!has_chat_template || (!assume_start_of_context && current_position != 0) || !g_model || !g_context) {
        return false;
    }
    const llama_vocab * vocab = llama_model_get_vocab(g_model);
    const llama_token   bos   = vocab ? llama_vocab_bos(vocab) : LLAMA_TOKEN_NULL;
    if (bos != LLAMA_TOKEN_NULL) {
        const std::string bos_piece = common_token_to_piece(g_context, bos, true);
        if (!bos_piece.empty() && formatted_text.rfind(bos_piece, 0) == 0) {
            return false;
        }
    }
    return true;
}

[[maybe_unused]] static std::vector<common_chat_msg> compact_history_for_delta(
    const std::vector<common_chat_msg> & src) {
    std::vector<common_chat_msg> out;
    out.reserve(src.size());
    for (size_t i = 0; i < src.size(); ++i) {
        common_chat_msg m = src[i];
        // Shrink bulky historical user/assistant bodies already committed in KV cache, but preserve ROLE_TOOL content
        // so specialized handlers (e.g. Gemma 4 convert_tool_responses) keep full structured tool history intact.
        if (m.role != ROLE_TOOL && m.content.size() > 128 && m.tool_calls.empty()) {
            m.content = "x";
        }
        if (m.reasoning_content.size() > 128) {
            m.reasoning_content = "x";
        }
        out.push_back(std::move(m));
    }
    return out;
}

static bool template_supports_reasoning_effort() {
    if (!g_use_jinja || !g_chat_templates || !g_chat_templates.get()) {
        return false;
    }
    if (g_caps_cache.ready) {
        return g_caps_cache.supports_reasoning_effort;
    }
    try {
        const auto caps = common_chat_templates_get_caps(g_chat_templates.get());
        const auto it   = caps.find("supports_reasoning_effort");
        if (it != caps.end() && it->second) {
            return true;
        }
    } catch (...) {
    }
    return false;
}

static void apply_reasoning_effort_kwarg(common_chat_templates_inputs & inputs) {
    if (g_thinking_effort.empty()) {
        return;
    }
    if (g_thinking_effort != "none" || (!g_enable_thinking && template_supports_reasoning_effort())) {
        inputs.chat_template_kwargs["reasoning_effort"] = "\"" + g_thinking_effort + "\"";
    }
}

static common_chat_params apply_chat_templates_safe(const common_chat_templates *  tmpls,
                                                    common_chat_templates_inputs & inputs) {
    try {
        return common_chat_templates_apply(tmpls, inputs);
    } catch (...) {
        if (inputs.chat_template_kwargs.erase("reasoning_effort") > 0) {
            LOGw("%s: common_chat_templates_apply failed with reasoning_effort; retrying without it.", __func__);
            return common_chat_templates_apply(tmpls, inputs);
        }
        throw;
    }
}

static common_chat_params render_jinja_params(const common_chat_templates_inputs & base_inputs,
                                              const std::vector<common_chat_msg> & msgs,
                                              bool                                 add_generation_prompt,
                                              bool                                 add_bos,
                                              bool                                 force_pure_content) {
    common_chat_templates_inputs in = base_inputs;
    in.use_jinja                    = true;
    in.force_pure_content           = force_pure_content;
    in.reasoning_format      = force_pure_content ? COMMON_REASONING_FORMAT_NONE : COMMON_REASONING_FORMAT_DEEPSEEK;
    in.add_eos               = false;
    in.add_bos               = add_bos;
    in.add_generation_prompt = add_generation_prompt;
    in.messages              = msgs;
    return apply_chat_templates_safe(g_chat_templates.get(), in);
}

static std::string render_pure_jinja(const common_chat_templates_inputs & base_inputs,
                                     const std::vector<common_chat_msg> & msgs,
                                     bool                                 add_generation_prompt,
                                     bool                                 add_bos) {
    return render_jinja_params(base_inputs, msgs, add_generation_prompt, add_bos, /* force_pure_content = */ true)
        .prompt;
}

static bool can_use_body_sentinel(const std::string & content) {
    if (content.size() <= 2048) {
        return false;
    }
    const unsigned char first = static_cast<unsigned char>(content.front());
    const unsigned char last  = static_cast<unsigned char>(content.back());
    if (first <= ' ' || last <= ' ') {
        return false;
    }
    if (content.find("__AGY_MSG_BODY_MARKER__") != std::string::npos) {
        return false;
    }
    return true;
}

static bool has_tool_history_in_chat() {
    for (const auto & m : chat_msgs) {
        if (m.role == ROLE_TOOL || !m.tool_calls.empty()) {
            return true;
        }
    }
    return false;
}

static void register_preserved_token_strings(const std::vector<std::string> & tokens_str,
                                             std::set<llama_token> *          out_token_set = nullptr) {
    if (!g_model || tokens_str.empty()) {
        return;
    }
    const llama_vocab * vocab = llama_model_get_vocab(g_model);
    if (!vocab) {
        return;
    }
    for (const auto & tok_str : tokens_str) {
        if (tok_str.empty()) {
            continue;
        }
        const auto ids = common_tokenize(vocab, tok_str, false, true);
        if (ids.size() == 1) {
            g_preserved_token_ids.insert(ids[0]);
            if (out_token_set) {
                out_token_set->insert(ids[0]);
            }
        }
    }
}

static void apply_turn_format_from_params(const common_chat_params & params) {
    g_current_turn_format.format             = params.format;
    g_current_turn_format.parser_source      = params.parser;
    g_current_turn_format.generation_prompt  = params.generation_prompt;
    g_current_turn_format.grammar            = params.grammar;
    g_current_turn_format.grammar_lazy       = params.grammar_lazy;
    g_current_turn_format.grammar_triggers   = params.grammar_triggers;
    g_current_turn_format.preserved_tokens   = params.preserved_tokens;
    g_current_turn_format.thinking_start_tag = params.thinking_start_tag;
    g_current_turn_format.thinking_end_tags  = params.thinking_end_tags;
    g_current_turn_format.additional_stops   = params.additional_stops;
    register_preserved_token_strings(g_current_turn_format.preserved_tokens);
}

static void ensure_native_tools_parser_cached(bool use_native_tools_apply) {
    if (!g_use_jinja || !g_chat_templates || !g_chat_templates.get()) {
        g_current_turn_format = NativeTurnFormat{};
        return;
    }

    const NativeToolChoice eff_policy = use_native_tools_apply ? g_tool_policy : NativeToolChoice::NONE;
    if (!g_native_parser_cache.valid || g_native_parser_cache.tools_version != g_tools_config_version ||
        g_native_parser_cache.tool_policy != eff_policy || g_native_parser_cache.enable_thinking != g_enable_thinking ||
        g_native_parser_cache.thinking_effort != g_thinking_effort) {
        try {
            const auto caps   = common_chat_templates_get_caps(g_chat_templates.get());
            const auto it_par = caps.find("supports_parallel_tool_calls");

            common_chat_templates_inputs meta_inputs;
            meta_inputs.use_jinja             = true;
            meta_inputs.force_pure_content    = false;
            meta_inputs.enable_thinking       = g_enable_thinking;
            meta_inputs.reasoning_format      = COMMON_REASONING_FORMAT_DEEPSEEK;
            meta_inputs.add_eos               = false;
            meta_inputs.add_bos               = false;
            meta_inputs.add_generation_prompt = true;
            meta_inputs.parallel_tool_calls   = (it_par != caps.end() && it_par->second);
            if (use_native_tools_apply) {
                meta_inputs.tools       = g_configured_tools;
                meta_inputs.tool_choice = (g_tool_policy == NativeToolChoice::NONE) ? COMMON_CHAT_TOOL_CHOICE_NONE :
                                          (g_tool_policy == NativeToolChoice::REQUIRED) ?
                                                                                      COMMON_CHAT_TOOL_CHOICE_REQUIRED :
                                                                                      COMMON_CHAT_TOOL_CHOICE_AUTO;
            }
            apply_reasoning_effort_kwarg(meta_inputs);
            common_chat_msg ping;
            ping.role            = ROLE_USER;
            ping.content         = "ping";
            meta_inputs.messages = { ping };

            common_chat_params meta_params;
            try {
                meta_params = apply_chat_templates_safe(g_chat_templates.get(), meta_inputs);
            } catch (...) {
                meta_inputs.force_pure_content = true;
                meta_inputs.reasoning_format   = COMMON_REASONING_FORMAT_NONE;
                meta_params                    = apply_chat_templates_safe(g_chat_templates.get(), meta_inputs);
            }

            g_native_parser_cache                    = CachedNativeParser{};
            g_native_parser_cache.valid              = true;
            g_native_parser_cache.tools_version      = g_tools_config_version;
            g_native_parser_cache.tool_policy        = eff_policy;
            g_native_parser_cache.enable_thinking    = g_enable_thinking;
            g_native_parser_cache.thinking_effort    = g_thinking_effort;
            g_native_parser_cache.format             = meta_params.format;
            g_native_parser_cache.parser_source      = meta_params.parser;
            g_native_parser_cache.generation_prompt  = meta_params.generation_prompt;
            g_native_parser_cache.grammar            = meta_params.grammar;
            g_native_parser_cache.grammar_lazy       = meta_params.grammar_lazy;
            g_native_parser_cache.grammar_triggers   = meta_params.grammar_triggers;
            g_native_parser_cache.preserved_tokens   = meta_params.preserved_tokens;
            g_native_parser_cache.thinking_start_tag = meta_params.thinking_start_tag;
            g_native_parser_cache.thinking_end_tags  = meta_params.thinking_end_tags;
            g_native_parser_cache.additional_stops   = meta_params.additional_stops;
        } catch (const std::exception & e) {
            LOGw("%s: ensure_native_tools_parser_cached exception: %s", __func__, e.what());
            g_native_parser_cache = CachedNativeParser{};
        } catch (...) {
            LOGw("%s: ensure_native_tools_parser_cached unknown exception", __func__);
            g_native_parser_cache = CachedNativeParser{};
        }
    }

    if (g_native_parser_cache.valid) {
        g_current_turn_format.format             = g_native_parser_cache.format;
        g_current_turn_format.parser_source      = g_native_parser_cache.parser_source;
        g_current_turn_format.generation_prompt  = g_native_parser_cache.generation_prompt;
        g_current_turn_format.grammar            = g_native_parser_cache.grammar;
        g_current_turn_format.grammar_lazy       = g_native_parser_cache.grammar_lazy;
        g_current_turn_format.grammar_triggers   = g_native_parser_cache.grammar_triggers;
        g_current_turn_format.preserved_tokens   = g_native_parser_cache.preserved_tokens;
        g_current_turn_format.thinking_start_tag = g_native_parser_cache.thinking_start_tag;
        g_current_turn_format.thinking_end_tags  = g_native_parser_cache.thinking_end_tags;
        g_current_turn_format.additional_stops   = g_native_parser_cache.additional_stops;
        register_preserved_token_strings(g_current_turn_format.preserved_tokens);
    } else {
        g_current_turn_format = NativeTurnFormat{};
    }
}

static bool format_chat_message(const common_chat_msg & msg, bool is_user, std::string & out_formatted) {
    try {
        if (!g_vocab_and_template_probed && g_model) {
            probe_model_template_and_vocab_once();
        }
        const bool has_tool_ctx           = (msg.role == ROLE_TOOL) || has_tool_history_in_chat();
        const bool use_native_tools_apply = (msg.role == ROLE_TOOL || has_tool_ctx) ? has_native_tool_template_caps() :
                                                                                      supports_native_tool_templates();
        ensure_native_tools_parser_cached(use_native_tools_apply || (has_native_tool_template_caps() && has_tool_ctx));

        if (!has_tool_ctx && !use_native_tools_apply) {
            out_formatted = common_chat_format_single(g_chat_templates.get(), chat_msgs, msg, is_user, g_use_jinja,
                                                      g_enable_thinking, g_thinking_effort);
            g_current_turn_format.prompt.clear();
            return true;
        }
        if (g_use_jinja && g_chat_templates && g_chat_templates.get()) {
            common_chat_templates_inputs base_inputs;
            base_inputs.use_jinja          = true;
            base_inputs.force_pure_content = true;
            base_inputs.enable_thinking    = g_enable_thinking;
            base_inputs.reasoning_format   = COMMON_REASONING_FORMAT_NONE;
            base_inputs.add_eos            = false;
            if (use_native_tools_apply) {
                const auto caps                 = common_chat_templates_get_caps(g_chat_templates.get());
                const auto it_par               = caps.find("supports_parallel_tool_calls");
                base_inputs.parallel_tool_calls = (it_par != caps.end() && it_par->second);
                base_inputs.tools               = g_configured_tools;
                base_inputs.tool_choice = (g_tool_policy == NativeToolChoice::NONE) ? COMMON_CHAT_TOOL_CHOICE_NONE :
                                          (g_tool_policy == NativeToolChoice::REQUIRED) ?
                                                                                      COMMON_CHAT_TOOL_CHOICE_REQUIRED :
                                                                                      COMMON_CHAT_TOOL_CHOICE_AUTO;
            }
            apply_reasoning_effort_kwarg(base_inputs);

            // Always try fast pure Jinja first (<50ms) to avoid recompiling the 15-tool PEG/GBNF autoparser
            // (which takes 45-98s on mobile CPUs). Fall back to specialized tool mode ONLY if pure Jinja fails
            // or produces an empty delta for ROLE_TOOL (e.g. Gemma 4 convert_tool_responses).
            const bool allow_specialized_fallback = (msg.role != ROLE_SYSTEM) && use_native_tools_apply && has_tool_ctx;
            const bool use_specialized_tool_mode  = false;

            std::vector<common_chat_msg> hist = chat_msgs;
            std::string                  prev_prompt;
            bool                         prev_used_specialized = false;
            std::string                  last_compact_key;
            if (!hist.empty()) {
                const auto & last_committed = chat_msgs.back();
                last_compact_key            = last_committed.content + "|" + last_committed.tool_call_id + "|" +
                                   std::to_string(last_committed.tool_calls.size());

                if (g_prev_prompt_cache.valid && g_prev_prompt_cache.msg_count == chat_msgs.size() &&
                    g_prev_prompt_cache.last_role == last_committed.role &&
                    g_prev_prompt_cache.last_content == last_compact_key &&
                    g_prev_prompt_cache.tools_version == g_tools_config_version &&
                    g_prev_prompt_cache.tool_policy == g_tool_policy &&
                    g_prev_prompt_cache.use_native_tools == use_native_tools_apply &&
                    g_prev_prompt_cache.enable_thinking == g_enable_thinking &&
                    g_prev_prompt_cache.thinking_effort == g_thinking_effort) {
                    prev_prompt           = g_prev_prompt_cache.rendered_prompt;
                    prev_used_specialized = use_specialized_tool_mode;
                } else {
                    bool rendered_prev = false;
                    if (use_specialized_tool_mode) {
                        try {
                            prev_prompt   = render_jinja_params(base_inputs, hist, false, true, false).prompt;
                            rendered_prev = true;
                        } catch (...) {
                            rendered_prev = false;
                        }
                    }
                    prev_used_specialized = rendered_prev;
                    if (!rendered_prev) {
                        prev_prompt                          = render_pure_jinja(base_inputs, hist, false, true);
                        g_prev_prompt_cache                  = CachedRenderedHistory{};
                        g_prev_prompt_cache.valid            = true;
                        g_prev_prompt_cache.msg_count        = chat_msgs.size();
                        g_prev_prompt_cache.last_role        = last_committed.role;
                        g_prev_prompt_cache.last_content     = last_compact_key;
                        g_prev_prompt_cache.tools_version    = g_tools_config_version;
                        g_prev_prompt_cache.tool_policy      = g_tool_policy;
                        g_prev_prompt_cache.use_native_tools = use_native_tools_apply;
                        g_prev_prompt_cache.enable_thinking  = g_enable_thinking;
                        g_prev_prompt_cache.thinking_effort  = g_thinking_effort;
                        g_prev_prompt_cache.rendered_prompt  = prev_prompt;
                    }
                }
            }

            const bool      use_sentinel = can_use_body_sentinel(msg.content);
            common_chat_msg render_msg   = msg;
            if (use_sentinel) {
                render_msg.content = "__AGY_MSG_BODY_MARKER__";
            }

            hist.push_back(render_msg);
            bool               rendered_full_specialized = false;
            std::string        full_prompt;
            common_chat_params full_tool_params;
            if (use_specialized_tool_mode) {
                try {
                    full_tool_params          = render_jinja_params(base_inputs, hist, is_user, true, false);
                    full_prompt               = full_tool_params.prompt;
                    rendered_full_specialized = true;
                    if (is_user) {
                        apply_turn_format_from_params(full_tool_params);
                    }
                } catch (...) {
                    rendered_full_specialized = false;
                }
            }
            if (!rendered_full_specialized) {
                full_prompt = render_pure_jinja(base_inputs, hist, is_user, true);
            }
            // Keep prev_prompt and full_prompt in identical force_pure_content mode before prefix diff
            if (!prev_prompt.empty() && prev_used_specialized != rendered_full_specialized) {
                std::vector<common_chat_msg> prev_only = hist;
                prev_only.pop_back();
                if (!rendered_full_specialized) {
                    prev_prompt           = render_pure_jinja(base_inputs, prev_only, false, true);
                    prev_used_specialized = false;
                } else {
                    try {
                        prev_prompt           = render_jinja_params(base_inputs, prev_only, false, true, false).prompt;
                        prev_used_specialized = true;
                    } catch (...) {
                        full_prompt               = render_pure_jinja(base_inputs, hist, is_user, true);
                        rendered_full_specialized = false;
                    }
                }
            }

            // Restore large payload in O(N) without passing MBs through Jinja VM
            auto splice_sentinel = [&](std::string & target) -> bool {
                if (!use_sentinel) {
                    return true;
                }
                const char * marker     = "__AGY_MSG_BODY_MARKER__";
                const size_t marker_len = sizeof("__AGY_MSG_BODY_MARKER__") - 1;
                const size_t pos        = target.find(marker);
                if (pos == std::string::npos || target.find(marker, pos + marker_len) != std::string::npos) {
                    return false;
                }
                std::string rebuilt;
                rebuilt.reserve(target.size() - marker_len + msg.content.size());
                rebuilt.append(target, 0, pos);
                rebuilt.append(msg.content);
                rebuilt.append(target, pos + marker_len, target.size() - (pos + marker_len));
                target = std::move(rebuilt);
                return true;
            };

            if (!is_user && !rendered_full_specialized && !use_sentinel && msg.tool_calls.empty()) {
                g_prev_prompt_cache                  = CachedRenderedHistory{};
                g_prev_prompt_cache.valid            = true;
                g_prev_prompt_cache.msg_count        = chat_msgs.size() + 1;
                g_prev_prompt_cache.last_role        = msg.role;
                g_prev_prompt_cache.last_content     = msg.content + "|" + msg.tool_call_id + "|0";
                g_prev_prompt_cache.tools_version    = g_tools_config_version;
                g_prev_prompt_cache.tool_policy      = g_tool_policy;
                g_prev_prompt_cache.use_native_tools = use_native_tools_apply;
                g_prev_prompt_cache.enable_thinking  = g_enable_thinking;
                g_prev_prompt_cache.thinking_effort  = g_thinking_effort;
                g_prev_prompt_cache.rendered_prompt  = full_prompt;
            }

            if (prev_prompt.empty()) {
                if (!splice_sentinel(full_prompt)) {
                    hist.back().content = msg.content;
                    full_prompt =
                        render_jinja_params(base_inputs, hist, is_user, true, !rendered_full_specialized).prompt;
                }
                g_current_turn_format.prompt = full_prompt;
                out_formatted                = std::move(full_prompt);
                return true;
            }

            if (full_prompt.size() >= prev_prompt.size() &&
                full_prompt.compare(0, prev_prompt.size(), prev_prompt) == 0) {
                std::string delta = full_prompt.substr(prev_prompt.size());
                if (!delta.empty()) {
                    if (!splice_sentinel(delta)) {
                        hist.back().content = msg.content;
                        const std::string raw_full =
                            render_jinja_params(base_inputs, hist, is_user, true, !rendered_full_specialized).prompt;
                        delta =
                            (raw_full.size() >= prev_prompt.size()) ? raw_full.substr(prev_prompt.size()) : raw_full;
                    }
                    g_current_turn_format.prompt.clear();
                    out_formatted = std::move(delta);
                    return true;
                }
            }
            if (allow_specialized_fallback && !rendered_full_specialized) {
                try {
                    std::vector<common_chat_msg> spec_prev_hist = hist;
                    spec_prev_hist.pop_back();
                    const std::string spec_prev =
                        render_jinja_params(base_inputs, spec_prev_hist, false, true, false).prompt;
                    const common_chat_params spec_full_params =
                        render_jinja_params(base_inputs, hist, is_user, true, false);
                    if (is_user) {
                        apply_turn_format_from_params(spec_full_params);
                    }
                    if (spec_full_params.prompt.size() >= spec_prev.size() &&
                        spec_full_params.prompt.compare(0, spec_prev.size(), spec_prev) == 0) {
                        std::string spec_delta = spec_full_params.prompt.substr(spec_prev.size());
                        if (!splice_sentinel(spec_delta)) {
                            hist.back().content = msg.content;
                            const std::string raw_spec =
                                render_jinja_params(base_inputs, hist, is_user, true, false).prompt;
                            spec_delta =
                                (raw_spec.size() >= spec_prev.size()) ? raw_spec.substr(spec_prev.size()) : raw_spec;
                        }
                        if (!spec_delta.empty()) {
                            g_current_turn_format.prompt.clear();
                            out_formatted = std::move(spec_delta);
                            return true;
                        }
                    }
                } catch (...) {
                }
            }
        }

        out_formatted = common_chat_format_single(g_chat_templates.get(), chat_msgs, msg, is_user, g_use_jinja,
                                                  g_enable_thinking, g_thinking_effort);
        g_current_turn_format.prompt.clear();
        return true;
    } catch (const std::exception & e) {
        LOGe("%s: Exception in common_chat_format_single: %s. Formatting aborted (no raw fallback).", __func__,
             e.what());
        return false;
    } catch (...) {
        LOGe("%s: Unknown exception in common_chat_format_single. Formatting aborted (no raw fallback).", __func__);
        return false;
    }
}

static bool format_tool_batch_delta(const std::vector<common_chat_msg> & tool_batch, std::string & out_formatted) {
    out_formatted.clear();
    if (tool_batch.empty()) {
        return false;
    }
    if (tool_batch.size() == 1) {
        return format_chat_message(tool_batch.front(), /* is_user = */ false, out_formatted);
    }
    if (g_use_jinja && g_chat_templates && g_chat_templates.get() && !chat_msgs.empty()) {
        try {
            const bool                   use_native_tools_apply = has_native_tool_template_caps();
            common_chat_templates_inputs base_inputs;
            base_inputs.use_jinja        = true;
            base_inputs.enable_thinking  = g_enable_thinking;
            base_inputs.reasoning_format = COMMON_REASONING_FORMAT_DEEPSEEK;
            base_inputs.add_eos          = false;
            if (use_native_tools_apply) {
                const auto caps                 = common_chat_templates_get_caps(g_chat_templates.get());
                const auto it_par               = caps.find("supports_parallel_tool_calls");
                base_inputs.parallel_tool_calls = (it_par != caps.end() && it_par->second);
                base_inputs.tools               = g_configured_tools;
                base_inputs.tool_choice = (g_tool_policy == NativeToolChoice::NONE) ? COMMON_CHAT_TOOL_CHOICE_NONE :
                                          (g_tool_policy == NativeToolChoice::REQUIRED) ?
                                                                                      COMMON_CHAT_TOOL_CHOICE_REQUIRED :
                                                                                      COMMON_CHAT_TOOL_CHOICE_AUTO;
            }
            apply_reasoning_effort_kwarg(base_inputs);
            std::vector<common_chat_msg> hist = chat_msgs;
            std::string                  prev_p;
            std::string                  full_p;
            try {
                prev_p = render_pure_jinja(base_inputs, hist, false, false);
                for (const auto & tm : tool_batch) {
                    hist.push_back(tm);
                }
                full_p = render_pure_jinja(base_inputs, hist, false, false);
                if (full_p.size() <= prev_p.size() || full_p.compare(0, prev_p.size(), prev_p) != 0) {
                    hist   = chat_msgs;
                    prev_p = render_jinja_params(base_inputs, hist, false, false, false).prompt;
                    for (const auto & tm : tool_batch) {
                        hist.push_back(tm);
                    }
                    full_p = render_jinja_params(base_inputs, hist, false, false, false).prompt;
                }
            } catch (...) {
                hist   = chat_msgs;
                prev_p = render_jinja_params(base_inputs, hist, false, false, false).prompt;
                for (const auto & tm : tool_batch) {
                    hist.push_back(tm);
                }
                full_p = render_jinja_params(base_inputs, hist, false, false, false).prompt;
            }
            if (full_p.size() >= prev_p.size() && full_p.compare(0, prev_p.size(), prev_p) == 0) {
                out_formatted = full_p.substr(prev_p.size());
                return !out_formatted.empty();
            }
        } catch (...) {
        }
    }
    std::vector<common_chat_msg> temp_hist = chat_msgs;
    std::string                  combined;
    for (const auto & tm : tool_batch) {
        std::string piece;
        if (g_chat_templates && g_chat_templates.get()) {
            try {
                piece = common_chat_format_single(g_chat_templates.get(), temp_hist, tm, false, g_use_jinja,
                                                  g_enable_thinking, g_thinking_effort);
            } catch (...) {
                piece.clear();
            }
        }
        if (piece.empty()) {
            piece = "\n<tool_response>\n" + tm.content + "\n</tool_response>\n";
        }
        combined += piece;
        temp_hist.push_back(tm);
    }
    out_formatted = std::move(combined);
    return !out_formatted.empty();
}

/**
 * Completion loop's short-term states:
 * - token chars caching
 * - current assistant message being generated
 */
static std::string        cached_token_chars;
static std::ostringstream assistant_ss;

static std::string trim_ascii_ws(const std::string & s) {
    const size_t first = s.find_first_not_of(" \r\n\t");
    if (first == std::string::npos) {
        return "";
    }
    const size_t last = s.find_last_not_of(" \r\n\t");
    return s.substr(first, last - first + 1);
}

static bool extract_balanced_block(const std::string & src,
                                   size_t              start_pos,
                                   char                open_ch,
                                   char                close_ch,
                                   std::string &       out_block,
                                   size_t &            out_end) {
    size_t p = src.find_first_not_of(" \r\n\t", start_pos);
    if (p == std::string::npos || src[p] != open_ch) {
        return false;
    }
    int  depth        = 0;
    bool in_json_str  = false;
    bool in_gemma_str = false;
    bool escaped      = false;
    for (size_t i = p; i < src.size(); ++i) {
        if (in_gemma_str) {
            if (src.compare(i, 5, "<|\"|>") == 0) {
                in_gemma_str = false;
                i += 4;
            }
            continue;
        }
        const char c = src[i];
        if (in_json_str) {
            if (escaped) {
                escaped = false;
            } else if (c == '\\') {
                escaped = true;
            } else if (c == '"') {
                in_json_str = false;
            }
            continue;
        }
        if (src.compare(i, 5, "<|\"|>") == 0) {
            in_gemma_str = true;
            i += 4;
            continue;
        }
        if (c == '"') {
            in_json_str = true;
        } else if (c == open_ch) {
            ++depth;
        } else if (c == close_ch) {
            --depth;
            if (depth == 0) {
                out_block = src.substr(p, i - p + 1);
                out_end   = i + 1;
                return true;
            }
        }
    }
    return false;
}

static std::string normalize_gemma4_dict_to_json(const std::string & raw_dict) {
    std::string out;
    out.reserve(raw_dict.size() + 32);
    std::vector<char> containers;
    bool              expect_key = false;
    for (size_t i = 0; i < raw_dict.size();) {
        if (raw_dict.compare(i, 5, "<|\"|>") == 0) {
            i += 5;
            const size_t      close_pos = raw_dict.find("<|\"|>", i);
            const std::string val =
                (close_pos != std::string::npos) ? raw_dict.substr(i, close_pos - i) : raw_dict.substr(i);
            out.push_back('"');
            for (const unsigned char uc : val) {
                switch (uc) {
                    case '"':
                        out += "\\\"";
                        break;
                    case '\\':
                        out += "\\\\";
                        break;
                    case '\n':
                        out += "\\n";
                        break;
                    case '\r':
                        out += "\\r";
                        break;
                    case '\t':
                        out += "\\t";
                        break;
                    default:
                        if (uc < 0x20) {
                            char buf[8];
                            snprintf(buf, sizeof(buf), "\\u%04x", static_cast<unsigned int>(uc));
                            out += buf;
                        } else {
                            out.push_back(static_cast<char>(uc));
                        }
                        break;
                }
            }
            out.push_back('"');
            expect_key = false;
            if (close_pos == std::string::npos) {
                break;
            }
            i = close_pos + 5;
            continue;
        }
        const char c = raw_dict[i];
        if (c == '"') {
            out.push_back(c);
            ++i;
            bool esc = false;
            while (i < raw_dict.size()) {
                const char sc = raw_dict[i++];
                out.push_back(sc);
                if (esc) {
                    esc = false;
                } else if (sc == '\\') {
                    esc = true;
                } else if (sc == '"') {
                    break;
                }
            }
            expect_key = false;
            continue;
        }
        if (c == '{') {
            containers.push_back('{');
            out.push_back(c);
            expect_key = true;
            ++i;
            continue;
        }
        if (c == '[') {
            containers.push_back('[');
            out.push_back(c);
            expect_key = false;
            ++i;
            continue;
        }
        if (c == '}' || c == ']') {
            if (!containers.empty()) {
                containers.pop_back();
            }
            out.push_back(c);
            expect_key = false;
            ++i;
            continue;
        }
        if (c == ',') {
            out.push_back(c);
            expect_key = (!containers.empty() && containers.back() == '{');
            ++i;
            continue;
        }
        if (c == ':') {
            out.push_back(c);
            expect_key = false;
            ++i;
            continue;
        }
        if (expect_key && !std::isspace(static_cast<unsigned char>(c))) {
            size_t k_end = i;
            while (k_end < raw_dict.size()) {
                const char kc = raw_dict[k_end];
                if (kc == ':' || std::isspace(static_cast<unsigned char>(kc)) || kc == ',' || kc == '}') {
                    break;
                }
                ++k_end;
            }
            size_t after_ws = k_end;
            while (after_ws < raw_dict.size() && std::isspace(static_cast<unsigned char>(raw_dict[after_ws]))) {
                ++after_ws;
            }
            if (k_end > i && after_ws < raw_dict.size() && raw_dict[after_ws] == ':') {
                out.push_back('"');
                out.append(raw_dict, i, k_end - i);
                out.push_back('"');
                i          = k_end;
                expect_key = false;
                continue;
            }
        }
        out.push_back(c);
        ++i;
    }
    return out;
}

static void finalize_generation(FinishReason reason) {
    if (g_gen_state.is_finalized) {
        return;
    }
    g_gen_state.is_finalized  = true;
    g_gen_state.is_active     = false;
    g_gen_state.finish_reason = reason;

    LOGi("%s: Finalizing generation with reason=%s. Raw tokens generated=%d, total length=%zu", __func__,
         finish_reason_to_string(reason), g_gen_state.generated_tokens_count, g_gen_state.generated_raw_text.size());

    LOGi(
        "[GENERATION COMPLETE SUMMARY]: Reason: %s | Raw tokens: %d (%zu chars) | Reasoning: %zu chars | Content: "
        "%zu chars | Tool calls: %zu",
        finish_reason_to_string(reason), g_gen_state.generated_tokens_count, g_gen_state.generated_raw_text.size(),
        g_gen_state.accumulated_reasoning.size(), g_gen_state.accumulated_content.size(),
        g_gen_state.accumulated_tool_calls.size());

    // 1. If native parser parameters were armed for this turn, run common_chat_parse once at turn completion
    bool        parsed_native_with_tools = false;
    std::string parsed_clean_content;
    std::string parsed_clean_reasoning;
    if (g_gen_state.accumulated_tool_calls.empty() && g_gen_state.has_parser &&
        !g_gen_state.generated_raw_text.empty()) {
        const bool      is_partial = (reason != FinishReason::EOG);
        common_chat_msg parsed_msg;
        bool            parse_ok = false;
        try {
            parsed_msg = common_chat_parse(g_gen_state.generated_raw_text, is_partial, g_gen_state.parser_params);
            parse_ok   = true;
        } catch (const std::exception & e) {
            LOGw("%s: common_chat_parse (partial=%d) failed: %s", __func__, is_partial ? 1 : 0, e.what());
        } catch (...) {
            LOGw("%s: common_chat_parse (partial=%d) failed (unknown exception)", __func__, is_partial ? 1 : 0);
        }
        if (!parse_ok && !is_partial) {
            try {
                parsed_msg = common_chat_parse(g_gen_state.generated_raw_text, true, g_gen_state.parser_params);
                parse_ok   = true;
            } catch (...) {
            }
        }
        if (parse_ok) {
            if (!parsed_msg.tool_calls.empty()) {
                for (auto & tc : parsed_msg.tool_calls) {
                    if (tc.id.empty()) {
                        tc.id = generate_tool_call_id();
                    }
                    if (tc.arguments.empty()) {
                        tc.arguments = "{}";
                    }
                }
                g_gen_state.accumulated_tool_calls     = parsed_msg.tool_calls;
                g_gen_state.last_parsed_msg.tool_calls = parsed_msg.tool_calls;
                parsed_native_with_tools               = true;
                parsed_clean_content                   = parsed_msg.content;
                parsed_clean_reasoning                 = parsed_msg.reasoning_content;
                LOGi("%s: common_chat_parse recovered %zu native tool call(s)", __func__, parsed_msg.tool_calls.size());
            }
        }
    }

    // 2. Fallback: If no tool calls parsed by native parser, scan raw text when policy != NONE
    //    or when Kotlin signaled active tool mode (g_tools_active_hint) for non-Jinja / assisted models
    if (g_gen_state.accumulated_tool_calls.empty() &&
        (g_tool_policy != NativeToolChoice::NONE || g_tools_active_hint)) {
        auto trim_clean_content_at = [&](const std::string & marker) {
            const size_t c_pos = g_gen_state.accumulated_content.find(marker);
            if (c_pos != std::string::npos) {
                parsed_clean_content = g_gen_state.accumulated_content.substr(0, c_pos);
            }
        };

        if (g_tool_markup_format == 2) {
            // Mistral [TOOL_CALLS][...] array or [TOOL_CALLS]name[ARGS]{...} (Ministral 3 / Small 3.2)
            const std::string mistral_tag = "[TOOL_CALLS]";
            const std::string args_tag    = "[ARGS]";
            const std::string call_id_tag = "[CALL_ID]";
            size_t            mpos        = g_gen_state.generated_raw_text.find(mistral_tag);
            while (mpos != std::string::npos) {
                const size_t after_tag = mpos + mistral_tag.size();
                const size_t first_ch  = g_gen_state.generated_raw_text.find_first_not_of(" \r\n\t", after_tag);
                if (first_ch == std::string::npos) {
                    break;
                }
                if (g_gen_state.generated_raw_text[first_ch] == '[') {
                    std::string arr_str;
                    size_t      arr_end = first_ch;
                    if (!extract_balanced_block(g_gen_state.generated_raw_text, first_ch, '[', ']', arr_str, arr_end)) {
                        arr_str = g_gen_state.generated_raw_text.substr(first_ch);
                        arr_end = g_gen_state.generated_raw_text.size();
                    }
                    try {
                        auto arr = common_json::parse_no_throw(arr_str);
                        if (!arr.is_discarded() && arr.is_array()) {
                            for (size_t k = 0; k < arr.size(); ++k) {
                                const auto & item = arr[k];
                                if (!item.is_object()) {
                                    continue;
                                }
                                std::string tname = item.value("name", "");
                                if (tname.empty() && item.contains("function") && item["function"].is_object()) {
                                    tname = item["function"].value("name", "");
                                }
                                if (!tname.empty()) {
                                    common_chat_tool_call tc;
                                    tc.id   = item.value("id", generate_tool_call_id());
                                    tc.name = tname;
                                    if (item.contains("arguments")) {
                                        tc.arguments = item["arguments"].is_string() ?
                                                           item["arguments"].get<std::string>() :
                                                           item["arguments"].dump();
                                    } else if (item.contains("function") && item["function"].is_object() &&
                                               item["function"].contains("arguments")) {
                                        const auto & fn_args = item["function"]["arguments"];
                                        tc.arguments =
                                            fn_args.is_string() ? fn_args.get<std::string>() : fn_args.dump();
                                    } else {
                                        tc.arguments = "{}";
                                    }
                                    g_gen_state.accumulated_tool_calls.push_back(tc);
                                    g_gen_state.last_parsed_msg.tool_calls.push_back(tc);
                                    parsed_native_with_tools = true;
                                    LOGi("%s: Fallback parsed [TOOL_CALLS][...] in raw text: name='%s', args='%s'",
                                         __func__, tc.name.c_str(), tc.arguments.c_str());
                                }
                            }
                        }
                    } catch (...) {
                    }
                    mpos = g_gen_state.generated_raw_text.find(mistral_tag, arr_end);
                } else {
                    const size_t next_tc  = g_gen_state.generated_raw_text.find(mistral_tag, first_ch);
                    const size_t args_pos = g_gen_state.generated_raw_text.find(args_tag, first_ch);
                    if (args_pos == std::string::npos || (next_tc != std::string::npos && args_pos > next_tc)) {
                        mpos = next_tc;
                        continue;
                    }
                    std::string  header = g_gen_state.generated_raw_text.substr(first_ch, args_pos - first_ch);
                    std::string  tname;
                    std::string  tid;
                    const size_t cid_pos = header.find(call_id_tag);
                    if (cid_pos != std::string::npos) {
                        tname = trim_ascii_ws(header.substr(0, cid_pos));
                        tid   = trim_ascii_ws(header.substr(cid_pos + call_id_tag.size()));
                    } else {
                        tname = trim_ascii_ws(header);
                    }
                    std::string args_block;
                    size_t      args_end = args_pos + args_tag.size();
                    if (!tname.empty() &&
                        extract_balanced_block(g_gen_state.generated_raw_text, args_pos + args_tag.size(), '{', '}',
                                               args_block, args_end)) {
                        try {
                            auto j = common_json::parse_no_throw(args_block);
                            if (!j.is_discarded() && j.is_object()) {
                                common_chat_tool_call tc;
                                tc.id        = !tid.empty() ? tid : generate_tool_call_id();
                                tc.name      = tname;
                                tc.arguments = j.dump();
                                g_gen_state.accumulated_tool_calls.push_back(tc);
                                g_gen_state.last_parsed_msg.tool_calls.push_back(tc);
                                parsed_native_with_tools = true;
                                LOGi("%s: Fallback parsed [TOOL_CALLS]name[ARGS] in raw text: name='%s', args='%s'",
                                     __func__, tc.name.c_str(), tc.arguments.c_str());
                            }
                        } catch (...) {
                        }
                    }
                    mpos = g_gen_state.generated_raw_text.find(mistral_tag, args_end);
                }
            }
            if (parsed_native_with_tools) {
                trim_clean_content_at(mistral_tag);
            }
        } else if (g_tool_markup_format == 3) {
            // Gemma 4: <|tool_call>call:name{...}<tool_call|>
            const std::string gemma_open  = "<|tool_call>";
            const std::string gemma_close = "<tool_call|>";
            size_t            gpos        = g_gen_state.generated_raw_text.find(gemma_open);
            while (gpos != std::string::npos) {
                size_t cur = g_gen_state.generated_raw_text.find_first_not_of(" \r\n\t", gpos + gemma_open.size());
                if (cur == std::string::npos) {
                    break;
                }
                if (g_gen_state.generated_raw_text.compare(cur, 5, "call:") == 0) {
                    cur += 5;
                }
                const size_t brace_pos = g_gen_state.generated_raw_text.find('{', cur);
                const size_t close_pos = g_gen_state.generated_raw_text.find(gemma_close, cur);
                if (brace_pos == std::string::npos || (close_pos != std::string::npos && brace_pos > close_pos)) {
                    if (close_pos == std::string::npos) {
                        break;
                    }
                    gpos = g_gen_state.generated_raw_text.find(gemma_open, close_pos + gemma_close.size());
                    continue;
                }
                const std::string tname = trim_ascii_ws(g_gen_state.generated_raw_text.substr(cur, brace_pos - cur));
                std::string       raw_args;
                size_t            args_end = brace_pos;
                if (!tname.empty() &&
                    extract_balanced_block(g_gen_state.generated_raw_text, brace_pos, '{', '}', raw_args, args_end)) {
                    try {
                        auto j = common_json::parse_no_throw(raw_args);
                        if (j.is_discarded() || !j.is_object()) {
                            const std::string normalized = normalize_gemma4_dict_to_json(raw_args);
                            j                            = common_json::parse_no_throw(normalized);
                        }
                        if (!j.is_discarded() && j.is_object()) {
                            common_chat_tool_call tc;
                            tc.id        = generate_tool_call_id();
                            tc.name      = tname;
                            tc.arguments = j.dump();
                            g_gen_state.accumulated_tool_calls.push_back(tc);
                            g_gen_state.last_parsed_msg.tool_calls.push_back(tc);
                            parsed_native_with_tools = true;
                            LOGi("%s: Fallback parsed <|tool_call>call: in raw text: name='%s', args='%s'", __func__,
                                 tc.name.c_str(), tc.arguments.c_str());
                        }
                    } catch (...) {
                    }
                }
                const size_t next_from = (close_pos != std::string::npos && close_pos >= args_end) ?
                                             (close_pos + gemma_close.size()) :
                                             args_end;
                gpos                   = g_gen_state.generated_raw_text.find(gemma_open, next_from);
            }
            if (parsed_native_with_tools) {
                trim_clean_content_at(gemma_open);
            }
        }

        if (g_gen_state.accumulated_tool_calls.empty() && g_tool_markup_format == 4) {
            const std::string harmony_marker = "to=functions.";
            size_t            hpos           = g_gen_state.generated_raw_text.find(harmony_marker);
            while (hpos != std::string::npos) {
                const size_t name_start = hpos + harmony_marker.size();
                size_t       name_end   = name_start;
                while (name_end < g_gen_state.generated_raw_text.size()) {
                    const char c = g_gen_state.generated_raw_text[name_end];
                    if (c == '<' || c == ' ' || c == '\t' || c == '\r' || c == '\n' || c == '{') {
                        break;
                    }
                    name_end++;
                }
                const std::string fn_name =
                    trim_ascii_ws(g_gen_state.generated_raw_text.substr(name_start, name_end - name_start));
                const size_t brace_open = g_gen_state.generated_raw_text.find('{', name_end);
                if (!fn_name.empty() && brace_open != std::string::npos) {
                    size_t brace_close = g_gen_state.generated_raw_text.find("<|call|>", brace_open);
                    if (brace_close == std::string::npos) {
                        brace_close = g_gen_state.generated_raw_text.find("<|end|>", brace_open);
                    }
                    std::string args_json =
                        (brace_close != std::string::npos) ?
                            g_gen_state.generated_raw_text.substr(brace_open, brace_close - brace_open) :
                            g_gen_state.generated_raw_text.substr(brace_open);
                    args_json = trim_ascii_ws(args_json);
                    auto j    = common_json::parse_no_throw(args_json);
                    if (!j.is_discarded() && j.is_object()) {
                        common_chat_tool_call tc;
                        tc.id        = generate_tool_call_id();
                        tc.name      = fn_name;
                        tc.arguments = j.dump();
                        g_gen_state.accumulated_tool_calls.push_back(tc);
                        g_gen_state.last_parsed_msg.tool_calls.push_back(tc);
                        parsed_native_with_tools = true;
                        trim_clean_content_at("<|channel|>commentary");
                        trim_clean_content_at(harmony_marker);
                        LOGi("%s: Fallback parsed Harmony tool call: name='%s', args='%s'", __func__, tc.name.c_str(),
                             tc.arguments.c_str());
                    }
                    if (brace_close == std::string::npos) {
                        break;
                    }
                    hpos = g_gen_state.generated_raw_text.find(harmony_marker, brace_close);
                } else {
                    break;
                }
            }
        }

        if (g_gen_state.accumulated_tool_calls.empty() &&
            (g_tool_markup_format == 0 || g_tool_markup_format == 1 || g_tool_markup_format == 4)) {
            const std::string open_tag  = "<tool_call>";
            const std::string close_tag = "</tool_call>";
            size_t            pos       = g_gen_state.generated_raw_text.find(open_tag);
            while (pos != std::string::npos) {
                const size_t start    = pos + open_tag.length();
                const size_t end      = g_gen_state.generated_raw_text.find(close_tag, start);
                std::string  json_str = (end != std::string::npos) ?
                                            g_gen_state.generated_raw_text.substr(start, end - start) :
                                            g_gen_state.generated_raw_text.substr(start);
                json_str              = trim_ascii_ws(json_str);
                if (!json_str.empty()) {
                    bool parsed_this_block = false;
                    try {
                        auto j = common_json::parse_no_throw(json_str);
                        if (!j.is_discarded() && j.is_object()) {
                            std::string tname = j.value("name", "");
                            if (tname.empty() && j.contains("function") && j["function"].is_object()) {
                                tname = j["function"].value("name", "");
                            }
                            if (!tname.empty()) {
                                common_chat_tool_call tc;
                                tc.id   = generate_tool_call_id();
                                tc.name = tname;
                                if (j.contains("arguments")) {
                                    if (j["arguments"].is_string()) {
                                        tc.arguments = j["arguments"].get<std::string>();
                                    } else {
                                        tc.arguments = j["arguments"].dump();
                                    }
                                } else {
                                    tc.arguments = "{}";
                                }
                                g_gen_state.accumulated_tool_calls.push_back(tc);
                                g_gen_state.last_parsed_msg.tool_calls.push_back(tc);
                                parsed_native_with_tools = true;
                                parsed_this_block        = true;
                                trim_clean_content_at(open_tag);
                                LOGi("%s: Fallback parsed <tool_call> JSON in raw text: name='%s', args='%s'", __func__,
                                     tc.name.c_str(), tc.arguments.c_str());
                            }
                        }
                    } catch (...) {
                    }
                    // XML fallback for Qwen3.5 (<function=NAME><parameter=KEY>VAL</parameter></function>)
                    // and MiniCPM5 (<function name="NAME"><param name="KEY">VAL</param></function>)
                    if (!parsed_this_block && json_str.find("<function") != std::string::npos) {
                        std::string tname;
                        size_t      fn_pos = json_str.find("<function");
                        size_t fn_gt = (fn_pos != std::string::npos) ? json_str.find('>', fn_pos) : std::string::npos;
                        if (fn_gt != std::string::npos) {
                            std::string fn_hdr = json_str.substr(fn_pos + 9, fn_gt - (fn_pos + 9));
                            fn_hdr             = trim_ascii_ws(fn_hdr);
                            if (!fn_hdr.empty() && fn_hdr[0] == '=') {
                                tname = trim_ascii_ws(fn_hdr.substr(1));
                            } else {
                                size_t npos = fn_hdr.find("name=");
                                if (npos != std::string::npos) {
                                    tname = trim_ascii_ws(fn_hdr.substr(npos + 5));
                                }
                            }
                            while (!tname.empty() && (tname.front() == '"' || tname.front() == '\'')) {
                                tname.erase(tname.begin());
                            }
                            while (!tname.empty() &&
                                   (tname.back() == '"' || tname.back() == '\'' || tname.back() == '/')) {
                                tname.pop_back();
                            }
                            if (!tname.empty()) {
                                common_json args_obj = common_json::object();
                                size_t      pcur     = fn_gt + 1;
                                while (pcur < json_str.size()) {
                                    size_t p_open = json_str.find("<param", pcur);
                                    if (p_open == std::string::npos) {
                                        break;
                                    }
                                    size_t p_gt = json_str.find('>', p_open);
                                    if (p_gt == std::string::npos) {
                                        break;
                                    }
                                    std::string phdr = json_str.substr(p_open + 1, p_gt - (p_open + 1));
                                    std::string pkey;
                                    size_t      eq = phdr.find('=');
                                    if (eq != std::string::npos) {
                                        pkey = trim_ascii_ws(phdr.substr(eq + 1));
                                        while (!pkey.empty() && (pkey.front() == '"' || pkey.front() == '\'')) {
                                            pkey.erase(pkey.begin());
                                        }
                                        while (!pkey.empty() &&
                                               (pkey.back() == '"' || pkey.back() == '\'' || pkey.back() == '/')) {
                                            pkey.pop_back();
                                        }
                                    }
                                    size_t p_close = json_str.find("</param", p_gt + 1);
                                    if (p_close == std::string::npos) {
                                        break;
                                    }
                                    std::string pval = trim_ascii_ws(json_str.substr(p_gt + 1, p_close - (p_gt + 1)));
                                    if (!pkey.empty()) {
                                        args_obj[pkey] = pval;
                                    }
                                    size_t close_gt = json_str.find('>', p_close);
                                    pcur = (close_gt != std::string::npos) ? (close_gt + 1) : json_str.size();
                                }
                                common_chat_tool_call tc;
                                tc.id        = generate_tool_call_id();
                                tc.name      = tname;
                                tc.arguments = args_obj.dump();
                                g_gen_state.accumulated_tool_calls.push_back(tc);
                                g_gen_state.last_parsed_msg.tool_calls.push_back(tc);
                                parsed_native_with_tools = true;
                                trim_clean_content_at(open_tag);
                                LOGi("%s: Fallback parsed <tool_call> XML in raw text: name='%s', args='%s'", __func__,
                                     tc.name.c_str(), tc.arguments.c_str());
                            }
                        }
                    }
                }
                if (end == std::string::npos) {
                    break;
                }
                pos = g_gen_state.generated_raw_text.find(open_tag, end + close_tag.length());
            }
        }
    }

    // Emit all accumulated tool calls as SemanticEventType::TOOL_CALL_DELTA to event queue
    if (!g_gen_state.accumulated_tool_calls.empty()) {
        const auto & first_call                = g_gen_state.accumulated_tool_calls.front();
        g_pending_tool_call.id                 = first_call.id;
        g_pending_tool_call.name               = first_call.name;
        g_pending_tool_call.arguments          = first_call.arguments;
        g_pending_tool_call.completed          = false;
        g_pending_tool_call.waiting_for_result = true;

        for (size_t idx = 0; idx < g_gen_state.accumulated_tool_calls.size(); ++idx) {
            const auto &        tc = g_gen_state.accumulated_tool_calls[idx];
            NativeSemanticEvent ev;
            ev.type            = SemanticEventType::TOOL_CALL_DELTA;
            ev.tool_index      = static_cast<int32_t>(idx);
            ev.tool_id         = tc.id;
            ev.tool_name       = tc.name;
            ev.tool_args_delta = tc.arguments;
            push_semantic_event(std::move(ev));
            LOGi("[FINALIZE EMIT TOOL_CALL]: idx=%zu id='%s' name='%s' args='%s'", idx, tc.id.c_str(), tc.name.c_str(),
                 tc.arguments.c_str());
        }
    }

    // Flush any residual un-emitted text from generated_raw_text (skip raw tool-call holdback when tool calls were parsed)
    if (g_gen_state.emitted_raw_pos < g_gen_state.generated_raw_text.size()) {
        std::string rem             = g_gen_state.generated_raw_text.substr(g_gen_state.emitted_raw_pos);
        g_gen_state.emitted_raw_pos = g_gen_state.generated_raw_text.size();
        if (g_gen_state.accumulated_tool_calls.empty()) {
            if (g_gen_state.in_reasoning) {
                g_gen_state.accumulated_reasoning += rem;
                g_gen_state.last_parsed_msg.reasoning_content += rem;
                NativeSemanticEvent ev;
                ev.type = SemanticEventType::REASONING_DELTA;
                ev.text = std::move(rem);
                push_semantic_event(std::move(ev));
            } else {
                g_gen_state.accumulated_content += rem;
                g_gen_state.last_parsed_msg.content += rem;
                NativeSemanticEvent ev;
                ev.type = SemanticEventType::CONTENT_DELTA;
                ev.text = std::move(rem);
                push_semantic_event(std::move(ev));
            }
        }
    }

    // Safety fallback when ThinkingProfile.OFF suppressed streaming <think> blocks
    // and the model produced no visible content outside reasoning tags or tool calls: never lose the response silently!
    const bool has_visible_content = g_gen_state.accumulated_content.find_first_not_of(" \t\r\n") != std::string::npos;
    if (!g_enable_thinking && !has_visible_content && !g_gen_state.accumulated_reasoning.empty() &&
        g_gen_state.accumulated_tool_calls.empty()) {
        std::string fallback_content = g_gen_state.accumulated_reasoning;
        g_gen_state.accumulated_content += fallback_content;
        g_gen_state.last_parsed_msg.content = g_gen_state.accumulated_content;
        NativeSemanticEvent ev;
        ev.type = SemanticEventType::CONTENT_DELTA;
        ev.text = std::move(fallback_content);
        push_semantic_event(std::move(ev));
    }

    // Flush any pending UTF-8 bytes at stream end directly to event queue
    {
        std::lock_guard<std::mutex> lock(g_event_queue_mutex);
        if (!g_pending_content_utf8.empty()) {
            NativeSemanticEvent flush_ev;
            flush_ev.generation_id    = g_current_generation_id.load();
            flush_ev.sequence         = g_event_sequence_counter.fetch_add(1);
            flush_ev.type             = SemanticEventType::CONTENT_DELTA;
            flush_ev.text             = std::move(g_pending_content_utf8);
            flush_ev.generated_tokens = g_gen_state.generated_tokens_count;
            g_pending_content_utf8.clear();
            g_native_event_queue.push_back(std::move(flush_ev));
        }
        if (!g_pending_reasoning_utf8.empty()) {
            NativeSemanticEvent flush_ev;
            flush_ev.generation_id    = g_current_generation_id.load();
            flush_ev.sequence         = g_event_sequence_counter.fetch_add(1);
            flush_ev.type             = SemanticEventType::REASONING_DELTA;
            flush_ev.text             = std::move(g_pending_reasoning_utf8);
            flush_ev.generated_tokens = g_gen_state.generated_tokens_count;
            g_pending_reasoning_utf8.clear();
            g_native_event_queue.push_back(std::move(flush_ev));
        }
    }

    if (!g_no_perf && g_context && g_sampler) {
        try {
            common_perf_print(g_context, g_sampler);
        } catch (...) {
        }
    }

    // Enqueue Finished event to decoupled queue
    NativeSemanticEvent finish_ev;
    finish_ev.type             = SemanticEventType::FINISHED;
    finish_ev.finish_reason    = reason;
    finish_ev.prompt_tokens    = g_gen_state.prompt_tokens_count;
    finish_ev.generated_tokens = g_gen_state.generated_tokens_count;
    push_semantic_event(std::move(finish_ev));

    // Persist structured assistant message to history (Guarantee zero desync)
    if (assistant_ss.tellp() > 0 || !g_gen_state.accumulated_content.empty() ||
        !g_gen_state.accumulated_reasoning.empty() || !g_gen_state.accumulated_tool_calls.empty()) {
        common_chat_msg assistant_msg;
        assistant_msg.role = ROLE_ASSISTANT;
        if (!g_gen_state.accumulated_content.empty() || !g_gen_state.accumulated_reasoning.empty() ||
            !g_gen_state.accumulated_tool_calls.empty()) {
            assistant_msg.content = parsed_native_with_tools ? parsed_clean_content : g_gen_state.accumulated_content;
            assistant_msg.reasoning_content =
                !g_gen_state.accumulated_reasoning.empty() ?
                    g_gen_state.accumulated_reasoning :
                    (parsed_native_with_tools ? parsed_clean_reasoning : g_gen_state.accumulated_reasoning);
            assistant_msg.tool_calls = g_gen_state.accumulated_tool_calls;
        } else {
            assistant_msg.content = assistant_ss.str();
        }
        chat_msgs.push_back(std::move(assistant_msg));
        LOGd("%s: Stored assistant message to history (content_len=%zu, reasoning_len=%zu, tool_calls=%zu)", __func__,
             chat_msgs.back().content.length(), chat_msgs.back().reasoning_content.length(),
             chat_msgs.back().tool_calls.size());

        // Register ALL pending tool calls if assistant returned tool calls and generation finished cleanly
        // (or cancelled right on closing </tool_call>)
        if ((reason == FinishReason::EOG || reason == FinishReason::CANCELLED) &&
            !chat_msgs.back().tool_calls.empty()) {
            g_pending_tool_calls.clear();
            g_staged_tool_msgs.clear();
            for (const auto & item_tc : chat_msgs.back().tool_calls) {
                PendingToolCallState st;
                st.id                 = item_tc.id;
                st.name               = item_tc.name;
                st.arguments          = item_tc.arguments;
                st.completed          = false;
                st.waiting_for_result = true;
                g_pending_tool_calls.push_back(std::move(st));
            }
            const auto & tc                        = g_pending_tool_calls.front();
            g_pending_tool_call.id                 = tc.id;
            g_pending_tool_call.name               = tc.name;
            g_pending_tool_call.arguments          = tc.arguments;
            g_pending_tool_call.completed          = false;
            g_pending_tool_call.waiting_for_result = true;
            LOGi("[TOOL_CALL_COMPLETE]: count=%zu first_id='%s', first_name='%s', args_bytes=%zu",
                 g_pending_tool_calls.size(), tc.id.c_str(), tc.name.c_str(), tc.arguments.size());
        }
    }
}

static void reset_short_term_states() {
    stop_generation_position = 0;
    cached_token_chars.clear();
    assistant_ss.str("");
    g_gen_state.reset();
    g_current_turn_format = NativeTurnFormat{};
    std::lock_guard<std::mutex> lock(g_event_queue_mutex);
    g_pending_content_utf8.clear();
    g_pending_reasoning_utf8.clear();
}

static int decode_tokens_in_batches(llama_context *      context,
                                    llama_batch &        batch,
                                    const llama_tokens & tokens,
                                    const llama_pos      start_pos,
                                    const bool           compute_last_logit) {
    LOGd("%s: Decode %d tokens starting at position %d", __func__, (int) tokens.size(), start_pos);

    const llama_pos checkpoint_pos      = start_pos;
    const size_t    checkpoint_kv_cells = g_kv_cells_used;

    int       current_batch_limit = (g_n_batch > 0) ? g_n_batch : BATCH_SIZE;
    llama_pos active_pos          = start_pos;
    size_t    local_kv_cells      = g_kv_cells_used;
    bool      has_shifted         = false;

    for (int i = 0; i < (int) tokens.size(); i += current_batch_limit) {
        const int cur_batch_size = std::min((int) tokens.size() - i, current_batch_limit);
        common_batch_clear(batch);
        LOGv("%s: Preparing a batch size of %d starting at: %d", __func__, cur_batch_size, i);

        // Dynamické klouzání (sliding KV cache) přímo během prefillu velkých textů/promptů
        int context_size = static_cast<int>(llama_n_ctx(context));
        if (local_kv_cells + cur_batch_size >= (size_t) (context_size - OVERFLOW_HEADROOM)) {
            LOGw(
                "%s: Current batch won't fit into context! KV cells: %zu, cur_batch: %d, limit: %d. Shifting "
                "dynamically...",
                __func__, local_kv_cells, cur_batch_size, context_size);
            if (!shift_context_ex(active_pos, local_kv_cells)) {
                if (i == 0 && !g_cached_tokens.empty() &&
                    rebuild_context_preserving_tail(active_pos, local_kv_cells, (int) tokens.size() + 16, false)) {
                    common_batch_clear(batch);
                    has_shifted = true;
                } else {
                    LOGe("%s: Dynamic in-prefill shift failed!", __func__);
                    if (!has_shifted) {
                        LOGw("%s: Rolling back partial prefill from pos %d back to %d", __func__, active_pos,
                             checkpoint_pos);
                        llama_memory_seq_rm(llama_get_memory(context), 0, checkpoint_pos, -1);
                    } else {
                        LOGe("%s: Shift already occurred, resetting context state", __func__);
                        reset_long_term_states(true, false);
                    }
                    return -1;
                }
            } else {
                has_shifted = true;
            }
        }

        // Add tokens to the batch with proper positions
        for (int j = 0; j < cur_batch_size; j++) {
            const llama_token token_id   = tokens[i + j];
            const llama_pos   position   = active_pos + j;
            const bool        want_logit = compute_last_logit && (i + j == (int) tokens.size() - 1);
            common_batch_add(batch, token_id, position, { 0 }, want_logit);
        }

        // Decode this batch
        const int decode_result = llama_decode(context, batch);
        if (decode_result) {
            LOGe("%s: llama_decode failed w/ %d", __func__, decode_result);
            if (!has_shifted) {
                LOGw("%s: Rolling back KV cache from pos %d back to checkpoint %d (cells: %zu)", __func__, active_pos,
                     checkpoint_pos, checkpoint_kv_cells);
                const bool rollback_ok = llama_memory_seq_rm(llama_get_memory(context), 0, checkpoint_pos, -1);
                if (!rollback_ok) {
                    LOGe(
                        "%s: llama_memory_seq_rm rollback FAILED! Context dirty, resetting long term state to ensure "
                        "consistency.",
                        __func__);
                    reset_long_term_states(true, false);
                } else {
                    current_position = checkpoint_pos;
                    g_kv_cells_used  = checkpoint_kv_cells;
                }
            } else {
                LOGe("%s: Decode failed after context shift - resetting to clean state", __func__);
                reset_long_term_states(true, false);
            }
            return decode_result;
        }

        active_pos += cur_batch_size;
        local_kv_cells += cur_batch_size;
    }

    current_position = active_pos;
    g_kv_cells_used  = local_kv_cells;
    return 0;
}

extern "C" JNIEXPORT jint JNICALL Java_com_arm_aichat_internal_InferenceEngineImpl_resizeContextNative(JNIEnv * env,
                                                                                                       jobject  thiz,
                                                                                                       jint new_n_ctx) {
    (void) env;
    (void) thiz;
    std::lock_guard<std::recursive_mutex> cfg_lock(g_config_mutex);
    if (!g_model || !g_context || !g_batch_initialized || g_gen_state.is_active || new_n_ctx <= 0) {
        LOGe("resizeContextNative: invalid state (model=%p, ctx=%p, active=%d, n_ctx=%d)", g_model, g_context,
             g_gen_state.is_active ? 1 : 0, (int) new_n_ctx);
        return -1;
    }

    const int old_n_ctx = static_cast<int>(llama_n_ctx(g_context));
    if (new_n_ctx == old_n_ctx) {
        return 1;
    }

    // Hexagon cDSP Virtual Address (VA) limit guard before freeing/reallocating context
    if (g_active_backend_info.rfind("Hexagon", 0) == 0 && g_hexagon_va_limit_mb > 0 && new_n_ctx > old_n_ctx) {
        const double  model_mb  = static_cast<double>(g_loaded_model_file_bytes) / (1024.0 * 1024.0);
        double        est_kv_mb = 0.0;
        const int64_t n_layer   = llama_model_n_layer(g_model);
        const int64_t n_head_kv = llama_model_n_head_kv(g_model);
        const int64_t n_head_q  = llama_model_n_head(g_model);
        const int64_t head_dim  = (n_head_q > 0) ? (llama_model_n_embd(g_model) / n_head_q) : 0;
        if (n_layer > 0 && n_head_kv > 0 && head_dim > 0) {
            double bytes_per_elem = 1.06;  // Q8_0 default
            switch (g_kv_cache_mode) {
                case 0:
                    bytes_per_elem = 2.0;
                    break;  // F16
                case 1:
                    bytes_per_elem = 1.06;
                    break;  // Q8_0
                case 2:
                    bytes_per_elem = 0.82;
                    break;  // Q6_K
                case 3:
                    bytes_per_elem = 0.69;
                    break;  // Q5_K
                case 4:
                    bytes_per_elem = 0.56;
                    break;  // Q4_K
                case 5:
                    bytes_per_elem = 0.75;
                    break;  // Q5_1
                case 6:
                    bytes_per_elem = 0.69;
                    break;  // Q5_0
                case 7:
                    bytes_per_elem = 0.63;
                    break;  // Q4_1
                case 8:
                    bytes_per_elem = 0.56;
                    break;  // Q4_0
                case 9:
                    bytes_per_elem = 0.56;
                    break;  // IQ4_NL
                case 10:
                case 11:
                    bytes_per_elem = 0.43;
                    break;  // IQ3_S
                case 12:
                case 13:
                    bytes_per_elem = 0.30;
                    break;  // IQ2_XS / IQ2_XXS
                case 14:
                    bytes_per_elem = 0.22;
                    break;  // IQ1_S
                default:
                    bytes_per_elem = 1.06;
                    break;
            }
            const double kv_bytes = static_cast<double>(n_layer) * static_cast<double>(n_head_kv) *
                                    static_cast<double>(head_dim) * 2.0 * static_cast<double>(new_n_ctx) *
                                    bytes_per_elem;
            est_kv_mb = (kv_bytes / (1024.0 * 1024.0)) + 256.0;
        } else {
            est_kv_mb = (static_cast<double>(new_n_ctx) / 4096.0) * 512.0 + 256.0;
        }
        const int    effective_ndev = std::max(1, s_hexagon_initialized_ndev);
        const double va_limit_mb    = static_cast<double>(g_hexagon_va_limit_mb) * static_cast<double>(effective_ndev);
        if (model_mb + est_kv_mb > va_limit_mb) {
            LOGw(
                "resizeContextNative: Hexagon cDSP VA guard rejected resize %d -> %d "
                "(model=%.1f MB + est_kv_scratch=%.1f MB = %.1f MB > limit=%.1f MB [per_dev=%d MB, ndev=%d]); "
                "preserving old context.",
                old_n_ctx, (int) new_n_ctx, model_mb, est_kv_mb, model_mb + est_kv_mb, va_limit_mb,
                g_hexagon_va_limit_mb, effective_ndev);
            return -1;
        }
    }

    const bool           has_non_text_media = (g_kv_cells_used != g_cached_tokens.size());
    std::vector<uint8_t> state_buf;
    size_t               copied_bytes = 0;
    if (g_kv_cells_used > 0) {
        try {
            llama_synchronize(g_context);
            const size_t est_size = llama_state_seq_get_size(g_context, 0);
            if (est_size > 0) {
                state_buf.resize(est_size);
                copied_bytes = llama_state_seq_get_data(g_context, state_buf.data(), state_buf.size(), 0);
                if (copied_bytes == 0) {
                    state_buf.clear();
                }
            }
        } catch (...) {
            state_buf.clear();
            copied_bytes = 0;
        }
    }

    // Free old llama_context BEFORE allocating resized context to avoid double KV cache spike (Android LMK)
    detach_backend_sampler_if_attached();
    llama_free(g_context);
    g_context = nullptr;

    auto * new_ctx = init_context(g_model, new_n_ctx);
    if (!new_ctx) {
        LOGe("resizeContextNative: OOM allocating n_ctx=%d, rolling back to %d...", (int) new_n_ctx, old_n_ctx);
        auto * rollback_ctx = init_context(g_model, old_n_ctx);
        if (rollback_ctx) {
            g_context = rollback_ctx;
            g_n_ctx   = static_cast<int>(llama_n_ctx(g_context));
            if (g_kv_cells_used > 0) {
                size_t rb_restored = 0;
                if (copied_bytes > 0) {
                    try {
                        rb_restored = llama_state_seq_set_data(g_context, state_buf.data(), copied_bytes, 0);
                    } catch (...) {
                        rb_restored = 0;
                    }
                }
                if (rb_restored == 0) {
                    if (!has_non_text_media && !g_cached_tokens.empty() && g_n_ctx > OVERFLOW_HEADROOM &&
                        static_cast<int>(g_cached_tokens.size()) < g_n_ctx - OVERFLOW_HEADROOM) {
                        llama_memory_clear(llama_get_memory(g_context), false);
                        current_position = 0;
                        g_kv_cells_used  = 0;
                        if (decode_tokens_in_batches(g_context, g_batch, g_cached_tokens, 0, false) != 0) {
                            reset_long_term_states(true, false);
                        }
                    } else {
                        reset_long_term_states(true, false);
                    }
                }
            }
            attach_backend_sampler_if_enabled();
        } else {
            LOGe("resizeContextNative: Double OOM during rollback to n_ctx=%d; clearing long-term state.", old_n_ctx);
            reset_long_term_states(false, false);
            reset_short_term_states();
        }
        return -1;
    }

    g_context = new_ctx;
    g_n_ctx   = static_cast<int>(llama_n_ctx(g_context));

    bool kv_preserved = true;
    if (g_kv_cells_used > 0) {
        const bool fits_new_ctx = (g_n_ctx > OVERFLOW_HEADROOM) &&
                                  (g_kv_cells_used < static_cast<size_t>(g_n_ctx - OVERFLOW_HEADROOM)) &&
                                  (current_position < g_n_ctx - OVERFLOW_HEADROOM);
        if (!fits_new_ctx) {
            LOGw("resizeContextNative: existing KV (%zu cells, pos=%d) does not fit in shrunk n_ctx=%d; clearing KV.",
                 g_kv_cells_used, current_position, g_n_ctx);
            kv_preserved = false;
            reset_long_term_states(true, false);
        } else {
            // Models with context-length-dependent RoPE scaling (e.g. Phi-3.5 LongRoPE short_factor vs long_factor,
            // YaRN) compute different K-vector RoPE frequencies when n_ctx crosses original_max_position_embeddings.
            // Copying raw binary KV state across n_ctx changes causes RoPE phase mismatch (word salad); replay tokens instead.
            bool force_token_replay = false;
            if (g_kv_replay_on_ctx_resize_mode == 1) {
                force_token_replay = true;
                LOGi(
                    "resizeContextNative: kv_replay_mode=ALWAYS; forcing clean token replay instead of binary KV "
                    "copy.");
            } else if (g_kv_replay_on_ctx_resize_mode == 2) {
                force_token_replay = false;
                LOGi("resizeContextNative: kv_replay_mode=NEVER; preferring binary KV copy.");
            } else {
                bool has_dynamic_rope_scaling = false;
                if (old_n_ctx != g_n_ctx && g_model) {
                    const int32_t meta_count = llama_model_meta_count(g_model);
                    char          key_buf[256];
                    for (int32_t i = 0; i < meta_count; ++i) {
                        if (llama_model_meta_key_by_index(g_model, i, key_buf, sizeof(key_buf)) > 0 &&
                            strstr(key_buf, "rope.scaling") != nullptr) {
                            has_dynamic_rope_scaling = true;
                            LOGi(
                                "resizeContextNative: detected %s; forcing clean token replay instead of binary KV "
                                "copy.",
                                key_buf);
                            break;
                        }
                    }
                }
                force_token_replay = has_dynamic_rope_scaling;
            }
            size_t restored = 0;
            if (copied_bytes > 0 && !force_token_replay) {
                try {
                    restored = llama_state_seq_set_data(g_context, state_buf.data(), copied_bytes, 0);
                } catch (...) {
                    restored = 0;
                }
            }
            if (restored == 0) {
                if (!has_non_text_media && !g_cached_tokens.empty() &&
                    static_cast<int>(g_cached_tokens.size()) < g_n_ctx - OVERFLOW_HEADROOM) {
                    LOGw("resizeContextNative: state_seq restore unavailable, replaying %zu cached tokens...",
                         g_cached_tokens.size());
                    llama_memory_clear(llama_get_memory(g_context), false);
                    current_position = 0;
                    g_kv_cells_used  = 0;
                    if (decode_tokens_in_batches(g_context, g_batch, g_cached_tokens, 0, false) != 0) {
                        kv_preserved = false;
                        reset_long_term_states(true, false);
                    }
                } else {
                    kv_preserved = false;
                    reset_long_term_states(true, false);
                }
            }
        }
    }

    attach_backend_sampler_if_enabled();
    LOGi("resizeContextNative: %d -> %d (kv_preserved=%d, pos=%d, cells=%zu)", old_n_ctx, g_n_ctx, kv_preserved ? 1 : 0,
         current_position, g_kv_cells_used);
    return kv_preserved ? 1 : 0;
}

extern "C" JNIEXPORT jint JNICALL
Java_com_arm_aichat_internal_InferenceEngineImpl_processSystemPrompt(JNIEnv * env,
                                                                     jobject  thiz,
                                                                     jstring  jsystem_prompt) {
    (void) thiz;
    std::lock_guard<std::recursive_mutex> cfg_lock(g_config_mutex);
    if (!g_model || !g_context || !g_batch_initialized) {
        LOGe("%s: Model, context or batch not initialized!", __func__);
        return 2;
    }
    // Reset long-term & short-term states (keep bitmap attached before first message!)
    reset_long_term_states(/* clear_kv_cache = */ true, /* keep_bitmap = */ true);
    reset_short_term_states();

    // Obtain system prompt from JEnv (preserve GGUF template default_system_message when it carries reasoning tags e.g. Ministral-3 [THINK])
    std::string formatted_system_prompt =
        merge_template_default_reasoning_system_prompt(jstring_to_utf8(env, jsystem_prompt));
    LOGi("%s: System prompt received (%zu chars)", __func__, formatted_system_prompt.length());

    common_chat_msg system_msg;
    system_msg.role    = ROLE_SYSTEM;
    system_msg.content = formatted_system_prompt;

    // Format system prompt if applicable
    const bool has_chat_template = (g_chat_templates && g_chat_templates.get()) ?
                                       common_chat_templates_was_explicit(g_chat_templates.get()) :
                                       false;
    if (has_chat_template) {
        const bool format_ok      = format_chat_message(system_msg, false, formatted_system_prompt);
        const bool prompt_dropped = formatted_system_prompt.empty() ||
                                    (!system_msg.content.empty() &&
                                     formatted_system_prompt.find(system_msg.content.substr(
                                         0, std::min<size_t>(20, system_msg.content.length()))) == std::string::npos);

        // Detect Jinja templates that emit system text right after BOS without system-role delimiters
        // (e.g. DeepSeek-R1: {{bos_token}}{{ns.system_prompt}}<|User|>...) so bare instructions cannot
        // be mistaken for the user's question by smaller reasoning models.
        bool unwrapped_bare_system = false;
        if (format_ok && !prompt_dropped && !system_msg.content.empty()) {
            std::string         stripped = formatted_system_prompt;
            const llama_vocab * vocab    = g_model ? llama_model_get_vocab(g_model) : nullptr;
            if (vocab) {
                const llama_token bos_id = llama_vocab_bos(vocab);
                const char * bos_txt     = (bos_id != LLAMA_TOKEN_NULL) ? llama_vocab_get_text(vocab, bos_id) : nullptr;
                if (bos_txt && *bos_txt) {
                    const size_t bos_len = strlen(bos_txt);
                    if (stripped.size() >= bos_len && stripped.compare(0, bos_len, bos_txt) == 0) {
                        stripped.erase(0, bos_len);
                    }
                }
            }
            auto trim_ws = [](const std::string & s) -> std::string {
                const size_t first = s.find_first_not_of(" \t\r\n");
                if (first == std::string::npos) {
                    return "";
                }
                const size_t last = s.find_last_not_of(" \t\r\n");
                return s.substr(first, last - first + 1);
            };
            const std::string trimmed_formatted = trim_ws(stripped);
            const std::string trimmed_sys       = trim_ws(system_msg.content);
            unwrapped_bare_system               = (!trimmed_sys.empty() && trimmed_formatted == trimmed_sys);
        }

        if (!format_ok || prompt_dropped || unwrapped_bare_system) {
            LOGw(
                "%s: Standalone system prompt template %s by model template. Deferring system prompt to first user "
                "turn.",
                __func__, unwrapped_bare_system ? "lacks system-role delimiters" : "unsupported or dropped");
            g_deferred_system_prompt = system_msg.content;
            return 0;
        }
    }

    // Tokenize system prompt
    const bool add_sys_bos   = should_add_bos_for_prompt(formatted_system_prompt, has_chat_template);
    const auto system_tokens = common_tokenize(g_context, formatted_system_prompt, add_sys_bos, has_chat_template);
    LOGi("%s: Tokenized system prompt into %zu tokens.", __func__, system_tokens.size());

    // Handle context overflow
    int       context_size   = static_cast<int>(llama_n_ctx(g_context));
    const int max_batch_size = context_size - OVERFLOW_HEADROOM;
    if ((int) system_tokens.size() > max_batch_size) {
        LOGe("%s: System prompt too long for context! %d tokens, max: %d", __func__, (int) system_tokens.size(),
             max_batch_size);
        return 1;
    }

    // Decode system tokens in batches
    if (decode_tokens_in_batches(g_context, g_batch, system_tokens, current_position)) {
        LOGe("%s: llama_decode() failed!", __func__);
        return 2;
    }

    // Commit system message only after prefill decoding succeeds!
    if (has_chat_template) {
        chat_msgs.push_back(std::move(system_msg));
    }
    system_prompt_position = current_position;
    g_cached_tokens        = system_tokens;
    LOGi("%s: System prompt prefilled and cached in Prefix Cache (%zu tokens).", __func__, g_cached_tokens.size());
    return 0;
}

extern "C" JNIEXPORT jboolean JNICALL
Java_com_arm_aichat_internal_InferenceEngineImpl_updateSystemPromptPreservingContextNative(JNIEnv * env,
                                                                                           jobject  thiz,
                                                                                           jstring  jsystem_prompt) {
    (void) thiz;
    std::lock_guard<std::recursive_mutex> cfg_lock(g_config_mutex);
    if (!g_model || !g_context || !g_batch_initialized || g_gen_state.is_active) {
        return JNI_FALSE;
    }
    const bool has_chat_template  = (g_chat_templates && g_chat_templates.get()) ?
                                        common_chat_templates_was_explicit(g_chat_templates.get()) :
                                        false;
    const bool has_non_text_media = (g_kv_cells_used != g_cached_tokens.size());
    if (!has_chat_template || !g_use_jinja || has_non_text_media || chat_msgs.size() <= 1 ||
        chat_msgs.front().role != ROLE_SYSTEM || g_cached_tokens.empty()) {
        return JNI_FALSE;
    }

    const std::string new_sys = merge_template_default_reasoning_system_prompt(jstring_to_utf8(env, jsystem_prompt));
    if (chat_msgs.front().content == new_sys && !g_configured_tools.empty()) {
        // Nothing changed structurally if both system text and tools are identical
    }

    try {
        std::vector<common_chat_msg> updated_msgs = chat_msgs;
        updated_msgs.front().content              = new_sys;

        const bool use_native_tools =
            supports_native_tool_templates() || (has_native_tool_template_caps() && has_tool_history_in_chat());
        const bool preserve_tool_format = use_native_tools && has_tool_history_in_chat();

        common_chat_templates_inputs inputs;
        inputs.use_jinja          = true;
        inputs.force_pure_content = !preserve_tool_format;
        inputs.enable_thinking    = g_enable_thinking;
        inputs.reasoning_format =
            preserve_tool_format ? COMMON_REASONING_FORMAT_DEEPSEEK : COMMON_REASONING_FORMAT_NONE;
        inputs.add_eos               = false;
        inputs.add_generation_prompt = false;
        if (use_native_tools) {
            const auto caps            = common_chat_templates_get_caps(g_chat_templates.get());
            const auto it_par          = caps.find("supports_parallel_tool_calls");
            inputs.parallel_tool_calls = (it_par != caps.end() && it_par->second);
            inputs.tools               = g_configured_tools;
            inputs.tool_choice         = (g_tool_policy == NativeToolChoice::NONE) ? COMMON_CHAT_TOOL_CHOICE_NONE :
                                         (g_tool_policy == NativeToolChoice::REQUIRED) ? COMMON_CHAT_TOOL_CHOICE_REQUIRED :
                                                                                         COMMON_CHAT_TOOL_CHOICE_AUTO;
        }
        apply_reasoning_effort_kwarg(inputs);

        // Render system-only prompt to compute updated system_prompt_position
        inputs.messages              = { updated_msgs.front() };
        const std::string sys_prompt = apply_chat_templates_safe(g_chat_templates.get(), inputs).prompt;
        if (!updated_msgs.front().content.empty() &&
            sys_prompt.find(updated_msgs.front().content.substr(
                0, std::min<size_t>(20, updated_msgs.front().content.size()))) == std::string::npos) {
            return JNI_FALSE;
        }
        const bool sys_bos    = should_add_bos_for_prompt(sys_prompt, true, true);
        const auto sys_tokens = common_tokenize(g_context, sys_prompt, sys_bos, true);

        // Render full updated conversation history without generation prompt (with pure-content fallback if needed)
        inputs.messages = updated_msgs;
        std::string full_str;
        try {
            full_str = apply_chat_templates_safe(g_chat_templates.get(), inputs).prompt;
        } catch (...) {
            if (!inputs.force_pure_content) {
                inputs.force_pure_content = true;
                inputs.reasoning_format   = COMMON_REASONING_FORMAT_NONE;
                full_str                  = apply_chat_templates_safe(g_chat_templates.get(), inputs).prompt;
            } else {
                throw;
            }
        }
        const bool full_bos   = should_add_bos_for_prompt(full_str, true, true);
        const auto new_tokens = common_tokenize(g_context, full_str, full_bos, true);

        const int context_size = static_cast<int>(llama_n_ctx(g_context));
        if (new_tokens.empty() || static_cast<int>(new_tokens.size()) >= context_size - OVERFLOW_HEADROOM) {
            return JNI_FALSE;
        }

        size_t       common_prefix = 0;
        const size_t min_len       = std::min(g_cached_tokens.size(), new_tokens.size());
        while (common_prefix < min_len && g_cached_tokens[common_prefix] == new_tokens[common_prefix]) {
            ++common_prefix;
        }

        llama_memory_t mem = llama_get_memory(g_context);
        if (!mem) {
            return JNI_FALSE;
        }

        llama_tokens suffix_tokens;
        bool         partial_ok = (common_prefix == g_cached_tokens.size() && common_prefix == new_tokens.size());
        if (!partial_ok) {
            partial_ok = llama_memory_seq_rm(mem, 0, static_cast<llama_pos>(common_prefix), -1);
        }

        if (partial_ok) {
            current_position = static_cast<llama_pos>(common_prefix);
            g_kv_cells_used  = common_prefix;
            if (common_prefix < new_tokens.size()) {
                suffix_tokens.assign(new_tokens.begin() + static_cast<ptrdiff_t>(common_prefix), new_tokens.end());
            }
        } else {
            // Hybrid/Recurrent KV caches may reject partial seq_rm > 0; clear & replay cleanly
            llama_memory_clear(mem, false);
            current_position = 0;
            g_kv_cells_used  = 0;
            suffix_tokens    = new_tokens;
        }

        if (!suffix_tokens.empty()) {
            if (decode_tokens_in_batches(g_context, g_batch, suffix_tokens, current_position, false) != 0) {
                reset_long_term_states(true, false);
                return JNI_FALSE;
            }
        }

        chat_msgs              = std::move(updated_msgs);
        system_prompt_position = static_cast<llama_pos>(sys_tokens.size());
        g_cached_tokens        = new_tokens;
        invalidate_turn_prompt_caches();
        reset_short_term_states();
        LOGi("%s: Re-rendered system prompt in-place preserving KV prefix (%zu/%zu tokens shared, total=%zu).",
             __func__, common_prefix, new_tokens.size(), chat_msgs.size());
        return JNI_TRUE;
    } catch (const std::exception & e) {
        LOGw("%s: fallback to full reset (%s)", __func__, e.what());
        return JNI_FALSE;
    } catch (...) {
        return JNI_FALSE;
    }
}

extern "C" JNIEXPORT jint JNICALL
Java_com_arm_aichat_internal_InferenceEngineImpl_processUserPrompt(JNIEnv * env,
                                                                   jobject  thiz,
                                                                   jstring  juser_prompt,
                                                                   jint     n_predict) {
    (void) thiz;
    std::lock_guard<std::recursive_mutex> cfg_lock(g_config_mutex);
    g_cancel_requested.store(false, std::memory_order_release);

    if (!g_model || !g_context || !g_batch_initialized) {
        LOGe("%s: Model or context not initialized", __func__);
        return 2;
    }

    // If the previous generation was interrupted/cancelled before finalize_generation was called,
    // finalize it now to commit the generated assistant turn into chat_msgs before starting the new user prompt!
    if (!g_gen_state.is_finalized && g_gen_state.is_active) {
        LOGw(
            "%s: Previous generation was active but not finalized (tokens=%d). Finalizing with CANCELLED before new "
            "prompt.",
            __func__, g_gen_state.generated_tokens_count);
        finalize_generation(FinishReason::CANCELLED);
    }

    // Preserve whether the preceding turn was CANCELLED (e.g. early stop at </tool_call> in Scenario B)
    // BEFORE reset_short_term_states() clears finish_reason, so we can prepend any missing template turn-end delimiter!
    const bool prev_turn_was_cancelled = (g_gen_state.finish_reason == FinishReason::CANCELLED);

    // Reset short-term states and discard any stale unconsumed events from cancelled/aborted turns
    reset_short_term_states();
    clear_native_event_queue();

    // Advance generation id and reset sequence for this new turn
    g_current_generation_id.fetch_add(1);
    g_event_sequence_counter.store(0);
    std::string formatted_user_prompt = jstring_to_utf8(env, juser_prompt);
    LOGi("%s: User prompt received (%zu chars)", __func__, formatted_user_prompt.length());

    if (!g_vocab_and_template_probed && g_model) {
        probe_model_template_and_vocab_once();
    }

    // Prevent hanging pending/staged tool state when starting a new non-empty user prompt
    if ((g_pending_tool_call.waiting_for_result || !g_pending_tool_calls.empty() || !g_staged_tool_msgs.empty()) &&
        !formatted_user_prompt.empty()) {
        LOGw("%s: Resetting hanging pending tool call ('%s') on new user prompt to prevent engine stall.", __func__,
             g_pending_tool_call.name.c_str());
        g_pending_tool_call = PendingToolCallState{};
        g_pending_tool_calls.clear();
        g_staged_tool_msgs.clear();
        if (!chat_msgs.empty() && chat_msgs.back().role == ROLE_ASSISTANT && !chat_msgs.back().tool_calls.empty()) {
            chat_msgs.back().tool_calls.clear();
            if (chat_msgs.back().content.empty()) {
                chat_msgs.back().content = "(tool call)";
            }
            invalidate_turn_prompt_caches();
        }
    }
    if (!formatted_user_prompt.empty()) {
        g_last_tool_formatted_delta.clear();
    }

    // Commit any staged batch tool responses if continuation starts before all tool results arrived
    if (!g_staged_tool_msgs.empty() && formatted_user_prompt.empty()) {
        std::string batch_delta;
        if (format_tool_batch_delta(g_staged_tool_msgs, batch_delta) && !batch_delta.empty()) {
            auto batch_tokens = common_tokenize(g_context, batch_delta, false, true);
            if (!batch_tokens.empty() &&
                decode_tokens_in_batches(g_context, g_batch, batch_tokens, current_position, true) == 0) {
                for (auto & tm : g_staged_tool_msgs) {
                    chat_msgs.push_back(std::move(tm));
                }
                g_cached_tokens.insert(g_cached_tokens.end(), batch_tokens.begin(), batch_tokens.end());
            }
        }
        g_staged_tool_msgs.clear();
        g_pending_tool_calls.clear();
        g_pending_tool_call = PendingToolCallState{};
    }

    const bool is_continuation_after_tool =
        (!chat_msgs.empty() && chat_msgs.back().role == ROLE_TOOL && formatted_user_prompt.empty());

    // If a system prompt was deferred because the template does not support standalone system role
    // (e.g. Gemma/Mistral) or lacks system-role delimiters (e.g. DeepSeek-R1),
    // merge it into the first user turn with explicit framing now (cleared only after decode succeeds).
    const bool used_deferred_system_prompt = (!g_deferred_system_prompt.empty() && !is_continuation_after_tool);
    if (used_deferred_system_prompt) {
        LOGw("%s: Merging deferred system prompt (%zu chars) into first user turn.", __func__,
             g_deferred_system_prompt.length());
        formatted_user_prompt = g_deferred_system_prompt + "\n\n" + formatted_user_prompt;
    }

    common_chat_msg candidate_user_msg;
    candidate_user_msg.role    = ROLE_USER;
    candidate_user_msg.content = formatted_user_prompt;

    // Format user prompt if applicable (without mutating chat_msgs yet!)
    const bool has_chat_template = (g_chat_templates && g_chat_templates.get()) ?
                                       common_chat_templates_was_explicit(g_chat_templates.get()) :
                                       false;
    if (has_chat_template && !is_continuation_after_tool) {
        if (!format_chat_message(candidate_user_msg, true, formatted_user_prompt)) {
            LOGw("%s: User prompt template formatting failed. Falling back to raw prompt.", __func__);
            formatted_user_prompt = candidate_user_msg.content;
        }

        if (!g_enable_thinking) {
            // Strip any prefilled thinking start tag from the end of formatted_user_prompt if present!
            // Some Jinja2 templates (Qwen 3.5, DeepSeek, etc.) unconditionally add <think>\n.
            // We must strip it so it does NOT get committed to the KV cache prefill.
            strip_trailing_think_start(formatted_user_prompt, __func__);
        }

        // CRITICAL FIX (Scenario B / Gemma 3 turn boundary):
        // If the previous assistant turn ended via CANCELLED (early stop on </tool_call>) before emitting its
        // template turn-end token (e.g. <end_of_turn>, <|im_end|>), prepend the missing delimiter now so KV cache
        // never contains an unclosed model turn concatenated directly with <start_of_turn>user!
        if (prev_turn_was_cancelled && !g_cached_tokens.empty() && !g_template_stop_strings.empty()) {
            std::string  tail_txt;
            const size_t tail_count = std::min<size_t>(6, g_cached_tokens.size());
            for (size_t i = g_cached_tokens.size() - tail_count; i < g_cached_tokens.size(); ++i) {
                if (g_cached_tokens[i] != LLAMA_TOKEN_NULL) {
                    tail_txt += common_token_to_piece(g_context, g_cached_tokens[i], true);
                }
            }
            const size_t      fp_first = formatted_user_prompt.find_first_not_of(" \r\n\t");
            const std::string fp_trimmed =
                (fp_first != std::string::npos) ? formatted_user_prompt.substr(fp_first) : "";
            bool has_turn_end = false;
            for (const auto & stop_str : g_template_stop_strings) {
                if (!stop_str.empty() &&
                    (tail_txt.find(stop_str) != std::string::npos || formatted_user_prompt.rfind(stop_str, 0) == 0 ||
                     fp_trimmed.rfind(stop_str, 0) == 0)) {
                    has_turn_end = true;
                    break;
                }
            }
            if (!has_turn_end) {
                const std::string & turn_end = g_template_stop_strings.front();
                if (!formatted_user_prompt.empty() && formatted_user_prompt.front() != '\n') {
                    formatted_user_prompt = turn_end + "\n" + formatted_user_prompt;
                } else {
                    formatted_user_prompt = turn_end + formatted_user_prompt;
                }
                LOGi("%s: Prepended missing template turn-end delimiter '%s' after early cancel (Scenario B).",
                     __func__, turn_end.c_str());
            }
        }
    }

    std::string cont_gen_prompt;
    if (is_continuation_after_tool && has_chat_template && g_chat_templates) {
        const bool                   use_native_tools_cont = has_native_tool_template_caps();
        const auto                   caps                  = common_chat_templates_get_caps(g_chat_templates.get());
        const auto                   it_par                = caps.find("supports_parallel_tool_calls");
        common_chat_templates_inputs cont_inputs;
        cont_inputs.use_jinja             = g_use_jinja;
        cont_inputs.force_pure_content    = true;
        cont_inputs.messages              = chat_msgs;
        cont_inputs.add_generation_prompt = true;
        cont_inputs.enable_thinking       = g_enable_thinking;
        cont_inputs.reasoning_format      = COMMON_REASONING_FORMAT_NONE;
        cont_inputs.parallel_tool_calls   = (it_par != caps.end() && it_par->second);
        cont_inputs.add_eos               = false;
        apply_reasoning_effort_kwarg(cont_inputs);
        if (use_native_tools_cont) {
            cont_inputs.tools       = g_configured_tools;
            cont_inputs.tool_choice = (g_tool_policy == NativeToolChoice::NONE)     ? COMMON_CHAT_TOOL_CHOICE_NONE :
                                      (g_tool_policy == NativeToolChoice::REQUIRED) ? COMMON_CHAT_TOOL_CHOICE_REQUIRED :
                                                                                      COMMON_CHAT_TOOL_CHOICE_AUTO;
        }

        auto apply_cont_inputs = [&](common_chat_templates_inputs & inputs_to_apply) {
            const common_chat_params cont_params = apply_chat_templates_safe(g_chat_templates.get(), inputs_to_apply);
            cont_gen_prompt                      = cont_params.generation_prompt;
            if (cont_gen_prompt.empty()) {
                inputs_to_apply.add_generation_prompt = false;
                const std::string no_gen = apply_chat_templates_safe(g_chat_templates.get(), inputs_to_apply).prompt;
                if (cont_params.prompt.size() > no_gen.size() &&
                    cont_params.prompt.compare(0, no_gen.size(), no_gen) == 0) {
                    cont_gen_prompt = cont_params.prompt.substr(no_gen.size());
                }
            }
            if (!inputs_to_apply.force_pure_content && !cont_params.parser.empty()) {
                g_current_turn_format.format             = cont_params.format;
                g_current_turn_format.parser_source      = cont_params.parser;
                g_current_turn_format.grammar            = cont_params.grammar;
                g_current_turn_format.grammar_lazy       = cont_params.grammar_lazy;
                g_current_turn_format.grammar_triggers   = cont_params.grammar_triggers;
                g_current_turn_format.preserved_tokens   = cont_params.preserved_tokens;
                g_current_turn_format.thinking_start_tag = cont_params.thinking_start_tag;
                g_current_turn_format.thinking_end_tags  = cont_params.thinking_end_tags;
                g_current_turn_format.additional_stops   = cont_params.additional_stops;
            } else if (use_native_tools_cont) {
                ensure_native_tools_parser_cached(true);
            }
            g_current_turn_format.generation_prompt = cont_gen_prompt;
        };

        try {
            apply_cont_inputs(cont_inputs);
        } catch (const std::exception & e) {
            LOGw("%s: Continuation template apply (pure=%d) failed: %s", __func__,
                 cont_inputs.force_pure_content ? 1 : 0, e.what());
            if (!cont_inputs.force_pure_content) {
                try {
                    cont_inputs.force_pure_content    = true;
                    cont_inputs.reasoning_format      = COMMON_REASONING_FORMAT_NONE;
                    cont_inputs.add_generation_prompt = true;
                    apply_cont_inputs(cont_inputs);
                } catch (...) {
                    LOGw("%s: Continuation fallback template apply also failed.", __func__);
                }
            }
        } catch (...) {
            LOGw("%s: Continuation template apply failed with unknown error.", __func__);
            if (!cont_inputs.force_pure_content) {
                try {
                    cont_inputs.force_pure_content    = true;
                    cont_inputs.reasoning_format      = COMMON_REASONING_FORMAT_NONE;
                    cont_inputs.add_generation_prompt = true;
                    apply_cont_inputs(cont_inputs);
                } catch (...) {
                }
            }
        }
        if (!g_enable_thinking && !cont_gen_prompt.empty()) {
            strip_trailing_think_start(cont_gen_prompt, __func__);
            g_current_turn_format.generation_prompt = cont_gen_prompt;
        }
    }

    // Build turn-level sampling params from immutable g_sparams (preserving user backend_sampling preference)
    common_params_sampling turn_sparams = g_sparams;
    turn_sparams.generation_prompt.clear();
    turn_sparams.grammar_lazy = false;
    turn_sparams.grammar_triggers.clear();
    turn_sparams.preserved_tokens.clear();
    turn_sparams.reasoning_budget_tokens = -1;
    turn_sparams.reasoning_budget_start.clear();
    turn_sparams.reasoning_budget_end.clear();
    turn_sparams.reasoning_budget_forced.clear();

    const bool arm_native_parser =
        (supports_native_tool_templates() || (has_native_tool_template_caps() && is_continuation_after_tool)) &&
        !g_current_turn_format.parser_source.empty();

    if (arm_native_parser) {
        try {
            g_gen_state.parser_params                      = common_chat_parser_params{};
            g_gen_state.parser_params.format               = g_current_turn_format.format;
            g_gen_state.parser_params.reasoning_format     = COMMON_REASONING_FORMAT_DEEPSEEK;
            g_gen_state.parser_params.reasoning_in_content = false;
            g_gen_state.parser_params.parse_tool_calls     = (g_tool_policy != NativeToolChoice::NONE);
            std::string adj_gen_prompt                     = g_current_turn_format.generation_prompt;
            if (!g_enable_thinking && !adj_gen_prompt.empty()) {
                strip_trailing_think_start(adj_gen_prompt, __func__);
            }
            g_gen_state.parser_params.generation_prompt = adj_gen_prompt;
            g_gen_state.parser_params.parser.load(g_current_turn_format.parser_source);
            g_gen_state.has_parser = true;
        } catch (const std::exception & e) {
            LOGw("%s: Failed to arm native tool parser: %s", __func__, e.what());
            g_gen_state.has_parser = false;
        } catch (...) {
            LOGw("%s: Failed to arm native tool parser (unknown exception)", __func__);
            g_gen_state.has_parser = false;
        }
    } else {
        g_gen_state.has_parser = false;
    }

    const std::string & effective_prompt_end = is_continuation_after_tool ? cont_gen_prompt : formatted_user_prompt;

    bool prompt_ends_with_think = false;
    g_gen_state.active_cot_close_tags.clear();
    for (const auto & rule : g_active_cot_rules) {
        const bool match_with_nl = effective_prompt_end.size() > rule.open_tag.size() &&
                                   effective_prompt_end.back() == '\n' &&
                                   effective_prompt_end.compare(effective_prompt_end.size() - rule.open_tag.size() - 1,
                                                                rule.open_tag.size(), rule.open_tag) == 0;
        const bool match_exact = !match_with_nl && effective_prompt_end.size() >= rule.open_tag.size() &&
                                 effective_prompt_end.compare(effective_prompt_end.size() - rule.open_tag.size(),
                                                              rule.open_tag.size(), rule.open_tag) == 0;
        if (match_with_nl || match_exact) {
            prompt_ends_with_think            = true;
            g_gen_state.active_cot_close_tags = rule.close_tags;
            g_model_thinking_start_tag        = rule.open_tag;
            if (!rule.close_tags.empty()) {
                g_model_thinking_end_tag = rule.close_tags.front();
            }
            break;
        }
    }
    g_gen_state.in_reasoning = g_enable_thinking && prompt_ends_with_think;
    if (!g_gen_state.in_reasoning) {
        g_gen_state.active_cot_close_tags.clear();
    }

    // Apply per-turn grammars & preserved_tokens following server contract:
    // explicit user setGrammarNative always takes exclusive precedence over native tool grammars.
    bool needs_turn_sampler_constraints = false;
    if (!g_sparams.grammar.empty()) {
        turn_sparams.grammar           = g_sparams.grammar;
        needs_turn_sampler_constraints = true;
    } else if (g_enable_native_tool_grammar && supports_native_tool_templates() &&
               !g_current_turn_format.grammar.empty()) {
        turn_sparams.grammar          = common_grammar(COMMON_GRAMMAR_TYPE_TOOL_CALLS, g_current_turn_format.grammar);
        turn_sparams.grammar_lazy     = g_current_turn_format.grammar_lazy;
        turn_sparams.grammar_triggers = g_current_turn_format.grammar_triggers;
        register_preserved_token_strings(g_current_turn_format.preserved_tokens, &turn_sparams.preserved_tokens);
        std::string adj_gen_prompt = g_current_turn_format.generation_prompt;
        if (!g_enable_thinking && !adj_gen_prompt.empty()) {
            strip_trailing_think_start(adj_gen_prompt, __func__);
        }
        turn_sparams.generation_prompt = std::move(adj_gen_prompt);
        needs_turn_sampler_constraints = true;
    }

    // Wire reasoning tags with unlimited budget (-1 -> INT_MAX) ONLY when lazy tool grammar is active,
    // so lazy tool-call grammar pauses during <think> without EVER truncating reasoning mid-thought.
    if (g_enable_thinking && turn_sparams.grammar_lazy) {
        const std::string        eff_think_start = !g_current_turn_format.thinking_start_tag.empty() ?
                                                       g_current_turn_format.thinking_start_tag :
                                                       g_model_thinking_start_tag;
        std::vector<std::string> eff_think_ends  = !g_current_turn_format.thinking_end_tags.empty() ?
                                                       g_current_turn_format.thinking_end_tags :
                                                       std::vector<std::string>{ g_model_thinking_end_tag };
        if (g_model && !eff_think_start.empty() && !eff_think_ends.empty()) {
            const llama_vocab * vocab = llama_model_get_vocab(g_model);
            if (vocab) {
                turn_sparams.reasoning_budget_start = common_tokenize(vocab, eff_think_start, false, true);
                for (const auto & et : eff_think_ends) {
                    if (!et.empty()) {
                        auto et_toks = common_tokenize(vocab, et, false, true);
                        if (!et_toks.empty()) {
                            turn_sparams.reasoning_budget_end.push_back(std::move(et_toks));
                        }
                    }
                }
                if (!turn_sparams.reasoning_budget_start.empty() && !turn_sparams.reasoning_budget_end.empty()) {
                    turn_sparams.reasoning_budget_forced = turn_sparams.reasoning_budget_end.front();
                    turn_sparams.reasoning_budget_tokens =
                        (g_reasoning_budget_tokens > 0) ? g_reasoning_budget_tokens : -1;
                    if (turn_sparams.generation_prompt.empty()) {
                        std::string adj_gen            = g_current_turn_format.generation_prompt;
                        turn_sparams.generation_prompt = std::move(adj_gen);
                    }
                    needs_turn_sampler_constraints = true;
                }
            }
        }
    }

    if (needs_turn_sampler_constraints || g_sampler_has_turn_constraints || !g_sampler) {
        detach_backend_sampler_if_attached();
        if (g_sampler) {
            common_sampler_free(g_sampler);
            g_sampler = nullptr;
        }
        apply_ignore_eos_logit_bias(turn_sparams);
        common_params_sampling sampler_copy = turn_sparams;
        try {
            g_sampler                      = common_sampler_init(g_model, sampler_copy);
            g_sampler_has_turn_constraints = needs_turn_sampler_constraints;
            attach_backend_sampler_if_enabled();
            LOGi("%s: Sampler initialized (turn_constraints=%d, lazy=%d, rbudget=%d).", __func__,
                 needs_turn_sampler_constraints ? 1 : 0, turn_sparams.grammar_lazy ? 1 : 0,
                 turn_sparams.reasoning_budget_tokens);
        } catch (const std::exception & e) {
            LOGw("%s: Turn-constrained sampler init failed (%s), falling back to clean sampler.", __func__, e.what());
            common_params_sampling clean_copy = g_sparams;
            clean_copy.grammar                = g_sparams.grammar;
            apply_ignore_eos_logit_bias(clean_copy);
            g_sampler                      = common_sampler_init(g_model, clean_copy);
            g_sampler_has_turn_constraints = !g_sparams.grammar.empty();
            attach_backend_sampler_if_enabled();
        } catch (...) {
            LOGw("%s: Turn-constrained sampler init failed (unknown error), falling back to clean sampler.", __func__);
            common_params_sampling clean_copy = g_sparams;
            apply_ignore_eos_logit_bias(clean_copy);
            g_sampler                      = common_sampler_init(g_model, clean_copy);
            g_sampler_has_turn_constraints = !g_sparams.grammar.empty();
            attach_backend_sampler_if_enabled();
        }
    }

    LOGi("processUserPrompt: g_cached_bmp=%s, prompt contains <__image__>=%s", g_cached_bmp ? "YES" : "NO",
         (formatted_user_prompt.find("<__image__>") != std::string::npos ||
          formatted_user_prompt.find(mtmd_default_marker()) != std::string::npos) ?
             "YES" :
             "NO");

    if (is_continuation_after_tool) {
        LOGi("%s: Continuation turn after tool result. Decoding generation prompt only.", __func__);
        if (!cont_gen_prompt.empty()) {
            auto gen_tokens = common_tokenize(g_context, cont_gen_prompt, false, true);
            if (!gen_tokens.empty()) {
                const int      context_size   = static_cast<int>(llama_n_ctx(g_context));
                const int      max_batch_size = context_size - OVERFLOW_HEADROOM;
                llama_memory_t mem            = llama_get_memory(g_context);
                const bool     has_non_text   = (g_kv_cells_used != g_cached_tokens.size());
                const bool     can_shift      = (mem && !has_non_text && llama_memory_can_shift(mem));
                if (!can_shift && g_kv_cells_used + gen_tokens.size() >= (size_t) max_batch_size) {
                    rebuild_context_preserving_tail(current_position, g_kv_cells_used, (int) gen_tokens.size() + 64,
                                                    false);
                }
                int decode_res = decode_tokens_in_batches(g_context, g_batch, gen_tokens, current_position, true);
                if (decode_res != 0) {
                    LOGe("%s: decode_tokens_in_batches failed for continuation generation prompt: %d", __func__,
                         decode_res);
                    return 2;
                }
                g_gen_state.prompt_tokens_count = static_cast<int32_t>(gen_tokens.size());
                g_cached_tokens.insert(g_cached_tokens.end(), gen_tokens.begin(), gen_tokens.end());
                LOGi("Continuation generation prompt decode OK (%zu tokens). New pos: %d, KV cells: %zu",
                     gen_tokens.size(), current_position, g_kv_cells_used);
            }
        }
    } else if (g_cached_bmp) {
        // AUTO-PREPEND / LEGACY MARKER NORMALIZATION:
        // AgentCoordinator posílá "<__image__>", zatímco mtmd_tokenize() štěpí podle mtmd_default_marker() ("<__media__>").
        // Pokud prompt obsahuje "<__image__>" a neobsahuje mtmd_default_marker(), nahradíme všechny výskyty "<__image__>";
        // pokud neobsahuje ani jeden, přidáme mtmd_default_marker() na začátek (Chat mode fallback).
        const std::string default_marker = mtmd_default_marker();
        const std::string legacy_marker  = "<__image__>";
        const bool        has_default    = (formatted_user_prompt.find(default_marker) != std::string::npos);
        const bool        has_legacy     = (formatted_user_prompt.find(legacy_marker) != std::string::npos);
        if (has_legacy) {
            size_t pos = 0;
            while ((pos = formatted_user_prompt.find(legacy_marker, pos)) != std::string::npos) {
                formatted_user_prompt.replace(pos, legacy_marker.length(), default_marker);
                pos += default_marker.length();
            }
            LOGi("Replaced legacy <__image__> marker(s) with %s.", default_marker.c_str());
        } else if (!has_default) {
            LOGi("No media marker found in prompt. Auto-prepending %s (Chat mode fallback).", default_marker.c_str());
            formatted_user_prompt = default_marker + formatted_user_prompt;
        }

        LOGi("Found cached image bitmap! Injecting it into prompt at media marker...");
        mtmd_input_text     text      = { formatted_user_prompt.c_str(), formatted_user_prompt.length(), false, false };
        const mtmd_bitmap * bitmaps[] = { g_cached_bmp };
        mtmd_input_chunks * chunks    = mtmd_input_chunks_init();

        if (mtmd_tokenize(g_mtmd_ctx, chunks, &text, bitmaps, 1) == 0) {
            llama_pos new_n_past = 0;

            // SECURITY MEASURE: Check if multimodal prompt fits into remaining KV cache
            // CRITICAL: Používáme g_kv_cells_used (skutečné obsazení buněk), NE current_position!
            // Pro M-RoPE modely: obrázek 36x36 zabere 1296 buněk, ale n_pos = max(36,36) = 36.
            // current_position by ukázal jen 36, ale KV cache spotřebuje 1296 buněk.
            int    context_size    = static_cast<int>(llama_n_ctx(g_context));
            size_t n_chunks_tokens = mtmd_helper_get_n_tokens(chunks);
            LOGi("Multimodal chunks: %zu tokens needed. KV cells used: %zu, Current pos: %d, Limit: %d",
                 n_chunks_tokens, g_kv_cells_used, current_position, context_size);

            if (g_kv_cells_used + n_chunks_tokens > (size_t) (context_size - OVERFLOW_HEADROOM)) {
                LOGe(
                    "%s: Cannot process multimodal prompt: Requires %zu tokens, KV cells used: %zu, limit: %d. "
                    "Multimodal KV cannot be shifted in-place.",
                    __func__, n_chunks_tokens, g_kv_cells_used, context_size);
                mtmd_input_chunks_free(chunks);
                mtmd_bitmap_free(g_cached_bmp);
                g_cached_bmp = nullptr;
                return 2;
            }

            const llama_pos checkpoint_pos      = current_position;
            const size_t    checkpoint_kv_cells = g_kv_cells_used;

            int32_t res = mtmd_helper_eval_chunks(g_mtmd_ctx, g_context, chunks, current_position, 0, g_n_batch, true,
                                                  &new_n_past);
            if (res == 0) {
                current_position = new_n_past;
                g_kv_cells_used += n_chunks_tokens;
                g_gen_state.prompt_tokens_count = static_cast<int32_t>(n_chunks_tokens);
                // COMMIT candidate_user_msg to committed history only now!
                if (has_chat_template) {
                    chat_msgs.push_back(std::move(candidate_user_msg));
                }
                if (used_deferred_system_prompt) {
                    g_deferred_system_prompt.clear();
                }
                LOGi("Multimodal eval OK. User message committed to history. New pos: %d, KV cells used: %zu",
                     current_position, g_kv_cells_used);
            } else {
                LOGe("mtmd_helper_eval_chunks failed with %d (rolling back KV to checkpoint_pos=%d)", res,
                     checkpoint_pos);
                llama_memory_seq_rm(llama_get_memory(g_context), 0, checkpoint_pos, -1);
                current_position = checkpoint_pos;
                g_kv_cells_used  = checkpoint_kv_cells;
                mtmd_input_chunks_free(chunks);
                mtmd_bitmap_free(g_cached_bmp);
                g_cached_bmp = nullptr;
                return 2;
            }
        } else {
            LOGe("mtmd_tokenize failed to parse the prompt chunks.");
            mtmd_input_chunks_free(chunks);
            mtmd_bitmap_free(g_cached_bmp);
            g_cached_bmp = nullptr;
            return 2;
        }

        mtmd_input_chunks_free(chunks);
        mtmd_bitmap_free(g_cached_bmp);
        g_cached_bmp = nullptr;
    } else {
        LOGi("No cached image bitmap. Processing text-only prompt.");
        std::string clean_prompt = formatted_user_prompt;
        {
            const std::string markers[] = { "<__image__>", "<__media__>" };
            for (const auto & m : markers) {
                size_t pos;
                while ((pos = clean_prompt.find(m)) != std::string::npos) {
                    clean_prompt.erase(pos, m.length());
                }
            }
        }
        // Pokud je current_position == 0 (napr. po odlozenem system promptu u Gemmy 3),
        // MUSI byt add_special = has_chat_template (true), aby se vlozil <bos> token!
        const bool add_bos_special = should_add_bos_for_prompt(clean_prompt, has_chat_template);
        auto       user_tokens     = common_tokenize(g_context, clean_prompt, add_bos_special, true);
        LOGi("%s: Tokenized user prompt into %zu tokens.", __func__, user_tokens.size());

        bool           can_shift          = false;
        llama_memory_t mem                = llama_get_memory(g_context);
        const bool     has_non_text_media = (g_kv_cells_used != g_cached_tokens.size());
        if (mem && !has_non_text_media) {
            can_shift = llama_memory_can_shift(mem);
        }

        const int user_prompt_size = (int) user_tokens.size();
        int       context_size     = static_cast<int>(llama_n_ctx(g_context));
        const int max_batch_size   = context_size - OVERFLOW_HEADROOM;

        if (!can_shift && (g_kv_cells_used + user_tokens.size() >= (size_t) max_batch_size)) {
            LOGw(
                "%s: Context near full for non-shiftable model (%zu used + %zu new >= %d max). Preserving system + "
                "tail tokens (120-500) via KV rebuild...",
                __func__, g_kv_cells_used, user_tokens.size(), max_batch_size);
            const bool rebuilt =
                rebuild_context_preserving_tail(current_position, g_kv_cells_used, user_prompt_size + 64, false);
            if (rebuilt) {
                prune_chat_history_keeping_recent(16);
                invalidate_turn_prompt_caches();
            } else {
                // Fallback only when token-tail rebuild is impossible (e.g. empty cache or giant single prompt)
                prune_chat_history_keeping_recent(2);
                const auto saved_msgs = chat_msgs;
                reset_long_term_states(true, false);
                chat_msgs = saved_msgs;
                if (has_chat_template && g_chat_templates) {
                    common_chat_templates_inputs prune_inputs;
                    prune_inputs.use_jinja          = g_use_jinja;
                    prune_inputs.enable_thinking    = g_enable_thinking;
                    prune_inputs.reasoning_format   = COMMON_REASONING_FORMAT_NONE;
                    prune_inputs.add_eos            = false;
                    prune_inputs.force_pure_content = true;
                    apply_reasoning_effort_kwarg(prune_inputs);
                    if (supports_native_tool_templates()) {
                        prune_inputs.tools       = g_configured_tools;
                        prune_inputs.tool_choice = (g_tool_policy == NativeToolChoice::REQUIRED) ?
                                                       COMMON_CHAT_TOOL_CHOICE_REQUIRED :
                                                       COMMON_CHAT_TOOL_CHOICE_AUTO;
                    }
                    auto rebuild_history = chat_msgs;
                    rebuild_history.push_back(candidate_user_msg);
                    prune_inputs.messages              = rebuild_history;
                    prune_inputs.add_generation_prompt = true;
                    try {
                        auto rebuild_params          = apply_chat_templates_safe(g_chat_templates.get(), prune_inputs);
                        g_current_turn_format.prompt = rebuild_params.prompt;
                        if (!rebuild_params.generation_prompt.empty()) {
                            g_current_turn_format.generation_prompt = rebuild_params.generation_prompt;
                        }
                        formatted_user_prompt = rebuild_params.prompt;
                    } catch (const std::exception & e) {
                        LOGw("%s: Failed to rebuild prompt template after reset: %s", __func__, e.what());
                    }
                }
                if (!g_enable_thinking && has_chat_template) {
                    strip_trailing_think_start(formatted_user_prompt, __func__);
                }
                const bool rebuild_bos = should_add_bos_for_prompt(formatted_user_prompt, has_chat_template);
                user_tokens            = common_tokenize(g_context, formatted_user_prompt, rebuild_bos, true);
            }
        } else if (user_prompt_size > max_batch_size) {
            if (!can_shift) {
                LOGe(
                    "%s: User prompt too long! %d tokens exceed max context capacity %d and backend does not support "
                    "shift.",
                    __func__, user_prompt_size, max_batch_size);
                return 1;
            }
            LOGi(
                "%s: Prompt size (%d tokens) exceeds context (%d). Sliding KV cache will dynamically slide during "
                "prefill.",
                __func__, user_prompt_size, context_size);
        }

        // Hard truncation safety net: Never send more tokens to decoder than max context capacity (preserve BOS at [0])
        if ((int) user_tokens.size() > max_batch_size) {
            LOGw(
                "%s: Prompt tokens (%zu) exceed max context capacity (%d). Clamping tokens to newest %d to fit context "
                "safely.",
                __func__, user_tokens.size(), max_batch_size, max_batch_size);
            const llama_vocab * trunc_vocab = g_model ? llama_model_get_vocab(g_model) : nullptr;
            const llama_token   bos_tok     = trunc_vocab ? llama_vocab_bos(trunc_vocab) : LLAMA_TOKEN_NULL;
            if (bos_tok != LLAMA_TOKEN_NULL && !user_tokens.empty() && user_tokens.front() == bos_tok &&
                max_batch_size > 1) {
                const size_t keep_tail = static_cast<size_t>(max_batch_size - 1);
                user_tokens.erase(user_tokens.begin() + 1, user_tokens.end() - keep_tail);
            } else {
                user_tokens.erase(user_tokens.begin(), user_tokens.end() - max_batch_size);
            }
        }

        int decode_res = decode_tokens_in_batches(g_context, g_batch, user_tokens, current_position, true);
        if (decode_res != 0) {
            LOGe("%s: decode_tokens_in_batches failed w/ %d (candidate user msg NOT committed, KV rolled back)",
                 __func__, decode_res);
            return 2;
        }

        g_gen_state.prompt_tokens_count = static_cast<int32_t>(user_tokens.size());
        g_cached_tokens.insert(g_cached_tokens.end(), user_tokens.begin(), user_tokens.end());
        // COMMIT candidate_user_msg to committed history only now (never push empty continuation user msg)
        if (has_chat_template && !is_continuation_after_tool) {
            chat_msgs.push_back(std::move(candidate_user_msg));
            LOGi("Text prompt decode OK. User message committed to history. New pos: %d, KV cells: %zu, cached: %zu",
                 current_position, g_kv_cells_used, g_cached_tokens.size());
        }
        if (used_deferred_system_prompt) {
            g_deferred_system_prompt.clear();
        }
    }

    if (n_predict < 0) {
        // Unlimited: generování do EOG nebo context size (context shifting se postará o overflow)
        stop_generation_position = INT32_MAX;
        LOGi("%s: n_predict=%d -> unlimited generation (until EOG or context shift)", __func__, n_predict);
    } else {
        stop_generation_position = current_position + n_predict;
        LOGi("%s: n_predict=%d -> stop at position %d", __func__, n_predict, stop_generation_position);
    }
    g_gen_state.is_active = true;
    return 0;
}

// --- Phase 1: Native Submit Tool Result ---
extern "C" JNIEXPORT jboolean JNICALL
Java_com_arm_aichat_internal_InferenceEngineImpl_submitToolResultNative(JNIEnv * env,
                                                                        jobject  thiz,
                                                                        jstring  jcall_id,
                                                                        jstring  jname,
                                                                        jstring  jcontent) {
    (void) thiz;
    std::lock_guard<std::recursive_mutex> cfg_lock(g_config_mutex);
    if (!g_context || !g_model || !g_batch_initialized) {
        LOGe("submitToolResultNative: Model or context not initialized");
        return JNI_FALSE;
    }
    if (g_gen_state.is_active) {
        LOGe("submitToolResultNative: Engine is currently generating");
        return JNI_FALSE;
    }
    std::string call_id = jstring_to_utf8(env, jcall_id);
    std::string name    = jstring_to_utf8(env, jname);
    std::string content = jstring_to_utf8(env, jcontent);

    // Resolve pending tool call across g_pending_tool_calls (parallel batches), legacy g_pending_tool_call,
    // or backwards in chat_msgs (past any already-submitted ROLE_TOOL turns).
    int matched_idx = -1;
    for (size_t idx = 0; idx < g_pending_tool_calls.size(); ++idx) {
        const auto & st = g_pending_tool_calls[idx];
        if (st.waiting_for_result && !st.completed && !call_id.empty() && call_id != "call_default" &&
            st.id == call_id) {
            matched_idx = static_cast<int>(idx);
            break;
        }
    }
    if (matched_idx < 0 && !name.empty()) {
        for (size_t idx = 0; idx < g_pending_tool_calls.size(); ++idx) {
            const auto & st = g_pending_tool_calls[idx];
            if (st.waiting_for_result && !st.completed && st.name == name) {
                matched_idx = static_cast<int>(idx);
                break;
            }
        }
    }
    if (matched_idx < 0 && (call_id.empty() || call_id == "call_default")) {
        for (size_t idx = 0; idx < g_pending_tool_calls.size(); ++idx) {
            const auto & st = g_pending_tool_calls[idx];
            if (st.waiting_for_result && !st.completed) {
                matched_idx = static_cast<int>(idx);
                break;
            }
        }
    }

    if (matched_idx >= 0) {
        const auto & matched = g_pending_tool_calls[static_cast<size_t>(matched_idx)];
        if (call_id.empty() || call_id == "call_default" ||
            (!matched.id.empty() && call_id != matched.id && name == matched.name)) {
            call_id = matched.id;
        }
        if (name.empty()) {
            name = matched.name;
        }
    } else if (g_pending_tool_call.waiting_for_result && !g_pending_tool_call.completed) {
        if (call_id.empty() || call_id == "call_default" || call_id == g_pending_tool_call.id) {
            call_id = g_pending_tool_call.id;
        } else if (name == g_pending_tool_call.name) {
            LOGi("submitToolResultNative: Tool name match '%s' overrides call_id ('%s' -> '%s')", name.c_str(),
                 call_id.c_str(), g_pending_tool_call.id.c_str());
            call_id = g_pending_tool_call.id;
        }
        if (name.empty()) {
            name = g_pending_tool_call.name;
        }
    } else {
        int asst_idx = static_cast<int>(chat_msgs.size()) - 1;
        while (asst_idx >= 0 && chat_msgs[static_cast<size_t>(asst_idx)].role == ROLE_TOOL) {
            --asst_idx;
        }
        if (asst_idx < 0 || chat_msgs[static_cast<size_t>(asst_idx)].role != ROLE_ASSISTANT ||
            chat_msgs[static_cast<size_t>(asst_idx)].tool_calls.empty()) {
            LOGw(
                "submitToolResultNative: No pending tool call and previous assistant turn has no tool_calls "
                "(chat_msgs=%zu). Returning false for Scenario B fallback.",
                chat_msgs.size());
            return JNI_FALSE;
        }
        LOGi("submitToolResultNative: Re-attaching tool result to assistant turn %d tool_call.", asst_idx);
        const auto & asst_calls = chat_msgs[static_cast<size_t>(asst_idx)].tool_calls;
        if (call_id.empty() || call_id == "call_default") {
            for (const auto & tc_item : asst_calls) {
                bool already_used = false;
                for (size_t k = static_cast<size_t>(asst_idx) + 1; k < chat_msgs.size(); ++k) {
                    if (chat_msgs[k].tool_call_id == tc_item.id) {
                        already_used = true;
                        break;
                    }
                }
                if (!already_used) {
                    for (const auto & st_msg : g_staged_tool_msgs) {
                        if (st_msg.tool_call_id == tc_item.id) {
                            already_used = true;
                            break;
                        }
                    }
                }
                if (!already_used) {
                    call_id = tc_item.id;
                    if (name.empty()) {
                        name = tc_item.name;
                    }
                    break;
                }
            }
            if (call_id.empty() || call_id == "call_default") {
                call_id = asst_calls.front().id.empty() ? "call_00000001" : asst_calls.front().id;
            }
        }
        if (name.empty()) {
            for (const auto & tc_item : asst_calls) {
                if (!tc_item.id.empty() && tc_item.id == call_id) {
                    name = tc_item.name;
                    break;
                }
            }
            if (name.empty()) {
                name = asst_calls.front().name;
            }
        }
    }

    LOGi("[TOOL_RESULT_BEGIN]: callId='%s', name='%s', content_len=%zu", call_id.c_str(), name.c_str(), content.size());

    // Construct and stage candidate tool msg
    common_chat_msg candidate_tool_msg;
    candidate_tool_msg.role         = ROLE_TOOL;
    candidate_tool_msg.tool_call_id = call_id;
    candidate_tool_msg.tool_name    = name;
    candidate_tool_msg.content      = content;

    g_staged_tool_msgs.push_back(std::move(candidate_tool_msg));
    if (matched_idx >= 0) {
        g_pending_tool_calls[static_cast<size_t>(matched_idx)].completed          = true;
        g_pending_tool_calls[static_cast<size_t>(matched_idx)].waiting_for_result = false;
    }

    size_t remaining_pending = 0;
    for (const auto & st : g_pending_tool_calls) {
        if (st.waiting_for_result && !st.completed) {
            ++remaining_pending;
        }
    }
    if (remaining_pending > 0) {
        LOGi("[TOOL_RESULT_STAGED]: Staged '%s' (%s); waiting for %zu more parallel tool result(s).", name.c_str(),
             call_id.c_str(), remaining_pending);
        return JNI_TRUE;
    }

    auto rollback_staged = [&]() {
        if (!g_staged_tool_msgs.empty()) {
            g_staged_tool_msgs.pop_back();
        }
        if (matched_idx >= 0 && static_cast<size_t>(matched_idx) < g_pending_tool_calls.size()) {
            g_pending_tool_calls[static_cast<size_t>(matched_idx)].completed          = false;
            g_pending_tool_calls[static_cast<size_t>(matched_idx)].waiting_for_result = true;
        }
    };

    std::string formatted_delta;
    format_tool_batch_delta(g_staged_tool_msgs, formatted_delta);
    g_last_tool_formatted_delta = formatted_delta;

    if (formatted_delta.empty()) {
        LOGw(
            "submitToolResultNative: Model template returned empty for tool role batch. Applying universal tool "
            "fallback format.");
        std::string fallback_combined;
        for (const auto & tm : g_staged_tool_msgs) {
            fallback_combined += "\n<tool_response>\n" + tm.content + "\n</tool_response>\n";
        }
        formatted_delta = std::move(fallback_combined);
    }

    // If the preceding assistant turn was cancelled early right at </tool_call> before generating its turn-end token
    // (e.g. <|im_end|>, <turn|>, <|eot_id|>, </s>), prepend the missing turn-end delimiter universally from model template!
    if (!g_cached_tokens.empty() && g_gen_state.finish_reason == FinishReason::CANCELLED &&
        !g_template_stop_strings.empty()) {
        std::string  tail_txt;
        const size_t tail_count = std::min<size_t>(6, g_cached_tokens.size());
        for (size_t i = g_cached_tokens.size() - tail_count; i < g_cached_tokens.size(); ++i) {
            if (g_cached_tokens[i] != LLAMA_TOKEN_NULL) {
                tail_txt += common_token_to_piece(g_context, g_cached_tokens[i], true);
            }
        }
        const size_t      fd_first     = formatted_delta.find_first_not_of(" \r\n\t");
        const std::string fd_trimmed   = (fd_first != std::string::npos) ? formatted_delta.substr(fd_first) : "";
        bool              has_turn_end = false;
        for (const auto & stop_str : g_template_stop_strings) {
            if (!stop_str.empty() && (tail_txt.find(stop_str) != std::string::npos ||
                                      formatted_delta.rfind(stop_str, 0) == 0 || fd_trimmed.rfind(stop_str, 0) == 0)) {
                has_turn_end = true;
                break;
            }
        }
        if (!has_turn_end) {
            const std::string & turn_end = g_template_stop_strings.front();
            if (!formatted_delta.empty() && formatted_delta.front() != '\n') {
                formatted_delta = turn_end + "\n" + formatted_delta;
            } else {
                formatted_delta = turn_end + formatted_delta;
            }
            LOGi("submitToolResultNative: Prepended missing template turn-end delimiter '%s' after early cancel.",
                 turn_end.c_str());
        }
    }

    LOGi("submitToolResultNative: Formatted delta (%zu chars, batch=%zu): `%s`", formatted_delta.size(),
         g_staged_tool_msgs.size(), formatted_delta.c_str());

    // Tokenize delta
    auto delta_tokens = common_tokenize(g_context, formatted_delta, false, true);
    if (delta_tokens.empty()) {
        LOGe("submitToolResultNative: Tokenization yielded 0 tokens for formatted delta!");
        rollback_staged();
        return JNI_FALSE;
    }

    // Capacity & shiftability check
    int       context_size   = static_cast<int>(llama_n_ctx(g_context));
    const int max_batch_size = context_size - OVERFLOW_HEADROOM;
    if (g_kv_cells_used + delta_tokens.size() > (size_t) max_batch_size) {
        bool           can_shift          = false;
        llama_memory_t mem                = llama_get_memory(g_context);
        const bool     has_non_text_media = (g_kv_cells_used != g_cached_tokens.size());
        if (mem && !has_non_text_media) {
            can_shift = llama_memory_can_shift(mem);
        }
        if (!can_shift) {
            if (!rebuild_context_preserving_tail(current_position, g_kv_cells_used, (int) delta_tokens.size() + 64,
                                                 false)) {
                LOGe(
                    "submitToolResultNative: Required %zu tokens (+ used %zu) exceeds context limit %d and tail "
                    "rebuild failed!",
                    delta_tokens.size(), g_kv_cells_used, max_batch_size);
                rollback_staged();
                return JNI_FALSE;
            }
        }
        LOGi("submitToolResultNative: Tool result size (%zu tokens, used=%zu) handled with context limit (%d).",
             delta_tokens.size(), g_kv_cells_used, context_size);
    }

    const llama_pos pos_before = current_position;

    // decode_tokens_in_batches internally handles dynamic in-prefill sliding, checkpointing, and rollback on error
    int decode_res = decode_tokens_in_batches(g_context, g_batch, delta_tokens, current_position, true);
    if (decode_res != 0) {
        LOGe("submitToolResultNative: decode_tokens_in_batches failed with code %d. Rollback handled by batch decoder.",
             decode_res);
        rollback_staged();
        return JNI_FALSE;
    }

    // Commit full batch to chat_msgs and mark pending call completed
    const size_t committed_batch_count = g_staged_tool_msgs.size();
    for (auto & tm : g_staged_tool_msgs) {
        chat_msgs.push_back(std::move(tm));
    }
    g_staged_tool_msgs.clear();
    g_pending_tool_calls.clear();
    g_pending_tool_call.completed          = true;
    g_pending_tool_call.waiting_for_result = false;
    g_cached_tokens.insert(g_cached_tokens.end(), delta_tokens.begin(), delta_tokens.end());

    LOGi(
        "[TOOL_RESULT_COMMIT]: batch=%zu, tokens=%zu, pos_before=%d, pos_after=%d, kv_cells=%zu, total_msgs=%zu, "
        "cached=%zu",
        committed_batch_count, delta_tokens.size(), pos_before, current_position, g_kv_cells_used, chat_msgs.size(),
        g_cached_tokens.size());

    return JNI_TRUE;
}

static bool is_valid_utf8(const char * string) {
    if (!string) {
        return true;
    }

    const auto * bytes = (const unsigned char *) string;
    while (*bytes != 0x00) {
        const unsigned char b0  = bytes[0];
        int                 num = 0;
        if (b0 <= 0x7F) {
            num = 1;
        } else if (b0 >= 0xC2 && b0 <= 0xDF) {
            num = 2;
        } else if (b0 >= 0xE0 && b0 <= 0xEF) {
            num = 3;
        } else if (b0 >= 0xF0 && b0 <= 0xF4) {
            num = 4;
        } else {
            return false;
        }

        for (int i = 1; i < num; ++i) {
            if ((bytes[i] & 0xC0) != 0x80) {
                return false;
            }
        }
        // RFC 3629 strict code point range & overlong/surrogate guards
        if (num == 3) {
            if (b0 == 0xE0 && bytes[1] < 0xA0) {
                return false;  // Overlong 3-byte
            }
            if (b0 == 0xED && bytes[1] >= 0xA0) {
                return false;  // UTF-16 surrogate U+D800..U+DFFF
            }
        } else if (num == 4) {
            if (b0 == 0xF0 && bytes[1] < 0x90) {
                return false;  // Overlong 4-byte
            }
            if (b0 == 0xF4 && bytes[1] > 0x8F) {
                return false;  // Above U+10FFFF
            }
        }
        bytes += num;
    }
    return true;
}

static jclass    g_string_class = nullptr;
static jmethodID g_string_ctor  = nullptr;
static jstring   g_utf8_charset = nullptr;
static jclass    g_event_cls    = nullptr;
static jmethodID g_event_ctor   = nullptr;

static bool is_pure_ascii(const std::string & str) {
    for (unsigned char c : str) {
        if (c >= 0x80) {
            return false;
        }
    }
    return true;
}

static jstring safe_new_string_utf(JNIEnv * env, const std::string & str) {
    if (str.empty()) {
        return nullptr;
    }
    // Fast path: Pure ASCII avoids all JNI overhead
    if (is_pure_ascii(str)) {
        jstring s = env->NewStringUTF(str.c_str());
        if (env->ExceptionCheck()) {
            env->ExceptionClear();
            return nullptr;
        }
        return s;
    }

    {
        std::lock_guard<std::recursive_mutex> cfg_lock(g_config_mutex);
        if (!g_string_class) {
            jclass local_cls = env->FindClass("java/lang/String");
            if (env->ExceptionCheck()) {
                env->ExceptionClear();
                local_cls = nullptr;
            }
            if (local_cls) {
                g_string_class = (jclass) env->NewGlobalRef(local_cls);
                g_string_ctor  = env->GetMethodID(g_string_class, "<init>", "([BLjava/lang/String;)V");
                if (env->ExceptionCheck()) {
                    env->ExceptionClear();
                    g_string_ctor = nullptr;
                }
                env->DeleteLocalRef(local_cls);
            }
            if (!g_utf8_charset) {
                jstring local_cs = env->NewStringUTF("UTF-8");
                if (local_cs) {
                    g_utf8_charset = (jstring) env->NewGlobalRef(local_cs);
                    env->DeleteLocalRef(local_cs);
                }
            }
        }
    }

    jsize      len   = static_cast<jsize>(str.size());
    jbyteArray bytes = env->NewByteArray(len);
    if (!bytes) {
        if (env->ExceptionCheck()) {
            env->ExceptionClear();
        }
        return env->NewStringUTF("");
    }
    env->SetByteArrayRegion(bytes, 0, len, reinterpret_cast<const jbyte *>(str.data()));
    jstring result = nullptr;
    if (g_string_class && g_string_ctor && g_utf8_charset) {
        result = static_cast<jstring>(env->NewObject(g_string_class, g_string_ctor, bytes, g_utf8_charset));
        if (env->ExceptionCheck()) {
            env->ExceptionClear();
            result = nullptr;
        }
    }
    env->DeleteLocalRef(bytes);
    return result ? result : env->NewStringUTF("");
}

static jstring safe_new_string_utf_nonnull(JNIEnv * env, const std::string & str) {
    if (str.empty()) {
        return env->NewStringUTF("");
    }
    jstring s = safe_new_string_utf(env, str);
    return s ? s : env->NewStringUTF("");
}

extern "C" JNIEXPORT jstring JNICALL Java_com_arm_aichat_internal_InferenceEngineImpl_generateNextToken(JNIEnv * env,
                                                                                                        jobject  thiz) {
    (void) thiz;
    std::lock_guard<std::recursive_mutex> cfg_lock(g_config_mutex);
    const int64_t                         t_tok_start = (g_timing_per_token && !g_no_perf) ? ggml_time_us() : 0;

    if (!g_model || !g_context || !g_batch_initialized || !g_sampler) {
        LOGe("%s: Engine not initialized", __func__);
        if (g_gen_state.is_active && !g_gen_state.is_finalized) {
            finalize_generation(FinishReason::ERROR_STATE);
        }
        return nullptr;
    }

    // Check atomic cancellation request before doing any computation
    if (g_cancel_requested.exchange(false, std::memory_order_acq_rel)) {
        LOGi("%s: Atomic cancel requested before token generation.", __func__);
        finalize_generation(FinishReason::CANCELLED);
        return nullptr;
    }

    // Infinite text generation via context shifting
    // Používáme g_kv_cells_used pro přesnou detekci přetečení (M-RoPE divergence fix)
    int context_size = static_cast<int>(llama_n_ctx(g_context));
    if (g_kv_cells_used >= (size_t) (context_size - OVERFLOW_HEADROOM)) {
        LOGw("%s: Context full! KV cells used: %zu, limit: %d. Shifting...", __func__, g_kv_cells_used, context_size);
        bool recovered = shift_context();
        if (!recovered) {
            recovered = rebuild_context_preserving_tail(current_position, g_kv_cells_used, 64, true);
        }
        if (!recovered) {
            LOGw("%s: Context shift/rebuild unsupported or failed at %zu/%d cells. Finalizing generation safely.",
                 __func__, g_kv_cells_used, context_size);
            finalize_generation(FinishReason::CONTEXT_FULL);
            return nullptr;
        }
    }

    // Stop if reaching the marked position
    if (current_position >= stop_generation_position) {
        LOGw("%s: STOP: hitting stop position: %d", __func__, stop_generation_position);
        finalize_generation(FinishReason::LENGTH);
        return nullptr;
    }

    // Sample next token
    const auto new_token_id = common_sampler_sample(g_sampler, g_context, -1);
    common_sampler_accept(g_sampler, new_token_id, true);

    // Populate the batch with new token, then decode
    common_batch_clear(g_batch);
    common_batch_add(g_batch, new_token_id, current_position, { 0 }, true);
    if (llama_decode(g_context, g_batch) != 0) {
        LOGe("%s: llama_decode() failed for generated token", __func__);
        finalize_generation(FinishReason::ERROR_STATE);
        return nullptr;
    }

    // Atomic token commit: update KV bookkeeping immediately after successful decode
    current_position++;
    g_kv_cells_used++;
    g_gen_state.generated_tokens_count++;
    g_cached_tokens.push_back(new_token_id);

    // Check atomic cancellation request after token is committed into KV bookkeeping
    if (g_cancel_requested.exchange(false, std::memory_order_acq_rel)) {
        LOGi("%s: Atomic cancel requested after token decode and commit.", __func__);
        finalize_generation(FinishReason::CANCELLED);
        return nullptr;
    }

    // Check for EOG (End of Generation) / EOS or template-level stop token ID
    // Only if NOT ignored by user setting
    if (!g_ignore_eos && (llama_vocab_is_eog(llama_model_get_vocab(g_model), new_token_id) ||
                          g_additional_stop_token_ids.count(new_token_id) != 0)) {
        LOGd("id: %d,\tIS EOG/STOP!\nSTOP.", new_token_id);
        finalize_generation(FinishReason::EOG);
        return nullptr;
    }

    // If not EOG, convert to text
    auto new_token_chars = common_token_to_piece(g_context, new_token_id);

    // Append to accumulated raw generated text for this generation
    g_gen_state.generated_raw_text += new_token_chars;

    // Check multi-token template/turn stop sequences without leaking them (exact, +LF, or +CRLF)
    auto check_stop_list = [&](const std::vector<std::string> & stops) -> bool {
        const std::string & raw = g_gen_state.generated_raw_text;
        for (const auto & stop_str : stops) {
            if (stop_str.empty()) {
                continue;
            }
            const size_t slen = stop_str.size();
            if (raw.size() >= slen && raw.compare(raw.size() - slen, slen, stop_str) == 0) {
                LOGi("Template stop sequence matched: '%s'. Finalizing generation.", stop_str.c_str());
                g_gen_state.generated_raw_text.erase(raw.size() - slen);
                return true;
            }
            if (raw.size() >= slen + 1 && raw.back() == '\n' &&
                raw.compare(raw.size() - slen - 1, slen, stop_str) == 0) {
                LOGi("Template stop sequence+LF matched: '%s'. Finalizing generation.", stop_str.c_str());
                g_gen_state.generated_raw_text.erase(raw.size() - slen - 1);
                return true;
            }
            if (raw.size() >= slen + 2 && raw[raw.size() - 2] == '\r' && raw.back() == '\n' &&
                raw.compare(raw.size() - slen - 2, slen, stop_str) == 0) {
                LOGi("Template stop sequence+CRLF matched: '%s'. Finalizing generation.", stop_str.c_str());
                g_gen_state.generated_raw_text.erase(raw.size() - slen - 2);
                return true;
            }
        }
        return false;
    };
    if (!g_ignore_eos &&
        (check_stop_list(g_template_stop_strings) || check_stop_list(g_current_turn_format.additional_stops))) {
        finalize_generation(FinishReason::EOG);
        return nullptr;
    }

    auto compute_partial_stop_holdback = [&](std::string_view sv) -> size_t {
        if (g_ignore_eos || sv.empty()) {
            return 0;
        }
        size_t max_hold   = 0;
        auto   scan_stops = [&](const std::vector<std::string> & stops) {
            for (const auto & stop_str : stops) {
                if (stop_str.size() <= 1) {
                    continue;
                }
                const size_t max_l = std::min(sv.size(), stop_str.size() - 1);
                for (size_t l = max_l; l >= 1; --l) {
                    if (stop_str.compare(0, l, sv.data() + sv.size() - l, l) == 0) {
                        if (l > max_hold) {
                            max_hold = l;
                        }
                        break;
                    }
                }
            }
        };
        scan_stops(g_template_stop_strings);
        scan_stops(g_current_turn_format.additional_stops);
        return max_hold;
    };

    // Fast zero-alloc semantic streaming with multi-token tag & stop boundary support
    if (!new_token_chars.empty()) {
        const std::string & raw = g_gen_state.generated_raw_text;
        while (g_gen_state.emitted_raw_pos < raw.size()) {
            const std::string_view rem_sv(raw.data() + g_gen_state.emitted_raw_pos,
                                          raw.size() - g_gen_state.emitted_raw_pos);
            if (!g_gen_state.in_reasoning) {
                const size_t stop_hold      = compute_partial_stop_holdback(rem_sv);
                const bool   has_tmpl_think = !g_current_turn_format.thinking_start_tag.empty();
                const bool   tmpl_think_lead =
                    has_tmpl_think &&
                    rem_sv.find(g_current_turn_format.thinking_start_tag[0]) != std::string_view::npos;
                if (stop_hold == 0 && !tmpl_think_lead &&
                    rem_sv.find_first_of(g_cot_lead_chars) == std::string_view::npos) {
                    std::string c(rem_sv);
                    g_gen_state.accumulated_content += c;
                    g_gen_state.last_parsed_msg.content += c;
                    g_gen_state.emitted_raw_pos = raw.size();
                    NativeSemanticEvent ev;
                    ev.type = SemanticEventType::CONTENT_DELTA;
                    ev.text = std::move(c);
                    push_semantic_event(std::move(ev));
                    break;
                }

                size_t            earliest_rel_pos = std::string_view::npos;
                const CoTTagDef * matched_tag      = nullptr;
                CoTTagDef         tmpl_tag_def;
                if (has_tmpl_think) {
                    const size_t p_tmpl = rem_sv.find(g_current_turn_format.thinking_start_tag);
                    if (p_tmpl != std::string_view::npos) {
                        earliest_rel_pos      = p_tmpl;
                        tmpl_tag_def.open_tag = g_current_turn_format.thinking_start_tag;
                        if (!g_current_turn_format.thinking_end_tags.empty()) {
                            tmpl_tag_def.close_tags = g_current_turn_format.thinking_end_tags;
                        } else if (!g_model_thinking_end_tag.empty()) {
                            tmpl_tag_def.close_tags = { g_model_thinking_end_tag };
                        } else {
                            tmpl_tag_def.close_tags = { "</think>" };
                        }
                        matched_tag = &tmpl_tag_def;
                    }
                }
                for (const auto & tag_def : g_active_cot_rules) {
                    const size_t p = rem_sv.find(tag_def.open_tag);
                    if (p != std::string_view::npos &&
                        (earliest_rel_pos == std::string_view::npos || p < earliest_rel_pos)) {
                        earliest_rel_pos = p;
                        matched_tag      = &tag_def;
                    }
                }

                if (matched_tag && earliest_rel_pos != std::string_view::npos) {
                    if (earliest_rel_pos > 0) {
                        std::string c(rem_sv.substr(0, earliest_rel_pos));
                        g_gen_state.accumulated_content += c;
                        g_gen_state.last_parsed_msg.content += c;
                        NativeSemanticEvent ev;
                        ev.type = SemanticEventType::CONTENT_DELTA;
                        ev.text = std::move(c);
                        push_semantic_event(std::move(ev));
                    }
                    g_gen_state.emitted_raw_pos += earliest_rel_pos + matched_tag->open_tag.size();
                    g_gen_state.in_reasoning          = true;
                    g_gen_state.active_cot_close_tags = matched_tag->close_tags;
                    g_model_thinking_start_tag        = matched_tag->open_tag;
                    if (!matched_tag->close_tags.empty()) {
                        g_model_thinking_end_tag = matched_tag->close_tags.front();
                    }
                    continue;
                }

                size_t max_partial_len = stop_hold;
                if (has_tmpl_think && g_current_turn_format.thinking_start_tag.size() > 1) {
                    const std::string & tt    = g_current_turn_format.thinking_start_tag;
                    const size_t        max_l = std::min(rem_sv.size(), tt.size() - 1);
                    for (size_t l = max_l; l >= 1; --l) {
                        if (tt.compare(0, l, rem_sv.data() + rem_sv.size() - l, l) == 0) {
                            if (l > max_partial_len) {
                                max_partial_len = l;
                            }
                            break;
                        }
                    }
                }
                for (const auto & tag_def : g_active_cot_rules) {
                    const size_t tlen = tag_def.open_tag.size();
                    if (tlen <= 1) {
                        continue;
                    }
                    const size_t max_l = std::min(rem_sv.size(), tlen - 1);
                    for (size_t l = max_l; l >= 1; --l) {
                        if (tag_def.open_tag.compare(0, l, rem_sv.data() + rem_sv.size() - l, l) == 0) {
                            if (l > max_partial_len) {
                                max_partial_len = l;
                            }
                            break;
                        }
                    }
                }

                const size_t safe_len = (rem_sv.size() > max_partial_len) ? (rem_sv.size() - max_partial_len) : 0;
                if (safe_len > 0) {
                    std::string c(rem_sv.substr(0, safe_len));
                    g_gen_state.accumulated_content += c;
                    g_gen_state.last_parsed_msg.content += c;
                    g_gen_state.emitted_raw_pos += safe_len;
                    NativeSemanticEvent ev;
                    ev.type = SemanticEventType::CONTENT_DELTA;
                    ev.text = std::move(c);
                    push_semantic_event(std::move(ev));
                }
                break;
            } else {
                static const std::vector<std::string> k_default_close_tags = { "</think>" };
                const std::vector<std::string> &      close_candidates =
                    !g_gen_state.active_cot_close_tags.empty() ?
                             g_gen_state.active_cot_close_tags :
                             (!g_current_turn_format.thinking_end_tags.empty() ? g_current_turn_format.thinking_end_tags :
                                                                                 k_default_close_tags);

                size_t earliest_rel_pos  = std::string_view::npos;
                size_t matched_close_len = 0;
                for (const auto & close_tag : close_candidates) {
                    const size_t p = rem_sv.find(close_tag);
                    if (p != std::string_view::npos &&
                        (earliest_rel_pos == std::string_view::npos || p < earliest_rel_pos)) {
                        earliest_rel_pos  = p;
                        matched_close_len = close_tag.size();
                    }
                }

                if (matched_close_len > 0 && earliest_rel_pos != std::string_view::npos) {
                    if (earliest_rel_pos > 0) {
                        std::string r(rem_sv.substr(0, earliest_rel_pos));
                        g_gen_state.accumulated_reasoning += r;
                        g_gen_state.last_parsed_msg.reasoning_content += r;
                        NativeSemanticEvent ev;
                        ev.type = SemanticEventType::REASONING_DELTA;
                        ev.text = std::move(r);
                        push_semantic_event(std::move(ev));
                    }
                    g_gen_state.emitted_raw_pos += earliest_rel_pos + matched_close_len;
                    g_gen_state.in_reasoning = false;
                    g_gen_state.active_cot_close_tags.clear();
                    continue;
                }

                size_t max_partial_len = compute_partial_stop_holdback(rem_sv);
                for (const auto & close_tag : close_candidates) {
                    const size_t tlen = close_tag.size();
                    if (tlen <= 1) {
                        continue;
                    }
                    const size_t max_l = std::min(rem_sv.size(), tlen - 1);
                    for (size_t l = max_l; l >= 1; --l) {
                        if (close_tag.compare(0, l, rem_sv.data() + rem_sv.size() - l, l) == 0) {
                            if (l > max_partial_len) {
                                max_partial_len = l;
                            }
                            break;
                        }
                    }
                }

                const size_t safe_len = (rem_sv.size() > max_partial_len) ? (rem_sv.size() - max_partial_len) : 0;
                if (safe_len > 0) {
                    std::string r(rem_sv.substr(0, safe_len));
                    g_gen_state.emitted_raw_pos += safe_len;
                    g_gen_state.accumulated_reasoning += r;
                    g_gen_state.last_parsed_msg.reasoning_content += r;
                    NativeSemanticEvent ev;
                    ev.type = SemanticEventType::REASONING_DELTA;
                    ev.text = std::move(r);
                    push_semantic_event(std::move(ev));
                }
                break;
            }
        }
    }

    cached_token_chars += new_token_chars;

    // Create and return a valid UTF-8 Java string (never return nullptr for empty pieces while active)
    jstring result = nullptr;
    if (is_valid_utf8(cached_token_chars.c_str())) {
        result = safe_new_string_utf_nonnull(env, cached_token_chars);
        LOGv("id: %d,\tcached: `%s`,\tnew: `%s`", new_token_id, cached_token_chars.c_str(), new_token_chars.c_str());
        assistant_ss << cached_token_chars;
        cached_token_chars.clear();
    } else {
        LOGv("id: %d,\tappend to cache", new_token_id);
        result = env->NewStringUTF("");
    }
    if (t_tok_start > 0) {
        const double tok_ms = static_cast<double>(ggml_time_us() - t_tok_start) / 1000.0;
        LOGd("token_timing_ms=%.2f (id=%d)", tok_ms, new_token_id);
    }
    return result;
}

extern "C" JNIEXPORT void JNICALL
Java_com_arm_aichat_internal_InferenceEngineImpl_cancelGenerationNative(JNIEnv * env, jobject thiz) {
    (void) env;
    (void) thiz;
    LOGi("cancelGenerationNative invoked from JNI - setting atomic cancel flag.");
    g_cancel_requested.store(true, std::memory_order_release);
}

extern "C" JNIEXPORT void JNICALL
Java_com_arm_aichat_internal_InferenceEngineImpl_finalizeCancelledGenerationNative(JNIEnv * env, jobject thiz) {
    (void) env;
    (void) thiz;
    std::lock_guard<std::recursive_mutex> cfg_lock(g_config_mutex);
    if (!g_gen_state.is_finalized && g_gen_state.is_active) {
        LOGi("finalizeCancelledGenerationNative: Synchronizing cancelled generation state before event drain.");
        finalize_generation(FinishReason::CANCELLED);
    }
}

extern "C" JNIEXPORT jobjectArray JNICALL
Java_com_arm_aichat_internal_InferenceEngineImpl_pollGenerationEventsNative(JNIEnv * env,
                                                                            jobject  thiz,
                                                                            jint     maxEvents) {
    (void) thiz;
    std::vector<NativeSemanticEvent> events;
    {
        std::lock_guard<std::mutex> lock(g_event_queue_mutex);
        int                         count = 0;
        int                         limit = (maxEvents > 0) ? maxEvents : 64;
        while (!g_native_event_queue.empty() && count < limit) {
            events.push_back(std::move(g_native_event_queue.front()));
            g_native_event_queue.pop_front();
            count++;
        }
    }

    {
        std::lock_guard<std::recursive_mutex> cfg_lock(g_config_mutex);
        if (!g_event_cls) {
            jclass local_cls = env->FindClass("com/arm/aichat/NativeGenerationEvent");
            if (env->ExceptionCheck()) {
                env->ExceptionClear();
                local_cls = nullptr;
            }
            if (local_cls) {
                g_event_cls  = (jclass) env->NewGlobalRef(local_cls);
                g_event_ctor = env->GetMethodID(
                    g_event_cls, "<init>",
                    "(JJILjava/lang/String;ILjava/lang/String;Ljava/lang/String;Ljava/lang/String;III)V");
                if (env->ExceptionCheck()) {
                    env->ExceptionClear();
                    g_event_ctor = nullptr;
                }
                env->DeleteLocalRef(local_cls);
            }
        }
    }
    if (!g_event_cls || !g_event_ctor) {
        LOGe("Failed to find class com/arm/aichat/NativeGenerationEvent or constructor");
        return nullptr;
    }
    jclass    event_cls = g_event_cls;
    jmethodID ctor      = g_event_ctor;

    jobjectArray result = env->NewObjectArray(static_cast<jsize>(events.size()), event_cls, nullptr);
    if (!result) {
        LOGe("Failed to allocate jobjectArray for NativeGenerationEvent");
        return nullptr;
    }

    for (size_t i = 0; i < events.size(); i++) {
        const auto & ev        = events[i];
        jstring      jtext     = !ev.text.empty() ? safe_new_string_utf(env, ev.text) : nullptr;
        jstring      jtoolId   = !ev.tool_id.empty() ? safe_new_string_utf(env, ev.tool_id) : nullptr;
        jstring      jtoolName = !ev.tool_name.empty() ? safe_new_string_utf(env, ev.tool_name) : nullptr;
        jstring      jtoolArgs = !ev.tool_args_delta.empty() ? safe_new_string_utf(env, ev.tool_args_delta) : nullptr;

        jobject obj = env->NewObject(event_cls, ctor, (jlong) ev.generation_id, (jlong) ev.sequence,
                                     (jint) static_cast<int32_t>(ev.type), jtext, (jint) ev.tool_index, jtoolId,
                                     jtoolName, jtoolArgs, (jint) static_cast<int32_t>(ev.finish_reason),
                                     (jint) ev.prompt_tokens, (jint) ev.generated_tokens);

        env->SetObjectArrayElement(result, static_cast<jsize>(i), obj);

        if (jtext) {
            env->DeleteLocalRef(jtext);
        }
        if (jtoolId) {
            env->DeleteLocalRef(jtoolId);
        }
        if (jtoolName) {
            env->DeleteLocalRef(jtoolName);
        }
        if (jtoolArgs) {
            env->DeleteLocalRef(jtoolArgs);
        }
        env->DeleteLocalRef(obj);
    }

    return result;
}

extern "C" JNIEXPORT void JNICALL Java_com_arm_aichat_internal_InferenceEngineImpl_resetContextNative(JNIEnv * env,
                                                                                                      jobject  thiz) {
    (void) env;
    (void) thiz;
    std::lock_guard<std::recursive_mutex> cfg_lock(g_config_mutex);
    LOGi("Performing explicit CONTEXT RESET via JNI...");
    g_cancel_requested.store(false, std::memory_order_release);
    if (!g_gen_state.is_finalized && g_gen_state.is_active) {
        finalize_generation(FinishReason::CANCELLED);
    }
    // Reset internal states (KV cache, history, pointers)
    reset_long_term_states();
    reset_short_term_states();
    clear_native_event_queue();

    // Reset sampler and clear stale grammar constraints
    if (g_sampler) {
        if (!g_sparams.grammar.empty() || g_sampler_has_turn_constraints) {
            detach_backend_sampler_if_attached();
            common_sampler_free(g_sampler);
            g_sampler                      = nullptr;
            g_sparams.grammar              = common_grammar();
            g_sampler_has_turn_constraints = false;
            g_sampling_cache.valid         = false;
            if (g_model) {
                common_params_sampling reset_copy = g_sparams;
                apply_ignore_eos_logit_bias(reset_copy);
                g_sampler = common_sampler_init(g_model, reset_copy);
                attach_backend_sampler_if_enabled();
            }
        } else {
            common_sampler_reset(g_sampler);
        }
    } else {
        g_sparams.grammar              = common_grammar();
        g_sampler_has_turn_constraints = false;
        g_sampling_cache.valid         = false;
    }
    {
        std::lock_guard<std::mutex> qlock(g_event_queue_mutex);
        g_pending_content_utf8.clear();
        g_pending_reasoning_utf8.clear();
    }
    LOGi("Context reset finished.");
}

extern "C" JNIEXPORT void JNICALL Java_com_arm_aichat_internal_InferenceEngineImpl_unload(JNIEnv * env, jobject thiz) {
    (void) env;
    (void) thiz;
    std::lock_guard<std::recursive_mutex> cfg_lock(g_config_mutex);
    g_cancel_requested.store(false, std::memory_order_release);
    // Reset long-term & short-term states
    reset_long_term_states();
    reset_short_term_states();
    clear_native_event_queue();

    // Detach and free sampler + grammar state
    detach_backend_sampler_if_attached();
    if (g_sampler) {
        common_sampler_free(g_sampler);
        g_sampler = nullptr;
    }
    g_sparams.grammar              = common_grammar();
    g_sampler_has_turn_constraints = false;
    g_chat_templates.reset();
    g_configured_tools.clear();
    g_last_tools_json_str.clear();
    g_tool_policy = NativeToolChoice::AUTO;
    g_active_cot_rules.clear();
    g_custom_cot_rules_set    = false;
    g_allow_native_tools_hint = true;
    g_tools_active_hint       = false;
    g_tool_markup_format      = 0;
    invalidate_model_template_caches();
    if (g_batch_initialized) {
        llama_batch_free(g_batch);
        g_batch             = {};
        g_batch_initialized = false;
    }
    if (g_cached_bmp) {
        mtmd_bitmap_free(g_cached_bmp);
        g_cached_bmp = nullptr;
    }
    if (g_mtmd_ctx) {
        mtmd_free(g_mtmd_ctx);
        g_mtmd_ctx = nullptr;
        LOGi("mtmd_context vision projector was freed.");
    }
    if (g_context) {
        llama_free(g_context);
        g_context = nullptr;
    }
    if (g_model) {
        llama_model_free(g_model);
        g_model = nullptr;
    }
}

extern "C" JNIEXPORT void JNICALL Java_com_arm_aichat_internal_InferenceEngineImpl_shutdown(JNIEnv * env,
                                                                                            jobject  thiz) {
    (void) thiz;
    std::lock_guard<std::recursive_mutex> cfg_lock(g_config_mutex);
    clear_native_event_queue();
    if (g_string_class) {
        env->DeleteGlobalRef(g_string_class);
        g_string_class = nullptr;
        g_string_ctor  = nullptr;
    }
    if (g_utf8_charset) {
        env->DeleteGlobalRef(g_utf8_charset);
        g_utf8_charset = nullptr;
    }
    if (g_event_cls) {
        env->DeleteGlobalRef(g_event_cls);
        g_event_cls  = nullptr;
        g_event_ctor = nullptr;
    }
    llama_backend_free();
}
