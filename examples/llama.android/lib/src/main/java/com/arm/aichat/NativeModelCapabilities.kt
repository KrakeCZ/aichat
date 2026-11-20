package com.arm.aichat

/**
 * Diagnostic model capabilities reported natively by the llama.cpp chat template engine.
 */
data class NativeModelCapabilities @JvmOverloads constructor(
    val hasChatTemplate: Boolean,
    val supportsJinja: Boolean,
    val supportsThinking: Boolean,
    val supportsTools: Boolean,
    val supportsToolCalls: Boolean,
    val detectedFormat: Int,
    val handlerName: String,
    val supportsReasoningEffort: Boolean = false,
    val contextLength: Int = 0,
    val chatTemplate: String = "",
    val supportsSystemRole: Boolean = true,
    val supportsParallelToolCalls: Boolean = false
) {
    val hasToolUseTemplate: Boolean
        get() = supportsTools && supportsToolCalls
}
