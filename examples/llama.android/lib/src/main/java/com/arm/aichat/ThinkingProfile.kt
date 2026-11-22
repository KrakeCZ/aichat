package com.arm.aichat

/**
 * Effort Levels controlling test-time compute / reasoning depth without mid-thought token truncation.
 *
 * maxTokens = -1 with isEnabled = true guarantees no hard token cutoff (rbudget disabled);
 * effort steers model/template/API reasoning intensity ("none", "minimal", "low", "medium", "high", "xhigh", "max"),
 * while OEM ("") preserves each model's factory default without overrides.
 */
enum class ThinkingProfile(
    val id: Int,
    val maxTokens: Int,
    val effort: String,
    val displayName: String
) {
    OEM(4, -1, "", "OEM"),
    OFF(0, 0, "none", "Off"),
    MINIMAL(5, -1, "minimal", "Min"),
    LOW(1, -1, "low", "Low"),
    MEDIUM(2, -1, "medium", "Med"),
    DEEP(3, -1, "high", "High"),
    XHIGH(6, -1, "xhigh", "XHigh"),
    MAX(7, -1, "max", "Max");

    val isEnabled: Boolean
        get() = this != OFF

    companion object {
        fun fromId(id: Int): ThinkingProfile = values().firstOrNull { it.id == id } ?: MEDIUM
    }
}
