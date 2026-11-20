package com.arm.aichat.telemetry

/**
 * Release no-op stub for AdbTelemetry (zero runtime overhead and no logcat output in release builds).
 */
object AdbTelemetry {
    enum class Backend(val wireName: String) {
        CPU("CPU"),
        GPU("GPU"),
        VULKAN("Vulkan"),
        OPENCL("OpenCL"),
        OPENGL("OpenGL"),
        NPU("NPU"),
        QNN("QNN"),
        QNN_HTP("QNN-HTP"),
        QNN_GPU("QNN-GPU"),
        QNN_CPU("QNN-CPU"),
        QNN_DSP("QNN-DSP"),
        DSP("DSP")
    }

    private val NO_OP = Operation()

    fun resolveBackend(activeBackendInfo: String): Backend = Backend.CPU

    fun start(backend: Backend): Operation = NO_OP

    class Operation internal constructor() {
        fun beat() {}
        fun done(profileConfirmed: Boolean = false) {}
        fun fallback(to: Backend) {}
        fun error() {}
    }
}
