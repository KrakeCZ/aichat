package com.arm.aichat.telemetry

import android.os.SystemClock
import android.util.Log
import java.util.Locale
import java.util.concurrent.atomic.AtomicLong

/**
 * Debug-only telemetry emitter compatible with ADB Hardware Monitor (tag: "ADBTelemetry").
 * Emits only from actual backend execution call sites in debug builds.
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

    private const val TAG = "ADBTelemetry"
    private val nextId = AtomicLong()

    fun resolveBackend(activeBackendInfo: String): Backend {
        val info = activeBackendInfo.lowercase(Locale.ROOT)
        return when {
            info.contains("hexagon") || info.contains("htp") -> Backend.QNN_HTP
            info.contains("opencl") -> Backend.OPENCL
            info.contains("vulkan") -> Backend.VULKAN
            else -> Backend.CPU
        }
    }

    /** Call directly before submitting a request to [backend]. */
    fun start(backend: Backend): Operation {
        val operation = Operation(backend, nextId.incrementAndGet(), SystemClock.elapsedRealtimeNanos())
        Log.i(TAG, "backend=${backend.wireName} event=start id=${operation.id}")
        return operation
    }

    class Operation internal constructor(
        private val backend: Backend,
        internal val id: Long,
        private val startedNs: Long
    ) {
        private var finished = false

        @Synchronized
        fun beat() {
            if (!finished) emit("beat")
        }

        /** Wait for completion and verify backend selection before calling this. */
        @Synchronized
        fun done(profileConfirmed: Boolean = false) {
            val elapsedMs = (SystemClock.elapsedRealtimeNanos() - startedNs) / 1_000_000.0
            val proof = if (profileConfirmed) "profile" else "runtime"
            terminal("done", " proof=$proof ms=${String.format(Locale.ROOT, "%.3f", elapsedMs)}")
        }

        @Synchronized
        fun fallback(to: Backend) {
            if (to != backend) terminal("fallback", " to=${to.wireName}")
        }

        @Synchronized
        fun error() {
            terminal("error")
        }

        private fun terminal(event: String, suffix: String = "") {
            if (finished) return
            finished = true
            emit(event, suffix)
        }

        private fun emit(event: String, suffix: String = "") {
            Log.i(TAG, "backend=${backend.wireName} event=$event id=$id$suffix")
        }
    }
}
