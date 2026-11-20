package com.example.llama.agent.adapter

import android.util.Log
import com.arm.aichat.NativeModelCapabilities
import com.arm.aichat.gguf.GgufMetadata
import com.example.llama.agent.adapter.policy.*
import com.example.llama.agent.adapter.template.*

/**
 * Selektivní metadatový router a továrna oddělující:
 * 1. ModelIdentityProfile (Base vs Instruct vs Reasoning vs Tool-trained + původ schopností)
 * 2. TemplateProtocolModule (způsob komunikace a doručení nástrojů do enginu)
 * 3. FamilySemanticPolicy + ModelFamilyAdapter (specifika rodiny bez přepisování osobnosti modelu)
 */
object ModelAdapterFactory {
    private const val TAG = "ModelAdapterFactory"

    fun getAdapter(
        architecture: String? = null,
        modelPath: String? = null,
        handlerName: String? = null,
        chatTemplate: String? = null,
        capabilities: NativeModelCapabilities? = null,
        ggufMetadata: GgufMetadata? = null,
        remoteCapabilities: RemoteDeclaredCapabilities? = null,
        identityOverride: ModelIdentityProfile? = null
    ): ModelFamilyAdapter {
        val identity = identityOverride ?: ModelIdentityDiscovery.discover(
            architecture = architecture,
            modelPath = modelPath,
            handlerName = handlerName,
            chatTemplate = chatTemplate,
            capabilities = capabilities,
            ggufMetadata = ggufMetadata,
            remoteCapabilities = remoteCapabilities
        )
        val archLower = architecture?.lowercase() ?: ""
        val pathLower = modelPath?.lowercase() ?: ""
        val handlerLower = handlerName?.lowercase() ?: ""
        val effectiveTemplate = capabilities?.chatTemplate?.takeIf { it.isNotBlank() }
            ?: chatTemplate?.takeIf { it.isNotBlank() }
            ?: ggufMetadata?.tokenizer?.chatTemplate?.takeIf { it.isNotBlank() }
        val tmplLower = effectiveTemplate?.lowercase() ?: ""

        // --- 1. Dimenze protokolu šablony (Template Protocol Module) ---
        val hasNativeToolCap = identity.nativeToolsSignal.supported ||
            (capabilities?.hasToolUseTemplate == true && capabilities.supportsJinja)
        val isDesktopLanHandler = identity.executionKind == EngineExecutionKind.DESKTOP_LAN
        val isCloudApiHandler = identity.executionKind == EngineExecutionKind.CLOUD_API
        val isCloudHandler = isDesktopLanHandler || isCloudApiHandler
        val resolvedEngineKind = identity.executionKind

        val templateModule: TemplateProtocolModule = when {
            // A) Cloud / Desktop LAN: systémový kontext a definice nástrojů jdou po HTTP/API
            isCloudHandler -> {
                Log.i(TAG, "🔧 Routing: JinjaPromptInjectedModule ($resolvedEngineKind)")
                JinjaPromptInjectedModule()
            }

            // B) Raw BASE model bez chat šablony -> nikdy nezabalovat do umělých Jinja tool/instruct šablon
            identity.isBaseWithoutTemplate -> {
                Log.i(TAG, "🔧 Routing: NonJinjaLegacyModule (raw BASE model without chat_template)")
                NonJinjaLegacyModule()
            }

            // C) Nativní Jinja2 nástroje v lokálním C++ enginu
            hasNativeToolCap -> {
                Log.i(TAG, "🔧 Routing: Jinja2NativeToolsModule (native tool branching)")
                Jinja2NativeToolsModule()
            }

            // D) User-Injected modely (Gemma 2/3 nebo šablony bez nativní podpory system role)
            (identity.systemRoleSignal.source == CapabilitySource.VERIFIED && !identity.systemRoleSignal.supported && identity.chatTemplateSignal.supported) ||
                (identity.primaryFamily == ModelFamily.GEMMA && !identity.systemRoleSignal.supported) ||
                (identityOverride == null &&
                    (archLower.contains("gemma") || pathLower.contains("gemma") || handlerLower.contains("gemma")) &&
                    !archLower.contains("gemma4") && !archLower.contains("gemma-4") &&
                    !pathLower.contains("gemma-4") && !pathLower.contains("gemma4") &&
                    !(capabilities?.hasChatTemplate == true && capabilities.supportsJinja && capabilities.supportsSystemRole)) -> {
                Log.i(TAG, "🔧 Routing: UserInjectedTemplateModule (standalone system role not supported natively)")
                UserInjectedTemplateModule()
            }

            // E) ChatML modely (explicitní <|im_start|> nebo handler 'chatml')
            tmplLower.contains("<|im_start|>") || handlerLower.contains("chatml") -> {
                Log.i(TAG, "🔧 Routing: ChatMlTemplateModule (ChatML protocol)")
                ChatMlTemplateModule()
            }

            // F) Jinja modely bez tool filtrů (včetně Qwen-arch destilátů jako DeepSeek-R1 s <|User|>/<|Assistant|>)
            capabilities?.supportsJinja == true ||
                identity.chatTemplateSignal.detail.contains("Jinja", ignoreCase = true) ||
                tmplLower.contains("{%") -> {
                Log.i(TAG, "🔧 Routing: JinjaPromptInjectedModule")
                JinjaPromptInjectedModule()
            }

            // G) Fallback pro starší Qwen GGUF bez dostupné Jinja šablony (mimo DeepSeek/R1 rodinu)
            (identity.primaryFamily == ModelFamily.QWEN ||
                (identityOverride == null && identity.primaryFamily == ModelFamily.GENERIC &&
                    (archLower.contains("qwen") || pathLower.contains("qwen")))) &&
                identity.primaryFamily != ModelFamily.DEEPSEEK &&
                !archLower.contains("deepseek") && !pathLower.contains("deepseek") && !pathLower.contains("r1") -> {
                Log.i(TAG, "🔧 Routing: ChatMlTemplateModule (Qwen fallback without Jinja)")
                ChatMlTemplateModule()
            }

            // H) Legacy / Non-Jinja
            else -> {
                Log.i(TAG, "🔧 Routing: NonJinjaLegacyModule")
                NonJinjaLegacyModule()
            }
        }

        // --- 2. Dimenze rodinné sémantiky (Family Semantic Policy) + konkrétní rodinný adaptér ---
        val (familyPolicy, familyAdapter) = when (identity.primaryFamily) {
            ModelFamily.DEEPSEEK -> DeepSeekReasoningPolicy() to DeepSeekModelAdapter()
            ModelFamily.GEMINI -> GeminiFamilyPolicy() to GeminiModelAdapter()
            ModelFamily.GEMMA -> GemmaFamilyPolicy() to GemmaModelAdapter()
            ModelFamily.QWEN -> QwenFamilyPolicy() to QwenModelAdapter()
            ModelFamily.MISTRAL -> MistralFamilyPolicy() to MistralModelAdapter()
            ModelFamily.PHI -> PhiFamilyPolicy() to PhiModelAdapter()
            ModelFamily.MINICPM -> {
                val policy = MiniCpmFamilyPolicy()
                policy to GenericModelAdapter(familyName = policy.familyName, familyPolicy = policy)
            }
            ModelFamily.GLM -> {
                val policy = GlmFamilyPolicy()
                policy to GenericModelAdapter(familyName = policy.familyName, familyPolicy = policy)
            }
            ModelFamily.LLAMA -> LlamaFamilyPolicy() to LlamaModelAdapter()
            ModelFamily.GPT_OSS -> {
                val policy = GenericFamilyPolicy(familyName = "GPT-OSS", primaryFamily = ModelFamily.GPT_OSS)
                policy to GenericModelAdapter(familyName = policy.familyName, familyPolicy = policy)
            }
            ModelFamily.OPENAI -> {
                val policy = GenericFamilyPolicy(familyName = "OpenAI", primaryFamily = ModelFamily.OPENAI)
                policy to GenericModelAdapter(familyName = policy.familyName, familyPolicy = policy)
            }
            ModelFamily.CLAUDE -> {
                val policy = GenericFamilyPolicy(familyName = "Claude", primaryFamily = ModelFamily.CLAUDE)
                policy to GenericModelAdapter(familyName = policy.familyName, familyPolicy = policy)
            }
            ModelFamily.GROK -> {
                val policy = GenericFamilyPolicy(familyName = "Grok", primaryFamily = ModelFamily.GROK)
                policy to GenericModelAdapter(familyName = policy.familyName, familyPolicy = policy)
            }
            ModelFamily.GENERIC -> GenericFamilyPolicy() to GenericModelAdapter()
        }

        val controller = ModelAgentController(
            templateModule = templateModule,
            familyPolicy = familyPolicy,
            familyAdapter = familyAdapter,
            isRemoteEngineAdapter = isCloudHandler,
            engineKind = resolvedEngineKind,
            identity = identity
        )

        Log.i(
            TAG,
            "✅ Composed Controller: ${controller.familyName} | kind=${identity.trainingKind} | " +
                "variants=${identity.variants} | effort=${identity.effortMechanism} | " +
                "sysRole=${identity.systemRoleSignal.supported}(${identity.systemRoleSignal.source}) | " +
                "allowNativeTools=${controller.runtimeSpec.allowNativeTools}(${identity.nativeToolsSignal.source}) | " +
                "parallelTools=${controller.runtimeSpec.supportsParallelToolCalls}(${identity.parallelToolCallsSignal.source}) | " +
                "detectedFormat=${controller.runtimeSpec.detectedChatFormat} | " +
                "markup=${controller.runtimeSpec.toolMarkupFormat} | detectedCot=${identity.detectedTemplateCotTags} | " +
                "arch='$architecture', path='$modelPath'"
        )
        return controller
    }
}
