package com.example.llama

import android.content.Intent
import android.graphics.Bitmap
import android.net.Uri
import android.os.Bundle
import android.text.Editable
import android.text.TextWatcher
import android.util.Log
import android.view.MotionEvent
import android.view.ScaleGestureDetector
import android.view.View
import android.view.ViewGroup
import android.view.ViewStub
import android.widget.EditText
import android.widget.FrameLayout
import android.widget.ImageButton
import android.widget.ProgressBar
import android.widget.TextView
import android.widget.Toast
import androidx.activity.addCallback
import androidx.activity.enableEdgeToEdge
import androidx.activity.viewModels
import androidx.appcompat.app.AppCompatActivity
import androidx.constraintlayout.widget.ConstraintLayout
import androidx.core.view.GravityCompat
import androidx.core.view.ViewCompat
import androidx.core.view.WindowCompat
import androidx.core.view.WindowInsetsCompat
import androidx.core.view.WindowInsetsControllerCompat
import androidx.core.view.updatePadding
import androidx.drawerlayout.widget.DrawerLayout
import androidx.lifecycle.lifecycleScope
import androidx.recyclerview.widget.LinearLayoutManager
import androidx.recyclerview.widget.RecyclerView
import com.arm.aichat.AiChat
import com.arm.aichat.InferenceEngine
import com.arm.aichat.ThinkingProfile
import com.arm.aichat.isModelLoaded
import com.arm.aichat.isUninterruptible
import com.google.android.material.button.MaterialButton
import com.google.android.material.button.MaterialButtonToggleGroup
import com.example.llama.agent.skills.SkillRegistry
import com.example.llama.agent.skills.SkillsUiManager
import com.example.llama.agent.tools.web.WebToolsDispatcher
import com.example.llama.agent.tools.web.ToolCallRequest
import com.example.llama.data.AiModel
import com.example.llama.infrastructure.voice.AndroidVoiceManager
import com.example.llama.managers.ChatSessionManager
import com.example.llama.managers.FileSystemManager
import com.example.llama.managers.LlamaServiceManager
import com.example.llama.managers.ModelMetadataManager
import com.example.llama.managers.PermissionManager
import com.example.llama.managers.ThemeManager
import com.example.llama.agent.vision.VisionSourceManager
import com.example.llama.service.GenerationState
import com.example.llama.service.LlamaGenerationService
import com.example.llama.service.LlamaNotificationHelper
import com.example.llama.ui.ModelInfoDialogFragment
import com.example.llama.ui.dialogs.DialogManager
import com.google.android.material.appbar.MaterialToolbar
import com.google.android.material.navigation.NavigationView
import java.util.UUID
import kotlin.math.abs
import kotlinx.coroutines.Dispatchers
import kotlinx.coroutines.Job
import kotlinx.coroutines.flow.onCompletion
import kotlinx.coroutines.launch
import kotlinx.coroutines.sync.Mutex
import kotlinx.coroutines.sync.withLock
import kotlinx.coroutines.withContext

class MainActivity : AppCompatActivity() {
    private val viewModel: MainViewModel by viewModels()
    private lateinit var voiceManager: AndroidVoiceManager
    private lateinit var voiceInputManager: com.example.llama.agent.voice.AgentVoiceSTT
    private lateinit var ggufTv: TextView
    private lateinit var messagesRv: RecyclerView
    private lateinit var userInputEt: EditText
    private lateinit var sendButton: ImageButton
    private lateinit var speedTv: TextView
    private lateinit var toggleThinkingProfile: MaterialButtonToggleGroup
    private lateinit var btnSkillsSelector: MaterialButton
    private lateinit var btnToolsSelector: MaterialButton
    private lateinit var btnWebPipSelector: MaterialButton
    private var pipStub: ViewStub? = null
    private var pipRootView: View? = null
    private var pipContainer: FrameLayout? = null
    private var pipWebView: android.webkit.WebView? = null
    private var pipUrlTv: TextView? = null
    private var pipProgressBar: ProgressBar? = null
    private var pipExpandBtn: TextView? = null
    private var pipCloseBtn: ImageButton? = null
    private var isPipExpanded: Boolean = false
    @Volatile
    private var isPipOfflineLocked: Boolean = false
    @Volatile
    private var currentPipUrl: String = ""
    @Volatile
    private var pipPageFinishedWaiter: kotlinx.coroutines.CompletableDeferred<Boolean>? = null
    private val pipNavMutex = Mutex()
    private lateinit var subagentIndicatorTv: TextView
    private lateinit var tokenCountTv: TextView
    private lateinit var genTimeTv: TextView
    private lateinit var drawerLayout: DrawerLayout
    private lateinit var navView: NavigationView
    private lateinit var toolbar: MaterialToolbar
    private lateinit var mainContentContainer: ConstraintLayout
    private lateinit var engine: InferenceEngine
    private var generationJob: Job? = null
    private val messages = mutableListOf<Message>()
    private lateinit var messageAdapter: MessageAdapter
    private lateinit var historyManager: ChatHistoryManager
    private lateinit var settingsManager: SettingsManager
    private lateinit var fileManager: FileSystemManager
    private lateinit var sessionManager: ChatSessionManager
    private lateinit var permissionManager: PermissionManager
    private var currentModelPath: String? = null
    private var currentProjectorPath: String? = null
    private var currentModelArchitecture: String? = null
    private var currentModelChatTemplate: String? = null
    private var lastAppliedContextSizeSetting: Int? = null
    private var lastHandledCompleteState: GenerationState.Complete? = null
    private lateinit var serviceManager: LlamaServiceManager
    private var isFirstAgentEmit = true
    // Pending image flag: Když je true, bypass Agent routing a jdi přímo do LLM chat
    // (C++ vrstva má g_cached_bmp a auto-prepend <__media__> marker)
    @Volatile private var hasPendingImage = false
    private var lastAppliedSkillId: String? = null
    private var lastAppliedToolsMode: SettingsManager.ToolsMode? = null
    private var lastAppliedSystemMessage: String? = null
    private var lastObservedBehaviorSignature: String? = null
    private var isSessionSystemPromptSet = false
    private var lastActiveSessionId: String? = null

    override fun onCreate(savedInstanceState: Bundle?) {
        super.onCreate(savedInstanceState)
        enableEdgeToEdge()
        setContentView(R.layout.activity_main)

        drawerLayout = findViewById(R.id.drawer_layout)
        ViewCompat.setOnApplyWindowInsetsListener(drawerLayout) { view, insets ->
            val systemBars = insets.getInsets(WindowInsetsCompat.Type.systemBars())
            view.updatePadding(
                    left = systemBars.left,
                    right = systemBars.right,
                    bottom = systemBars.bottom
            )
            insets
        }

        historyManager = ChatHistoryManager(this)
        settingsManager = SettingsManager(this)
        fileManager = FileSystemManager(this)
        fileManager.onModelSelectedCallback = { uri -> handleSelectedModelImport(uri) }
        permissionManager = PermissionManager(this)

        // CRITICAL: Inicializovat audio orchestrátor pro chat TTS
        // (Idempotentní — bezpečné volat i pokud již inicializován z AccessibilityService)
        com.example.llama.agent.core.audio.AgentAudioOrchestrator.initialize(applicationContext)

        // Web Tools Dispatcher & Search Engine Config Init (DataSaver, FTS5 cache, User RAG)
        WebToolsDispatcher.initialize(applicationContext)
        settingsManager.syncToWebSearchEngineConfig()
        // Synchronize settings on resume (keep lastAppliedSystemMessage for diff check in buildEffectivePromptToSend)
        if (lastAppliedSystemMessage == null) {
            lastAppliedSystemMessage = settingsManager.systemMessage
        }

        initPipController()

        voiceManager = AndroidVoiceManager(this, lifecycleScope)
        lifecycle.addObserver(voiceManager)

        // INTEGRACE: Standard Chat Dictation
        voiceInputManager = com.example.llama.agent.voice.AgentVoiceSTT(this)
        // Race Condition Fix: Do NOT initialize here if Agent might be using it.
        // Initialization happens lazily in Observer or explicitly later.

        messageAdapter =
                MessageAdapter(messages, this) { text ->
                    if (settingsManager.isTtsEnabled) {
                        val currentState = voiceManager.state.value
                        if (currentState is com.example.llama.domain.voice.VoiceState.Speaking &&
                                        currentState.text == text
                        ) {
                            voiceManager.stopSpeaking()
                        } else {
                            voiceManager.speak(text)
                        }
                    } else {
                        Toast.makeText(this, "TTS is disabled in settings", Toast.LENGTH_SHORT)
                                .show()
                    }
                }
        messageAdapter.setHasStableIds(true)
        messageAdapter.thinkingProfile = settingsManager.thinkingProfile

        ggufTv = findViewById(R.id.gguf)
        messagesRv = findViewById(R.id.messages)
        val layoutManager = LinearLayoutManager(this).apply { stackFromEnd = true }
        messagesRv.layoutManager = layoutManager
        messagesRv.itemAnimator = null // CRITICAL for performance
        messagesRv.adapter = messageAdapter
        messageAdapter.fontSizeSp = settingsManager.chatFontSizeSp

        val chatScaleDetector = ScaleGestureDetector(this, object : ScaleGestureDetector.SimpleOnScaleGestureListener() {
            override fun onScale(detector: ScaleGestureDetector): Boolean {
                val scaleFactor = detector.scaleFactor
                if (scaleFactor != 1.0f) {
                    val currentSize = messageAdapter.fontSizeSp
                    val newSize = (currentSize * scaleFactor).coerceIn(11.0f, 34.0f)
                    if (kotlin.math.abs(newSize - currentSize) >= 0.2f) {
                        messageAdapter.fontSizeSp = newSize
                        settingsManager.chatFontSizeSp = newSize
                    }
                }
                return true
            }
        })

        messagesRv.addOnItemTouchListener(object : RecyclerView.SimpleOnItemTouchListener() {
            override fun onInterceptTouchEvent(rv: RecyclerView, e: MotionEvent): Boolean {
                if (e.pointerCount >= 2) {
                    chatScaleDetector.onTouchEvent(e)
                    return chatScaleDetector.isInProgress
                }
                return false
            }

            override fun onTouchEvent(rv: RecyclerView, e: MotionEvent) {
                chatScaleDetector.onTouchEvent(e)
            }
        })
        userInputEt = findViewById(R.id.user_input)
        sendButton = findViewById(R.id.send_button)
        speedTv = findViewById(R.id.tv_speed)
        subagentIndicatorTv = findViewById(R.id.tv_subagent_indicator)
        toggleThinkingProfile = findViewById(R.id.toggle_thinking_profile)
        tokenCountTv = findViewById(R.id.tv_token_count)
        genTimeTv = findViewById(R.id.tv_gen_time)
        navView = findViewById(R.id.nav_view)
        toolbar = findViewById(R.id.toolbar)
        mainContentContainer = findViewById(R.id.main_content)
        sessionManager =
                ChatSessionManager(
                        activity = this,
                        adapter = messageAdapter,
                        messages = messages,
                        historyManager = historyManager,
                        settingsManager = settingsManager,
                        drawerLayout = drawerLayout,
                        navView = navView
                )
        sessionManager.onSettingsClicked = { openSettings() }
        sessionManager.onModelManagementClicked = { showModelSelector() }
        sessionManager.onSkillsClicked = {
            SkillsUiManager.showSkillsManager(this) {
                updateSkillsButtonLabel()
            }
        }

        setupToolbar()
        sessionManager.refreshNavigationMenu()
        setupInputListeners()
        setupThinkingProfileToggle()
        btnSkillsSelector = findViewById(R.id.btn_skills_selector)
        updateSkillsButtonLabel()
        btnSkillsSelector.setOnClickListener {
            SkillsUiManager.showQuickSkillSelector(this) {
                updateSkillsButtonLabel()
            }
        }
        btnToolsSelector = findViewById(R.id.btn_tools_selector)
        updateToolsButtonLabel()
        btnToolsSelector.setOnClickListener {
            showToolsSelectorDialog()
        }
        btnToolsSelector.setOnLongClickListener {
            toggleToolsQuick()
            true
        }
        btnWebPipSelector = findViewById(R.id.btn_web_pip_selector)
        pipStub = findViewById(R.id.stub_in_chat_pip)
        updateWebPipButtonLabel()
        btnWebPipSelector.setOnClickListener {
            showWebPipSelectorDialog()
        }
        btnWebPipSelector.setOnLongClickListener {
            toggleWebPipQuick()
            true
        }
        setupVoiceInputObserver() // Moved here to ensure Views are initialized
        setupVoiceObserver() // Moved here
        setupViewModelObservers()
        permissionManager.checkAndRequestNotificationPermission()
        ggufTv.setOnClickListener { handleModelInfoClick() }
        sendButton.setOnClickListener { handleSendButtonClick() }

        setupMultiModalCallbacks()

        serviceManager = LlamaServiceManager(this, lifecycleScope)
        val runBackground = settingsManager.backgroundGeneration
        serviceManager.connect(asResident = runBackground)

        lifecycleScope.launch {
            serviceManager.generationState.collect { state -> handleGenerationState(state) }
        }

        lifecycleScope.launch(Dispatchers.Default) {
            try {
                engine = com.example.llama.agent.core.EngineProvider.get(applicationContext)
                val tp = settingsManager.thinkingProfile
                engine.configureThinking(tp.isEnabled, tp.maxTokens, tp.effort)
                withContext(Dispatchers.Main) {
                    setupEngineStateObserver()
                    setupRobustOnBackPressed()
                }
            } catch (e: Exception) {
                Log.e(TAG, "Engine init failed", e)
                withContext(Dispatchers.Main) {
                    Toast.makeText(
                                    this@MainActivity,
                                    "Engine Init Failed: ${e.message}",
                                    Toast.LENGTH_LONG
                            )
                            .show()
                }
            }
        }
        updateSendButtonState()
        handleIntentExtras(intent)
    }

    override fun onNewIntent(intent: android.content.Intent) {
        super.onNewIntent(intent)
        setIntent(intent)
        handleIntentExtras(intent)
    }

    private fun handleIntentExtras(intent: android.content.Intent?) {
        val modelTarget = intent?.getStringExtra("load_model")
        val promptToSend = intent?.getStringExtra("send_prompt")

        if (modelTarget != null) {
            lifecycleScope.launch {
                var attempts = 0
                while (attempts < 30 && (!::engine.isInitialized || viewModel.savedModels.value.isEmpty())) {
                    kotlinx.coroutines.delay(200)
                    attempts++
                }
                val models = viewModel.savedModels.value
                val matched = models.firstOrNull { it.displayName.contains(modelTarget, ignoreCase = true) }
                if (matched != null) {
                    Log.i(TAG, "🤖 Intent Auto-Load Model: ${matched.displayName}")
                    withContext(Dispatchers.Main) {
                        handleLoadLocalModel(matched)
                    }
                    if (promptToSend != null) {
                        while (!::engine.isInitialized || !engine.state.value.isModelLoaded) {
                            kotlinx.coroutines.delay(300)
                        }
                        kotlinx.coroutines.delay(800)
                        withContext(Dispatchers.Main) {
                            userInputEt.setText(promptToSend)
                            handleUserInput()
                        }
                    }
                } else {
                    Log.w(TAG, "⚠️ Model matching '$modelTarget' not found in saved models: ${models.map { it.displayName }}")
                }
            }
        } else if (promptToSend != null) {
            lifecycleScope.launch {
                while (!::engine.isInitialized || !engine.state.value.isModelLoaded) {
                    kotlinx.coroutines.delay(300)
                }
                withContext(Dispatchers.Main) {
                    userInputEt.setText(promptToSend)
                    handleUserInput()
                }
            }
        }
    }

    private fun setupMultiModalCallbacks() {
        // Zpracování textových souborů a dokumentů
        fileManager.onTextAttachedCallback = { textToAppend ->
            val currentText = userInputEt.text.toString().trim()
            val formatted = if (currentText.isNotBlank()) {
                "$currentText\n\n[Přiložený text / soubor k analýze]:\n```\n$textToAppend\n```"
            } else {
                "[Přiložený text / soubor k analýze]:\n```\n$textToAppend\n```\n"
            }
            userInputEt.setText(formatted)
            userInputEt.setSelection(userInputEt.text.length)
        }

        // Zpracování statických obrázků do kontextu
        fileManager.onImageAttachedCallback = { imageBytes ->
            lifecycleScope.launch(Dispatchers.IO) {
                try {
                    val isPdfPayload = imageBytes.size >= 5 &&
                        imageBytes[0] == '%'.code.toByte() &&
                        imageBytes[1] == 'P'.code.toByte() &&
                        imageBytes[2] == 'D'.code.toByte() &&
                        imageBytes[3] == 'F'.code.toByte() &&
                        imageBytes[4] == '-'.code.toByte()
                    val compressed = if (isPdfPayload && settingsManager.isRemoteInferenceActive) {
                        imageBytes
                    } else {
                        com.example.llama.agent.vision.VisionSourceManager.processAndCompressImageBlocking(imageBytes)
                    }
                    if (compressed != null) {
                        var modelNotLoaded = false
                        var missingProjector = false
                        val processed = com.example.llama.agent.core.EngineOrchestrator.withForegroundLock(
                            context = applicationContext,
                            owner = com.example.llama.agent.core.EngineOwner.FOREGROUND_CHAT,
                            config = null
                        ) { orchEngine ->
                            if (!orchEngine.state.value.isModelLoaded) {
                                modelNotLoaded = true
                                false
                            } else if (!orchEngine.hasProjector() && !settingsManager.isRemoteInferenceActive) {
                                missingProjector = true
                                false
                            } else {
                                orchEngine.processImage(compressed)
                            }
                        }
                        if (!processed) {
                            hasPendingImage = false
                            withContext(Dispatchers.Main) {
                                if (modelNotLoaded) {
                                    Toast.makeText(this@MainActivity, "Model není načten", Toast.LENGTH_SHORT).show()
                                } else {
                                    Toast.makeText(
                                        this@MainActivity,
                                        if (missingProjector) {
                                            "Model nemá načtený multimodální projektor (.mmproj) nebo nepodporuje obrázky."
                                        } else {
                                            "Zpracování přílohy selhalo."
                                        },
                                        Toast.LENGTH_LONG
                                    ).show()
                                }
                            }
                            return@launch
                        }
                        hasPendingImage = true // Signalizace pro handleUserInput()

                        // ============ MEDIA PREVIEW (podmíněné) ============
                        // Thumbnail se vytváří JEN pokud je preview zapnutý.
                        // inSampleSize=4 → ~4x menší Bitmap → nízká RAM zátěž.
                        val thumbnail = if (!isPdfPayload && settingsManager.isMediaPreviewEnabled) {
                            try {
                                val opts = android.graphics.BitmapFactory.Options().apply {
                                    inSampleSize = 4 // Downscale: 4000x3000 → 1000x750
                                    inPreferredConfig = android.graphics.Bitmap.Config.RGB_565 // 16-bit = 50% RAM vs ARGB_8888
                                }
                                android.graphics.BitmapFactory.decodeByteArray(imageBytes, 0, imageBytes.size, opts)
                            } catch (e: Exception) {
                                Log.w(TAG, "Thumbnail decode failed (non-critical)", e)
                                null
                            }
                        } else null

                        withContext(Dispatchers.Main) {
                            // Přidat user zprávu s náhledem do chatu
                            val placeholderText = if (isPdfPayload) {
                                "[📄 PDF dokument přijat (${imageBytes.size / 1024} KB) — zadej dotaz k dokumentu]"
                            } else {
                                "[📷 Obrázek přijat — zeptej se 'Co je na obrázku?']"
                            }
                            val msg = Message(
                                id = java.util.UUID.randomUUID().toString(),
                                contentBuilder = java.lang.StringBuilder(placeholderText),
                                isUser = true,
                                attachmentBitmap = thumbnail,
                                attachmentType = if (thumbnail != null) AttachmentType.IMAGE else AttachmentType.NONE
                            )
                            messages.add(msg)
                            messageAdapter.notifyItemInserted(messages.size - 1)
                            messagesRv.scrollToPosition(messages.size - 1)
                        }
                    } else {
                        hasPendingImage = false
                        withContext(Dispatchers.Main) {
                            Toast.makeText(this@MainActivity, "Nepodařilo se dekódovat obrázek", Toast.LENGTH_SHORT).show()
                        }
                    }
                } catch (e: Exception) {
                    hasPendingImage = false
                    withContext(Dispatchers.Main) {
                        Toast.makeText(this@MainActivity, "Chyba Image: ${e.message}", Toast.LENGTH_SHORT).show()
                    }
                }
            }
        }

        // Zpracování videa (extrakce klíčového snímku pro LLM, plný stream v plánu)
        fileManager.onVideoAttachedCallback = { videoUri ->
            Log.i(TAG, "Video attached: $videoUri")
            lifecycleScope.launch(Dispatchers.IO) {
                try {
                    val retriever = android.media.MediaMetadataRetriever()
                    retriever.setDataSource(this@MainActivity, videoUri)
                    // Vytáhnu první reprezentativní snímek z 1. sekundy
                    val frameBitmap = retriever.getFrameAtTime(1000000, android.media.MediaMetadataRetriever.OPTION_CLOSEST_SYNC)
                    retriever.release()

                    if (frameBitmap != null) {
                        val compressed = com.example.llama.agent.vision.VisionSourceManager().processAndCompressImageBlocking(frameBitmap, recycleSource = true)
                        if (compressed != null) {
                            var modelNotLoaded = false
                            var missingProjector = false
                            val processed = com.example.llama.agent.core.EngineOrchestrator.withForegroundLock(
                                context = applicationContext,
                                owner = com.example.llama.agent.core.EngineOwner.FOREGROUND_CHAT,
                                config = null
                            ) { orchEngine ->
                                if (!orchEngine.state.value.isModelLoaded) {
                                    modelNotLoaded = true
                                    false
                                } else if (!orchEngine.hasProjector() && !settingsManager.isRemoteInferenceActive) {
                                    missingProjector = true
                                    false
                                } else {
                                    orchEngine.processImage(compressed)
                                }
                            }
                            if (!processed) {
                                hasPendingImage = false
                                withContext(Dispatchers.Main) {
                                    if (modelNotLoaded) {
                                        Toast.makeText(this@MainActivity, "Model není načten", Toast.LENGTH_SHORT).show()
                                    } else {
                                        Toast.makeText(
                                            this@MainActivity,
                                            if (missingProjector) {
                                                "Model nemá načtený multimodální projektor (.mmproj) nebo nepodporuje video/obrazový vstup."
                                            } else {
                                                "Zpracování video snímku selhalo."
                                            },
                                            Toast.LENGTH_LONG
                                        ).show()
                                    }
                                }
                                return@launch
                            }
                            hasPendingImage = true // Signalizace pro handleUserInput() (Video Bypass)

                            // ============ VIDEO THUMBNAIL (podmíněné) ============
                            val thumbnail = if (settingsManager.isMediaPreviewEnabled) {
                                try {
                                    // Nový retriever pro thumbnail (originální bitmap byl recyklován uvnitř VisionSourceMngr)
                                    val thumbRetriever = android.media.MediaMetadataRetriever()
                                    thumbRetriever.setDataSource(this@MainActivity, videoUri)
                                    val thumb = thumbRetriever.getFrameAtTime(1000000, android.media.MediaMetadataRetriever.OPTION_CLOSEST_SYNC)
                                    thumbRetriever.release()
                                    // Downscale pro šetření RAM
                                    if (thumb != null) {
                                        val scaled = android.graphics.Bitmap.createScaledBitmap(thumb, 300, 200, true)
                                        if (scaled !== thumb) thumb.recycle()
                                        scaled
                                    } else null
                                } catch (e: Exception) {
                                    Log.w(TAG, "Video thumbnail failed (non-critical)", e)
                                    null
                                }
                            } else null

                            withContext(Dispatchers.Main) {
                                val msg = Message(
                                    id = java.util.UUID.randomUUID().toString(),
                                    contentBuilder = java.lang.StringBuilder("[🎬 Video frame přijat — zeptej se 'Analyzuj video']"),
                                    isUser = true,
                                    attachmentBitmap = thumbnail,
                                    attachmentType = if (thumbnail != null) AttachmentType.VIDEO else AttachmentType.NONE
                                )
                                messages.add(msg)
                                messageAdapter.notifyItemInserted(messages.size - 1)
                                messagesRv.scrollToPosition(messages.size - 1)
                            }
                        } else {
                            hasPendingImage = false
                        }
                    } else {
                        hasPendingImage = false
                        withContext(Dispatchers.Main) {
                            Toast.makeText(this@MainActivity, "Nelze získat náhled videa", Toast.LENGTH_SHORT).show()
                        }
                    }
                } catch (e: Exception) {
                    hasPendingImage = false
                    Log.e(TAG, "Video extraction failed: ${e.message}", e)
                    withContext(Dispatchers.Main) {
                        Toast.makeText(this@MainActivity, "Chyba u videa: ${e.message}", Toast.LENGTH_SHORT).show()
                    }
                }
            }
        }

        // Zpracování audia — mtmd automaticky detekuje formát přes magic bytes (wav/mp3/flac)
        fileManager.onAudioAttachedCallback = { audioBytes ->
            lifecycleScope.launch(Dispatchers.IO) {
                try {
                    var modelNotLoaded = false
                    var missingProjector = false
                    val success = com.example.llama.agent.core.EngineOrchestrator.withForegroundLock(
                        context = applicationContext,
                        owner = com.example.llama.agent.core.EngineOwner.FOREGROUND_CHAT,
                        config = null
                    ) { orchEngine ->
                        if (!orchEngine.state.value.isModelLoaded) {
                            modelNotLoaded = true
                            false
                        } else if (!orchEngine.hasProjector() && !settingsManager.isRemoteInferenceActive) {
                            missingProjector = true
                            false
                        } else {
                            // mtmd_helper_bitmap_init_from_buf rozpozná audio přes magic bytes
                            // a vytvoří audio mel spectrogram automaticky
                            orchEngine.processImage(audioBytes)
                        }
                    }
                    if (success) {
                        hasPendingImage = true // Signalizace pro handleUserInput() (Audio Bypass)
                    } else {
                        hasPendingImage = false
                    }
                    withContext(Dispatchers.Main) {
                        if (modelNotLoaded) {
                            Toast.makeText(this@MainActivity, "Model není načten", Toast.LENGTH_SHORT).show()
                        } else if (success) {
                            val msg = Message(
                                id = java.util.UUID.randomUUID().toString(),
                                contentBuilder = java.lang.StringBuilder("[🎵 Audio přijato — zeptej se 'Co je řečeno v nahrávce?']"),
                                isUser = true,
                                attachmentType = AttachmentType.AUDIO
                                // Audio nemá bitmap thumbnail — AttachmentType.AUDIO zapne audio formát bubliny
                            )
                            messages.add(msg)
                            messageAdapter.notifyItemInserted(messages.size - 1)
                            messagesRv.scrollToPosition(messages.size - 1)
                        } else {
                            Toast.makeText(
                                this@MainActivity,
                                if (missingProjector) {
                                    "Model nemá načtený multimodální projektor (.mmproj) pro audio vstup."
                                } else {
                                    "Audio processing selhal \u2014 model nemusí podporovat audio vstup."
                                },
                                Toast.LENGTH_LONG
                            ).show()
                        }
                    }
                } catch (e: Exception) {
                    hasPendingImage = false
                    Log.e(TAG, "Audio processing failed", e)
                    withContext(Dispatchers.Main) {
                        Toast.makeText(this@MainActivity,
                            "Chyba Audio: ${e.message}", Toast.LENGTH_SHORT).show()
                    }
                }
            }
        }
    }

    // --- SERVICE STATE HANDLER OPTIMIZED ---
    private fun handleGenerationState(state: GenerationState) {
        when (state) {
            is GenerationState.Idle -> {
                updateSendButtonState()
            }
            is GenerationState.Standby -> {
                messages.lastOrNull { !it.isUser }?.let {
                    messageAdapter.setMessageStreaming(it.id, false)
                }
                updateSendButtonState()
            }
            is GenerationState.Generating -> {
                // 1. APPEND MODE (Fastest)
                if (state.reasoningToken.isNotEmpty()) {
                    messageAdapter.appendReasoningToLastMessage(state.reasoningToken)
                }
                if (state.token.isNotEmpty()) {
                    messageAdapter.appendTokenToLastMessage(state.token)
                }

                // 2. DESYNC CHECK (Independent content + reasoning drift + tool-step reset)
                val lastMsg = messages.lastOrNull { !it.isUser }
                if (lastMsg != null && !lastMsg.isThoughtsCollapsedByUser && state.partialReasoning.isNotBlank()) {
                    lastMsg.isThoughtsExpanded = true
                }
                val contentClearedForTool = lastMsg != null && lastMsg.content.isNotEmpty() && state.partialResponse.isEmpty()
                val contentDrift = lastMsg != null && abs(lastMsg.content.length - state.partialResponse.length) > 10
                val reasoningDrift = lastMsg != null && abs(lastMsg.reasoning.length - state.partialReasoning.length) > 10

                if (contentClearedForTool || contentDrift || reasoningDrift) {
                    Log.w(
                            TAG,
                            "Sync drift detected: Local(c=${lastMsg?.content?.length}, r=${lastMsg?.reasoning?.length}), Service(c=${state.partialResponse.length}, r=${state.partialReasoning.length}). Forcing update."
                    )
                    messageAdapter.updateLastMessageState(state.partialResponse, state.partialReasoning)
                }

                if (!settingsManager.disableAutoScroll) {
                    // Smart Autoscroll: Scroll only if near bottom
                    if (!messagesRv.canScrollVertically(1) && messages.isNotEmpty()) {
                        messagesRv.post {
                            val count = messageAdapter.itemCount
                            if (count > 0) messagesRv.smoothScrollToPosition(count - 1)
                        }
                    }
                }

                // Update Stats
                if (settingsManager.showStats) {
                    speedTv.text = getString(R.string.info_speed_format, state.tokensPerSecond)
                }
                if (settingsManager.showTokenCount) {
                    tokenCountTv.text = getString(R.string.stats_tokens_format, state.tokenCount)
                }
                if (settingsManager.showGenTime) {
                    genTimeTv.text = getString(R.string.stats_time_format, state.elapsedTime)
                }
                updateSendButtonState()
            }
            is GenerationState.Complete -> {
                if (state === lastHandledCompleteState) return
                lastHandledCompleteState = state
                // Final hard sync to ensure markdown processing, search reports & preserved reasoning
                val lastAssistant = messages.lastOrNull { !it.isUser }
                if (lastAssistant != null) {
                    messageAdapter.setMessageStreaming(lastAssistant.id, false)
                    if (state.searchReports.isNotEmpty() && lastAssistant.searchReports.isEmpty()) {
                        lastAssistant.searchReports.addAll(state.searchReports)
                    }
                    val hasVisibleThoughts = state.fullReasoning.isNotBlank() &&
                        (settingsManager.showThoughtInProgress || state.fullText.isBlank() || settingsManager.thinkingProfile != ThinkingProfile.OFF)
                    if (hasVisibleThoughts && !lastAssistant.isThoughtsCollapsedByUser) {
                        lastAssistant.isThoughtsExpanded = true
                    } else if (lastAssistant.isThoughtsCollapsedByUser || state.fullReasoning.isBlank()) {
                        lastAssistant.isThoughtsExpanded = false
                    }
                }
                messageAdapter.updateLastMessageState(state.fullText, state.fullReasoning)
                if (lastAssistant != null) {
                    val idx = messages.indexOf(lastAssistant)
                    if (idx != -1) messageAdapter.notifyItemChanged(idx)
                }
                sessionManager.saveCurrentState()
                updateSendButtonState()
            }
            is GenerationState.Error -> {
                messages.lastOrNull { !it.isUser }?.let { lastAssistant ->
                    if (!lastAssistant.content.contains(state.message)) {
                        messageAdapter.appendTokenToLastMessage("\n⚠️ ${state.message}")
                    }
                    messageAdapter.setMessageStreaming(lastAssistant.id, false)
                }
                sessionManager.saveCurrentState()
                Toast.makeText(this, "Generation Error: ${state.message}", Toast.LENGTH_LONG).show()
                updateSendButtonState()
            }
        }
    }

    private fun setupVoiceObserver() {
        lifecycleScope.launch {
            voiceManager.state.collect { state ->
                val speakingText =
                        if (state is com.example.llama.domain.voice.VoiceState.Speaking) state.text
                        else null
                messageAdapter.updateSpeakingState(speakingText)

                if (state is com.example.llama.domain.voice.VoiceState.Error) {
                    val msg = state.message
                    if (state.isCritical) {
                        Toast.makeText(this@MainActivity, msg, Toast.LENGTH_LONG).show()
                    } else {
                        Log.w("VoiceManager", msg)
                    }
                }
            }
        }
    }

    private fun setupVoiceInputObserver() {
        lifecycleScope.launch {
            voiceInputManager.state.collect { state ->
                when (state) {
                    is com.example.llama.agent.voice.AgentVoiceSTT.STTState.Listening -> {
                        userInputEt.hint = "Poslouchám..."
                    }
                    is com.example.llama.agent.voice.AgentVoiceSTT.STTState.Result -> {
                        val currentText = userInputEt.text.toString()
                        val newText =
                                if (currentText.isEmpty()) state.text
                                else "$currentText ${state.text}"
                        userInputEt.setText(newText)
                        userInputEt.setSelection(userInputEt.text.length)
                        userInputEt.hint = null
                    }
                    is com.example.llama.agent.voice.AgentVoiceSTT.STTState.Error -> {
                        Toast.makeText(this@MainActivity, state.message, Toast.LENGTH_SHORT).show()
                        userInputEt.hint = null
                    }
                    else -> {
                        if (userInputEt.hint == "Poslouchám...") userInputEt.hint = null
                    }
                }
            }
        }
    }

    private fun handleModelInfoClick() {
        if (settingsManager.isRemoteInferenceActive) {
            val provider = settingsManager.aiProviderType
            val modelName = settingsManager.getSelectedModelForProvider(provider).ifBlank { "nevybrán" }
            val endpointUrl = settingsManager.getActiveRemoteEndpointUrl(provider)
            val apiKey = settingsManager.getActiveRemoteApiKey(provider)
            val adapter = com.example.llama.agent.remote.RemoteAdapterFactory.getAdapter(provider)
            val declaredCaps = adapter.describeModelCapabilities(modelName, endpointUrl, apiKey)
            val nativeCaps = runCatching {
                if (::engine.isInitialized) engine.getModelCapabilities() else null
            }.getOrNull()
            val detail = com.example.llama.data.ModelDetailState(
                name = modelName,
                filePath = endpointUrl.ifBlank { settingsManager.getActiveRemoteModelUri(provider) },
                contextSize = (declaredCaps.contextLength ?: nativeCaps?.contextLength ?: 0).toLong(),
                modelSize = provider.displayName,
                parameters = buildString {
                    append("Tools=${nativeCaps?.supportsTools ?: declaredCaps.supportsNativeTools}")
                    append(" | Vision=${declaredCaps.supportsVision ?: "auto"}")
                    append(" | Effort=${declaredCaps.effortMechanism.name} (${declaredCaps.source.name})")
                },
                quantization = buildString {
                    append("MaxOut=${declaredCaps.maxOutputTokens ?: "Auto"}")
                    if (declaredCaps.supportedEfforts.isNotEmpty()) {
                        append(" | ${declaredCaps.supportedEfforts.joinToString("/")}")
                    }
                },
                chatTemplate = buildString {
                    appendLine("Provider: ${provider.displayName}")
                    appendLine("Adapter: ${adapter.handlerId}")
                    appendLine("Endpoint: ${endpointUrl.ifBlank { "default" }}")
                    appendLine("Capabilities Source: ${declaredCaps.source.name}")
                    appendLine("Thinking Toggle: ${declaredCaps.supportsThinkingToggle}, Effort: ${declaredCaps.supportsReasoningEffort}")
                    appendLine("Native Tools: ${declaredCaps.supportsNativeTools}, Vision: ${declaredCaps.supportsVision ?: "auto"}")
                }.trim(),
                architecture = "${provider.shortName} (${adapter.handlerId})"
            )
            val tag = "ModelInfoDialog"
            if (supportFragmentManager.findFragmentByTag(tag) == null) {
                ModelInfoDialogFragment.newInstance(detail).show(supportFragmentManager, tag)
            }
            return
        }
        if (currentModelPath == null) {
            Toast.makeText(this, "No model loaded", Toast.LENGTH_SHORT).show()
            return
        }
        val currentModels = viewModel.savedModels.value
        val model = currentModels.find { it.sourceUri == currentModelPath }
        if (model != null) {
            viewModel.showModelInfo(model.id)
        } else {
            Toast.makeText(this, "Model info not available in DB", Toast.LENGTH_SHORT).show()
        }
    }

    private fun setupRobustOnBackPressed() {
        onBackPressedDispatcher.addCallback(this) {
            if (drawerLayout.isDrawerOpen(GravityCompat.START)) {
                drawerLayout.closeDrawer(GravityCompat.START)
            } else if (::engine.isInitialized && engine.state.value.isUninterruptible) {
                Toast.makeText(this@MainActivity, R.string.msg_wait_operation, Toast.LENGTH_SHORT)
                        .show()
            } else {
                Log.w(TAG, "Back press ignored to prevent accidental exit.")
            }
        }
    }

    private fun setupEngineStateObserver() {
        lifecycleScope.launch {
            if (!::engine.isInitialized) return@launch
            engine.state.collect { state ->
                val isModelLoaded = state.isModelLoaded
                userInputEt.isEnabled = isModelLoaded
                if (isModelLoaded && settingsManager.isRemoteInferenceActive) {
                    currentModelPath = settingsManager.getActiveRemoteModelUri()
                    currentModelArchitecture = when (settingsManager.aiProviderType) {
                        SettingsManager.AiProviderType.GOOGLE_GEMINI -> "gemini"
                        SettingsManager.AiProviderType.REMOTE_DESKTOP -> "remote_desktop"
                        else -> "cloud"
                    }
                    currentModelChatTemplate = null
                    currentProjectorPath = null
                    ggufTv.text = buildRemoteStatusText()
                    userInputEt.hint = null
                }
                updateSendButtonState()

                if (state is InferenceEngine.State.Error) {
                    val errorMsg = getString(R.string.status_error_prefix, state.exception.message)
                    ggufTv.text = errorMsg
                    Toast.makeText(this@MainActivity, errorMsg, Toast.LENGTH_LONG).show()
                }
            }
        }
    }

    private fun setupViewModelObservers() {
        lifecycleScope.launch {
            viewModel.modelStatus.collect { status ->
                if (status.startsWith("Error") || status.startsWith("Importing")) {
                    if (!status.startsWith("Info Error")) {
                        ggufTv.text = status
                    } else {
                        Toast.makeText(this@MainActivity, status, Toast.LENGTH_SHORT).show()
                    }
                }
            }
        }

        lifecycleScope.launch {
            viewModel.isLoading.collect { loading ->
                if (loading) {
                    userInputEt.hint = getString(R.string.status_importing)
                    userInputEt.isEnabled = false
                } else {
                    userInputEt.hint = null
                    if (::engine.isInitialized && engine.state.value.isModelLoaded) {
                        userInputEt.isEnabled = true
                    }
                    updateSendButtonState()
                }
            }
        }

        lifecycleScope.launch {
            viewModel.modelDetail.collect { detail ->
                val tag = "model_info_dialog"
                if (supportFragmentManager.findFragmentByTag(tag) == null) {
                    ModelInfoDialogFragment.newInstance(detail).show(supportFragmentManager, tag)
                }
            }
        }

        lifecycleScope.launch {
            viewModel.savedModels.collect { models ->
                Log.d(TAG, "Database updated. Available models: ${models.size}")
            }
        }

        lifecycleScope.launch {
            viewModel.isAgentEnabled.collect { isEnabled ->
                try {
                    // Basic UI Update - Safe Call
                    updateToolbarAgentIcon(isEnabled)

                    if (isFirstAgentEmit) {
                        isFirstAgentEmit = false
                        // Initial state sync (silent)
                    } else {
                        // User Action / State Change
                        val msg = if (isEnabled) "Agent zapnut" else "Agent vypnut"
                        Toast.makeText(this@MainActivity, msg, Toast.LENGTH_SHORT).show()
                        applyDisplaySettings() // Update bottom bar mic visibility

                        // Lazy Load / Unload Agent Modules
                        if (isEnabled) {
                            // INTEGRACE: Auto-start Open Mic if enabled
                            if (settingsManager.agentVoiceMode == SettingsManager.VOICEMODE_OPEN_MIC
                            ) {
                                // Check permission again just in case (though we checked before
                                // toggle)
                                if (permissionManager.hasAudioPermission()) {
                                    // voiceInputManager.startContinuousListening()
                                    Log.i(
                                            "MainActivity",
                                            "Agent enabled - delegating Open Mic to Service"
                                    )
                                    Toast.makeText(
                                                    this@MainActivity,
                                                    "Open Mic aktivován (Service)",
                                                    Toast.LENGTH_SHORT
                                            )
                                            .show()
                                }
                            }

                            // INTEGRACE: Immediate Permission Check on Enable
                            checkStartupPermissions()
                        } else {
                            if (::voiceInputManager.isInitialized) {
                                voiceInputManager.shutdown()
                            }
                        }

                        // Notify Service to refresh its state/notification
                        if (::serviceManager.isInitialized && serviceManager.isServiceConnected()) {
                            try {
                                val intent =
                                        Intent(
                                                        this@MainActivity,
                                                        LlamaGenerationService::class.java
                                                )
                                                .apply {
                                                    action =
                                                            LlamaNotificationHelper
                                                                    .ACTION_START_RESIDENT
                                                }
                                startService(intent)
                            } catch (e: Exception) {
                                Log.e(TAG, "Failed to start service intent", e)
                            }
                        }
                    }
                } catch (e: Exception) {
                    Log.e(TAG, "CRITICAL: Error in Agent State Observer", e)
                    // Don't crash app, just log
                }
            }
        }

        lifecycleScope.launch {
            com.example.llama.agent.perception.PerceptionEventBus.events.collect { event ->
                if (event is com.example.llama.agent.perception.UiActionEvent && event.action == com.example.llama.agent.perception.UiAction.TOGGLE_FULLSCREEN_BOXES) {
                    val newState = !settingsManager.isSmartVisionBoundingBoxEnabled
                    settingsManager.isSmartVisionBoundingBoxEnabled = newState
                    updateToolbarSmartVisionIcon(newState)
                    val msg = if (newState) "SmartVision ZAPNUT" else "SmartVision VYPNUT"
                    Toast.makeText(this@MainActivity, msg, Toast.LENGTH_SHORT).show()
                }
            }
        }
    }

    private var isReturningFromSettings = false

    private fun buildBehaviorSignature(): String {
        return listOf(
            settingsManager.runtimeBehaviorMode.name,
            settingsManager.desktopLanBehaviorMode.name,
            settingsManager.cloudApiBehaviorMode.name,
            settingsManager.includeDateTimeAnchor.toString(),
            settingsManager.devExecutionProfileOverride.id,
            settingsManager.thinkingProfile.name,
            settingsManager.toolsMode.name,
            settingsManager.searchIntentMode.name,
            settingsManager.queryFidelityGuardMode.id,
            settingsManager.contextSize.toString(),
            settingsManager.remoteDesktopNumCtx.toString(),
            settingsManager.remoteMaxOutputTokens.toString(),
            settingsManager.devCustomContextSizeTokens.toString(),
            settingsManager.noPerf.toString(),
            settingsManager.timingPerToken.toString(),
            settingsManager.gpuLayers.toString(),
            settingsManager.llamaBackend.id,
            settingsManager.devHexagonVaLimitMb.toString(),
            settingsManager.devHexagonNdev.toString(),
            settingsManager.devKvReplayOnCtxResizeMode.id
        ).joinToString("|")
    }

    private fun buildCurrentEngineConfig(modelPath: String? = currentModelPath): com.example.llama.agent.core.EngineConfiguration? {
        val path = modelPath?.takeIf { it.isNotBlank() } ?: return null
        if (path.startsWith("cloud://") || path.startsWith("remote_desktop://")) return null
        return com.example.llama.agent.core.EngineConfiguration(
            modelPath = path,
            projectorPath = currentProjectorPath,
            nCtx = settingsManager.contextSize,
            useMmap = settingsManager.useMmap,
            nBatch = settingsManager.batchSize,
            nUBatch = settingsManager.ubatchSize,
            flashAttn = settingsManager.flashAttention,
            kvCacheMode = settingsManager.kvCacheMode,
            llamaBackend = settingsManager.llamaBackend,
            strictBackend = settingsManager.strictBackend,
            gpuLayers = settingsManager.gpuLayers,
            hexagonVaLimitMb = settingsManager.devHexagonVaLimitMb,
            hexagonNdev = settingsManager.devHexagonNdev,
            kvReplayOnCtxResizeMode = settingsManager.devKvReplayOnCtxResizeMode.nativeCode
        )
    }

    override fun onResume() {
        super.onResume()
        syncCloudApiModeUiOnResume()
        findViewById<com.google.android.material.button.MaterialButton>(R.id.btn_quick_backend)?.let { btn ->
            updateQuickBackendButton(btn, settingsManager.llamaBackend)
        }
        findViewById<com.google.android.material.button.MaterialButton>(R.id.btn_runtime_mode)?.let { btn ->
            updateRuntimeModeButton(btn)
        }
        updateToolsButtonLabel()
        updateSkillsButtonLabel()
        refreshModelCapabilitiesAndUi()
        applyDisplaySettings()
        val currentBehaviorSig = buildBehaviorSignature()
        if (lastObservedBehaviorSignature != null && lastObservedBehaviorSignature != currentBehaviorSig) {
            if (settingsManager.refreshSystemPromptOnModeSwitch) {
                isSessionSystemPromptSet = false
            }
        }
        lastObservedBehaviorSignature = currentBehaviorSig
        if (lastAppliedSystemMessage != null && lastAppliedSystemMessage != settingsManager.systemMessage) {
            Log.i(TAG, "🔄 User System Prompt changed in Settings. Marking session system prompt for refresh.")
            isSessionSystemPromptSet = false
            if (messages.isNotEmpty() && messages[0].content.startsWith("SYSTEM:")) {
                messages[0].content = "SYSTEM: ${settingsManager.systemMessage}"
                messageAdapter.notifyItemChanged(0)
            }
        }
        settingsManager.syncToWebSearchEngineConfig()
        if (!settingsManager.isRemoteInferenceActive &&
            ::engine.isInitialized &&
            engine.state.value.isModelLoaded &&
            generationJob?.isActive != true &&
            serviceManager.generationState.value !is GenerationState.Generating
        ) {
            val activeCfg = com.example.llama.agent.core.EngineOrchestrator.activeConfig
            val desiredBase = buildCurrentEngineConfig()
            if (activeCfg != null && desiredBase != null) {
                val contextSettingChanged = lastAppliedContextSizeSetting != null &&
                    lastAppliedContextSizeSetting != settingsManager.contextSize
                val loadParamsChanged = activeCfg.kvCacheMode != desiredBase.kvCacheMode ||
                    activeCfg.flashAttn != desiredBase.flashAttn ||
                    activeCfg.useMmap != desiredBase.useMmap ||
                    activeCfg.nBatch != desiredBase.nBatch ||
                    activeCfg.nUBatch != desiredBase.nUBatch ||
                    activeCfg.gpuLayers != desiredBase.gpuLayers ||
                    activeCfg.strictBackend != desiredBase.strictBackend ||
                    activeCfg.llamaBackend != desiredBase.llamaBackend ||
                    activeCfg.hexagonVaLimitMb != desiredBase.hexagonVaLimitMb ||
                    activeCfg.hexagonNdev != desiredBase.hexagonNdev ||
                    activeCfg.kvReplayOnCtxResizeMode != desiredBase.kvReplayOnCtxResizeMode
                val explicitCtxChanged = settingsManager.contextSize > 0 && activeCfg.nCtx != settingsManager.contextSize
                if (contextSettingChanged || explicitCtxChanged || loadParamsChanged) {
                    lastAppliedContextSizeSetting = settingsManager.contextSize
                    val targetCfg = if (!contextSettingChanged && settingsManager.contextSize == 0 && activeCfg.nCtx > 0) {
                        desiredBase.copy(nCtx = activeCfg.nCtx)
                    } else {
                        desiredBase
                    }
                    lifecycleScope.launch(Dispatchers.IO) {
                        try {
                            com.example.llama.agent.core.EngineOrchestrator.withForegroundLock(
                                context = applicationContext,
                                owner = com.example.llama.agent.core.EngineOwner.FOREGROUND_CHAT,
                                config = targetCfg
                            ) { _ -> }
                            if (!com.example.llama.agent.core.EngineOrchestrator.lastResizePreservedKv) {
                                isSessionSystemPromptSet = false
                            }
                            withContext(Dispatchers.Main) {
                                refreshModelCapabilitiesAndUi()
                            }
                        } catch (e: Exception) {
                            Log.e(TAG, "onResume engine reconfiguration failed: ${e.message}", e)
                        }
                    }
                }
            }
        }
        // Synchronize settings on resume (keep lastAppliedSystemMessage for diff check in buildEffectivePromptToSend)
        if (lastAppliedSystemMessage == null) {
            lastAppliedSystemMessage = settingsManager.systemMessage
        }
        messageAdapter.refreshMarkwon()
        updateSendButtonState()

        if (!settingsManager.alwaysShowSidebar && drawerLayout.isDrawerOpen(GravityCompat.START)) {
            drawerLayout.closeDrawer(GravityCompat.START)
        }

        // CRITICAL FIX (2026-02-17):
        // Re-evaluate background/foreground nastavení po návratu ze Settings.
        // Dříve toto maskovalo HyperCallService (InCallService) s vysokou OOM_ADJ prioritou.
        // Po jejím odstranění je NUTNÉ explicitně (re)startovat foreground službu.
        val shouldBeResident = settingsManager.backgroundGeneration
        if (shouldBeResident) {
            // Zajistit, že služba běží jako Foreground (pokud ještě neběží)
            val intent =
                    android.content.Intent(
                                    this,
                                    com.example.llama.service.LlamaGenerationService::class.java
                            )
                            .apply {
                                action =
                                        com.example.llama.service.LlamaNotificationHelper
                                                .ACTION_START_RESIDENT
                            }
            if (android.os.Build.VERSION.SDK_INT >= android.os.Build.VERSION_CODES.O) {
                startForegroundService(intent)
            } else {
                startService(intent)
            }
        } else {
            // Background OFF → zastavit foreground režim (služba zůstane pro binding, ale bez
            // priority)
            try {
                val stopIntent =
                        android.content.Intent(
                                        this,
                                        com.example.llama.service.LlamaGenerationService::class.java
                                )
                                .apply {
                                    action =
                                            com.example.llama.service.LlamaNotificationHelper
                                                    .ACTION_STOP_RESIDENT
                                }
                startService(stopIntent)
            } catch (e: Exception) {
                // Služba nemusí běžet — ignorovat
            }
        }

        // INTEGRACE: Startup Checks (Accessibility & Battery)
        // checkStartupPermissions() // DISABLED per user request
    }

    private fun checkStartupPermissions() {
        // 1. Check Accessibility Service
        val isServiceEnabled =
                com.example.llama.agent.core.AgentHelper.isAccessibilityServiceEnabled(this)

        if (viewModel.isAgentEnabled.value) {
            if (!isServiceEnabled) {
                speakAndOpenSettings(
                        "Potřebuji povolení pro usnadnění přístupu, otevírám nastavení.",
                        com.example.llama.agent.core.AgentHelper.getAccessibilitySettingsIntent()
                )
                return // Prioritize one at a time
            }
        }
    }

    private var lastSpokenTime = 0L
    private fun speakAndOpenSettings(text: String, intent: Intent) {
        // Debounce to prevent loops
        val now = System.currentTimeMillis()
        if (now - lastSpokenTime < 3000) return
        lastSpokenTime = now

        Toast.makeText(this, text, Toast.LENGTH_LONG).show()

        if (settingsManager.isTtsEnabled) {
            voiceManager.speak(text)
        }

        // Delay intent slightly to let TTS start
        lifecycleScope.launch {
            kotlinx.coroutines.delay(2000)
            try {
                isReturningFromSettings = true // SET FLAG
                intent.addFlags(Intent.FLAG_ACTIVITY_NEW_TASK)
                startActivity(intent)
            } catch (e: Exception) {
                Log.e(TAG, "Failed to open settings", e)
            }
        }
    }

    private fun setupToolbar() {
        toolbar.setNavigationOnClickListener { drawerLayout.openDrawer(GravityCompat.START) }
        toolbar.inflateMenu(R.menu.toolbar_menu)
        // Sync initial state of Agent Icon
        updateToolbarAgentIcon(viewModel.isAgentEnabled.value)
        updateToolbarIrisIcon(settingsManager.isIrisMotionEnabled)
        updateToolbarSmartVisionIcon(settingsManager.isSmartVisionBoundingBoxEnabled)
        updateToolbarWorkersState()

        val btnRuntimeMode: com.google.android.material.button.MaterialButton? = findViewById(R.id.btn_runtime_mode)
        btnRuntimeMode?.let { btn ->
            updateRuntimeModeButton(btn)
            btn.setOnClickListener {
                if (settingsManager.isRemoteInferenceActive) {
                    val provider = settingsManager.aiProviderType
                    val nextRemoteMode = if (settingsManager.getEffectiveRemoteBehaviorMode(provider) ==
                        SettingsManager.RemoteBehaviorMode.AUTO
                    ) {
                        SettingsManager.RemoteBehaviorMode.NATIVE
                    } else {
                        SettingsManager.RemoteBehaviorMode.AUTO
                    }
                    if (provider == SettingsManager.AiProviderType.REMOTE_DESKTOP) {
                        settingsManager.desktopLanBehaviorMode = nextRemoteMode
                    } else {
                        settingsManager.cloudApiBehaviorMode = nextRemoteMode
                    }
                    if (settingsManager.refreshSystemPromptOnModeSwitch) {
                        isSessionSystemPromptSet = false
                    }
                    lastObservedBehaviorSignature = buildBehaviorSignature()
                    updateRuntimeModeButton(btn)
                    applyToolsModeToEngine(settingsManager.toolsMode)
                    refreshModelCapabilitiesAndUi()
                    Toast.makeText(this, "${provider.shortName}: ${nextRemoteMode.displayName}", Toast.LENGTH_SHORT).show()
                } else {
                    val nextMode = when (settingsManager.runtimeBehaviorMode) {
                        SettingsManager.RuntimeBehaviorMode.AUTO_BY_ENGINE -> SettingsManager.RuntimeBehaviorMode.NATIVE_FIRST
                        SettingsManager.RuntimeBehaviorMode.NATIVE_FIRST -> SettingsManager.RuntimeBehaviorMode.ASSISTED_LEGACY
                        SettingsManager.RuntimeBehaviorMode.ASSISTED_LEGACY -> SettingsManager.RuntimeBehaviorMode.AUTO_BY_ENGINE
                    }
                    settingsManager.runtimeBehaviorMode = nextMode
                    if (settingsManager.refreshSystemPromptOnModeSwitch) {
                        isSessionSystemPromptSet = false
                    }
                    lastObservedBehaviorSignature = buildBehaviorSignature()
                    updateRuntimeModeButton(btn)
                    applyToolsModeToEngine(settingsManager.toolsMode)
                    refreshModelCapabilitiesAndUi()
                    Toast.makeText(this, "Režim chování: ${nextMode.displayName}", Toast.LENGTH_SHORT).show()
                }
            }
            btn.setOnLongClickListener {
                startActivity(Intent(this, SettingsActivity::class.java))
                true
            }
        }

        val btnQuickBackend: com.google.android.material.button.MaterialButton? = findViewById(R.id.btn_quick_backend)
        btnQuickBackend?.let { btn ->
            updateQuickBackendButton(btn, settingsManager.llamaBackend)

            btn.setOnClickListener {
                if (settingsManager.isRemoteInferenceActive) {
                    showCloudQuickSwitchDialog(btn)
                    return@setOnClickListener
                }
                val current = settingsManager.llamaBackend
                val all = SettingsManager.LlamaBackend.entries
                val nextIdx = (all.indexOf(current) + 1) % all.size
                val next = all[nextIdx]
                settingsManager.llamaBackend = next
                updateQuickBackendButton(btn, next)
                Toast.makeText(this, "GGML Backend: ${next.displayName}", Toast.LENGTH_SHORT).show()

                // Pokud je načten model, přenačteme jej s novým backendem
                currentModelPath?.let { modelPath ->
                    lifecycleScope.launch {
                        loadModelToEngine(modelPath)
                    }
                }
            }

            btn.setOnLongClickListener {
                val intent = Intent(this, SettingsActivity::class.java).apply {
                    putExtra("open_tab", 5) // Tab Advanced (Pokročilé)
                }
                startActivity(intent)
                true
            }
        }

        toolbar.setOnMenuItemClickListener { item ->
            when (item.itemId) {
                R.id.action_agent -> {
                    try {
                        // Prevent crash: Check dependencies before enabling
                        if (viewModel.isAgentEnabled.value) {
                            viewModel.toggleAgentMode() // Turning OFF - safe
                        } else {
                            // Turning ON - Check Runtime Permissions (Audio, Call, SMS...)
                            permissionManager.checkAndRequestAgentPermissions { granted ->
                                try {
                                    if (granted) {
                                        // 2. Check System Settings Permission (Write Settings) -
                                        // Category C
                                        if (!permissionManager.hasWriteSettingsPermission()) {
                                            permissionManager.requestWriteSettingsPermission()
                                            // Non-blocking, but user needs to grant it for
                                            // brightness/rotation
                                        }

                                        // 3. Check Accessibility Service - Category A/General
                                        val isServiceEnabled =
                                                com.example.llama.agent.core.AgentHelper
                                                        .isAccessibilityServiceEnabled(
                                                                this@MainActivity
                                                        )
                                        if (!isServiceEnabled) {
                                            Toast.makeText(
                                                            this@MainActivity,
                                                            "Pro plnou funkčnost povolte Agenta v Usnadnění (Accessibility).",
                                                            Toast.LENGTH_LONG
                                                    )
                                                    .show()
                                            // Optional: Open Accessibility Settings directly?
                                            // startActivity(com.example.llama.agent.core.AgentHelper.getAccessibilitySettingsIntent())
                                        }

                                        viewModel.toggleAgentMode()
                                    } else {
                                        Toast.makeText(
                                                        this@MainActivity,
                                                        "Agent vyžaduje oprávnění (Mikrofon, Kontakty) pro plnou funkčnost.",
                                                        Toast.LENGTH_SHORT
                                                )
                                                .show()
                                    }
                                } catch (e: Exception) {
                                    Log.e("MainActivity", "Error in permission callback", e)
                                    Toast.makeText(
                                                    this@MainActivity,
                                                    "Chyba při inicializaci agenta: ${e.message}",
                                                    Toast.LENGTH_SHORT
                                            )
                                            .show()
                                }
                            }
                        }
                    } catch (e: Exception) {
                        Log.e("MainActivity", "Error processing agent toggle", e)
                        Toast.makeText(this@MainActivity, "Chyba: ${e.message}", Toast.LENGTH_SHORT).show()
                    }
                    true
                }
                R.id.action_smart_vision -> {
                    toggleSmartVision()
                    true
                }
                R.id.action_iris_motion -> {
                    toggleIrisMotion()
                    true
                }
                R.id.action_load_model -> {
                    showModelSelector()
                    true
                }
                R.id.action_eject_model -> {
                    handleEjectModel()
                    true
                }
                R.id.action_attach_file -> {
                    if (::engine.isInitialized && engine.state.value.isModelLoaded) {
                        fileManager.pickMultiModalAsset()
                    } else {
                        Toast.makeText(this, R.string.user_input_hint_no_model, Toast.LENGTH_SHORT)
                                .show()
                    }
                    true
                }
                R.id.action_export -> {
                    showExportSelector()
                    true
                }
                R.id.action_delete -> {
                    sessionManager.confirmDeleteCurrentSession()
                    true
                }
                R.id.action_toggle_vision -> {
                    settingsManager.isPerceptionEnabled = !settingsManager.isPerceptionEnabled
                    updateToolbarWorkersState()
                    true
                }
                R.id.action_toggle_audio -> {
                    settingsManager.isAudioPerceptionEnabled = !settingsManager.isAudioPerceptionEnabled
                    updateToolbarWorkersState()
                    true
                }
                R.id.action_toggle_ocr -> {
                    settingsManager.isOcrEnabled = !settingsManager.isOcrEnabled
                    updateToolbarWorkersState()
                    true
                }
                R.id.action_toggle_summarization -> {
                    settingsManager.isSummarizationEnabled = !settingsManager.isSummarizationEnabled
                    updateToolbarWorkersState()
                    true
                }
                R.id.action_toggle_transcript -> {
                    settingsManager.isTranscriptEnabled = !settingsManager.isTranscriptEnabled
                    updateToolbarWorkersState()
                    true
                }
                else -> false
            }
        }
    }

    private fun updateToolbarWorkersState() {
        val menu = toolbar.menu ?: return
        val visionItem = menu.findItem(R.id.action_toggle_vision)
        val audioItem = menu.findItem(R.id.action_toggle_audio)
        val ocrItem = menu.findItem(R.id.action_toggle_ocr)
        val sumItem = menu.findItem(R.id.action_toggle_summarization)
        val transItem = menu.findItem(R.id.action_toggle_transcript)

        visionItem?.title = if (settingsManager.isPerceptionEnabled) "Vision: ON" else "Vision: OFF"
        audioItem?.title = if (settingsManager.isAudioPerceptionEnabled) "Audio: ON" else "Audio: OFF"
        ocrItem?.title = if (settingsManager.isOcrEnabled) "OCR: ON" else "OCR: OFF"
        sumItem?.title = if (settingsManager.isSummarizationEnabled) "Sum: ON" else "Sum: OFF"
        transItem?.title = if (settingsManager.isTranscriptEnabled) "Titulky: ON" else "Titulky: OFF"
    }

    private fun updateToolbarAgentIcon(isEnabled: Boolean) {
        try {
            val menu = toolbar.menu
            val item = menu.findItem(R.id.action_agent)
            if (item != null) {
                if (isEnabled) {
                    item.setIcon(R.drawable.ic_smart_toy_24)
                    item.title = "Agent (ON)"
                    // Visual indication: Full Opacity or Tint
                    item.icon?.alpha = 255
                    val typedValue = android.util.TypedValue()
                    theme.resolveAttribute(
                            com.google.android.material.R.attr.colorPrimary,
                            typedValue,
                            true
                    )
                    item.icon?.setTint(typedValue.data)
                } else {
                    item.setIcon(R.drawable.ic_smart_toy_24)
                    item.title = "Agent (OFF)"
                    // Visual indication: Dimmed or Gray
                    item.icon?.alpha = 120
                    item.icon?.setTintList(null) // Remove tint to show original/gray
                }
            }
        } catch (e: Exception) {
            Log.e(TAG, "Error updating toolbar icon", e)
        }
    }

    private fun toggleIrisMotion() {
        val newState = !settingsManager.isIrisMotionEnabled

        if (newState) {
            permissionManager.checkAndRequestCameraPermission { granted ->
                if (granted) {
                    settingsManager.isIrisMotionEnabled = true
                    updateToolbarIrisIcon(true)
                    Toast.makeText(this, "IrisMotion ZAPNUT", Toast.LENGTH_SHORT).show()
                } else {
                    Toast.makeText(this, "Oprávnění kamery zamítnuto", Toast.LENGTH_SHORT).show()
                }
            }
        } else {
            settingsManager.isIrisMotionEnabled = false
            updateToolbarIrisIcon(false)
            Toast.makeText(this, "IrisMotion VYPNUT", Toast.LENGTH_SHORT).show()
        }
    }

    private fun toggleSmartVision() {
        val newState = !settingsManager.isSmartVisionBoundingBoxEnabled

        if (newState) {
            // Check pro Accessibility Service / SYSTEM_ALERT_WINDOW by mohl být tu
            settingsManager.isSmartVisionBoundingBoxEnabled = true
            updateToolbarSmartVisionIcon(true)
            Toast.makeText(this, "SmartVision ZAPNUT", Toast.LENGTH_SHORT).show()
        } else {
            settingsManager.isSmartVisionBoundingBoxEnabled = false
            updateToolbarSmartVisionIcon(false)
            Toast.makeText(this, "SmartVision VYPNUT", Toast.LENGTH_SHORT).show()
        }
    }

    private fun updateToolbarSmartVisionIcon(isEnabled: Boolean) {
        try {
            val menu = toolbar.menu
            val item = menu.findItem(R.id.action_smart_vision)
            if (item != null) {
                if (isEnabled) {
                    item.title = "SmartVision (ON)"
                    item.icon?.alpha = 255
                    val typedValue = android.util.TypedValue()
                    theme.resolveAttribute(
                            com.google.android.material.R.attr.colorPrimary,
                            typedValue,
                            true
                    )
                    item.icon?.setTint(typedValue.data)
                } else {
                    item.title = "SmartVision (OFF)"
                    item.icon?.alpha = 120
                    item.icon?.setTintList(null)
                }
            }
        } catch (e: Exception) {
            Log.e(TAG, "Error updating SmartVision icon", e)
        }
    }


    private fun updateToolbarIrisIcon(isEnabled: Boolean) {
        try {
            val menu = toolbar.menu
            val item = menu.findItem(R.id.action_iris_motion)
            if (item != null) {
                if (isEnabled) {
                    item.title = "Iris (ON)"
                    item.icon?.alpha = 255
                    val typedValue = android.util.TypedValue()
                    theme.resolveAttribute(
                            com.google.android.material.R.attr.colorPrimary,
                            typedValue,
                            true
                    )
                    item.icon?.setTint(typedValue.data)
                } else {
                    item.title = "Iris (OFF)"
                    item.icon?.alpha = 120
                    item.icon?.setTintList(null)
                }
            }
        } catch (e: Exception) {
            Log.e(TAG, "Error updating iris toolbar icon", e)
        }
    }

    private fun showModelSelector() {
        val models = viewModel.savedModels.value

        DialogManager.showModelSelector(
                context = this,
                models = models,
                onModelSelect = { selectedModel ->
                    if (selectedModel.displayName.contains("mmproj", ignoreCase = true)) {
                        Toast.makeText(this, "Toto je Vision projektor (\uD83D\uDC41\uFE0F mmproj).\nNelze ho načíst jako hlavní textový model. Zvolte ho v Nastavení -> SmartVision.", Toast.LENGTH_LONG).show()
                    } else {
                        handleLoadLocalModel(selectedModel)
                    }
                },
                onImportClick = { fileManager.pickModel { uri -> handleSelectedModelImport(uri) } },
                onDeleteClick = { showDeleteModelConfirm(models) },
                onCloudApiSelect = {
                    syncCloudApiModeUiOnResume()
                    Toast.makeText(this, "☁️ Aktivován ${settingsManager.getActiveRemoteDisplayName()}", Toast.LENGTH_SHORT).show()
                }
        )
    }

    private fun showDeleteModelConfirm(models: List<AiModel>) {
        DialogManager.showDeleteModelConfirm(
                context = this,
                models = models,
                onConfirmDelete = { modelToDelete ->
                    viewModel.deleteModel(modelToDelete)
                    Toast.makeText(this, R.string.msg_model_deleted, Toast.LENGTH_SHORT).show()
                }
        )
    }

    private fun showExportSelector() {
        if (messages.isEmpty()) {
            Toast.makeText(this, "Chat is empty", Toast.LENGTH_SHORT).show()
            return
        }

        DialogManager.showExportFormatSelector(this) { format -> exportChat(format) }
    }

    private fun exportChat(format: ChatHistoryManager.ExportFormat) {
        lifecycleScope.launch {
            fileManager.exportAndShare(
                    historyManager,
                    sessionManager.currentSessionId,
                    messages,
                    format
            )
        }
    }

    private fun handleLoadLocalModel(model: AiModel) {
        if (!::engine.isInitialized) {
            Toast.makeText(this, "Engine is initializing, please wait...", Toast.LENGTH_SHORT)
                    .show()
            return
        }
        ggufTv.text = getString(R.string.status_loading_named_model, model.displayName)
        viewModel.loadLocalModel(
                modelId = model.id,
                onPathReady = { nativePath ->
                    lifecycleScope.launch { loadModelToEngine(nativePath) }
                },
                onError = { error ->
                    Toast.makeText(
                                    this,
                                    getString(R.string.status_error_prefix, error),
                                    Toast.LENGTH_LONG
                            )
                            .show()
                    ggufTv.text = getString(R.string.status_error_prefix, error)
                }
        )
    }

    private suspend fun loadModelToEngine(procPath: String) =
            withContext(Dispatchers.IO) {
                if (settingsManager.aiProviderType != SettingsManager.AiProviderType.LOCAL_OFFLINE) {
                    settingsManager.aiProviderType = SettingsManager.AiProviderType.LOCAL_OFFLINE
                    com.example.llama.agent.core.EngineProvider.syncWithSettings(applicationContext)
                }
                currentModelPath = procPath

                withContext(Dispatchers.Main) {
                    userInputEt.hint = getString(R.string.info_loading_model)
                    ggufTv.text = "Reading metadata..."
                }
                Log.i(TAG, "Loading model from: $procPath")

                val meta = ModelMetadataManager.readMetadata(procPath)
                ModelMetadataManager.displayMetadata(meta, ggufTv)
                currentModelArchitecture = meta?.architecture?.architecture
                currentModelChatTemplate = meta?.tokenizer?.chatTemplate

                // ===================== MULTIMODAL: Database / Settings Managed mmproj =====================
                var discoveredProjectorPath: String? = null
                val visionModelId = settingsManager.activeVisionModelId

                if (settingsManager.isNativeVisionModel) {
                    Log.i(TAG, "\uD83D\uDD0D All-In-One Vision Mode: Using main model path as projector ($procPath)")
                    discoveredProjectorPath = procPath
                } else if (visionModelId.isNotEmpty()) {
                    try {
                        val db = com.example.llama.data.LlamaDatabase.getDatabase(applicationContext)
                        val visionModel = db.aiModelDao().getModelById(visionModelId)
                        if (visionModel != null && java.io.File(visionModel.sourceUri).exists()) {
                            discoveredProjectorPath = visionModel.sourceUri
                            Log.i(TAG, "\uD83D\uDD0D Database-selected vision projector: $discoveredProjectorPath")
                        } else {
                            Log.w(TAG, "\u26A0\uFE0F Selected vision model not found on disk or DB")
                        }
                    } catch (e: Exception) {
                        Log.e(TAG, "Failed to load vision model from DB", e)
                    }
                }

                if (discoveredProjectorPath == null && meta != null && ModelMetadataManager.isMultimodalModel(meta)) {
                    Log.i(TAG, "🔍 Metadata auto-detected Multimodal/Omni architecture (${meta.architecture?.architecture}): Using main model as projector ($procPath)")
                    discoveredProjectorPath = procPath
                }

                if (discoveredProjectorPath == null) {
                    Log.i(TAG, "\u2139\uFE0F No mmproj selected or found \u2192 text-only mode")
                }

                val useMmap = settingsManager.useMmap
                val kvCacheMode = settingsManager.kvCacheMode
                val nCtx = settingsManager.contextSize
                val nBatch = settingsManager.batchSize
                val nUBatch = settingsManager.ubatchSize
                val flashAttn = settingsManager.flashAttention
                val llamaBackend = settingsManager.llamaBackend
                val strictBackend = settingsManager.strictBackend
                val gpuLayers = settingsManager.gpuLayers
                currentProjectorPath = discoveredProjectorPath
                lastAppliedContextSizeSetting = nCtx
                try {
                    if (!::engine.isInitialized) {
                        throw IllegalStateException("Engine not initialized")
                    }
                    val config = com.example.llama.agent.core.EngineConfiguration(
                        modelPath = procPath,
                        projectorPath = discoveredProjectorPath,
                        nCtx = nCtx,
                        useMmap = useMmap,
                        nBatch = nBatch,
                        nUBatch = nUBatch,
                        flashAttn = flashAttn,
                        kvCacheMode = kvCacheMode,
                        llamaBackend = llamaBackend,
                        strictBackend = strictBackend,
                        gpuLayers = gpuLayers,
                        hexagonVaLimitMb = settingsManager.devHexagonVaLimitMb,
                        hexagonNdev = settingsManager.devHexagonNdev,
                        kvReplayOnCtxResizeMode = settingsManager.devKvReplayOnCtxResizeMode.nativeCode
                    )
                    com.example.llama.agent.core.EngineOrchestrator.withForegroundLock(
                        context = applicationContext,
                        owner = com.example.llama.agent.core.EngineOwner.FOREGROUND_CHAT,
                        config = config
                    ) { orchestratorEngine ->
                        orchestratorEngine.resetContext()
                    }
                    isSessionSystemPromptSet = false
                    withContext(Dispatchers.Main) {
                        userInputEt.hint = null
                        findViewById<com.google.android.material.button.MaterialButton>(R.id.btn_runtime_mode)
                            ?.let { updateRuntimeModeButton(it) }
                        val btnQuickBackend: com.google.android.material.button.MaterialButton? = findViewById(R.id.btn_quick_backend)
                        btnQuickBackend?.let { updateQuickBackendButton(it, settingsManager.llamaBackend) }
                        refreshModelCapabilitiesAndUi()
                        updateSendButtonState()
                        if (messages.isEmpty()) sessionManager.startNewSession(skipEngineReset = true)
                        val isMultimodal = ::engine.isInitialized && engine.hasProjector()
                        val successMsg = if (isMultimodal) {
                            getString(R.string.msg_load_success) + " \uD83D\uDC41\uFE0F Multimodal"
                        } else {
                            getString(R.string.msg_load_success)
                        }
                        Toast.makeText(
                                        this@MainActivity,
                                        successMsg,
                                        Toast.LENGTH_SHORT
                                )
                                .show()
                    }
                } catch (e: Exception) {
                    Log.e(TAG, "CRITICAL: Engine Load Failed", e)
                    withContext(Dispatchers.Main) {
                        userInputEt.hint = null
                        val isBackendError = e is com.arm.aichat.BackendInitializationException
                        val displayMsg = if (isBackendError) {
                            "❌ Selhala inicializace backendu (${settingsManager.llamaBackend.displayName}): ${e.message}"
                        } else {
                            getString(R.string.status_error_prefix, e.message)
                        }
                        Toast.makeText(this@MainActivity, displayMsg, Toast.LENGTH_LONG).show()
                        if (!ggufTv.text.contains("Metadata") && !ggufTv.text.contains("GGUF")) {
                            ggufTv.text = displayMsg
                        }
                    }
                }
            }

    private fun resolveLoadedModelTrainContextLength(): Int {
        return if (::engine.isInitialized) {
            runCatching { engine.getModelCapabilities().contextLength }.getOrDefault(0)
        } else {
            0
        }
    }

    private fun buildRemoteStatusText(): String {
        val budget = settingsManager.getEffectiveBudgetProfile(
            modelTrainContextLength = resolveLoadedModelTrainContextLength()
        )
        val transportTag = if (settingsManager.aiProviderType == SettingsManager.AiProviderType.REMOTE_DESKTOP) {
            "Desktop ${settingsManager.remoteDesktopTransportMode}"
        } else {
            "Online API (0 MB RAM)"
        }
        val base = "${settingsManager.getActiveRemoteDisplayName()} • $transportTag"
        val ctxBadge = formatContextStatusBadge()
        val withCtx = if (ctxBadge != null) "$base • $ctxBadge" else base
        return if (budget.isOverridden) {
            val tag = when (budget.overrideMode) {
                com.example.llama.agent.remote.ExecutionProfileOverride.FORCE_MOBILE_GGUF ->
                    "DEV Simulace n_ctx=${budget.simulatedContextSizeTokens}/GGUF"
                com.example.llama.agent.remote.ExecutionProfileOverride.CUSTOM_DEV_CLAMP -> {
                    val ctxStr = if (budget.simulatedContextSizeTokens > 0) "${budget.simulatedContextSizeTokens}" else "auto"
                    "DEV Ořez (ctx=$ctxStr)"
                }
                else -> "DEV: ${budget.effectiveTier.id}"
            }
            "$withCtx • ⚙️ $tag"
        } else {
            withCtx
        }
    }

    private fun updateRuntimeModeButton(btn: com.google.android.material.button.MaterialButton) {
        if (settingsManager.isRemoteInferenceActive) {
            val (label, colorHex) = when (settingsManager.getEffectiveRemoteBehaviorMode()) {
                SettingsManager.RemoteBehaviorMode.NATIVE -> "Native" to "#00E676"
                SettingsManager.RemoteBehaviorMode.AUTO -> "Auto" to "#00E5FF"
            }
            btn.text = label
            val colorInt = android.graphics.Color.parseColor(colorHex)
            btn.setTextColor(colorInt)
            btn.strokeColor = android.content.res.ColorStateList.valueOf(colorInt)
            return
        }
        val mode = settingsManager.runtimeBehaviorMode
        val (label, colorHex) = when (mode) {
            SettingsManager.RuntimeBehaviorMode.AUTO_BY_ENGINE -> "Auto" to "#80D8FF"
            SettingsManager.RuntimeBehaviorMode.NATIVE_FIRST -> "Nová" to "#00E676"
            SettingsManager.RuntimeBehaviorMode.ASSISTED_LEGACY -> "Stará" to "#FFB74D"
        }
        btn.text = label
        val colorInt = android.graphics.Color.parseColor(colorHex)
        btn.setTextColor(colorInt)
        btn.strokeColor = android.content.res.ColorStateList.valueOf(colorInt)
    }

    private fun updateQuickBackendButton(
        btn: com.google.android.material.button.MaterialButton,
        backend: SettingsManager.LlamaBackend
    ) {
        if (settingsManager.isRemoteInferenceActive) {
            btn.text = when (settingsManager.aiProviderType) {
                SettingsManager.AiProviderType.REMOTE_DESKTOP -> "🖥️ LAN"
                SettingsManager.AiProviderType.GOOGLE_GEMINI -> "☁️ Gemini"
                SettingsManager.AiProviderType.OPENAI_COMPATIBLE -> "☁️ API"
                SettingsManager.AiProviderType.LOCAL_OFFLINE -> "GGUF"
            }
            val cloudColor = android.graphics.Color.parseColor(
                if (settingsManager.aiProviderType == SettingsManager.AiProviderType.REMOTE_DESKTOP) "#FFB74D" else "#00E5FF"
            )
            btn.setTextColor(cloudColor)
            btn.strokeColor = android.content.res.ColorStateList.valueOf(cloudColor)
            return
        }
        btn.text = backend.shortName
        val colorHex = when (backend) {
            SettingsManager.LlamaBackend.CPU -> "#00E676"     // Neon Green
            SettingsManager.LlamaBackend.OPENCL -> "#29B6F6"  // Cyan / Blue
            SettingsManager.LlamaBackend.HEXAGON -> "#AB47BC" // Purple
            SettingsManager.LlamaBackend.HTP_V75 -> "#FF7043" // Deep Orange
            SettingsManager.LlamaBackend.HTP_V73 -> "#FFA726" // Amber
            SettingsManager.LlamaBackend.VULKAN -> "#FF5252"  // Vulkan Red
        }
        val colorInt = android.graphics.Color.parseColor(colorHex)
        btn.setTextColor(colorInt)
        btn.strokeColor = android.content.res.ColorStateList.valueOf(colorInt)
    }

    private fun syncCloudApiModeUiOnResume() {
        val hybrid = com.example.llama.agent.core.EngineProvider.getHybrid(applicationContext)
        hybrid.syncWithSettings()
        findViewById<com.google.android.material.button.MaterialButton>(R.id.btn_runtime_mode)
            ?.let { updateRuntimeModeButton(it) }
        val btnQuickBackend: com.google.android.material.button.MaterialButton? = findViewById(R.id.btn_quick_backend)
        btnQuickBackend?.let { updateQuickBackendButton(it, settingsManager.llamaBackend) }

        if (settingsManager.isRemoteInferenceActive) {
            val remoteUri = settingsManager.getActiveRemoteModelUri()
            val expectedRemoteArch = when (settingsManager.aiProviderType) {
                SettingsManager.AiProviderType.GOOGLE_GEMINI -> "gemini"
                SettingsManager.AiProviderType.REMOTE_DESKTOP -> "remote_desktop"
                else -> "cloud"
            }
            val switchedEngine = currentModelPath != remoteUri ||
                currentModelArchitecture != expectedRemoteArch ||
                currentModelChatTemplate != null
            if (switchedEngine) {
                currentModelPath = remoteUri
                currentProjectorPath = null
                lastAppliedContextSizeSetting = null
                currentModelArchitecture = expectedRemoteArch
                currentModelChatTemplate = null
                isSessionSystemPromptSet = false
                if (messages.isNotEmpty() && hybrid.remoteEngine.getConversationHistorySnapshot().isEmpty()) {
                    hybrid.restoreRemoteHistoryFromUiMessages(messages)
                }
            }
            ggufTv.text = buildRemoteStatusText()
            userInputEt.isEnabled = true
            userInputEt.hint = null
            refreshModelCapabilitiesAndUi()
            updateSendButtonState()
        } else if (currentModelPath?.startsWith("cloud://") == true || currentModelPath?.startsWith("remote_desktop://") == true) {
            currentModelPath = null
            currentProjectorPath = null
            lastAppliedContextSizeSetting = null
            currentModelArchitecture = null
            currentModelChatTemplate = null
            isSessionSystemPromptSet = false
            ggufTv.text = getString(R.string.user_input_hint_no_model)
            userInputEt.isEnabled = ::engine.isInitialized && engine.state.value.isModelLoaded
            refreshModelCapabilitiesAndUi()
            updateSendButtonState()
        }
    }

    private fun showCloudQuickSwitchDialog(btn: com.google.android.material.button.MaterialButton) {
        val provider = settingsManager.aiProviderType
        val cached = settingsManager.getCachedRemoteModels(provider)
        if (cached.isEmpty()) {
            Toast.makeText(this, "⏳ Načítám dostupné modely pro ${provider.shortName}...", Toast.LENGTH_SHORT).show()
            lifecycleScope.launch(Dispatchers.IO) {
                val hybrid = com.example.llama.agent.core.EngineProvider.getHybrid(applicationContext)
                val res = hybrid.remoteEngine.fetchAvailableModels(provider)
                withContext(Dispatchers.Main) {
                    res.onSuccess { fetched ->
                        settingsManager.setCachedRemoteModels(provider, fetched)
                        if (fetched.isNotEmpty() && settingsManager.getSelectedModelForProvider(provider).isBlank()) {
                            val firstModel = fetched.first().first
                            if (provider == SettingsManager.AiProviderType.REMOTE_DESKTOP) {
                                settingsManager.remoteDesktopModelName = firstModel
                            } else {
                                settingsManager.cloudModelName = firstModel
                            }
                        }
                        syncCloudApiModeUiOnResume()
                        presentCloudQuickSwitchDialogWithModels(btn, provider, fetched)
                    }.onFailure { err ->
                        Toast.makeText(
                            this@MainActivity,
                            "❌ Nelze načíst modely: ${err.localizedMessage}",
                            Toast.LENGTH_LONG
                        ).show()
                        presentCloudQuickSwitchDialogWithModels(btn, provider, emptyList())
                    }
                }
            }
        } else {
            presentCloudQuickSwitchDialogWithModels(btn, provider, cached)
        }
    }

    private fun presentCloudQuickSwitchDialogWithModels(
        btn: com.google.android.material.button.MaterialButton,
        provider: SettingsManager.AiProviderType,
        models: List<Pair<String, String>>
    ) {
        val selectedId = settingsManager.getSelectedModelForProvider(provider)
        val icon = if (provider == SettingsManager.AiProviderType.REMOTE_DESKTOP) "🖥️ " else "☁️ "
        val modelItems = models.map { (id, label) ->
            val check = if (selectedId == id) "✅ " else icon
            "$check$label"
        }
        val items = (modelItems + listOf(
            "🔄 Obnovit seznam dostupných modelů",
            "📱 Přepnout zpět na Offline GGUF (Lokální NPU/GPU/CPU)",
            "⚙️ Otevřít nastavení poskytovatele a API klíčů..."
        )).toTypedArray()

        val title = if (provider == SettingsManager.AiProviderType.REMOTE_DESKTOP) {
            "🖥️ Desktop modely (${settingsManager.remoteDesktopBaseUrl})"
        } else {
            "☁️ Dostupné modely (${provider.shortName})"
        }

        com.google.android.material.dialog.MaterialAlertDialogBuilder(this)
            .setTitle(title)
            .setItems(items) { _, which ->
                when {
                    which < models.size -> {
                        val chosenModel = models[which].first
                        if (provider == SettingsManager.AiProviderType.REMOTE_DESKTOP) {
                            settingsManager.remoteDesktopModelName = chosenModel
                        } else {
                            settingsManager.cloudModelName = chosenModel
                        }
                        syncCloudApiModeUiOnResume()
                        Toast.makeText(this, "$icon Aktivní model: $chosenModel", Toast.LENGTH_SHORT).show()
                    }
                    which == models.size -> {
                        settingsManager.setCachedRemoteModels(provider, emptyList())
                        showCloudQuickSwitchDialog(btn)
                    }
                    which == models.size + 1 -> {
                        settingsManager.aiProviderType = SettingsManager.AiProviderType.LOCAL_OFFLINE
                        syncCloudApiModeUiOnResume()
                        updateQuickBackendButton(btn, settingsManager.llamaBackend)
                        Toast.makeText(this, "📱 Přepnuto na Offline GGUF režim", Toast.LENGTH_SHORT).show()
                    }
                    else -> {
                        startActivity(Intent(this, SettingsActivity::class.java))
                    }
                }
            }
            .setNegativeButton(android.R.string.cancel, null)
            .show()
    }

    private fun setupInputListeners() {
        userInputEt.addTextChangedListener(
                object : TextWatcher {
                    override fun beforeTextChanged(
                            s: CharSequence?,
                            start: Int,
                            count: Int,
                            after: Int
                    ) {}
                    override fun onTextChanged(
                            s: CharSequence?,
                            start: Int,
                            before: Int,
                            count: Int
                    ) {
                        updateSendButtonState()
                    }
                    override fun afterTextChanged(s: Editable?) {}
                }
        )
    }

    private fun resolveActiveModelController(): com.example.llama.agent.adapter.ModelAgentController? {
        val modelLoaded = if (::engine.isInitialized) engine.state.value.isModelLoaded else false
        if (!settingsManager.isRemoteInferenceActive && !modelLoaded && currentModelPath.isNullOrBlank()) {
            return null
        }
        val caps = if (::engine.isInitialized && (settingsManager.isRemoteInferenceActive || modelLoaded)) {
            runCatching { engine.getModelCapabilities() }.getOrNull()
        } else {
            null
        }
        return com.example.llama.agent.core.AgentToolStepHelper.resolveAdapter(
            settingsManager = settingsManager,
            capabilities = caps,
            architecture = currentModelArchitecture,
            modelPath = currentModelPath,
            chatTemplate = currentModelChatTemplate
        ) as? com.example.llama.agent.adapter.ModelAgentController
    }

    private fun resolveActiveThinkingSupport(): com.example.llama.agent.adapter.ModelThinkingSupport {
        val controller = resolveActiveModelController()
        if (controller != null) {
            return controller.thinkingSupport
        }
        return com.example.llama.agent.adapter.ModelThinkingSupport(
            mechanism = com.example.llama.agent.adapter.EffortMechanism.NONE,
            supportedProfiles = listOf(ThinkingProfile.OEM),
            supportedEfforts = emptySet(),
            supportsThinkingOff = false
        )
    }

    private fun formatContextStatusBadge(): String? {
        if (!::engine.isInitialized) return null
        val caps = runCatching { engine.getModelCapabilities() }.getOrNull()
        val activeNCtx = com.example.llama.agent.core.EngineOrchestrator.activeConfig?.nCtx ?: 0
        val allocatedCtx = when {
            activeNCtx > 0 -> activeNCtx
            !settingsManager.isRemoteInferenceActive && settingsManager.contextSize > 0 -> settingsManager.contextSize
            !settingsManager.isRemoteInferenceActive && engine.state.value.isModelLoaded -> 4096
            else -> 0
        }
        val trainCtx = caps?.contextLength ?: 0
        return when {
            allocatedCtx > 0 && trainCtx > 0 -> "n_ctx=$allocatedCtx / max=$trainCtx"
            allocatedCtx > 0 -> "n_ctx=$allocatedCtx"
            trainCtx > 0 -> "max_ctx=$trainCtx"
            else -> null
        }
    }

    private fun updateContextInfoInGgufTv() {
        if (!::ggufTv.isInitialized || !::engine.isInitialized) return
        if (settingsManager.isRemoteInferenceActive) {
            ggufTv.text = buildRemoteStatusText()
            return
        }
        if (!engine.state.value.isModelLoaded) return
        val badge = formatContextStatusBadge() ?: return
        val raw = ggufTv.text?.toString().orEmpty()
        val cleaned = raw
            .replace(Regex("""\s*\|\s*n_ctx=[^\n]+$"""), "")
            .replace(Regex("""\s*\|\s*max_ctx=[^\n]+$"""), "")
        if (cleaned.isNotBlank() && !cleaned.startsWith("Error")) {
            ggufTv.text = "$cleaned | $badge"
        }
    }

    private fun refreshModelCapabilitiesAndUi() {
        val controller = resolveActiveModelController()
        if (controller != null) {
            MessageAdapter.updateActiveFamilyCoTRules(controller.cotRules)
        }
        val support = controller?.thinkingSupport ?: resolveActiveThinkingSupport()
        val effective = support.resolveEffectiveProfile(settingsManager.thinkingProfile)
        if (::messageAdapter.isInitialized) {
            messageAdapter.thinkingProfile = effective
        }
        if (::engine.isInitialized && (settingsManager.isRemoteInferenceActive || engine.state.value.isModelLoaded)) {
            val isGeneratingNow = generationJob?.isActive == true ||
                serviceManager.generationState.value is GenerationState.Generating
            if (!isGeneratingNow && controller != null) {
                val toolsJson = com.example.llama.agent.tools.web.WebToolsDispatcher.getToolsJson()
                val toolsEnabled = settingsManager.toolsMode != SettingsManager.ToolsMode.OFF && toolsJson.isNotBlank()
                val policyOverride = if (settingsManager.toolsMode == SettingsManager.ToolsMode.OFF) 2 else 0
                controller.configureEngine(engine, toolsEnabled, toolsJson, toolPolicyOverride = policyOverride)
            }
            engine.configureThinking(effective.isEnabled, effective.maxTokens, effective.effort)
            updateContextInfoInGgufTv()
        }
        syncThinkingProfileToggleUi(effective, support)
    }

    private fun syncThinkingProfileToggleUi(
        profile: ThinkingProfile,
        support: com.example.llama.agent.adapter.ModelThinkingSupport = resolveActiveThinkingSupport()
    ) {
        if (!::toggleThinkingProfile.isInitialized) return
        val btnOem = findViewById<com.google.android.material.button.MaterialButton>(R.id.btn_think_oem)
        val btnOff = findViewById<com.google.android.material.button.MaterialButton>(R.id.btn_think_off)
        val btnLow = findViewById<com.google.android.material.button.MaterialButton>(R.id.btn_think_low)
        val btnMed = findViewById<com.google.android.material.button.MaterialButton>(R.id.btn_think_med)
        val btnDeep = findViewById<com.google.android.material.button.MaterialButton>(R.id.btn_think_deep)

        val supported = support.supportedProfiles
        val effective = support.resolveEffectiveProfile(profile)

        val enableOem = true
        val enableOff = ThinkingProfile.OFF in supported
        val enableLow = !support.isBinarySwitchOnly &&
            (ThinkingProfile.LOW in supported || ThinkingProfile.MINIMAL in supported)
        val enableMed = !support.isBinarySwitchOnly && (ThinkingProfile.MEDIUM in supported)
        val enableDeep = ThinkingProfile.DEEP in supported ||
            ThinkingProfile.XHIGH in supported ||
            ThinkingProfile.MAX in supported

        fun applyBtnState(btn: com.google.android.material.button.MaterialButton?, enabled: Boolean, label: String? = null) {
            if (btn == null) return
            btn.isEnabled = enabled
            btn.alpha = if (enabled) 1.0f else 0.35f
            if (label != null) btn.text = label
        }

        applyBtnState(btnOem, enableOem, "OEM")
        applyBtnState(btnOff, enableOff, "Off")
        applyBtnState(
            btnLow,
            enableLow,
            if (effective == ThinkingProfile.MINIMAL || (ThinkingProfile.LOW !in supported && ThinkingProfile.MINIMAL in supported)) "Min" else "Low"
        )
        applyBtnState(btnMed, enableMed, "Med")
        applyBtnState(
            btnDeep,
            enableDeep,
            when {
                support.isBinarySwitchOnly -> "On"
                effective == ThinkingProfile.XHIGH -> "XHigh"
                effective == ThinkingProfile.MAX -> "Max"
                else -> "High"
            }
        )

        val checkId = when {
            !support.supportsAnyControl -> R.id.btn_think_oem
            effective == ThinkingProfile.OEM -> R.id.btn_think_oem
            effective == ThinkingProfile.OFF && enableOff -> R.id.btn_think_off
            (effective == ThinkingProfile.MINIMAL || effective == ThinkingProfile.LOW) && enableLow -> R.id.btn_think_low
            effective == ThinkingProfile.MEDIUM && enableMed -> R.id.btn_think_med
            (effective == ThinkingProfile.DEEP || effective == ThinkingProfile.XHIGH || effective == ThinkingProfile.MAX) && enableDeep ->
                R.id.btn_think_deep
            else -> R.id.btn_think_oem
        }
        if (toggleThinkingProfile.checkedButtonId != checkId) {
            toggleThinkingProfile.check(checkId)
        }
    }

    private fun applySelectedThinkingProfile(newProfile: ThinkingProfile) {
        val support = resolveActiveThinkingSupport()
        val effective = support.resolveEffectiveProfile(newProfile)
        val previousProfile = settingsManager.thinkingProfile
        settingsManager.thinkingProfile = effective
        if (::messageAdapter.isInitialized) {
            messageAdapter.thinkingProfile = effective
        }
        if (::engine.isInitialized) {
            engine.configureThinking(effective.isEnabled, effective.maxTokens, effective.effort)
        }
        syncThinkingProfileToggleUi(effective, support)
        Log.i(TAG, "Effort Level updated ($support): $previousProfile -> $effective (requested=$newProfile, enabled=${effective.isEnabled}, effort='${effective.effort}')")
    }

    private fun showEffortLevelPickerDialog() {
        val support = resolveActiveThinkingSupport()
        val profiles = support.supportedProfiles
        if (profiles.size <= 1) {
            Toast.makeText(
                this,
                "Aktivní model podporuje pouze výchozí chování OEM (nemá řízení hloubky uvažování).",
                Toast.LENGTH_SHORT
            ).show()
            return
        }
        val labels = profiles.map {
            when (it) {
                ThinkingProfile.OEM -> "OEM (tovární výchozí chování modelu)"
                ThinkingProfile.OFF -> "Off / none (vypnout uvažování)"
                ThinkingProfile.MINIMAL -> "Minimal (minimální úvaha)"
                ThinkingProfile.LOW -> "Low (krátká úvaha)"
                ThinkingProfile.MEDIUM -> "Medium (vyvážená hloubka)"
                ThinkingProfile.DEEP -> when (support.mechanism) {
                    com.example.llama.agent.adapter.EffortMechanism.NONE ->
                        "On / Zkusit (asistovaný CoT)"
                    com.example.llama.agent.adapter.EffortMechanism.ENABLE_THINKING_SWITCH,
                    com.example.llama.agent.adapter.EffortMechanism.TEMPLATE_DEFAULT_ONLY ->
                        "On / Zapnuto (binární přepínač)"
                    else -> "High (důkladný rozbor)"
                }
                ThinkingProfile.XHIGH -> "Extra High / xhigh (hluboká analýza)"
                ThinkingProfile.MAX -> "Max (maximální výpočetní hloubka)"
            }
        }.toTypedArray()
        val effective = support.resolveEffectiveProfile(settingsManager.thinkingProfile)
        val currentIdx = profiles.indexOf(effective).coerceAtLeast(0)
        val mechHint = when (support.mechanism) {
            com.example.llama.agent.adapter.EffortMechanism.ENABLE_THINKING_SWITCH,
            com.example.llama.agent.adapter.EffortMechanism.TEMPLATE_DEFAULT_ONLY -> "Binární přepínač"
            com.example.llama.agent.adapter.EffortMechanism.GEMINI_THINKING_LEVEL -> "Gemini Thinking Level"
            com.example.llama.agent.adapter.EffortMechanism.ANTHROPIC_OUTPUT_EFFORT -> "Claude Output Effort"
            com.example.llama.agent.adapter.EffortMechanism.NATIVE_REASONING_EFFORT -> "Reasoning Effort"
            com.example.llama.agent.adapter.EffortMechanism.NONE -> "OEM / Off / On (Zkusit CoT)"
        }
        androidx.appcompat.app.AlertDialog.Builder(this)
            .setTitle("Úroveň uvažování ($mechHint)")
            .setSingleChoiceItems(labels, currentIdx) { dialog, which ->
                applySelectedThinkingProfile(profiles[which])
                dialog.dismiss()
            }
            .setNegativeButton("Zrušit", null)
            .show()
    }

    private fun setupThinkingProfileToggle() {
        if (!::toggleThinkingProfile.isInitialized) return
        refreshModelCapabilitiesAndUi()

        toggleThinkingProfile.addOnButtonCheckedListener { _, checkedId, isChecked ->
            if (isChecked) {
                val support = resolveActiveThinkingSupport()
                val cur = support.resolveEffectiveProfile(settingsManager.thinkingProfile)
                val requested = when (checkedId) {
                    R.id.btn_think_oem -> ThinkingProfile.OEM
                    R.id.btn_think_off -> ThinkingProfile.OFF
                    R.id.btn_think_low ->
                        if (ThinkingProfile.LOW !in support.supportedProfiles && ThinkingProfile.MINIMAL in support.supportedProfiles) {
                            ThinkingProfile.MINIMAL
                        } else if (cur == ThinkingProfile.MINIMAL) {
                            ThinkingProfile.MINIMAL
                        } else {
                            ThinkingProfile.LOW
                        }
                    R.id.btn_think_med -> ThinkingProfile.MEDIUM
                    R.id.btn_think_deep -> when (cur) {
                        ThinkingProfile.XHIGH -> ThinkingProfile.XHIGH
                        ThinkingProfile.MAX -> ThinkingProfile.MAX
                        else -> ThinkingProfile.DEEP
                    }
                    else -> ThinkingProfile.OEM
                }
                val newProfile = support.resolveEffectiveProfile(requested)
                if (newProfile != settingsManager.thinkingProfile || newProfile != cur) {
                    applySelectedThinkingProfile(newProfile)
                }
            }
        }
        listOf(R.id.btn_think_oem, R.id.btn_think_off, R.id.btn_think_low, R.id.btn_think_med, R.id.btn_think_deep)
            .forEach { btnId ->
                findViewById<android.view.View>(btnId)?.setOnLongClickListener {
                    showEffortLevelPickerDialog()
                    true
                }
            }
    }

    private fun updateSkillsButtonLabel() {
        if (!::btnSkillsSelector.isInitialized) return
        val registry = SkillRegistry.getInstance(this)
        val activeId = registry.activeSkillId
        val label = when (activeId) {
            SkillRegistry.SKILL_AUTO -> "🧠 Skills: Auto"
            SkillRegistry.SKILL_NONE -> "🧠 Skills: Off"
            else -> {
                val skill = registry.getSkillById(activeId)
                val shortName = (skill?.name ?: activeId).take(10)
                "🧠 $shortName"
            }
        }
        btnSkillsSelector.text = label
    }

    private fun updateToolsButtonLabel() {
        if (!::btnToolsSelector.isInitialized) return
        when (settingsManager.toolsMode) {
            SettingsManager.ToolsMode.AUTO -> {
                val total = com.example.llama.agent.tools.web.WebToolsDispatcher.ALL_TOOL_ITEMS.size
                val active = settingsManager.enabledAgentTools.size
                val suffix = if (active < total) " ($active/$total)" else ""
                btnToolsSelector.text = "🛠️ Tools: Auto$suffix"
                val color = android.graphics.Color.parseColor("#FFA726")
                btnToolsSelector.setTextColor(color)
                btnToolsSelector.strokeColor = android.content.res.ColorStateList.valueOf(color)
            }
            SettingsManager.ToolsMode.ON -> {
                val total = com.example.llama.agent.tools.web.WebToolsDispatcher.ALL_TOOL_ITEMS.size
                val active = settingsManager.enabledAgentTools.size
                val suffix = if (active < total) " ($active/$total)" else ""
                btnToolsSelector.text = "🛠️ Tools: ON$suffix"
                val color = android.graphics.Color.parseColor("#00E676")
                btnToolsSelector.setTextColor(color)
                btnToolsSelector.strokeColor = android.content.res.ColorStateList.valueOf(color)
            }
            SettingsManager.ToolsMode.OFF -> {
                btnToolsSelector.text = "🛠️ Tools: Off"
                val color = android.graphics.Color.parseColor("#757575")
                btnToolsSelector.setTextColor(color)
                btnToolsSelector.strokeColor = android.content.res.ColorStateList.valueOf(color)
            }
        }
    }

    private fun showToolsSelectorDialog() {
        val searchIntentLabel = when (settingsManager.searchIntentMode) {
            SettingsManager.SearchIntentMode.AUTO -> "Auto"
            SettingsManager.SearchIntentMode.ON -> "Vždy ON"
            SettingsManager.SearchIntentMode.OFF -> "OFF"
        }
        val modes = arrayOf(
            "⚡ Auto (dle dotazu na internet / věda / fakta)",
            "🟢 Vždy zapnuto (ON - model má vždy web tools)",
            "🔴 Vypnuto (OFF - offline bez nástrojů)",
            "🎛️ Výběr jednotlivých nástrojů (Realtime kombinace)...",
            "🔍 Search Intent (aktuality pro starší modely): $searchIntentLabel...",
            "ℹ️ Informace o dostupných nástrojích..."
        )
        val currentIdx = when (settingsManager.toolsMode) {
            SettingsManager.ToolsMode.AUTO -> 0
            SettingsManager.ToolsMode.ON -> 1
            SettingsManager.ToolsMode.OFF -> 2
        }

        com.google.android.material.dialog.MaterialAlertDialogBuilder(this)
            .setTitle("🛠️ Autonomní Web Nástroje")
            .setSingleChoiceItems(modes, currentIdx) { dialog, which ->
                when (which) {
                    0 -> {
                        settingsManager.toolsMode = SettingsManager.ToolsMode.AUTO
                        updateToolsButtonLabel()
                        applyToolsModeToEngine(SettingsManager.ToolsMode.AUTO)
                        dialog.dismiss()
                    }
                    1 -> {
                        settingsManager.toolsMode = SettingsManager.ToolsMode.ON
                        updateToolsButtonLabel()
                        applyToolsModeToEngine(SettingsManager.ToolsMode.ON)
                        dialog.dismiss()
                    }
                    2 -> {
                        settingsManager.toolsMode = SettingsManager.ToolsMode.OFF
                        updateToolsButtonLabel()
                        applyToolsModeToEngine(SettingsManager.ToolsMode.OFF)
                        dialog.dismiss()
                    }
                    3 -> {
                        dialog.dismiss()
                        showIndividualToolsToggleDialog()
                    }
                    4 -> {
                        dialog.dismiss()
                        showSearchIntentModeDialog()
                    }
                    5 -> {
                        dialog.dismiss()
                        showToolsHelpDialog()
                    }
                }
            }
            .setNegativeButton(android.R.string.cancel, null)
            .show()
    }

    private fun showSearchIntentModeDialog() {
        val options = arrayOf(
            "⚡ Auto (podle schopností nástrojů a dotazu)",
            "🟢 Vždy aktivní (ON – doplnit web pro aktuální fakta)",
            "🔴 Vypnuto (OFF – spoléhat jen na volání modelu)"
        )
        val currentIdx = when (settingsManager.searchIntentMode) {
            SettingsManager.SearchIntentMode.AUTO -> 0
            SettingsManager.SearchIntentMode.ON -> 1
            SettingsManager.SearchIntentMode.OFF -> 2
        }
        com.google.android.material.dialog.MaterialAlertDialogBuilder(this)
            .setTitle("🔍 Chytré vyhledávání aktualit (Search Intent)")
            .setSingleChoiceItems(options, currentIdx) { dialog, which ->
                val chosen = when (which) {
                    1 -> SettingsManager.SearchIntentMode.ON
                    2 -> SettingsManager.SearchIntentMode.OFF
                    else -> SettingsManager.SearchIntentMode.AUTO
                }
                settingsManager.searchIntentMode = chosen
                Toast.makeText(this, "Search Intent: ${chosen.name}", Toast.LENGTH_SHORT).show()
                dialog.dismiss()
            }
            .setNegativeButton(android.R.string.cancel, null)
            .show()
    }

    private fun applyToolsModeToEngine(mode: SettingsManager.ToolsMode) {
        if (lastAppliedToolsMode != null && lastAppliedToolsMode != mode) {
            isSessionSystemPromptSet = false
        }
        lastAppliedToolsMode = mode
        lifecycleScope.launch(Dispatchers.Default) {
            try {
                com.example.llama.agent.core.EngineOrchestrator.withForegroundLock(
                    context = applicationContext,
                    owner = com.example.llama.agent.core.EngineOwner.FOREGROUND_CHAT
                ) { eng ->
                    val toolsJson = com.example.llama.agent.tools.web.WebToolsDispatcher.getToolsJson()
                    val caps = if (eng.state.value.isModelLoaded) eng.getModelCapabilities() else null
                    val toolsEnabled = mode != SettingsManager.ToolsMode.OFF && toolsJson.isNotBlank()
                    val policyOverride = if (mode == SettingsManager.ToolsMode.OFF) 2 else 0
                    val adapter = if (caps != null || currentModelArchitecture != null || currentModelPath != null || settingsManager.isRemoteInferenceActive) {
                        com.example.llama.agent.core.AgentToolStepHelper.resolveAdapter(
                            settingsManager = settingsManager,
                            capabilities = caps,
                            architecture = currentModelArchitecture,
                            modelPath = currentModelPath,
                            chatTemplate = currentModelChatTemplate
                        )
                    } else {
                        null
                    }
                    if (adapter is com.example.llama.agent.adapter.ModelAgentController) {
                        adapter.configureEngine(eng, toolsEnabled, toolsJson, toolPolicyOverride = policyOverride)
                    } else {
                        eng.setToolPolicy(2)
                        eng.setToolDefinitions("")
                    }
                    Log.i(TAG, "Applied tools policy $mode to engine (hasTools=${toolsJson.isNotBlank()}, caps=${caps?.handlerName}).")
                }
            } catch (e: Throwable) {
                Log.w(TAG, "applyToolsModeToEngine note: ${e.message}")
            }
        }
    }

    private fun toggleToolsQuick() {
        val nextMode = when (settingsManager.toolsMode) {
            SettingsManager.ToolsMode.AUTO -> SettingsManager.ToolsMode.ON
            SettingsManager.ToolsMode.ON -> SettingsManager.ToolsMode.OFF
            SettingsManager.ToolsMode.OFF -> SettingsManager.ToolsMode.AUTO
        }
        settingsManager.toolsMode = nextMode
        updateToolsButtonLabel()
        applyToolsModeToEngine(nextMode)
        Toast.makeText(this, "Nástroje: ${nextMode.name}", Toast.LENGTH_SHORT).show()
    }

    private fun showIndividualToolsToggleDialog() {
        val allItems = com.example.llama.agent.tools.web.WebToolsDispatcher.ALL_TOOL_ITEMS
        val labels = allItems.map { it.second }.toTypedArray()
        val currentEnabled = settingsManager.enabledAgentTools.toMutableSet()
        val checkedItems = BooleanArray(allItems.size) { i ->
            currentEnabled.contains(allItems[i].first)
        }

        com.google.android.material.dialog.MaterialAlertDialogBuilder(this)
            .setTitle("🎛️ Kombinace nástrojů (Realtime: ${currentEnabled.size}/${allItems.size})")
            .setMultiChoiceItems(labels, checkedItems) { _, which, isChecked ->
                val toolId = allItems[which].first
                if (isChecked) {
                    currentEnabled.add(toolId)
                } else {
                    currentEnabled.remove(toolId)
                }
                settingsManager.enabledAgentTools = currentEnabled
                if (settingsManager.refreshSystemPromptOnModeSwitch) {
                    isSessionSystemPromptSet = false
                }
                applyToolsModeToEngine(settingsManager.toolsMode)
                updateToolsButtonLabel()
            }
            .setPositiveButton("Hotovo") { d, _ ->
                updateToolsButtonLabel()
                Toast.makeText(
                    this,
                    "Aktivní nástroje: ${currentEnabled.size}/${allItems.size} (aplikováno v reálném čase)",
                    Toast.LENGTH_SHORT
                ).show()
                d.dismiss()
            }
            .setNeutralButton("Zapnout vše") { _, _ ->
                val allSet = allItems.map { it.first }.toSet()
                settingsManager.enabledAgentTools = allSet
                if (settingsManager.refreshSystemPromptOnModeSwitch) {
                    isSessionSystemPromptSet = false
                }
                applyToolsModeToEngine(settingsManager.toolsMode)
                updateToolsButtonLabel()
                Toast.makeText(this, "Všech ${allSet.size} nástrojů zapnuto", Toast.LENGTH_SHORT).show()
            }
            .setNegativeButton("Jen Web+RAG") { _, _ ->
                val coreSet = setOf("web_search", "rag_search", "browser_research_page", "read_url_content", "realtime_status")
                settingsManager.enabledAgentTools = coreSet
                if (settingsManager.refreshSystemPromptOnModeSwitch) {
                    isSessionSystemPromptSet = false
                }
                applyToolsModeToEngine(settingsManager.toolsMode)
                updateToolsButtonLabel()
                Toast.makeText(this, "Zvolen profil: Web + Bot + RAG (5 nástrojů)", Toast.LENGTH_SHORT).show()
            }
            .show()
    }

    private fun showToolsHelpDialog() {
        com.google.android.material.dialog.MaterialAlertDialogBuilder(this)
            .setTitle("🛠️ Autonomní nástroje modelu (Agentic Tools)")
            .setMessage(
                "Model má k dispozici 13 vestavěných nástrojů ve 4 oblastech:\n\n" +
                "🌐 Web & Hledání:\n" +
                "• web_search - Paralelní multi-source vyhledávání (DDG, Wikipedie, OpenAlex, arXiv, Scholar).\n" +
                "• read_url_content - Extrakce textu a článků s DataSaver kompresí (Opera Mini styl).\n" +
                "• browser_open / scroll / click / find / back - Virtuální prohlížeč pro navigaci po webu.\n\n" +
                "📁 Souborový Sandbox & Lokální RAG:\n" +
                "• file_write / read / list / delete - Bezpečné úložiště v sandboxu aplikace.\n" +
                "• rag_search - Fulltextové vyhledávání v dokumentech sandboxu bez nutnosti sítě.\n\n" +
                "⏱️ Systémová Telemetrie:\n" +
                "• realtime_status - Přesný čas, časová zóna, baterie, RAM a stav připojení.\n\n" +
                "Režimy:\n" +
                "• AUTO: Nástroje se aktivují automaticky při dotazech na fakta, novinky, soubory či čas.\n" +
                "• ON: Nástroje jsou aktivní v každém dotazu modelu.\n" +
                "• OFF: Čistý offline režim bez volání nástrojů."
            )
            .setPositiveButton("OK", null)
            .setNeutralButton("⚙️ Nastavení Web Agenta") { _, _ ->
                val intent = Intent(this, SettingsActivity::class.java).apply {
                    putExtra("open_tab", 5) // Tab Rozšířené (Advanced) s konfigurací Web Agenta a Deep Research
                }
                startActivity(intent)
            }
            .show()
    }

    private fun updateWebPipButtonLabel() {
        if (!::btnWebPipSelector.isInitialized) return
        when (settingsManager.webDisplayMode) {
            SettingsManager.WebDisplayMode.DIRECT -> {
                btnWebPipSelector.text = "🌐 Web: Direct"
                val color = android.graphics.Color.parseColor("#80CBC4")
                btnWebPipSelector.setTextColor(color)
                btnWebPipSelector.strokeColor = android.content.res.ColorStateList.valueOf(color)
            }
            SettingsManager.WebDisplayMode.PIP -> {
                btnWebPipSelector.text = "🌐 Web: PiP"
                val color = android.graphics.Color.parseColor("#00E5FF")
                btnWebPipSelector.setTextColor(color)
                btnWebPipSelector.strokeColor = android.content.res.ColorStateList.valueOf(color)
            }
        }
    }

    private fun showWebPipSelectorDialog() {
        val modes = arrayOf(
            "⚡ Direct (Tichý headless režim na pozadí)",
            "📺 In-Chat PiP (Živé okno prohlížeče v chatu)",
            "ℹ️ Informace o webovém zobrazení..."
        )
        val currentIdx = when (settingsManager.webDisplayMode) {
            SettingsManager.WebDisplayMode.DIRECT -> 0
            SettingsManager.WebDisplayMode.PIP -> 1
        }

        com.google.android.material.dialog.MaterialAlertDialogBuilder(this)
            .setTitle("🌐 Režim zobrazení webu")
            .setSingleChoiceItems(modes, currentIdx) { dialog, which ->
                when (which) {
                    0 -> {
                        settingsManager.webDisplayMode = SettingsManager.WebDisplayMode.DIRECT
                        destroyPipWebView()
                        updateWebPipButtonLabel()
                        dialog.dismiss()
                    }
                    1 -> {
                        settingsManager.webDisplayMode = SettingsManager.WebDisplayMode.PIP
                        updateWebPipButtonLabel()
                        dialog.dismiss()
                    }
                    2 -> {
                        dialog.dismiss()
                        showWebPipHelpDialog()
                    }
                }
            }
            .setNegativeButton(android.R.string.cancel, null)
            .show()
    }

    private fun toggleWebPipQuick() {
        val nextMode = when (settingsManager.webDisplayMode) {
            SettingsManager.WebDisplayMode.DIRECT -> SettingsManager.WebDisplayMode.PIP
            SettingsManager.WebDisplayMode.PIP -> {
                destroyPipWebView()
                SettingsManager.WebDisplayMode.DIRECT
            }
        }
        settingsManager.webDisplayMode = nextMode
        updateWebPipButtonLabel()
        Toast.makeText(this, "Režim webu: ${nextMode.name}", Toast.LENGTH_SHORT).show()
    }

    private fun showWebPipHelpDialog() {
        com.google.android.material.dialog.MaterialAlertDialogBuilder(this)
            .setTitle("🌐 Režim zobrazení webu")
            .setMessage(
                "Možnosti zobrazení při práci agenta na webu:\n\n" +
                "1. Direct (Doporučeno pro maximální rychlost):\n" +
                "   Model načítá data z webu na pozadí přes rychlý headless parser. Spotřebovává 0 MB RAM pro vykreslování a data okamžitě předává modelu.\n\n" +
                "2. In-Chat PiP (Picture-in-Picture):\n" +
                "   Přímo v okně chatu se otevře živý WebView, kde sledujete práci bota v reálném čase. " +
                "Po dokončení hledání se dle nastavení buď přepne do Offline zámku (0 % sítě a JS, stránku lze scrollovat a číst), nebo se ihned skryje a zničí."
            )
            .setPositiveButton("Rozumím", null)
            .show()
    }

    private fun initPipController() {
        com.example.llama.agent.tools.web.AutonomousBrowserSubagent.pipController =
            object : com.example.llama.agent.tools.web.PipBrowserController {
                override fun onLoadUrl(url: String) {
                    if (settingsManager.webDisplayMode != SettingsManager.WebDisplayMode.PIP) return
                    runOnUiThread {
                        showPipWithUrl(url)
                    }
                }

                override fun onScroll(direction: String, steps: Int) {
                    if (settingsManager.webDisplayMode != SettingsManager.WebDisplayMode.PIP) return
                    runOnUiThread {
                        val wv = pipWebView ?: return@runOnUiThread
                        val dir = direction.lowercase().trim()
                        val px = (480 * steps.coerceIn(1, 6))
                        val js = when (dir) {
                            "up" -> "window.scrollBy({top: -$px, behavior: 'smooth'});"
                            "top" -> "window.scrollTo({top: 0, behavior: 'smooth'});"
                            "bottom" -> "window.scrollTo({top: document.body.scrollHeight, behavior: 'smooth'});"
                            else -> "window.scrollBy({top: $px, behavior: 'smooth'});"
                        }
                        wv.evaluateJavascript(js, null)
                    }
                }

                override fun onDismissBanners() {
                    if (settingsManager.webDisplayMode != SettingsManager.WebDisplayMode.PIP) return
                    runOnUiThread {
                        val wv = pipWebView ?: return@runOnUiThread
                        wv.postDelayed({
                            val dismissJs = com.example.llama.agent.tools.web.WebPrivacyManager.buildStealthAndRejectCookiesJs(
                                scrollByPx = 320
                            )
                            pipWebView?.evaluateJavascript(dismissJs, null)
                        }, 900L)
                    }
                }

                override fun onNavigateBack() {
                    if (settingsManager.webDisplayMode != SettingsManager.WebDisplayMode.PIP) return
                    runOnUiThread {
                        if (pipWebView?.canGoBack() == true) {
                            pipWebView?.goBack()
                        }
                    }
                }

                override suspend fun navigateScrollAndExtractDom(
                    url: String,
                    scrollSteps: Int,
                    waitMs: Long
                ): com.example.llama.agent.tools.workspace.WebViewHeadlessBridge.DomSnapshotResult? {
                    if (settingsManager.webDisplayMode != SettingsManager.WebDisplayMode.PIP) return null
                    return pipNavMutex.withLock {
                        try {
                            val pageFinishedDeferred = kotlinx.coroutines.CompletableDeferred<Boolean>()
                            withContext(Dispatchers.Main) {
                                pipPageFinishedWaiter?.complete(false)
                                pipPageFinishedWaiter = pageFinishedDeferred
                                showPipWithUrl(url)
                            }
                            val loadedOk = kotlinx.coroutines.withTimeoutOrNull(waitMs.coerceIn(3000L, 5000L)) {
                                pageFinishedDeferred.await()
                            } ?: false
                            withContext(Dispatchers.Main) {
                                if (pipPageFinishedWaiter === pageFinishedDeferred) {
                                    pipPageFinishedWaiter = null
                                }
                            }
                            // Krátký čas pro ustálení JS/SPA po onPageFinished + viditelný lidský scroll
                            kotlinx.coroutines.delay(if (loadedOk) 450L else 250L)
                            val targetScrollPx = scrollSteps.coerceIn(1, 5) * 420
                            withContext(Dispatchers.Main) {
                                val hash = com.example.llama.agent.tools.web.WebPrivacyManager.sha256Short(url)
                                pipUrlTv?.text = "🤖 #$hash · Bot čte a scrolluje..."
                                pipWebView?.evaluateJavascript(
                                    "window.scrollTo({top: $targetScrollPx, behavior: 'smooth'});",
                                    null
                                )
                            }
                            kotlinx.coroutines.delay(450L)
                            val rawResult: String? = withContext(Dispatchers.Main) {
                                val wv = pipWebView ?: return@withContext null
                                kotlinx.coroutines.suspendCancellableCoroutine { cont ->
                                    val extractJs = com.example.llama.agent.tools.web.WebPrivacyManager.buildDomExtractionJs(
                                        scrollTopPx = null,
                                        maxChars = 14000,
                                        maxLinks = 35
                                    )
                                    wv.evaluateJavascript(extractJs) { value ->
                                        if (cont.isActive) {
                                            cont.resumeWith(Result.success(value))
                                        }
                                    }
                                }
                            }
                            if (rawResult.isNullOrBlank() || rawResult == "null") {
                                null
                            } else {
                                withContext(Dispatchers.Default) {
                                    com.example.llama.agent.tools.web.WebPrivacyManager.parseDomExtractionJson(rawResult)
                                }
                            }
                        } catch (e: Exception) {
                            pipPageFinishedWaiter?.complete(false)
                            pipPageFinishedWaiter = null
                            null
                        }
                    }
                }
            }
    }

    @androidx.annotation.MainThread
    private fun updatePipCardHeight(expanded: Boolean) {
        isPipExpanded = expanded
        val card = pipRootView ?: return
        val targetDp = if (expanded) 400 else 220
        val targetPx = (targetDp * resources.displayMetrics.density).toInt()
        card.layoutParams?.let { lp ->
            lp.height = targetPx
            card.layoutParams = lp
            card.requestLayout()
        }
        pipExpandBtn?.text = if (expanded) "⤡" else "⤢"
    }

    @androidx.annotation.MainThread
    private fun showPipWithUrl(url: String) {
        if (!com.example.llama.agent.tools.web.WebPrivacyManager.isSafeNavigationUrl(url)) {
            Log.w(TAG, "⛔ Blocked unsafe or disallowed PiP URL")
            destroyPipWebView()
            return
        }

        // Lazy inflate ViewStub if not yet inflated
        if (pipRootView == null && pipStub != null) {
            try {
                pipRootView = pipStub?.inflate()
                pipContainer = pipRootView?.findViewById(R.id.container_pip_webview)
                pipUrlTv = pipRootView?.findViewById(R.id.tv_pip_url)
                pipProgressBar = pipRootView?.findViewById(R.id.pb_pip_loading)
                pipExpandBtn = pipRootView?.findViewById(R.id.btn_pip_expand)
                pipCloseBtn = pipRootView?.findViewById(R.id.btn_pip_close)

                pipExpandBtn?.setOnClickListener {
                    updatePipCardHeight(!isPipExpanded)
                }
                pipCloseBtn?.setOnClickListener {
                    destroyPipWebView()
                }
            } catch (e: Exception) {
                Log.e(TAG, "Failed to inflate PiP ViewStub: ${e.message}")
            }
        }

        isPipOfflineLocked = false
        currentPipUrl = url
        pipContainer?.visibility = View.VISIBLE
        pipRootView?.visibility = View.VISIBLE
        val initialHash = com.example.llama.agent.tools.web.WebPrivacyManager.sha256Short(url)
        pipUrlTv?.text = "🌐 #$initialHash · Načítám..."
        pipProgressBar?.visibility = View.VISIBLE

        // Lazy create WebView if not created
        if (pipWebView == null && pipContainer != null) {
            pipContainer?.descendantFocusability = ViewGroup.FOCUS_AFTER_DESCENDANTS
            val isDataSaver = settingsManager.webDataSaverEnabled
            val wv = android.webkit.WebView(this).apply {
                layoutParams = ViewGroup.LayoutParams(ViewGroup.LayoutParams.MATCH_PARENT, ViewGroup.LayoutParams.MATCH_PARENT)
                isFocusable = true
                isFocusableInTouchMode = false
                isClickable = true
                com.example.llama.agent.tools.web.WebPrivacyManager.configureWebViewPrivacy(
                    webView = this,
                    blockImages = isDataSaver
                )
                webChromeClient = com.example.llama.agent.tools.web.WebPrivacyManager.createSafeWebChromeClient { progress ->
                    pipProgressBar?.progress = progress
                }
                webViewClient = object : android.webkit.WebViewClient() {
                    override fun shouldOverrideUrlLoading(
                        view: android.webkit.WebView?,
                        request: android.webkit.WebResourceRequest?
                    ): Boolean {
                        val target = request?.url?.toString()
                        val origin = view?.url ?: currentPipUrl
                        if (request?.isForMainFrame != false && isPipOfflineLocked) {
                            val allowInteractive =
                                com.example.llama.agent.tools.web.WebPrivacyManager.securityPolicy.allowPipInteractiveUnlockAfterWork
                            if (!allowInteractive) {
                                val lockHash = com.example.llama.agent.tools.web.WebPrivacyManager.sha256Short(origin)
                                pipUrlTv?.text = "🔒 #$lockHash · OFFLINE ZÁMEK (odkaz blokován)"
                                Toast.makeText(
                                    this@MainActivity,
                                    "🔒 PiP je po práci odpojeno od internetu. Proklik lze povolit v Dev menu.",
                                    Toast.LENGTH_SHORT
                                ).show()
                                return true
                            }
                            if (!com.example.llama.agent.tools.web.WebPrivacyManager.isSafeNavigationUrl(
                                    url = target,
                                    originUrl = origin
                                )
                            ) {
                                return true
                            }
                            isPipOfflineLocked = false
                            currentPipUrl = target ?: origin
                            view?.settings?.javaScriptEnabled = true
                            pipProgressBar?.visibility = View.VISIBLE
                            val nextHash = com.example.llama.agent.tools.web.WebPrivacyManager.sha256Short(currentPipUrl)
                            pipUrlTv?.text = "🌐 #$nextHash · Prohlížení odemčeno..."
                            return false
                        }
                        val allowed = com.example.llama.agent.tools.web.WebPrivacyManager.isSafeNavigationUrl(
                            url = target,
                            originUrl = origin
                        )
                        if (allowed && request?.isForMainFrame != false && !target.isNullOrBlank()) {
                            currentPipUrl = target
                        }
                        return !allowed
                    }

                    override fun shouldInterceptRequest(
                        view: android.webkit.WebView?,
                        request: android.webkit.WebResourceRequest?
                    ): android.webkit.WebResourceResponse? {
                        if (isPipOfflineLocked) {
                            return com.example.llama.agent.tools.web.WebPrivacyManager.emptyBlockedResponse()
                        }
                        val reqUrl = request?.url?.toString()
                        if (com.example.llama.agent.tools.web.WebPrivacyManager.shouldBlockRequest(
                                url = reqUrl,
                                blockMedia = settingsManager.webDataSaverEnabled
                            )
                        ) {
                            return com.example.llama.agent.tools.web.WebPrivacyManager.emptyBlockedResponse()
                        }
                        return super.shouldInterceptRequest(view, request)
                    }

                    override fun onPageStarted(view: android.webkit.WebView?, pageUrl: String?, favicon: Bitmap?) {
                        if (pageUrl == "about:blank") return
                        val activeUrl = pageUrl ?: currentPipUrl
                        currentPipUrl = activeUrl
                        pipProgressBar?.visibility = View.VISIBLE
                        val hash = com.example.llama.agent.tools.web.WebPrivacyManager.sha256Short(activeUrl)
                        pipUrlTv?.text = "🌐 #$hash · Načítám..."
                        view?.evaluateJavascript(
                            com.example.llama.agent.tools.web.WebPrivacyManager.EARLY_STEALTH_JS,
                            null
                        )
                    }

                    override fun onReceivedSslError(
                        view: android.webkit.WebView?,
                        handler: android.webkit.SslErrorHandler?,
                        error: android.net.http.SslError?
                    ) {
                        val hash = com.example.llama.agent.tools.web.WebPrivacyManager.sha256Short(view?.url ?: currentPipUrl)
                        com.example.llama.agent.tools.web.WebPrivacyManager.handleWebViewSslError(handler, hash)
                    }

                    override fun onPageFinished(view: android.webkit.WebView?, pageUrl: String?) {
                        if (pageUrl == "about:blank") return
                        pipProgressBar?.visibility = View.GONE
                        val activeUrl = pageUrl ?: currentPipUrl
                        val hash = com.example.llama.agent.tools.web.WebPrivacyManager.sha256Short(activeUrl)
                        pipUrlTv?.text = "🌐 #$hash · Živě"

                        // Privacy First: automatické odmítnutí cookie lišt (Reject all) a odstranění fixed overlay prvků bez souhlasu
                        val rejectCookiesJs = com.example.llama.agent.tools.web.WebPrivacyManager.buildStealthAndRejectCookiesJs()
                        view?.evaluateJavascript(rejectCookiesJs, null)

                        if (settingsManager.webDataSaverEnabled) {
                            view?.evaluateJavascript(
                                "document.querySelectorAll('img, video, iframe, picture, svg').forEach(e => e.style.display = 'none');",
                                null
                            )
                        }
                        pipPageFinishedWaiter?.complete(true)
                    }

                    override fun onReceivedError(
                        view: android.webkit.WebView?,
                        request: android.webkit.WebResourceRequest?,
                        error: android.webkit.WebResourceError?
                    ) {
                        if (request?.isForMainFrame == false) return
                        pipProgressBar?.visibility = View.GONE
                        val hash = com.example.llama.agent.tools.web.WebPrivacyManager.sha256Short(currentPipUrl)
                        pipUrlTv?.text = "⚠️ #$hash · Chyba načtení"
                        pipPageFinishedWaiter?.complete(false)
                    }
                }
            }
            pipContainer?.removeAllViews()
            pipContainer?.addView(wv)
            pipWebView = wv
        } else {
            pipWebView?.settings?.javaScriptEnabled = true
            pipWebView?.onResume()
        }

        pipWebView?.loadUrl(url)
    }

    @androidx.annotation.MainThread
    private fun destroyOnlyPipWebViewEngine() {
        isPipOfflineLocked = false
        currentPipUrl = ""
        pipPageFinishedWaiter?.complete(false)
        pipPageFinishedWaiter = null
        val wv = pipWebView
        pipWebView = null
        com.example.llama.agent.tools.web.WebPrivacyManager.clearWebViewPrivacyData(wv)
        pipContainer?.removeAllViews()
    }

    @androidx.annotation.MainThread
    private fun completePipWebActivity(elapsedMs: Long, sourceCount: Int, seedInput: String) {
        if (settingsManager.pipPostSearchAction == SettingsManager.PipPostSearchAction.AUTO_CLOSE) {
            destroyPipWebView()
            return
        }

        val wv = pipWebView
        if (wv == null) {
            pipRootView?.visibility = View.GONE
            return
        }

        isPipOfflineLocked = true
        pipProgressBar?.visibility = View.GONE
        com.example.llama.agent.tools.web.WebPrivacyManager.wipeStoragePreserveLiveView(wv)

        pipContainer?.visibility = View.VISIBLE
        pipRootView?.visibility = View.VISIBLE

        val secStr = String.format(java.util.Locale.US, "%.1fs", (elapsedMs / 1000.0).coerceAtLeast(0.1))
        val hash = com.example.llama.agent.tools.web.WebPrivacyManager.sha256Short(
            currentPipUrl.ifBlank { "${System.nanoTime()}:$seedInput" }
        )
        val srcLabel = when {
            sourceCount == 1 -> "1 zdroj"
            sourceCount in 2..4 -> "$sourceCount zdroje"
            else -> "$sourceCount zdrojů"
        }
        val interactiveHint = if (com.example.llama.agent.tools.web.WebPrivacyManager.securityPolicy.allowPipInteractiveUnlockAfterWork) {
            " · klik=odemč."
        } else {
            ""
        }
        pipUrlTv?.text = "🔒 #$hash · OFFLINE · $srcLabel · $secStr$interactiveHint"
    }

    @androidx.annotation.MainThread
    private fun destroyPipWebView() {
        updatePipCardHeight(false)
        destroyOnlyPipWebViewEngine()
        pipProgressBar?.visibility = View.GONE
        pipRootView?.visibility = View.GONE
    }

    @androidx.annotation.MainThread
    private fun showSubagentWorkingUi(statusLabel: String) {
        val mode = settingsManager.subagentIndicatorMode
        if (mode == SettingsManager.SubagentIndicatorMode.TOP_BADGE ||
            mode == SettingsManager.SubagentIndicatorMode.BOTH
        ) {
            subagentIndicatorTv.text = "🌐 $statusLabel"
            subagentIndicatorTv.visibility = View.VISIBLE
        } else {
            subagentIndicatorTv.visibility = View.GONE
        }

        if (mode == SettingsManager.SubagentIndicatorMode.METRICS ||
            mode == SettingsManager.SubagentIndicatorMode.BOTH
        ) {
            speedTv.text = "🌐 $statusLabel"
            speedTv.visibility = View.VISIBLE
        }
    }

    @androidx.annotation.MainThread
    private fun hideSubagentWorkingUi(restoreSpeedText: String? = null) {
        subagentIndicatorTv.visibility = View.GONE
        speedTv.visibility = if (settingsManager.showStats) View.VISIBLE else View.GONE
        if (!restoreSpeedText.isNullOrBlank()) {
            speedTv.text = restoreSpeedText
        }
    }

    private fun updateSendButtonState() {
        val isAgentActive = settingsManager.isAgentEnabled
        val inputEmpty = userInputEt.text.isNullOrEmpty()

        // Agent mode: send button funguje i bez engine (Tier 1/2 nepotřebují LLM)
        if (!::engine.isInitialized) {
            sendButton.setImageResource(R.drawable.outline_send_24)
            sendButton.isEnabled = isAgentActive && !inputEmpty
            return
        }

        val isGeneratingViaService =
                serviceManager.generationState.value is GenerationState.Generating
        val isGeneratingInteractive = generationJob?.isActive == true
        val isGenerating = isGeneratingViaService || isGeneratingInteractive

        val isModelLoaded = engine.state.value.isModelLoaded

        if (isGenerating) {
            sendButton.setImageResource(R.drawable.ic_stop_24)
            sendButton.isEnabled = true
        } else {
            sendButton.setImageResource(R.drawable.outline_send_24)
            sendButton.isEnabled = (isModelLoaded || isAgentActive) && !inputEmpty
        }
    }

    private fun handleUserInput() {
        // PENDING IMAGE BYPASS: Pokud uživatel uploadnul obrázek, bypass Agent
        // a pošli prompt přímo do LLM chat pipeline. C++ vrstva má g_cached_bmp
        // a auto-prepend <__media__> marker zajistí, že model obrázek uvidí.
        if (hasPendingImage) {
            Log.i(TAG, "📷 Pending image detected — bypassing Agent, routing to LLM chat pipeline")
            // Padáme do standardního LLM chat; hasPendingImage se vyčistí až po validaci vstupu před odesláním.
        } else if (settingsManager.isAgentEnabled) {
            // AGENT ROUTING: Pokud je agent zapnutý a NENÍ pending image
            val userMsg = userInputEt.text.toString().trim()
            if (userMsg.isEmpty()) {
                Toast.makeText(this, getString(R.string.info_input_empty), Toast.LENGTH_SHORT)
                        .show()
                return
            }
            userInputEt.text = null
            val agentService =
                    com.example.llama.agent.service.AiAgentAccessibilityService.getInstance()
            if (agentService != null) {
                Log.i(TAG, "🤖 Routing text to Agent: \"$userMsg\"")
                agentService.activateAgent(userMsg)
            } else {
                Log.e(TAG, "⚠️ Agent enabled but Accessibility Service not running!")
                Toast.makeText(
                                this,
                                "Agent Service není aktivní — povolte v Usnadnění přístupu",
                                Toast.LENGTH_LONG
                        )
                        .show()
            }
            return
        }
        // Standard LLM chat (agent je vypnutý)
        if (settingsManager.backgroundGeneration) {
            serviceManager.connect(asResident = true)
            if (serviceManager.isServiceConnected()) {
                delegateGenerationToService()
                return
            }
            Log.w(TAG, "Background Service requested but binder not connected yet; running interactively.")
        }
        runGenerationInteractive()
    }

    private fun delegateGenerationToService() {
        val userMsg = userInputEt.text.toString()
        if (userMsg.isEmpty()) {
            Toast.makeText(this, getString(R.string.info_input_empty), Toast.LENGTH_SHORT).show()
            return
        }

        if (!serviceManager.isServiceConnected()) {
            Toast.makeText(this, "Service not connected yet", Toast.LENGTH_SHORT).show()
            return
        }

        if (serviceManager.generationState.value is GenerationState.Generating) {
            Toast.makeText(this, "Generování právě probíhá...", Toast.LENGTH_SHORT).show()
            return
        }

        if (settingsManager.backgroundGeneration) {
            serviceManager.connect(asResident = true)
        }
        lastHandledCompleteState = null

        hasPendingImage = false
        userInputEt.text = null
        destroyPipWebView()
        applyDisplaySettings()
        if (lastAppliedSystemMessage != null && lastAppliedSystemMessage != settingsManager.systemMessage) {
            Log.i(TAG, "🔄 User System Prompt changed in Settings. Marking session system prompt for refresh.")
            isSessionSystemPromptSet = false
            if (messages.isNotEmpty() && messages[0].content.startsWith("SYSTEM:")) {
                messages[0].content = "SYSTEM: ${settingsManager.systemMessage}"
                messageAdapter.notifyItemChanged(0)
            }
        }

        val currentSessionId = sessionManager.currentSessionId
        val sessionChanged = lastActiveSessionId != currentSessionId
        if (sessionChanged) {
            Log.i(TAG, "Session change detected ($lastActiveSessionId -> $currentSessionId). Invalidating context.")
            lastActiveSessionId = currentSessionId
            isSessionSystemPromptSet = false
            lastAppliedSkillId = null
            lastAppliedToolsMode = null
        }

        val sc = settingsManager.loadSamplingConfig()
        Log.i(TAG, getString(R.string.info_generation_params_log, sc.temperature, sc.topK, sc.topP))
        val isFirstUserMsgNow = messages.none { it.isUser }

        messages.add(Message(UUID.randomUUID().toString(), StringBuilder(userMsg), true))
        val assistantMsgId = UUID.randomUUID().toString()
        messages.add(Message(assistantMsgId, StringBuilder(""), false, isStreaming = true))

        messageAdapter.notifyItemRangeInserted(messages.size - 2, 2)
        if (!settingsManager.disableAutoScroll) {
            messagesRv.post {
                val count = messageAdapter.itemCount
                if (count > 0) messagesRv.smoothScrollToPosition(count - 1)
            }
        }

        lifecycleScope.launch(Dispatchers.Default) {
            try {
                val promptToSend = buildEffectivePromptToSend(userMsg, isFirstUserMsgNow)
                val targetConfig = evaluateContextExpansion(userMsg)
                var preservedKvOnExpand = false
                if (targetConfig != null) {
                    com.example.llama.agent.core.EngineOrchestrator.withForegroundLock(
                        context = applicationContext,
                        owner = com.example.llama.agent.core.EngineOwner.FOREGROUND_CHAT,
                        config = targetConfig
                    ) { /* expanded context in orchestrator */ }
                    preservedKvOnExpand = com.example.llama.agent.core.EngineOrchestrator.lastResizePreservedKv
                    if (!preservedKvOnExpand) {
                        isSessionSystemPromptSet = false
                    }
                }

                val activeCaps = if (::engine.isInitialized) {
                    try { engine.getModelCapabilities() } catch (_: Exception) { null }
                } else null
                val adapter = com.example.llama.agent.core.AgentToolStepHelper.resolveAdapter(
                    settingsManager = settingsManager,
                    capabilities = activeCaps,
                    architecture = currentModelArchitecture,
                    modelPath = currentModelPath,
                    chatTemplate = currentModelChatTemplate
                )
                val executionKind = com.example.llama.agent.core.AgentToolStepHelper.resolveEngineExecutionKind(adapter)
                val isAssistedMode = settingsManager.isAssistedModeForEngine(executionKind)
                val budget = settingsManager.getEffectiveBudgetProfile(
                    modelTrainContextLength = activeCaps?.contextLength ?: 0
                )
                val effectiveMaxTokens = if (settingsManager.isRemoteInferenceActive &&
                    !budget.useGgufSamplingMaxTokensSlider
                ) {
                    settingsManager.remoteMaxOutputTokens
                } else {
                    budget.resolveLocalOutputTokenLimit(sc.maxTokens)
                }
                val isFirstUserMsg = messages.count { it.isUser } <= 1
                val didResetKvForSystemPrompt = !isFirstUserMsg && !isSessionSystemPromptSet && !settingsManager.isRemoteInferenceActive
                val isFollowUpSearchCmd = isAssistedMode && (userMsg.trim().lowercase() in setOf(
                    "vyhledej to", "najdi to", "hledej to", "prohledej to", "zkus to najít", "zkus to vyhledat", "vyhledej", "hledej"
                ))
                val prevUserMsgForSearch = if (isFollowUpSearchCmd) {
                    messages.dropLast(2).lastOrNull { it.isUser && it.content.toString().trim().length > 12 }?.content?.toString()?.trim() ?: ""
                } else ""
                val effectiveSearchUserMsg = if (prevUserMsgForSearch.isNotBlank()) prevUserMsgForSearch else userMsg
                val effectiveSysPrompt = if (isFirstUserMsg || !isSessionSystemPromptSet || settingsManager.isRemoteInferenceActive) {
                    buildEffectiveSystemPrompt(
                        userMsg = effectiveSearchUserMsg,
                        architecture = currentModelArchitecture,
                        capabilities = activeCaps
                    )
                } else {
                    ""
                }
                isSessionSystemPromptSet = true

                val needsHistoryRecovery = (targetConfig != null && !preservedKvOnExpand)
                val historyLimit = if (budget.overrideMode == com.example.llama.agent.remote.ExecutionProfileOverride.CUSTOM_DEV_CLAMP ||
                    budget.overrideMode == com.example.llama.agent.remote.ExecutionProfileOverride.FORCE_MOBILE_GGUF
                ) {
                    budget.maxHistoryTurns.coerceAtLeast(1)
                } else {
                    10
                }
                val promptWithRecoveredHistory = if (needsHistoryRecovery && messages.count { it.isUser } > 1) {
                    val pastTurns = messages.dropLast(2)
                        .filter { msg ->
                            val raw = msg.content.trim()
                            raw.isNotEmpty() && (msg.isUser || (!raw.startsWith("SYSTEM:") && !raw.startsWith("⚠️") && !raw.startsWith("❌")))
                        }
                        .takeLast(historyLimit)
                        .joinToString("\n") { msg ->
                            val role = if (msg.isUser) "User" else "Assistant"
                            "$role: ${msg.content.trim()}"
                        }
                    if (pastTurns.isNotBlank()) {
                        "[Kontext konverzace v této relaci:\n$pastTurns]\n\n$promptToSend"
                    } else {
                        promptToSend
                    }
                } else {
                    promptToSend
                }

                serviceManager.startGeneration(
                        sessionId = sessionManager.currentSessionId,
                        currentHistory = messages.dropLast(1),
                        prompt = promptWithRecoveredHistory,
                        maxTokens = effectiveMaxTokens,
                        assistantMessageId = assistantMsgId,
                        systemPrompt = if (isFirstUserMsg || didResetKvForSystemPrompt || settingsManager.isRemoteInferenceActive) {
                            effectiveSysPrompt
                        } else {
                            null
                        },
                        resetContext = sessionChanged,
                        architecture = currentModelArchitecture,
                        modelPath = currentModelPath,
                        allowSystemPromptAutoBuild = false
                )
            } catch (e: Exception) {
                Log.e(TAG, "Configuration Error", e)
            }
        }
    }

    private fun buildEffectiveSystemPrompt(
        userMsg: String,
        architecture: String? = null,
        capabilities: com.arm.aichat.NativeModelCapabilities? = null,
        executionProfile: com.example.llama.agent.adapter.AgentExecutionProfile? = null
    ): String {
        val registry = SkillRegistry.getInstance(this)
        val sysMsg = settingsManager.systemMessage
        val toolsMode = settingsManager.toolsMode
        val toolsEnabled = toolsMode != SettingsManager.ToolsMode.OFF

        val modelCaps = capabilities ?: if (::engine.isInitialized) {
            try { engine.getModelCapabilities() } catch (_: Exception) { null }
        } else null
        val adapter = com.example.llama.agent.core.AgentToolStepHelper.resolveAdapter(
            settingsManager = settingsManager,
            capabilities = modelCaps,
            architecture = architecture ?: currentModelArchitecture,
            modelPath = currentModelPath,
            chatTemplate = currentModelChatTemplate
        )

        val tp = settingsManager.thinkingProfile
        val resolvedProfile = executionProfile ?: com.example.llama.agent.adapter.AgentExecutionProfile.build(
            adapter = adapter,
            settingsManager = settingsManager,
            toolsEnabled = toolsEnabled,
            toolsJsonNotBlank = com.example.llama.agent.tools.web.WebToolsDispatcher.getToolsJson().isNotBlank()
        )
        val isAssistedMode = resolvedProfile.isAssisted
        val skillsInstruction = registry.getPromptInjection(
            userPrompt = userMsg,
            includeLegacyToolSyntax = isAssistedMode && toolsEnabled
        )
        return if (adapter is com.example.llama.agent.adapter.ModelAgentController) {
            adapter.buildSystemPromptWithProfile(
                context = this,
                userSystemMessage = sysMsg,
                skillsInstruction = skillsInstruction,
                toolsEnabled = toolsEnabled,
                thinkingProfile = tp,
                supportsNativeThinkingControl = (modelCaps?.supportsThinking == true),
                executionProfile = resolvedProfile
            )
        } else {
            adapter.buildSystemPrompt(
                context = this,
                userSystemMessage = sysMsg,
                skillsInstruction = skillsInstruction,
                toolsEnabled = toolsEnabled,
                thinkingProfile = tp,
                supportsNativeThinkingControl = (modelCaps?.supportsThinking == true)
            )
        }
    }

    private fun buildEffectivePromptToSend(
        userMsg: String,
        isFirstTurn: Boolean = messages.none { it.isUser }
    ): String {
        val registry = SkillRegistry.getInstance(this)
        val activeId = registry.activeSkillId
        val toolsMode = settingsManager.toolsMode
        val sysMsg = settingsManager.systemMessage
        val toolsEnabled = toolsMode != SettingsManager.ToolsMode.OFF
        val executionKind = when {
            !settingsManager.isRemoteInferenceActive -> com.example.llama.agent.adapter.template.EngineExecutionKind.LOCAL_GGUF
            settingsManager.aiProviderType == SettingsManager.AiProviderType.REMOTE_DESKTOP ->
                com.example.llama.agent.adapter.template.EngineExecutionKind.DESKTOP_LAN
            else -> com.example.llama.agent.adapter.template.EngineExecutionKind.CLOUD_API
        }
        val skillsInstruction = registry.getPromptInjection(
            userPrompt = userMsg,
            includeLegacyToolSyntax = settingsManager.isAssistedModeForEngine(executionKind) && toolsEnabled
        )

        // 1. První uživatelská zpráva relace -> plný System Prompt (včetně dovedností a datumu)
        // je nastaven nativně přes orchestratorEngine.setSystemPrompt() v C++ jádru (s podporou Jinja/ROLE_SYSTEM
        // a automatického sloučení v ai_chat.cpp). Do uživatelské zprávy se posílá čistý userMsg, aby nedocházelo
        // ke zdvojení systémového kontextu v KV cache!
        if (isFirstTurn) {
            lastAppliedSystemMessage = sysMsg
            lastAppliedSkillId = activeId
            lastAppliedToolsMode = toolsMode
            return userMsg
        }

        // 2. Dynamická změna System Promptu, skillu nebo toolsMode uprostřed existujícího chatu (Mid-Chat Switch)
        val skillChanged = lastAppliedSkillId != null && lastAppliedSkillId != activeId
        val toolsChanged = lastAppliedToolsMode != null && lastAppliedToolsMode != toolsMode
        val sysMsgChanged = lastAppliedSystemMessage != null && lastAppliedSystemMessage != sysMsg
        if (sysMsgChanged || skillChanged || toolsChanged) {
            lastAppliedSystemMessage = sysMsg
            lastAppliedSkillId = activeId
            lastAppliedToolsMode = toolsMode
            if (sysMsgChanged || skillChanged || settingsManager.refreshSystemPromptOnModeSwitch) {
                isSessionSystemPromptSet = false
            }
        }

        // 3. Režim AUTO: Pokud zpráva obsahuje trigger slova aktivního skillu a čerstvý System Prompt
        // se v tomto tahu neposílá do KV cache / remote požadavku, připojí se definice skillu k userMsg bez duplikace.
        val willSendFreshSystemPrompt = isFirstTurn || !isSessionSystemPromptSet || settingsManager.isRemoteInferenceActive
        if (activeId == SkillRegistry.SKILL_AUTO && skillsInstruction.isNotBlank() && !willSendFreshSystemPrompt) {
            return "$skillsInstruction\n\n$userMsg"
        }

        // 4. Běžná zpráva (systémový kontext i definice nástrojů jsou zavedeny v KV cache / systémové instrukci)
        return userMsg
    }

    fun resetEngineContext(onResetComplete: (() -> Unit)? = null) {
        stopGeneration()
        hasPendingImage = false
        lastAppliedSkillId = null
        lastAppliedToolsMode = null
        isSessionSystemPromptSet = false
        lifecycleScope.launch(Dispatchers.IO) {
            var hadLoadedEngine = false
            try {
                com.example.llama.agent.core.EngineOrchestrator.withForegroundLock(
                    context = applicationContext,
                    owner = com.example.llama.agent.core.EngineOwner.FOREGROUND_CHAT,
                    config = null
                ) { orchEngine ->
                    (orchEngine as? com.example.llama.agent.core.HybridInferenceEngine)?.let { hyb ->
                        hyb.desktopLanEngine.resetContext()
                        hyb.geminiCloudEngine.resetContext()
                        hyb.openAiCloudEngine.resetContext()
                    }
                    if (orchEngine.state.value.isModelLoaded) {
                        hadLoadedEngine = true
                        orchEngine.resetContext()
                    }
                }
            } catch (e: Throwable) {
                Log.w(TAG, "resetEngineContext note: ${e.message}")
            } finally {
                withContext(Dispatchers.Main) {
                    if (hadLoadedEngine) {
                        Log.i(TAG, "Engine context reset completed.")
                    }
                    onResetComplete?.invoke()
                }
            }
        }
    }

    private fun resolveMaxRamAllowedContextTokens(): Int {
        return try {
            val am = getSystemService(android.content.Context.ACTIVITY_SERVICE) as? android.app.ActivityManager
            val mi = android.app.ActivityManager.MemoryInfo()
            am?.getMemoryInfo(mi)
            val availMb = if (mi.availMem > 0L) (mi.availMem / (1024L * 1024L)) else 4096L
            val usableMb = (availMb - 512L).coerceAtLeast(256L)
            val steps4k = (usableMb / 40L).coerceAtLeast(1L)
            (steps4k * 4096L).coerceIn(4096L, Int.MAX_VALUE.toLong()).toInt()
        } catch (_: Throwable) {
            Int.MAX_VALUE
        }
    }

    private suspend fun evaluateContextExpansion(userMsg: String): com.example.llama.agent.core.EngineConfiguration? {
        if (settingsManager.isRemoteInferenceActive) return null
        if (settingsManager.contextSize > 0) return null
        val activeCaps = if (::engine.isInitialized) {
            try { engine.getModelCapabilities() } catch (_: Exception) { null }
        } else null
        val budget = settingsManager.getEffectiveBudgetProfile(
            modelTrainContextLength = activeCaps?.contextLength ?: 0
        )
        val totalChars = messages.sumOf { it.content.length + it.reasoning.length } + userMsg.length
        // Přesný rozpočtovač tokenů (2.5 znaku/token): započítat jen reálný systémový prompt + schéma nástrojů
        // a navyšovat plynule až při dosažení 75 % aktuální hranice nCtx (ne na každém vstupu).
        val toolsEnabledNow = settingsManager.toolsMode != SettingsManager.ToolsMode.OFF
        val hasNativeTools = activeCaps?.supportsToolCalls == true
        val toolsSchemaChars = when {
            !toolsEnabledNow -> 0
            hasNativeTools -> com.example.llama.agent.tools.web.WebToolsDispatcher.getToolsJson().length
            else -> com.example.llama.agent.tools.web.WebToolsDispatcher.TOOL_INSTRUCTIONS.length
        }
        val systemAndToolsChars = settingsManager.systemMessage.length + toolsSchemaChars
        val estimatedTokens = ((totalChars + systemAndToolsChars) / 2.5).toInt()
        val baseConfig = com.example.llama.agent.core.EngineOrchestrator.activeConfig
        if (baseConfig != null) {
            val currentCtx = if (baseConfig.nCtx <= 0) 4096 else baseConfig.nCtx
            val ramMax = resolveMaxRamAllowedContextTokens()
            val trainMax = activeCaps?.contextLength?.takeIf { it > 0 } ?: Int.MAX_VALUE
            val devSimMax = if (
                (budget.overrideMode == com.example.llama.agent.remote.ExecutionProfileOverride.CUSTOM_DEV_CLAMP ||
                    budget.overrideMode == com.example.llama.agent.remote.ExecutionProfileOverride.FORCE_MOBILE_GGUF) &&
                budget.simulatedContextSizeTokens > 0
            ) {
                budget.simulatedContextSizeTokens
            } else {
                Int.MAX_VALUE
            }
            val isHexagonBackend = baseConfig.llamaBackend == SettingsManager.LlamaBackend.HEXAGON ||
                baseConfig.llamaBackend == SettingsManager.LlamaBackend.HTP_V75 ||
                baseConfig.llamaBackend == SettingsManager.LlamaBackend.HTP_V73
            val hexagonVaMax = if (isHexagonBackend) {
                val totalVaLimitMb = settingsManager.devHexagonVaLimitMb.toLong() * settingsManager.devHexagonNdev.toLong()
                val modelFileMb = try {
                    java.io.File(baseConfig.modelPath).takeIf { it.exists() }?.length()?.div(1024L * 1024L) ?: 0L
                } catch (_: Throwable) {
                    0L
                }
                val remainingVaMb = (totalVaLimitMb - modelFileMb).coerceAtLeast(0L)
                val vaAllowedCtx = if (remainingVaMb >= 40L) {
                    ((remainingVaMb / 40L) * 4096L).coerceIn(4096L, Int.MAX_VALUE.toLong()).toInt()
                } else {
                    currentCtx
                }
                if (vaAllowedCtx < minOf(ramMax, trainMax, devSimMax)) {
                    Log.i(
                        TAG,
                        "🛡️ Hexagon VA limit cap: backend=${baseConfig.llamaBackend.id}, model=${modelFileMb}MB, vaLimit=${totalVaLimitMb}MB (${settingsManager.devHexagonVaLimitMb}MB×${settingsManager.devHexagonNdev}), kvBudget=${remainingVaMb}MB -> maxCtx=$vaAllowedCtx"
                    )
                }
                vaAllowedCtx
            } else {
                Int.MAX_VALUE
            }
            val maxAllowedCtx = minOf(ramMax, trainMax, devSimMax, hexagonVaMax)
            val expandThreshold = (currentCtx * 0.75).toInt()
            if (estimatedTokens >= expandThreshold && currentCtx < maxAllowedCtx) {
                // Plynulé zaokrouhlení nahoru na nejbližší násobek 4096 podle dostupné RAM
                val neededWithHeadroom = ((estimatedTokens.toLong() * 4L) / 3L)
                val targetStep = (((neededWithHeadroom + 4095L) / 4096L) * 4096L)
                    .coerceAtLeast(currentCtx.toLong() + 4096L)
                val newCtx = targetStep.coerceIn(currentCtx.toLong(), maxAllowedCtx.toLong()).toInt()
                if (newCtx > currentCtx) {
                    Log.i(TAG, "📈 Dynamické navýšení kontextu (Auto 0, >=75%): $currentCtx -> $newCtx tokenů (odhad: $estimatedTokens, práh: $expandThreshold, cap: $maxAllowedCtx)")
                    withContext(Dispatchers.Main) {
                        Toast.makeText(this@MainActivity, "Navýšení kontextu na $newCtx tokenů...", Toast.LENGTH_SHORT).show()
                    }
                    return baseConfig.copy(nCtx = newCtx)
                }
            }
        }
        // DŮLEŽITÉ: Pokud odhad nepřekračuje currentCtx, vrací null -> ŽÁDNÉ ZMENŠOVÁNÍ!
        return null
    }

    private fun appendSeparatedReasoningChunk(chunk: String, needsParagraphSeparator: Boolean): Boolean {
        if (chunk.isEmpty()) return needsParagraphSeparator
        val targetChunk = if (needsParagraphSeparator) {
            val lastAssistant = messages.lastOrNull { !it.isUser }
            if (lastAssistant != null && lastAssistant.reasoning.isNotBlank() && !lastAssistant.reasoning.endsWith("\n\n")) {
                "\n\n$chunk"
            } else {
                chunk
            }
        } else {
            chunk
        }
        messageAdapter.appendReasoningToLastMessage(targetChunk)
        return false
    }

    private fun extractInlineThoughtsFromMessage(msg: Message) {
        if (msg.content.isBlank()) return
        val (inlineReasoning, remainingContent) = MessageAdapter.extractReasoningAndContent(
            rawReasoning = msg.reasoning,
            rawContent = msg.content,
            isStreaming = false
        )
        msg.reasoning = inlineReasoning
        msg.content = remainingContent
    }

    private fun runGenerationInteractive() {
        if (generationJob?.isActive == true) {
            Toast.makeText(this, "Generování právě probíhá...", Toast.LENGTH_SHORT).show()
            return
        }
        val userMsg = userInputEt.text.toString()
        if (userMsg.isEmpty()) {
            Toast.makeText(this, getString(R.string.info_input_empty), Toast.LENGTH_SHORT).show()
            return
        }
        hasPendingImage = false
        userInputEt.text = null
        destroyPipWebView()
        applyDisplaySettings()
        if (lastAppliedSystemMessage != null && lastAppliedSystemMessage != settingsManager.systemMessage) {
            Log.i(TAG, "🔄 User System Prompt changed in Settings. Marking session system prompt for refresh.")
            isSessionSystemPromptSet = false
            if (messages.isNotEmpty() && messages[0].content.startsWith("SYSTEM:")) {
                messages[0].content = "SYSTEM: ${settingsManager.systemMessage}"
                messageAdapter.notifyItemChanged(0)
            }
        }

        val currentSessionId = sessionManager.currentSessionId
        val sessionChanged = lastActiveSessionId != currentSessionId
        if (sessionChanged) {
            Log.i(TAG, "Session change detected ($lastActiveSessionId -> $currentSessionId). Invalidating context.")
            lastActiveSessionId = currentSessionId
            isSessionSystemPromptSet = false
            lastAppliedSkillId = null
            lastAppliedToolsMode = null
        }

        val sc = settingsManager.loadSamplingConfig()
        Log.i(TAG, getString(R.string.info_generation_params_log, sc.temperature, sc.topK, sc.topP))
        val isFirstUserMsgNow = messages.none { it.isUser }
        messages.add(Message(UUID.randomUUID().toString(), StringBuilder(userMsg), true))
        messages.add(
                Message(UUID.randomUUID().toString(), StringBuilder(""), false, isStreaming = true)
        )
        messageAdapter.notifyItemRangeInserted(messages.size - 2, 2)
        val fullResponseBuilder = StringBuilder()

        generationJob =
                lifecycleScope.launch(Dispatchers.Default) {
                    val uiBuffer = StringBuilder()
                    val reasoningBuffer = StringBuilder()
                    var accumulatedGenMs = 0L
                    var accumulatedConfirmedTokens = 0
                    var activePassStartMs = 0L
                    var activePassTokens = 0
                    var activePassFinishedTokens: Int? = null
                    var lastDisplayedSpeedText = ""
                    var lastUiUpdate = 0L
                    var lastReasoningUiUpdate = 0L
                    var hasExecutedTool = false
                    var needsReasoningSeparatorPending = false
                    var resolvedAdapterForRestore: com.example.llama.agent.adapter.ModelFamilyAdapter? = null
                    val promptToSend = buildEffectivePromptToSend(userMsg, isFirstUserMsgNow)
                    val targetConfig = evaluateContextExpansion(userMsg)
                    try {
                        com.example.llama.agent.core.EngineOrchestrator.withForegroundLock(
                            context = applicationContext,
                            owner = com.example.llama.agent.core.EngineOwner.FOREGROUND_CHAT,
                            config = targetConfig
                        ) { orchestratorEngine ->
                            if (sessionChanged) {
                                orchestratorEngine.resetContext()
                            }
                            // Dynamické navýšení kontextu: zkontrolujeme, zda resize zachoval KV cache in-place
                            val preservedKvOnExpand = if (targetConfig != null) {
                                val preserved = com.example.llama.agent.core.EngineOrchestrator.lastResizePreservedKv
                                if (!preserved) {
                                    Log.i(TAG, "📈 Dynamické navýšení kontextu na ${targetConfig.nCtx}. Obnovuji kontext session.")
                                    isSessionSystemPromptSet = false
                                } else {
                                    Log.i(TAG, "📈 Dynamické navýšení kontextu na ${targetConfig.nCtx} zachovalo KV cache.")
                                }
                                preserved
                            } else {
                                false
                            }

                            val toolsMode = settingsManager.toolsMode
                            val toolsEnabled = toolsMode != SettingsManager.ToolsMode.OFF
                            val modelCaps = try { orchestratorEngine.getModelCapabilities() } catch (_: Exception) { null }
                            val modelArch = currentModelArchitecture
                            val adapter = com.example.llama.agent.core.AgentToolStepHelper.resolveAdapter(
                                settingsManager = settingsManager,
                                capabilities = modelCaps,
                                architecture = modelArch,
                                modelPath = currentModelPath,
                                chatTemplate = currentModelChatTemplate
                            )
                            resolvedAdapterForRestore = adapter
                            val executionKind = com.example.llama.agent.core.AgentToolStepHelper.resolveEngineExecutionKind(adapter)
                            val activeToolsJson = com.example.llama.agent.tools.web.WebToolsDispatcher.getToolsJson()
                            val executionProfile = com.example.llama.agent.adapter.AgentExecutionProfile.build(
                                adapter = adapter,
                                settingsManager = settingsManager,
                                toolsEnabled = toolsEnabled,
                                toolsJsonNotBlank = activeToolsJson.isNotBlank()
                            )
                            val isAssistedMode = executionProfile.isAssisted
                            Log.i(TAG, "🤖 Active Model Adapter: ${adapter.familyName}, arch=$modelArch | Profile(${executionProfile.toLogSummary()})")

                            val tp = settingsManager.thinkingProfile
                            val effectiveTp = if (adapter is com.example.llama.agent.adapter.ModelAgentController) {
                                MessageAdapter.updateActiveFamilyCoTRules(adapter.cotRules)
                                adapter.resolveEffectiveThinkingProfile(tp)
                            } else {
                                tp
                            }
                            orchestratorEngine.configureThinking(effectiveTp.isEnabled, effectiveTp.maxTokens, effectiveTp.effort)

                            val policyOverride = if (settingsManager.toolsMode == SettingsManager.ToolsMode.OFF) 2 else 0
                            if (adapter is com.example.llama.agent.adapter.ModelAgentController) {
                                adapter.configureEngine(
                                    engine = orchestratorEngine,
                                    toolsEnabled = toolsEnabled,
                                    toolsJson = activeToolsJson,
                                    toolPolicyOverride = policyOverride,
                                    executionProfile = executionProfile
                                )
                            } else {
                                orchestratorEngine.setToolPolicy(2) // NONE
                                orchestratorEngine.setToolDefinitions("")
                            }

                            // 1. Prefill Native System Prompt on first message of session nebo bez ztráty KV (pro Cloud vždy čerstvý)
                            val isFirstUserMsg = messages.count { it.isUser } <= 1
                            var didResetKvForSystemPrompt = !isFirstUserMsg && !isSessionSystemPromptSet && !settingsManager.isRemoteInferenceActive
                            val isFollowUpSearchCmd = executionProfile.allowFollowUpSearchRewrite && (userMsg.trim().lowercase() in setOf(
                                "vyhledej to", "najdi to", "hledej to", "prohledej to", "zkus to najít", "zkus to vyhledat", "vyhledej", "hledej"
                            ))
                            val prevUserMsgForSearch = if (isFollowUpSearchCmd) {
                                messages.dropLast(2).lastOrNull { it.isUser && it.content.toString().trim().length > 12 }?.content?.toString()?.trim() ?: ""
                            } else ""
                            val effectiveSearchUserMsg = if (prevUserMsgForSearch.isNotBlank()) prevUserMsgForSearch else userMsg
                            val cachedBaseSysPrompt by lazy(LazyThreadSafetyMode.NONE) {
                                buildEffectiveSystemPrompt(effectiveSearchUserMsg, modelArch, modelCaps, executionProfile)
                            }
                            if (isFirstUserMsg || !isSessionSystemPromptSet || settingsManager.isRemoteInferenceActive) {
                                val sysPrompt = cachedBaseSysPrompt
                                try {
                                    if (didResetKvForSystemPrompt) {
                                        val preservedSysKv = orchestratorEngine.updateSystemPromptPreservingContext(sysPrompt)
                                        didResetKvForSystemPrompt = !preservedSysKv
                                        Log.i(TAG, "System Prompt (${sysPrompt.length} chars) updated via ${adapter.familyName} adapter (kvPreserved=$preservedSysKv).")
                                    } else {
                                        orchestratorEngine.setSystemPrompt(sysPrompt)
                                        Log.i(TAG, "System Prompt (${sysPrompt.length} chars) set via ${adapter.familyName} adapter.")
                                    }
                                    isSessionSystemPromptSet = true
                                    lastAppliedToolsMode = toolsMode
                                } catch (e: Exception) {
                                    Log.w(TAG, "Could not set native system prompt: ${e.message}")
                                }
                            }
                            isSessionSystemPromptSet = true

                            val sc = settingsManager.loadSamplingConfig()
                            orchestratorEngine.configureSampling(
                                    sc.temperature,
                                    sc.topK,
                                    sc.topP,
                                    sc.minP,
                                    sc.xtcProbability,
                                    sc.xtcThreshold,
                                    sc.typicalP,
                                    sc.dynamicTempRange,
                                    sc.dynamicTempExponent,
                                    sc.penaltyRepeat,
                                    sc.penaltyFreq,
                                    sc.penaltyPresent,
                                    sc.penaltyLastN,
                                    sc.mirostat,
                                    sc.mirostatTau,
                                    sc.mirostatEta,
                                    sc.samplersSequence,
                                    sc.useBackendSampling,
                                    ignoreEos = sc.ignoreEos,
                                    noPerf = sc.noPerf,
                                    timingPerToken = sc.timingPerToken
                            )

                            val activeBudget = settingsManager.getEffectiveBudgetProfile(
                                modelTrainContextLength = modelCaps?.contextLength ?: 0
                            )
                            val historyLimit = if (activeBudget.overrideMode == com.example.llama.agent.remote.ExecutionProfileOverride.CUSTOM_DEV_CLAMP ||
                                activeBudget.overrideMode == com.example.llama.agent.remote.ExecutionProfileOverride.FORCE_MOBILE_GGUF
                            ) {
                                activeBudget.maxHistoryTurns.coerceAtLeast(1)
                            } else {
                                10
                            }
                            // Pokud došlo k dynamickému navýšení kontextu nebo resetu systémového promptu a KV nebyla zachována, obnovíme historii v KV okně
                            val needsHistoryRecovery = ((targetConfig != null && !preservedKvOnExpand) || didResetKvForSystemPrompt)
                            val effectivePrompt = if (needsHistoryRecovery && messages.count { it.isUser } > 1) {
                                val pastTurns = messages.dropLast(2)
                                    .filter { msg ->
                                        val raw = msg.content.trim()
                                        raw.isNotEmpty() && (msg.isUser || (!raw.startsWith("SYSTEM:") && !raw.startsWith("⚠️") && !raw.startsWith("❌")))
                                    }
                                    .takeLast(historyLimit)
                                    .joinToString("\n") { msg ->
                                        val role = if (msg.isUser) "User" else "Assistant"
                                        "$role: ${msg.content.trim()}"
                                    }
                                if (pastTurns.isNotBlank()) {
                                    "[Kontext konverzace v této relaci:\n$pastTurns]\n\n$promptToSend"
                                } else {
                                    promptToSend
                                }
                            } else {
                                promptToSend
                            }

                            // Čistá autonomní AI inference: Uživatelský prompt směřuje IHNED přímo do modelu.
                            // Žádné předřazené síťové zdržení ani preemptivní vyhledávání na webu.
                            // Pokud uživatel připojil soubor nebo se ptá offline, AI reaguje okamžitě v řádu milisekund.
                            // Pokud AI potřebuje externí fakta, zavolá autonomně nástroj (web_search) v agentní smyčce níže.
                            val isFirstTurn = messages.count { it.isUser } <= 1
                            val resolvedEffectivePrompt = if (prevUserMsgForSearch.isNotBlank()) {
                                "$effectivePrompt (Téma k vyhledání na webu z předchozí zprávy: $prevUserMsgForSearch)"
                            } else {
                                effectivePrompt
                            }
                            var currentPrompt = if (adapter is com.example.llama.agent.adapter.ModelAgentController) {
                                adapter.prepareUserPrompt(resolvedEffectivePrompt, cachedBaseSysPrompt, isFirstTurn)
                            } else {
                                resolvedEffectivePrompt
                            }

                            if (settingsManager.isRemoteInferenceActive) {
                                val hybridEngine = com.example.llama.agent.core.EngineProvider.getHybrid(applicationContext)
                                if (hybridEngine.remoteEngine.getConversationHistorySnapshot().isEmpty()) {
                                    val priorUiMessages = if (messages.size >= 2) messages.dropLast(2) else emptyList()
                                    val hasPriorTurns = priorUiMessages.any {
                                        val raw = it.content.trim()
                                        raw.isNotEmpty() && (it.isUser || !raw.startsWith("SYSTEM:"))
                                    }
                                    if (hasPriorTurns) {
                                        hybridEngine.restoreRemoteHistoryFromUiMessages(priorUiMessages)
                                    }
                                }
                            }
                            val effectiveInteractiveMaxTokens = if (settingsManager.isRemoteInferenceActive &&
                                !activeBudget.useGgufSamplingMaxTokensSlider
                            ) {
                                settingsManager.remoteMaxOutputTokens
                            } else {
                                activeBudget.resolveLocalOutputTokenLimit(sc.maxTokens)
                            }

                            val interactiveUsesNativeTools = adapter is com.example.llama.agent.adapter.ModelAgentController &&
                                adapter.runtimeSpec.allowNativeTools
                            val initialSystemAndToolsChars = cachedBaseSysPrompt.length +
                                if (toolsEnabled && interactiveUsesNativeTools) com.example.llama.agent.tools.web.WebToolsDispatcher.getToolsJson().length else 0
                            var cumulativeInteractiveChars = initialSystemAndToolsChars + promptToSend.length +
                                messages.dropLast(2).sumOf { it.content.length + it.reasoning.length }
                            var agentStep = 0
                            val maxAgentSteps = activeBudget.maxAgentSteps
                            val executedToolCounts = mutableMapOf<String, Int>()
                            var pipTurnActive = false
                            var pipTurnStartMs = 0L
                            var pipTurnTotalSources = 0
                            var pipTurnSeedInput = ""
                            var lastToolResponseContent = ""
                            var botDeepResearchCompleted = false
                            var interactiveStreamError: String? = null

                            while (agentStep < maxAgentSteps && interactiveStreamError == null) {
                                agentStep++
                                val stepContentBuilder = StringBuilder()
                                val stepReasoningBuilder = StringBuilder()
                                var needsStepReasoningSeparator = (agentStep > 1)
                                if (needsStepReasoningSeparator) {
                                    needsReasoningSeparatorPending = true
                                }
                                val nativeAccumulator = com.example.llama.agent.core.AgentToolStepHelper.NativeToolCallAccumulator()
                                activePassStartMs = 0L
                                activePassTokens = 0
                                activePassFinishedTokens = null
                                lastUiUpdate = 0L
                                lastReasoningUiUpdate = 0L

                                val turnFlow = if (currentPrompt.isEmpty()) {
                                    orchestratorEngine.continueGeneration(effectiveInteractiveMaxTokens)
                                } else {
                                    orchestratorEngine.sendUserPromptEvents(currentPrompt, effectiveInteractiveMaxTokens)
                                }
                                turnFlow
                                    .collect { event ->
                                        when (event) {
                                            is com.arm.aichat.GenerationEvent.ReasoningDelta -> {
                                                val cleanReasoningText = event.text
                                                if (activePassStartMs == 0L) activePassStartMs = System.currentTimeMillis()
                                                if (event.generatedTokens > activePassTokens) {
                                                    activePassTokens = event.generatedTokens
                                                } else {
                                                    activePassTokens++
                                                }

                                                val isThinkingOff = false
                                                val activeStepBuf = if (isThinkingOff) stepContentBuilder else stepReasoningBuilder
                                                if (isThinkingOff) {
                                                    uiBuffer.append(cleanReasoningText)
                                                    fullResponseBuilder.append(cleanReasoningText)
                                                    stepContentBuilder.append(cleanReasoningText)
                                                } else {
                                                    reasoningBuffer.append(cleanReasoningText)
                                                    stepReasoningBuilder.append(cleanReasoningText)
                                                }

                                                if (toolsEnabled && !botDeepResearchCompleted &&
                                                    executionKind != com.example.llama.agent.adapter.template.EngineExecutionKind.CLOUD_API &&
                                                    agentStep < maxAgentSteps &&
                                                    com.example.llama.agent.core.AgentToolStepHelper.shouldStopStreamOnTextToolCall(
                                                        adapter = adapter,
                                                        buffer = activeStepBuf,
                                                        latestChunk = cleanReasoningText
                                                    )
                                                ) {
                                                    Log.i(TAG, "⚡ Detected complete tool call in ReasoningDelta, stopping step generation early to execute tool.")
                                                    orchestratorEngine.cancelGeneration()
                                                }

                                                val now = System.currentTimeMillis()
                                                val relevantLastUpdate = if (isThinkingOff) lastUiUpdate else lastReasoningUiUpdate
                                                if (now - relevantLastUpdate >= 40L) {
                                                    val chunk = if (isThinkingOff) uiBuffer.toString() else reasoningBuffer.toString()
                                                    if (isThinkingOff) {
                                                        uiBuffer.setLength(0)
                                                        lastUiUpdate = now
                                                    } else {
                                                        reasoningBuffer.setLength(0)
                                                        lastReasoningUiUpdate = now
                                                    }
                                                    withContext(Dispatchers.Main) {
                                                        if (isThinkingOff) {
                                                            messageAdapter.appendTokenToLastMessage(chunk)
                                                        } else {
                                                            needsStepReasoningSeparator = appendSeparatedReasoningChunk(chunk, needsStepReasoningSeparator)
                                                            needsReasoningSeparatorPending = needsStepReasoningSeparator
                                                        }
                                                        val passElapsedMs = (now - activePassStartMs).coerceAtLeast(1L)
                                                        val speed = activePassTokens / (passElapsedMs / 1000.0)
                                                        val totalLiveTokens = accumulatedConfirmedTokens + activePassTokens
                                                        val totalActiveSec = (accumulatedGenMs + passElapsedMs) / 1000.0
                                                        val speedText = getString(R.string.info_speed_format, speed)
                                                        lastDisplayedSpeedText = speedText
                                                        if (!settingsManager.disableAutoScroll && !messagesRv.canScrollVertically(1) && messages.isNotEmpty()) {
                                                            messagesRv.post {
                                                                val count = messageAdapter.itemCount
                                                                if (count > 0) messagesRv.scrollToPosition(count - 1)
                                                            }
                                                        }
                                                        if (settingsManager.showStats)
                                                            speedTv.text = speedText
                                                        if (settingsManager.showTokenCount)
                                                            tokenCountTv.text = getString(R.string.stats_tokens_format, totalLiveTokens)
                                                        if (settingsManager.showGenTime)
                                                            genTimeTv.text = getString(R.string.stats_time_format, totalActiveSec)
                                                    }
                                                }
                                            }
                                            is com.arm.aichat.GenerationEvent.ContentDelta -> {
                                                val cleanContentText = event.text
                                                if (activePassStartMs == 0L) activePassStartMs = System.currentTimeMillis()
                                                if (event.generatedTokens > activePassTokens) {
                                                    activePassTokens = event.generatedTokens
                                                } else {
                                                    activePassTokens++
                                                }

                                                if (reasoningBuffer.isNotEmpty()) {
                                                    val rChunk = reasoningBuffer.toString()
                                                    reasoningBuffer.setLength(0)
                                                    withContext(Dispatchers.Main) {
                                                        needsStepReasoningSeparator = appendSeparatedReasoningChunk(rChunk, needsStepReasoningSeparator)
                                                    }
                                                }

                                                uiBuffer.append(cleanContentText)
                                                fullResponseBuilder.append(cleanContentText)
                                                stepContentBuilder.append(cleanContentText)
                                                if (toolsEnabled && !botDeepResearchCompleted && agentStep < maxAgentSteps) {
                                                    if (com.example.llama.agent.core.AgentToolStepHelper.shouldStopStreamOnTextToolCall(
                                                            adapter = adapter,
                                                            buffer = stepContentBuilder,
                                                            latestChunk = cleanContentText,
                                                            executionProfile = executionProfile
                                                        )
                                                    ) {
                                                        Log.i(TAG, "⚡ Detected complete tool call in ContentDelta, stopping step generation early to execute tool.")
                                                        orchestratorEngine.cancelGeneration()
                                                    } else if (com.example.llama.agent.core.AgentToolStepHelper.shouldCancelStreamForProseToolCall(
                                                            adapter = adapter,
                                                            stepContent = stepContentBuilder,
                                                            latestChunk = cleanContentText,
                                                            activePassTokens = activePassTokens,
                                                            effectiveSearchUserMsg = effectiveSearchUserMsg,
                                                            isAssistedMode = isAssistedMode,
                                                            agentStep = agentStep,
                                                            executionProfile = executionProfile
                                                        )
                                                    ) {
                                                        Log.i(TAG, "⚡ Detected prose tool call description in ContentDelta, stopping fake prose to execute real tool.")
                                                        orchestratorEngine.cancelGeneration()
                                                    }
                                                }

                                                val now = System.currentTimeMillis()
                                                if (now - lastUiUpdate >= 40L) {
                                                    val chunk = uiBuffer.toString()
                                                    uiBuffer.setLength(0)
                                                    lastUiUpdate = now

                                                    withContext(Dispatchers.Main) {
                                                        messageAdapter.appendTokenToLastMessage(chunk)

                                                        val passElapsedMs = (now - activePassStartMs).coerceAtLeast(1L)
                                                        val speed = activePassTokens / (passElapsedMs / 1000.0)
                                                        val totalLiveTokens = accumulatedConfirmedTokens + activePassTokens
                                                        val totalActiveSec = (accumulatedGenMs + passElapsedMs) / 1000.0
                                                        val speedText = getString(R.string.info_speed_format, speed)
                                                        lastDisplayedSpeedText = speedText

                                                        if (!settingsManager.disableAutoScroll) {
                                                            if (!messagesRv.canScrollVertically(1) && messages.isNotEmpty()) {
                                                                messagesRv.post {
                                                                    val count = messageAdapter.itemCount
                                                                    if (count > 0) messagesRv.scrollToPosition(count - 1)
                                                                }
                                                            }
                                                        }
                                                        if (settingsManager.showStats)
                                                            speedTv.text = speedText
                                                        if (settingsManager.showTokenCount)
                                                            tokenCountTv.text = getString(R.string.stats_tokens_format, totalLiveTokens)
                                                        if (settingsManager.showGenTime)
                                                            genTimeTv.text = getString(R.string.stats_time_format, totalActiveSec)
                                                    }
                                                }
                                            }
                                            is com.arm.aichat.GenerationEvent.Finished -> {
                                                if (event.generatedTokens > 0) {
                                                    activePassFinishedTokens = event.generatedTokens
                                                }
                                                Log.i(TAG, "Generation finished event: reason=${event.reason}, promptTokens=${event.promptTokens}, genTokens=${event.generatedTokens}")
                                            }
                                            is com.arm.aichat.GenerationEvent.ToolCallDelta -> {
                                                nativeAccumulator.onDelta(event)
                                            }
                                            is com.arm.aichat.GenerationEvent.Error -> {
                                                Log.e(TAG, "Generation error event: ${event.message}")
                                                interactiveStreamError = event.message
                                                val errNotice = "\n⚠️ ${event.message}"
                                                uiBuffer.append(errNotice)
                                                fullResponseBuilder.append(errNotice)
                                                stepContentBuilder.append(errNotice)
                                            }
                                            else -> {}
                                        }
                                    }

                                if (activePassStartMs > 0L) {
                                    accumulatedGenMs += (System.currentTimeMillis() - activePassStartMs).coerceAtLeast(1L)
                                    activePassStartMs = 0L
                                }
                                val resolvedStepTokens = activePassFinishedTokens?.takeIf { it > 0 } ?: activePassTokens
                                accumulatedConfirmedTokens += resolvedStepTokens
                                activePassTokens = 0
                                activePassFinishedTokens = null

                                // Flush residual buffers after this turn
                                if (reasoningBuffer.isNotEmpty()) {
                                    val rChunk = reasoningBuffer.toString()
                                    reasoningBuffer.setLength(0)
                                    withContext(Dispatchers.Main) {
                                        needsStepReasoningSeparator = appendSeparatedReasoningChunk(rChunk, needsStepReasoningSeparator)
                                    }
                                }
                                if (uiBuffer.isNotEmpty()) {
                                    val cChunk = uiBuffer.toString()
                                    uiBuffer.setLength(0)
                                    withContext(Dispatchers.Main) {
                                        messageAdapter.appendTokenToLastMessage(cChunk)
                                    }
                                }

                                // Autonomous Agent: Check for tool calls (Native parallel/single > Content channel)
                                val stepToolsEnabled = settingsManager.toolsMode != SettingsManager.ToolsMode.OFF && !botDeepResearchCompleted
                                val stepToolCalls = if (stepToolsEnabled) {
                                    com.example.llama.agent.core.AgentToolStepHelper.extractStepToolCalls(
                                        adapter = adapter,
                                        nativeAccumulator = nativeAccumulator,
                                        stepContent = stepContentBuilder.toString(),
                                        stepReasoning = stepReasoningBuilder.toString(),
                                        effectiveUserMsg = effectiveSearchUserMsg,
                                        rawUserMsg = userMsg,
                                        searchIntentMode = settingsManager.searchIntentMode,
                                        agentStep = agentStep,
                                        isAssistedMode = isAssistedMode
                                    )
                                } else {
                                    emptyList()
                                }

                                if (interactiveStreamError != null) {
                                    Log.w(TAG, "Stopping interactive agent loop due to stream error: $interactiveStreamError")
                                    break
                                }

                                if (stepToolCalls.isNotEmpty() && agentStep < maxAgentSteps) {
                                    val batch = com.example.llama.agent.core.AgentToolStepHelper.prepareToolBatch(
                                        stepToolCalls = stepToolCalls,
                                        effectiveSearchUserMsg = effectiveSearchUserMsg,
                                        executedToolCounts = executedToolCounts
                                    )
                                    if (batch.allowedCount == 0) {
                                        Log.w(TAG, "🔁 Loop Guard: Model attempted duplicate tool call batch. Breaking agent loop to prevent cycle.")
                                        withContext(Dispatchers.Main) {
                                            val lastAssistant = messages.lastOrNull { !it.isUser }
                                            if (lastAssistant != null) {
                                                val cleanedGuardContent = com.example.llama.agent.tools.web.WebToolsDispatcher
                                                    .cleanToolCallMarkup(lastAssistant.content)
                                                    .trim()
                                                val isBareEchoedQuery = batch.calls.any { c ->
                                                    c.cleanQuery.isNotBlank() &&
                                                        cleanedGuardContent.equals(c.cleanQuery.trim(), ignoreCase = true)
                                                } || cleanedGuardContent.equals(effectiveSearchUserMsg.trim(), ignoreCase = true)
                                                lastAssistant.content = if (isBareEchoedQuery) "" else cleanedGuardContent
                                                val idx = messages.indexOf(lastAssistant)
                                                if (idx != -1) messageAdapter.notifyItemChanged(idx)
                                            }
                                        }
                                        break
                                    }

                                    hasExecutedTool = true

                                    // Clean up assistant bubble before executing tools while preserving ALL step reasoning intact
                                    withContext(Dispatchers.Main) {
                                        val lastAssistant = messages.lastOrNull { !it.isUser }
                                        if (lastAssistant != null) {
                                            lastAssistant.reasoning = com.example.llama.agent.core.AgentToolStepHelper.mergeStepThoughtsIntoReasoning(
                                                existingReasoning = lastAssistant.reasoning,
                                                rawStepContent = lastAssistant.content
                                            )
                                            lastAssistant.content = ""
                                            if (!lastAssistant.isThoughtsCollapsedByUser && lastAssistant.reasoning.isNotBlank()) {
                                                lastAssistant.isThoughtsExpanded = true
                                            }
                                            val idx = messages.indexOf(lastAssistant)
                                            if (idx != -1) messageAdapter.notifyItemChanged(idx)
                                        }
                                    }

                                    val stepResponses = mutableListOf<Pair<com.example.llama.agent.core.AgentToolStepHelper.ExtractedToolCallItem, com.example.llama.agent.tools.web.ToolCallResponse>>()
                                    for (prep in batch.calls) {
                                        val toolRequest = prep.reconciledRequest
                                        val toolName = toolRequest.name
                                        val cleanQ = prep.cleanQuery
                                        val canonicalTool = prep.canonicalTool
                                        Log.i(TAG, "🔧 Autonomous Agent Tool Call detected: $toolName, args=${toolRequest.arguments}")

                                        if (prep.isDuplicateSkipped) {
                                            val dupResp = com.example.llama.agent.tools.web.ToolCallResponse(
                                                toolName = toolName,
                                                queryOrUrl = cleanQ,
                                                content = "Tento identický dotaz už byl dříve proveden. Použij již získané výsledky a napiš přímou odpověď uživateli.",
                                                results = emptyList(),
                                                isSuccess = false
                                            )
                                            stepResponses.add(prep.call to dupResp)
                                            continue
                                        }

                                        val shouldSuppressWebSearchForFile = com.example.llama.agent.core.AgentToolStepHelper.shouldSuppressWebSearchForFile(
                                            userMsg = userMsg,
                                            toolName = toolName,
                                            cleanQuery = cleanQ
                                        )

                                        val isWebNavigationTool = canonicalTool in setOf(
                                            "web_search",
                                            "read_url_content",
                                            "browser_open",
                                            "browser_research_page",
                                            "academic_patent_search",
                                            "browser_scroll",
                                            "browser_click",
                                            "browser_find",
                                            "browser_back"
                                        )
                                        val isDirectUrlTool = canonicalTool in setOf("read_url_content", "browser_open", "browser_research_page")
                                        if (settingsManager.webDisplayMode == SettingsManager.WebDisplayMode.PIP && isWebNavigationTool && !shouldSuppressWebSearchForFile) {
                                            if (!pipTurnActive) {
                                                pipTurnActive = true
                                                pipTurnStartMs = System.currentTimeMillis()
                                            }
                                            if (isDirectUrlTool) {
                                                val directUrl = WebToolsDispatcher.extractCleanUrl(
                                                    toolRequest.arguments,
                                                    toolRequest.arguments.toString()
                                                )
                                                if (directUrl.isNotBlank() && (directUrl.startsWith("http://", ignoreCase = true) || directUrl.startsWith("https://", ignoreCase = true))) {
                                                    withContext(Dispatchers.Main) {
                                                        showPipWithUrl(directUrl)
                                                    }
                                                }
                                            } else if ((canonicalTool == "web_search" || canonicalTool == "academic_patent_search") && !settingsManager.webAutonomousBotDeepRead) {
                                                val q = cleanQ.ifBlank { (toolRequest.arguments["query"] ?: toolRequest.arguments["q"] ?: "").trim() }
                                                val searchPreviewUrl = if (q.isNotBlank() && q != "{}" && q != "[]" && q != "null") {
                                                    val encodedQ = java.net.URLEncoder.encode(q, "UTF-8")
                                                    val topEngine = settingsManager.webSearchEnginesOrder.firstOrNull { settingsManager.webEnabledEngines.contains(it) } ?: settingsManager.webPrimarySearchEngine
                                                    when (topEngine.lowercase().trim()) {
                                                        "brave" -> "https://search.brave.com/search?q=$encodedQ"
                                                        "google", "google_lite" -> "https://www.google.com/search?q=$encodedQ"
                                                        "searxng" -> "https://searx.be/search?q=$encodedQ"
                                                        "seznam" -> "https://search.seznam.cz/?q=$encodedQ"
                                                        "wikipedia" -> "https://cs.wikipedia.org/wiki/Special:Search?search=$encodedQ"
                                                        else -> "https://duckduckgo.com/?q=$encodedQ"
                                                    }
                                                } else ""
                                                if (searchPreviewUrl.isNotBlank()) {
                                                    withContext(Dispatchers.Main) {
                                                        showPipWithUrl(searchPreviewUrl)
                                                    }
                                                }
                                            } else if (pipWebView != null) {
                                                withContext(Dispatchers.Main) {
                                                    pipProgressBar?.visibility = View.VISIBLE
                                                    pipUrlTv?.text = "🤖 Bot pracuje ($canonicalTool)..."
                                                }
                                            }
                                        }

                                        val subagentUiActive = !shouldSuppressWebSearchForFile
                                        if (subagentUiActive) {
                                            val indicatorLabel = if (isWebNavigationTool) "Web" else toolName.take(14)
                                            withContext(Dispatchers.Main) {
                                                showSubagentWorkingUi(indicatorLabel)
                                            }
                                        }

                                        val toolResponse = try {
                                            if (shouldSuppressWebSearchForFile) {
                                                Log.i(TAG, "🛡️ Suppressing web search for local attached document. Returning instant local guidance.")
                                                com.example.llama.agent.tools.web.ToolCallResponse(
                                                    toolName = toolName,
                                                    queryOrUrl = cleanQ,
                                                    content = "Tento soubor je přiložen lokálně přímo v uživatelském promptu výše. Není na veřejném internetu. Zpracuj a analyzuj přímo poskytnutý text z promptu bez volání webových nástrojů.",
                                                    results = emptyList<com.example.llama.agent.tools.web.WebSearchResult>(),
                                                    isSuccess = true
                                                )
                                            } else {
                                                withContext(Dispatchers.IO) {
                                                    WebToolsDispatcher.executeTool(
                                                        name = toolName,
                                                        rawArgs = com.example.llama.agent.core.AgentToolStepHelper.resolveRawArgsJson(prep),
                                                        maxResults = settingsManager.webSearchMaxResults,
                                                        maxTokens = activeBudget.webExtractTokenLimit,
                                                        userPromptFallback = effectiveSearchUserMsg
                                                    )
                                                }
                                            }
                                        } finally {
                                            if (subagentUiActive) {
                                                withContext(Dispatchers.Main + kotlinx.coroutines.NonCancellable) {
                                                    hideSubagentWorkingUi(lastDisplayedSpeedText)
                                                }
                                            }
                                        }

                                        if (settingsManager.webDisplayMode == SettingsManager.WebDisplayMode.PIP &&
                                            isWebNavigationTool && !shouldSuppressWebSearchForFile
                                        ) {
                                            pipTurnTotalSources += toolResponse.results.size.coerceAtLeast(if (toolResponse.isSuccess) 1 else 0)
                                            val seedCandidate = toolResponse.queryOrUrl.ifBlank { cleanQ }
                                            if (seedCandidate.isNotBlank()) pipTurnSeedInput = seedCandidate
                                            if (pipWebView == null && toolResponse.isSuccess) {
                                                val firstResultUrl = toolResponse.results.firstOrNull {
                                                    val u = it.url.trim()
                                                    u.startsWith("http://", ignoreCase = true) || u.startsWith("https://", ignoreCase = true)
                                                }?.url
                                                if (!firstResultUrl.isNullOrBlank()) {
                                                    withContext(Dispatchers.Main) {
                                                        showPipWithUrl(firstResultUrl)
                                                    }
                                                }
                                            }
                                        }

                                        if (toolResponse.content.isNotBlank()) {
                                            lastToolResponseContent = toolResponse.content
                                        }
                                        val hasNativeLocalJinjaTools = executionProfile.useNativeToolsInEngine
                                        val completedDeepReadStep = canonicalTool in setOf("web_search", "browser_research_page", "academic_patent_search")
                                        val turnHadOnlySyntheticToolCalls = stepResponses.none { it.first.isNative } && !prep.call.isNative
                                        if (!settingsManager.isRemoteInferenceActive &&
                                            (!hasNativeLocalJinjaTools || turnHadOnlySyntheticToolCalls) &&
                                            completedDeepReadStep &&
                                            toolResponse.isSuccess && toolResponse.content.length > 160
                                        ) {
                                            botDeepResearchCompleted = true
                                        }

                                        // Add SearchReport to assistant message UI
                                        withContext(Dispatchers.Main) {
                                            val lastAssistant = messages.lastOrNull { !it.isUser }
                                            if (lastAssistant != null && (toolResponse.results.isNotEmpty() || lastAssistant.searchReports.isEmpty())) {
                                                val reportQuery = if (toolResponse.queryOrUrl.isNotBlank()) toolResponse.queryOrUrl else toolRequest.name
                                                lastAssistant.searchReports.add(
                                                    SearchReport(
                                                        query = reportQuery,
                                                        results = toolResponse.results,
                                                        isExpanded = false
                                                    )
                                                )
                                                val idx = messages.indexOf(lastAssistant)
                                                if (idx != -1) messageAdapter.notifyItemChanged(idx)
                                            }
                                            if (!settingsManager.disableAutoScroll && messages.isNotEmpty()) {
                                                messagesRv.post {
                                                    val count = messageAdapter.itemCount
                                                    if (count > 0) messagesRv.smoothScrollToPosition(count - 1)
                                                }
                                            }
                                        }

                                        stepResponses.add(prep.call to toolResponse)
                                    }

                                    cumulativeInteractiveChars += stepContentBuilder.length + stepReasoningBuilder.length +
                                        stepResponses.sumOf { it.second.content.length }
                                    if (settingsManager.contextSize == 0 && !settingsManager.isRemoteInferenceActive) {
                                        val activeCfg = com.example.llama.agent.core.EngineOrchestrator.activeConfig
                                        if (activeCfg != null) {
                                            val effectiveCurCtx = if (activeCfg.nCtx > 0) activeCfg.nCtx else 4096
                                            val estTokens = (cumulativeInteractiveChars / 2.5).toInt()
                                            if (estTokens >= (effectiveCurCtx * 0.75).toInt()) {
                                                val ramMax = resolveMaxRamAllowedContextTokens()
                                                val trainMax = modelCaps?.contextLength?.takeIf { it > 0 } ?: Int.MAX_VALUE
                                                val devSimMax = if (
                                                    (activeBudget.overrideMode == com.example.llama.agent.remote.ExecutionProfileOverride.CUSTOM_DEV_CLAMP ||
                                                        activeBudget.overrideMode == com.example.llama.agent.remote.ExecutionProfileOverride.FORCE_MOBILE_GGUF) &&
                                                    activeBudget.simulatedContextSizeTokens > 0
                                                ) {
                                                    activeBudget.simulatedContextSizeTokens
                                                } else {
                                                    Int.MAX_VALUE
                                                }
                                                val cap = minOf(ramMax, trainMax, devSimMax)
                                                val nextStep = (((estTokens / 4096) + 1) * 4096).coerceAtLeast(effectiveCurCtx + 4096)
                                                val targetCtx = nextStep.coerceAtMost(cap)
                                                if (targetCtx > effectiveCurCtx) {
                                                    val expanded = com.example.llama.agent.core.EngineOrchestrator
                                                        .tryExpandContextPreservingKvUnderLock(orchestratorEngine, targetCtx)
                                                    if (expanded) {
                                                        Log.i(TAG, "📈 Interactive in-place Auto nCtx expanded: $effectiveCurCtx -> $targetCtx (KV preserved)")
                                                    }
                                                }
                                            }
                                        }
                                    }

                                    val commitResult = com.example.llama.agent.core.AgentToolStepHelper.commitToolStepResponses(
                                        engine = orchestratorEngine,
                                        adapter = adapter,
                                        isRemoteEngine = settingsManager.isRemoteInferenceActive,
                                        isFinalStep = (agentStep >= maxAgentSteps - 1) || botDeepResearchCompleted,
                                        policyOverride = policyOverride,
                                        userMsg = effectiveSearchUserMsg,
                                        stepResponses = stepResponses,
                                        thinkingProfile = tp,
                                        isAssistedMode = isAssistedMode,
                                        allowNativeToolContinuation = executionProfile.useNativeToolsInEngine
                                    )
                                    currentPrompt = commitResult.nextPrompt
                                } else {
                                    break
                                }
                            }

                            // Pojistka: Pokud model po web vyhledávání vrátil pouze tool_call nebo <think> a obsah odpovědi pro uživatele zůstal prázdný
                            val lastAssistantCheck = messages.lastOrNull { !it.isUser }
                            var cleanContent = lastAssistantCheck?.content?.let { com.example.llama.agent.tools.web.WebToolsDispatcher.cleanToolCallMarkup(it).trim() } ?: ""
                            if (hasExecutedTool && cleanContent.isBlank() && lastAssistantCheck != null) {
                                val extractedFromCoT = com.example.llama.agent.tools.web.WebToolsDispatcher.extractAnswerFromUnclosedReasoning(lastAssistantCheck.reasoning)
                                if (extractedFromCoT != null) {
                                    withContext(Dispatchers.Main) {
                                        lastAssistantCheck.reasoning = extractedFromCoT.first
                                        lastAssistantCheck.content = extractedFromCoT.second
                                        val idx = messages.indexOf(lastAssistantCheck)
                                        if (idx != -1) messageAdapter.notifyItemChanged(idx)
                                    }
                                    cleanContent = extractedFromCoT.second
                                }
                            }
                            val allowExtraSynthesisCall = executionProfile.allowPostToolExtraSynthesis
                            val needsExtraSynthesis = com.example.llama.agent.core.AgentToolStepHelper.isBareUrlOrEchoAfterTools(
                                cleanedContent = cleanContent,
                                effectiveUserMsg = effectiveSearchUserMsg
                            )
                            if (hasExecutedTool && allowExtraSynthesisCall && needsExtraSynthesis && interactiveStreamError == null) {
                                Log.i(TAG, "🔧 Agent completed tools but visible answer content is blank/bare-URL. Running final direct answer synthesis pass...")
                                fullResponseBuilder.setLength(0)
                                withContext(Dispatchers.Main) {
                                    lastAssistantCheck?.content = ""
                                    val idx = messages.indexOf(lastAssistantCheck)
                                    if (idx != -1) messageAdapter.notifyItemChanged(idx)
                                }
                                orchestratorEngine.setToolPolicy(2) // NONE: Disable tools for final answer synthesis
                                val rawTpSynth = settingsManager.thinkingProfile
                                val tpSynth = (adapter as? com.example.llama.agent.adapter.ModelAgentController)
                                    ?.resolveEffectiveThinkingProfile(rawTpSynth)
                                    ?: rawTpSynth
                                orchestratorEngine.configureThinking(tpSynth.isEnabled, tpSynth.maxTokens, tpSynth.effort)
                                val finalPrompt =
                                    "Based on the tool results above, write a direct, complete final answer to the user's question (\"$effectiveSearchUserMsg\") in the same language as the user's question. Do not call any more tools."

                                val synthUiBuffer = StringBuilder()
                                val synthReasoningBuffer = StringBuilder()
                                var needsSynthReasoningSeparator = true
                                needsReasoningSeparatorPending = true
                                activePassStartMs = 0L
                                activePassTokens = 0
                                activePassFinishedTokens = null
                                lastUiUpdate = 0L
                                lastReasoningUiUpdate = 0L

                                orchestratorEngine.sendUserPromptEvents(finalPrompt, effectiveInteractiveMaxTokens)
                                    .collect { event ->
                                        when (event) {
                                            is com.arm.aichat.GenerationEvent.ContentDelta -> {
                                                val cleanContentText = event.text
                                                if (activePassStartMs == 0L) activePassStartMs = System.currentTimeMillis()
                                                if (event.generatedTokens > activePassTokens) {
                                                    activePassTokens = event.generatedTokens
                                                } else {
                                                    activePassTokens++
                                                }
                                                if (synthReasoningBuffer.isNotEmpty()) {
                                                    val rChunk = synthReasoningBuffer.toString()
                                                    synthReasoningBuffer.setLength(0)
                                                    withContext(Dispatchers.Main) {
                                                        needsSynthReasoningSeparator = appendSeparatedReasoningChunk(rChunk, needsSynthReasoningSeparator)
                                                        needsReasoningSeparatorPending = needsSynthReasoningSeparator
                                                    }
                                                }
                                                synthUiBuffer.append(cleanContentText)
                                                fullResponseBuilder.append(cleanContentText)

                                                val now = System.currentTimeMillis()
                                                if (now - lastUiUpdate >= 40L) {
                                                    val chunk = synthUiBuffer.toString()
                                                    synthUiBuffer.setLength(0)
                                                    lastUiUpdate = now
                                                    withContext(Dispatchers.Main) {
                                                        messageAdapter.appendTokenToLastMessage(chunk)
                                                        val passElapsedMs = (now - activePassStartMs).coerceAtLeast(1L)
                                                        val speed = activePassTokens / (passElapsedMs / 1000.0)
                                                        val totalLiveTokens = accumulatedConfirmedTokens + activePassTokens
                                                        val totalActiveSec = (accumulatedGenMs + passElapsedMs) / 1000.0
                                                        val speedText = getString(R.string.info_speed_format, speed)
                                                        lastDisplayedSpeedText = speedText
                                                        if (settingsManager.showStats)
                                                            speedTv.text = speedText
                                                        if (settingsManager.showTokenCount)
                                                            tokenCountTv.text = getString(R.string.stats_tokens_format, totalLiveTokens)
                                                        if (settingsManager.showGenTime)
                                                            genTimeTv.text = getString(R.string.stats_time_format, totalActiveSec)
                                                    }
                                                }
                                            }
                                            is com.arm.aichat.GenerationEvent.ReasoningDelta -> {
                                                val cleanReasoningText = event.text
                                                if (activePassStartMs == 0L) activePassStartMs = System.currentTimeMillis()
                                                if (event.generatedTokens > activePassTokens) {
                                                    activePassTokens = event.generatedTokens
                                                } else {
                                                    activePassTokens++
                                                }
                                                val isThinkingOff = false
                                                if (isThinkingOff) {
                                                    synthUiBuffer.append(cleanReasoningText)
                                                    fullResponseBuilder.append(cleanReasoningText)
                                                } else {
                                                    synthReasoningBuffer.append(cleanReasoningText)
                                                }
                                                val now = System.currentTimeMillis()
                                                val relevantLastUpdate = if (isThinkingOff) lastUiUpdate else lastReasoningUiUpdate
                                                if (now - relevantLastUpdate >= 40L) {
                                                    val chunk = if (isThinkingOff) synthUiBuffer.toString() else synthReasoningBuffer.toString()
                                                    if (isThinkingOff) {
                                                        synthUiBuffer.setLength(0)
                                                        lastUiUpdate = now
                                                    } else {
                                                        synthReasoningBuffer.setLength(0)
                                                        lastReasoningUiUpdate = now
                                                    }
                                                    withContext(Dispatchers.Main) {
                                                        if (isThinkingOff) {
                                                            messageAdapter.appendTokenToLastMessage(chunk)
                                                        } else {
                                                            needsSynthReasoningSeparator = appendSeparatedReasoningChunk(chunk, needsSynthReasoningSeparator)
                                                            needsReasoningSeparatorPending = needsSynthReasoningSeparator
                                                        }
                                                        val passElapsedMs = (now - activePassStartMs).coerceAtLeast(1L)
                                                        val speed = activePassTokens / (passElapsedMs / 1000.0)
                                                        val totalLiveTokens = accumulatedConfirmedTokens + activePassTokens
                                                        val totalActiveSec = (accumulatedGenMs + passElapsedMs) / 1000.0
                                                        val speedText = getString(R.string.info_speed_format, speed)
                                                        lastDisplayedSpeedText = speedText
                                                        if (settingsManager.showStats)
                                                            speedTv.text = speedText
                                                        if (settingsManager.showTokenCount)
                                                            tokenCountTv.text = getString(R.string.stats_tokens_format, totalLiveTokens)
                                                        if (settingsManager.showGenTime)
                                                            genTimeTv.text = getString(R.string.stats_time_format, totalActiveSec)
                                                    }
                                                }
                                            }
                                            is com.arm.aichat.GenerationEvent.Finished -> {
                                                if (event.generatedTokens > 0) {
                                                    activePassFinishedTokens = event.generatedTokens
                                                }
                                                Log.i(TAG, "Synthesis generation finished event: reason=${event.reason}, genTokens=${event.generatedTokens}")
                                            }
                                            is com.arm.aichat.GenerationEvent.Error -> {
                                                Log.e(TAG, "Synthesis generation error event: ${event.message}")
                                                interactiveStreamError = event.message
                                                val errNotice = "\n⚠️ ${event.message}"
                                                synthUiBuffer.append(errNotice)
                                                fullResponseBuilder.append(errNotice)
                                            }
                                            else -> {}
                                        }
                                    }

                                if (activePassStartMs > 0L) {
                                    accumulatedGenMs += (System.currentTimeMillis() - activePassStartMs).coerceAtLeast(1L)
                                    activePassStartMs = 0L
                                }
                                val resolvedSynthTokens = activePassFinishedTokens?.takeIf { it > 0 } ?: activePassTokens
                                accumulatedConfirmedTokens += resolvedSynthTokens
                                activePassTokens = 0
                                activePassFinishedTokens = null

                                if (synthReasoningBuffer.isNotEmpty()) {
                                    val chunk = synthReasoningBuffer.toString()
                                    synthReasoningBuffer.setLength(0)
                                    withContext(Dispatchers.Main) {
                                        needsSynthReasoningSeparator = appendSeparatedReasoningChunk(chunk, needsSynthReasoningSeparator)
                                    }
                                }
                                if (synthUiBuffer.isNotEmpty()) {
                                    val chunk = synthUiBuffer.toString()
                                    synthUiBuffer.setLength(0)
                                    withContext(Dispatchers.Main) {
                                        messageAdapter.appendTokenToLastMessage(chunk)
                                    }
                                }
                            }

                            val lastAssistantFin = messages.lastOrNull { !it.isUser }
                            if (lastAssistantFin != null) {
                                extractInlineThoughtsFromMessage(lastAssistantFin)
                                var cleaned = com.example.llama.agent.tools.web.WebToolsDispatcher.cleanToolCallMarkup(lastAssistantFin.content).trim()
                                if (cleaned.isBlank()) {
                                    val extractedFromCoT = if (hasExecutedTool || lastAssistantFin.reasoning.isNotBlank()) {
                                        com.example.llama.agent.tools.web.WebToolsDispatcher.extractAnswerFromUnclosedReasoning(lastAssistantFin.reasoning)
                                    } else {
                                        null
                                    }
                                    if (extractedFromCoT != null) {
                                        withContext(Dispatchers.Main) {
                                            lastAssistantFin.reasoning = extractedFromCoT.first
                                            lastAssistantFin.content = extractedFromCoT.second
                                            val idx = messages.indexOf(lastAssistantFin)
                                            if (idx != -1) messageAdapter.notifyItemChanged(idx)
                                        }
                                        cleaned = extractedFromCoT.second
                                    } else if (hasExecutedTool && lastAssistantFin.content.trim().isNotEmpty()) {
                                        withContext(Dispatchers.Main) {
                                            lastAssistantFin.content = ""
                                            val idx = messages.indexOf(lastAssistantFin)
                                            if (idx != -1) messageAdapter.notifyItemChanged(idx)
                                        }
                                    }
                                } else if (cleaned != lastAssistantFin.content.trim()) {
                                    withContext(Dispatchers.Main) {
                                        lastAssistantFin.content = cleaned
                                        val idx = messages.indexOf(lastAssistantFin)
                                        if (idx != -1) messageAdapter.notifyItemChanged(idx)
                                    }
                                }
                            }

                            if (settingsManager.webDisplayMode == SettingsManager.WebDisplayMode.PIP && pipTurnActive) {
                                val totalPipElapsedMs = (System.currentTimeMillis() - pipTurnStartMs).coerceAtLeast(1L)
                                withContext(Dispatchers.Main) {
                                    completePipWebActivity(
                                        elapsedMs = totalPipElapsedMs,
                                        sourceCount = pipTurnTotalSources,
                                        seedInput = pipTurnSeedInput
                                    )
                                }
                            }

                            val fullResponse = fullResponseBuilder.toString()
                            Log.i(TAG, "Generation complete. Total length: ${fullResponse.length}")
                            val rawRestoreTp = settingsManager.thinkingProfile
                            val restoreTp = (resolvedAdapterForRestore as? com.example.llama.agent.adapter.ModelAgentController)
                                ?.resolveEffectiveThinkingProfile(rawRestoreTp)
                                ?: rawRestoreTp
                            orchestratorEngine.configureThinking(restoreTp.isEnabled, restoreTp.maxTokens, restoreTp.effort)
                        }
                    } catch (e: kotlinx.coroutines.CancellationException) {
                        Log.i(TAG, "Generation cancelled: ${e.message}")
                        throw e
                    } catch (e: Exception) {
                        Log.e(TAG, "Generation Error", e)
                    } finally {
                        generationJob = null
                        if (activePassStartMs > 0L) {
                            accumulatedGenMs += (System.currentTimeMillis() - activePassStartMs).coerceAtLeast(1L)
                            activePassStartMs = 0L
                        }
                        val unclosedTokens = activePassFinishedTokens?.takeIf { it > 0 } ?: activePassTokens
                        if (unclosedTokens > 0) {
                            accumulatedConfirmedTokens += unclosedTokens
                            activePassTokens = 0
                            activePassFinishedTokens = null
                        }
                        try {
                            if (::engine.isInitialized) {
                                engine.setGrammar(null)
                                val rawRestoreTp = settingsManager.thinkingProfile
                                val restoreTp = (resolvedAdapterForRestore as? com.example.llama.agent.adapter.ModelAgentController)
                                    ?.resolveEffectiveThinkingProfile(rawRestoreTp)
                                    ?: rawRestoreTp
                                engine.configureThinking(restoreTp.isEnabled, restoreTp.maxTokens, restoreTp.effort)
                            }
                        } catch (_: Exception) {}
                        withContext(Dispatchers.Main + kotlinx.coroutines.NonCancellable) {
                            hideSubagentWorkingUi()
                            if (pipWebView != null && !isPipOfflineLocked) {
                                destroyPipWebView()
                            }
                            if (accumulatedConfirmedTokens > 0 && accumulatedGenMs > 0L) {
                                val finalElapsedSec = accumulatedGenMs / 1000.0
                                val finalSpeed = accumulatedConfirmedTokens / finalElapsedSec
                                if (settingsManager.showStats) {
                                    speedTv.text = getString(R.string.info_speed_format, finalSpeed)
                                }
                                if (settingsManager.showTokenCount) {
                                    tokenCountTv.text = getString(R.string.stats_tokens_format, accumulatedConfirmedTokens)
                                }
                                if (settingsManager.showGenTime) {
                                    genTimeTv.text = getString(R.string.stats_time_format, finalElapsedSec)
                                }
                            }
                            if (reasoningBuffer.isNotEmpty()) {
                                val rChunk = reasoningBuffer.toString()
                                reasoningBuffer.setLength(0)
                                needsReasoningSeparatorPending = appendSeparatedReasoningChunk(rChunk, needsReasoningSeparatorPending)
                            }
                            if (uiBuffer.isNotEmpty()) {
                                val cChunk = uiBuffer.toString()
                                uiBuffer.setLength(0)
                                messageAdapter.appendTokenToLastMessage(cChunk)
                            }
                            if (messages.isNotEmpty()) {
                                messageAdapter.setMessageStreaming(messages.last().id, false)
                                val lastAssistant = messages.lastOrNull { !it.isUser }
                                if (lastAssistant != null) {
                                    extractInlineThoughtsFromMessage(lastAssistant)
                                    val (extractedContent, extractedThought) = com.example.llama.agent.tools.web.WebToolsDispatcher.extractCleanContentAndReasoning(lastAssistant.content)
                                    if (extractedContent.isNotBlank() || extractedThought != null || lastAssistant.reasoning.isNotBlank()) {
                                        lastAssistant.content = extractedContent
                                    }
                                    if (extractedThought != null && extractedThought.isNotBlank()) {
                                        if (lastAssistant.reasoning.isBlank()) {
                                            lastAssistant.reasoning = extractedThought
                                        } else if (!lastAssistant.reasoning.contains(extractedThought)) {
                                            lastAssistant.reasoning += "\n\n" + extractedThought
                                        }
                                    } else if (lastAssistant.reasoning.isNotBlank()) {
                                        lastAssistant.reasoning = com.example.llama.agent.tools.web.WebToolsDispatcher.cleanToolCallMarkup(lastAssistant.reasoning)
                                    }
                                    // Záchrana prázdného obsahu v UI: Pokud content zůstal prázdný a neběžely nástroje, oddělíme odpověď jen pokud myšlenky zůstanou zachovány
                                    if (lastAssistant.content.isBlank() && lastAssistant.reasoning.isNotBlank()) {
                                        val rText = lastAssistant.reasoning.trim()
                                        val extracted = if (!hasExecutedTool) {
                                            com.example.llama.agent.tools.web.WebToolsDispatcher.extractAnswerFromUnclosedReasoning(rText)
                                        } else {
                                            null
                                        }
                                        if (extracted != null && extracted.first.isNotBlank()) {
                                            lastAssistant.reasoning = extracted.first
                                            lastAssistant.content = extracted.second
                                            Log.i(TAG, "Split unclosed reasoning into thought (${extracted.first.length} chars) and answer (${extracted.second.length} chars)")
                                        } else {
                                            // Keep unclosed thought intact in reasoning without truncating or erasing it
                                            Log.w(TAG, "⚠️ Final check: unclosed reasoning block (${rText.length} chars); keeping intact in reasoning.")
                                            lastAssistant.reasoning = rText
                                        }
                                    }
                                    // Preserve streamed thoughts visibly unless the user explicitly collapsed them
                                    val hasVisibleThoughts = lastAssistant.reasoning.isNotBlank() &&
                                        (settingsManager.showThoughtInProgress || lastAssistant.content.isBlank() || settingsManager.thinkingProfile != ThinkingProfile.OFF)
                                    if (hasVisibleThoughts && !lastAssistant.isThoughtsCollapsedByUser) {
                                        lastAssistant.isThoughtsExpanded = true
                                    } else if (lastAssistant.isThoughtsCollapsedByUser || lastAssistant.reasoning.isBlank()) {
                                        lastAssistant.isThoughtsExpanded = false
                                    }
                                    val idx = messages.indexOf(lastAssistant)
                                    if (idx != -1) messageAdapter.notifyItemChanged(idx)
                                }
                            }
                            sessionManager.saveCurrentState()
                            updateSendButtonState()
                            if (!settingsManager.disableAutoScroll && messages.isNotEmpty()) {
                                messagesRv.post {
                                    val count = messageAdapter.itemCount
                                    if (count > 0) messagesRv.smoothScrollToPosition(count - 1)
                                }
                            }
                        }
                    }
                }
        updateSendButtonState()
    }

    private fun handleSendButtonClick() {
        val isGeneratingViaService =
                serviceManager.generationState.value is GenerationState.Generating
        val isGeneratingInteractive = generationJob?.isActive == true

        if (isGeneratingViaService || isGeneratingInteractive) {
            stopGeneration()
        } else {
            handleUserInput()
        }
    }



    fun stopGeneration() {
        val wasGeneratingViaService =
                serviceManager.generationState.value is GenerationState.Generating
        val wasGeneratingInteractive = generationJob?.isActive == true
        val wasGenerating = wasGeneratingViaService || wasGeneratingInteractive

        if (wasGeneratingViaService) {
            serviceManager.stopGeneration()
        }
        generationJob?.cancel()
        generationJob = null
        if (::engine.isInitialized) {
            engine.cancelGeneration()
        }
        updateSendButtonState()
        if (wasGenerating) {
            Toast.makeText(this, getString(R.string.msg_generation_stopped), Toast.LENGTH_SHORT).show()
        }
    }

    private fun handleEjectModel() {
        if (!::engine.isInitialized || !engine.state.value.isModelLoaded) {
            Toast.makeText(this, "No model loaded to eject.", Toast.LENGTH_SHORT).show()
            return
        }

        stopGeneration()
        val wasRemote = settingsManager.isRemoteInferenceActive
        lifecycleScope.launch {
            val intent =
                    Intent(this@MainActivity, LlamaGenerationService::class.java).apply {
                        action = LlamaNotificationHelper.ACTION_RELEASE_RAM
                    }
            startService(intent)

            withContext(Dispatchers.IO) {
                try {
                    engine.cancelGeneration()
                } catch (_: Throwable) {}
                engine.cleanUp()
                com.example.llama.agent.core.EngineOrchestrator.resetConfiguration()
                if (wasRemote) {
                    settingsManager.aiProviderType = SettingsManager.AiProviderType.LOCAL_OFFLINE
                    com.example.llama.agent.core.EngineProvider.syncWithSettings(applicationContext)
                }
            }

            withContext(Dispatchers.Main) {
                hasPendingImage = false
                currentModelPath = null
                currentProjectorPath = null
                lastAppliedContextSizeSetting = null
                currentModelArchitecture = null
                currentModelChatTemplate = null
                isSessionSystemPromptSet = false
                if (wasRemote) {
                    syncCloudApiModeUiOnResume()
                    val btnQuickBackend: com.google.android.material.button.MaterialButton? = findViewById(R.id.btn_quick_backend)
                    btnQuickBackend?.let { updateQuickBackendButton(it, settingsManager.llamaBackend) }
                }
                ggufTv.text = "Model Ejected"
                userInputEt.hint = getString(R.string.user_input_hint_no_model)
                updateSendButtonState()
                Toast.makeText(
                                this@MainActivity,
                                "Model ejected & RAM released.",
                                Toast.LENGTH_SHORT
                        )
                        .show()
            }
        }
    }

    private fun applyDisplaySettings() {
        val keepStats = settingsManager.keepStatsVisible
        speedTv.visibility = if (settingsManager.showStats || keepStats) View.VISIBLE else View.GONE
        if (keepStats && !settingsManager.showStats) {
            val am = getSystemService(android.content.Context.ACTIVITY_SERVICE) as? android.app.ActivityManager
            val mi = android.app.ActivityManager.MemoryInfo()
            am?.getMemoryInfo(mi)
            val freeMb = mi.availMem / (1024L * 1024L)
            speedTv.text = "RAM: ${freeMb}MB volná"
        }
        tokenCountTv.visibility = if (settingsManager.showTokenCount) View.VISIBLE else View.GONE
        genTimeTv.visibility = if (settingsManager.showGenTime) View.VISIBLE else View.GONE
        val indicatorMode = settingsManager.subagentIndicatorMode
        if (indicatorMode != SettingsManager.SubagentIndicatorMode.TOP_BADGE &&
            indicatorMode != SettingsManager.SubagentIndicatorMode.BOTH
        ) {
            subagentIndicatorTv.visibility = View.GONE
        }

        // Enable Mic if Agent Enabled OR STT Enabled OR user showMicrophone flag enabled
        val isStt = settingsManager.isSttEnabled
        val isAgent = settingsManager.isAgentEnabled
        val showMic = isAgent || isStt || settingsManager.showMicrophone

        val micButton = findViewById<ImageButton>(R.id.mic_button)
        micButton.visibility = if (showMic) View.VISIBLE else View.GONE

        micButton.setOnClickListener {
            val pm = permissionManager
            pm.checkAndRequestAudioPermission { granted: Boolean ->
                if (granted) {
                    // INTEGRACE: Pokud je Agent zapnutý
                    if (settingsManager.isAgentEnabled) {
                        val voiceMode = settingsManager.agentVoiceMode

                        if (voiceMode == SettingsManager.VOICEMODE_PUSH_TO_TALK) {
                            // Push to Talk -> Aktivovat Agent Voice Command
                            val agentService =
                                    com.example.llama.agent.service.AiAgentAccessibilityService
                                            .getInstance()
                            if (agentService != null) {
                                if (agentService.isVoiceSpeaking()) {
                                    agentService.deactivateAgent("User interrupted speaking")
                                } else {
                                    agentService.startVoiceCommand()
                                    Toast.makeText(this, "Poslouchám...", Toast.LENGTH_SHORT).show()
                                }
                            } else {
                                // Fallback/Error state
                                Toast.makeText(
                                                this,
                                                "Agent Service není aktivní",
                                                Toast.LENGTH_SHORT
                                        )
                                        .show()
                            }
                        } else {
                            // Open Mic -> Button click might Toggle mute/unmute or restart
                            // listening?
                            // Prozatím restart poslechu nebo vizuální feedback
                            Toast.makeText(
                                            this,
                                            "Režim Open Mic je aktivní (Stále poslouchám)",
                                            Toast.LENGTH_SHORT
                                    )
                                    .show()
                            // Optional: Restart listening if stuck
                            voiceInputManager.startContinuousListening()
                        }
                        return@checkAndRequestAudioPermission
                    }

                    // Standardní diktování (Chat) - Fallback if Agent OFF but Mic clicked (e.g. STT
                    // enabled)
                    if (voiceInputManager.isListening()) {
                        voiceInputManager.stopListening()
                    } else {
                        voiceInputManager.startListening()
                    }
                }
            }
        }

        if (settingsManager.alwaysShowSidebar) {
            drawerLayout.setDrawerLockMode(DrawerLayout.LOCK_MODE_LOCKED_OPEN)
            drawerLayout.setScrimColor(android.graphics.Color.TRANSPARENT)
        } else {
            drawerLayout.setDrawerLockMode(DrawerLayout.LOCK_MODE_UNLOCKED)
        }

        val windowInsetsController = WindowCompat.getInsetsController(window, window.decorView)
        windowInsetsController.systemBarsBehavior =
                WindowInsetsControllerCompat.BEHAVIOR_SHOW_TRANSIENT_BARS_BY_SWIPE
        if (settingsManager.hideStatusBar) {
            windowInsetsController.hide(WindowInsetsCompat.Type.statusBars())
        } else {
            windowInsetsController.show(WindowInsetsCompat.Type.statusBars())
        }

        // Initialize stats text only if not already displaying valid metrics
        if (settingsManager.showStats && (speedTv.text.isNullOrBlank() || speedTv.text == "..")) {
            speedTv.text = getString(R.string.info_speed_format, 0.0)
        }
        if (settingsManager.showTokenCount && (tokenCountTv.text.isNullOrBlank() || tokenCountTv.text == "..")) {
            tokenCountTv.text = getString(R.string.stats_tokens_format, 0)
        }
        if (settingsManager.showGenTime && (genTimeTv.text.isNullOrBlank() || genTimeTv.text == "..")) {
            genTimeTv.text = getString(R.string.stats_time_format, 0.0)
        }

        messageAdapter.renderMarkdown = settingsManager.renderMarkdown
        messageAdapter.enableBubbleCopy = settingsManager.enableBubbleCopy
        messageAdapter.enableBubbleTranslate = settingsManager.enableBubbleTranslate
        messageAdapter.chatStyle = settingsManager.chatStyle
        messageAdapter.fontSizeSp = settingsManager.chatFontSizeSp
        messageAdapter.showThoughts = settingsManager.showThoughtInProgress
        val resolvedTpForDisplay = resolveActiveModelController()
            ?.resolveEffectiveThinkingProfile(settingsManager.thinkingProfile)
            ?: resolveActiveThinkingSupport().resolveEffectiveProfile(settingsManager.thinkingProfile)
        messageAdapter.thinkingProfile = resolvedTpForDisplay

        messageAdapter.isTtsEnabled = settingsManager.isTtsEnabled
        if (messages.isNotEmpty()) {
            messageAdapter.notifyItemRangeChanged(0, messages.size)
        }

        ThemeManager.applyTheme(this, settingsManager.chatStyle)
    }

    private fun openSettings() {
        startActivity(Intent(this, SettingsActivity::class.java))
    }

    private fun handleSelectedModelImport(uri: Uri) {
        Toast.makeText(this, getString(R.string.msg_import_wait), Toast.LENGTH_LONG).show()
        viewModel.importAndLoadModel(
                uri = uri,
                onPathReady = { nativePath ->
                    lifecycleScope.launch { loadModelToEngine(nativePath) }
                },
                onError = { error ->
                    val errText = getString(R.string.msg_import_failed, error)
                    Toast.makeText(this, errText, Toast.LENGTH_LONG).show()
                    ggufTv.text = getString(R.string.status_error_prefix, error)
                }
        )
    }

    override fun onStop() {
        super.onStop()
        destroyPipWebView()
        if (!settingsManager.backgroundGeneration) {
            generationJob?.cancel()
        }
    }

    override fun onDestroy() {
        com.example.llama.agent.tools.web.AutonomousBrowserSubagent.pipController = null
        destroyPipWebView()
        if (::voiceInputManager.isInitialized) {
            voiceInputManager.shutdown()
        }
        val agentService = com.example.llama.agent.service.AiAgentAccessibilityService.getInstance()
        agentService?.deactivateAgent("App destroyed")

        messageAdapter.cleanup()
        if (isFinishing) {
            Log.i(TAG, "Activity finishing - Killing service.")
            serviceManager.stopService()
        } else {
            Log.i(TAG, "Rotation - Detaching only.")
            serviceManager.disconnect()
        }
        super.onDestroy()
    }

    companion object {
        private val TAG = MainActivity::class.java.simpleName
    }
}
