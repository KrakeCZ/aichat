# ADB Hardware Monitor

Český dashboard pro ladění lokální inference na Androidu. Sběr používá běžné
ADB příkazy a funguje bez rootu. Konkrétní hodnoty závisejí na tom, co zpřístupní
firmware telefonu a jaké události pošle laděná aplikace.

## Spuštění ve Windows

1. Rozbal archiv a otevři **Spustit.cmd**. Potřebuješ
   [Python 3.10+ s Tcl/Tk](https://www.python.org/downloads/windows/) a
   [Android SDK Platform-Tools](https://developer.android.com/tools/releases/platform-tools).
2. Zapni ladění USB, připoj telefon a potvrď dialog s otiskem klíče.
   Pokud se ADB nenajde, použij **Vybrat ADB…** a otevři `adb.exe`.
3. Vyber zařízení, zadej název procesu a klikni na **Spustit**. Výchozí proces
   je `aichat.com.llama.example`. Pro oddělený proces lze zadat například
   `aichat.com.llama.example:llm`. Prázdné pole ponechá jen hardwarová měření.
4. Zapni workload až po zobrazení PID a informace o ADBTelemetry na kartě
   **Diagnostika**. První výřez logu při nalezení PID určí začátek měření.
5. **Vybrat backendy…** a **Vybrat teploty…** upravují dashboard i za běhu.
   Kliknutím na backend v tabulce přepneš graf latence. V teplotách vybereš
   až šest křivek, včetně konkrétních zón nalezených během sběru.
6. **Demo** funguje bez telefonu. **Export CSV…** ukládá vzorky celé relace,
   i když je některý signál na dashboardu skrytý.

Z PowerShellu lze spustit `py -3 .\AdbHardwareMonitor.pyw`. Na Linuxu použij
`python3 AdbHardwareMonitor.pyw` s Tkinterem. Skript přijímá `--demo` a
`--adb cesta`. Nepotřebuje další balíčky Pythonu.

## Jak číst dashboard

| Údaj | Co z něj zjistíš |
| --- | --- |
| CPU zařízení a jednotlivých jader | Aktivita v procentech z rozdílů `/proc/stat`, takt z čitelného `scaling_cur_freq`. První vzorek pracuje jako výchozí bod. |
| CPU procesu | CPU čas vybraného PID jako podíl kapacity všech CPU. Vyžaduje čitelné `/proc/PID/stat` a `/proc/stat`. Na osmijádru odpovídá jedno plně vytížené jádro přibližně 12,5 %. Měří celý proces včetně obsluhy GPU, UI a pre/postprocessingu. |
| Adreno GPU | Celková zátěž a takt z čitelných uzlů KGSL. Hodnoty zahrnují všechny klienty GPU včetně UI. |
| CPU/GPU/NPU/CDSP a baterie | Nejvyšší čitelná teplota podle kategorie. Individuální zóny mají názvy z telefonu. Zóna může patřit celému clusteru. Tepelný stav 0–6 hlásí Android za zařízení. |
| NPU a DSP takt | Čitelné domény devfreq odpovídající názvům NPU, NSP či CDSP. Frekvence ukazuje stav domény zařízení. Procento vytížení NPU ani jejího klienta z ní neurčíš. |
| Knihovna načtena | Nález známé knihovny v mapách vybraného procesu. Mapy se čtou přes `run-as`, obvykle v debuggable buildu. EGL/GLES může používat UI a FastRPC může posloužit více DSP klientům. |
| Odesláno · čeká | Aplikace hlásí začátek nebo heartbeat operace. Dashboard čeká na dokončení, chybu nebo fallback. |
| Dokončeno · runtime | Aplikace hlásí, že zvolený backend dokončil operaci. Přesnost závisí na tom, kam událost umístíš. |
| Dokončeno · profil | Aplikace deklaruje, že dokončení konkrétní operace potvrzuje profil backendu. Dashboard profil sám nečte. |
| Fallback / Chyba | Aplikace oznámila změnu backendu nebo neúspěch. Cílový backend hlásí vlastní start a dokončení. |
| Paměť procesu | PSS kategorií `Gfx dev`, `EGL mtrack`, `GL mtrack` a `Native Heap` v MiB. Jednotlivé řádky nelze přiřadit ke konkrétnímu výpočetnímu API. |

Graf nedostupné hodnoty vynechá a v CSV pro ni nevznikne řádek. Na kartě
**Diagnostika** uvidíš dostupnost zdrojů právě připojeného telefonu. Události
se čtou přibližně po 0,75 s přes krátký výřez logcatu vybraného PID. Při častém
logování mohou rychlé události z výřezu vypadnout. Pro inference loguj dávku nebo
celý požadavek, případně občasný heartbeat během dlouhého běhu. CSV obsahuje
poslední zachycenou dobu operace pro backend a počet událostí každého druhu
za vzorek. HW se čte podle zvoleného intervalu, teploty nejdříve po 2 s a
paměť, mapy a procesní CPU nejdříve po 3 s.

## Zapojení do aplikace

Přiložený `AdbTelemetry.kt` vlož do debug kódu aplikace a doplň deklaraci
balíčku podle projektu. Zapoj ho u skutečné implementace backendu. Když
dispatch probíhá v JNI/C++, značku můžeš emitovat i přímo tam se stejným
formátem. Pro běžné release sestavení události vypni.

1. Voláním `AdbTelemetry.start(AdbTelemetry.Backend.QNN_HTP)` získáš objekt
   operace. Udělej to až po skutečném výběru backendu, těsně před odesláním.
2. Po úspěšném dokončení zavolej `operation.done()`. U asynchronních API
   čekej na fence/event nebo hotový výsledek. Tím běžně měříš latenci z pohledu
   aplikace včetně čekání. Čistý HW čas můžeš místo toho získat z časových
   razítek daného API a poslat ručně.
3. Pokud runtime použije CPU, nejdřív označ `operation.fallback(AdbTelemetry.Backend.CPU)`
   a pro CPU vytvoř vlastní události. Pokud běh skončí chybou, volej `operation.error()`.
   U dlouhé operace můžeš přibližně každých 10 s volat `operation.beat()`.
   Bez průběžné události nejpozději po 45 s zmizí označení čekající operace.
4. `operation.done(profileConfirmed = true)` používej pouze tehdy, když máš
   pro tutéž operaci skutečnou exekuční událost profilu QNN/HTP.

Příklad zpráv pro `Log.i("ADBTelemetry", zpráva)`:

```text
backend=QNN-HTP event=start id=17
backend=QNN-HTP event=done id=17 proof=profile ms=32.617

backend=QNN-HTP event=start id=18
backend=QNN-HTP event=fallback id=18 to=CPU
backend=CPU event=start id=19
backend=CPU event=done id=19 proof=runtime ms=84.502
```

Seznam platných jmen: `CPU`, `GPU`, `Vulkan`, `OpenCL`, `OpenGL`, `NPU`,
`QNN`, `QNN-HTP`, `QNN-GPU`, `QNN-CPU`, `QNN-DSP`, `DSP`. Používej přesné
jméno skutečně zvoleného backendu. Obecné `QNN` neříká, které zařízení
požadavek vykonalo. HTP je větev pro NPU, DSP/FastRPC je širší pojem. Jednu
operaci na QNN HTP stačí značit `QNN-HTP`. Původní formát
`backend=Vulkan ms=32.617` funguje dál jako měření času bez stavu startu.

Události můžeš nezávisle ověřit v PowerShellu. Zástupný text `SERIAL` nahraď
hodnotou z `adb devices -l`:

```powershell
adb -s SERIAL logcat -b main -v threadtime ADBTelemetry:I '*:S'
```

## Galaxy S23 Ultra a S24 Ultra bez rootu

S23 Ultra má Snapdragon 8 Gen 2 for Galaxy, S24 Ultra Snapdragon 8 Gen 3
for Galaxy. Čitelnost kernelových uzlů otestuj na konkrétním telefonu s jeho
aktuálním firmwarem. Přenosný ADB čítač vytížení Hexagon NPU po procesech
dashboard nepředpokládá.

**Nejspolehlivější test pro NPU:** Pokud běžíš na QNN HTP, pro debug test
zakaž tichý návrat na CPU, pokud má tvůj runtime takovou možnost. U
ONNX Runtime QNN EP dokumentace uvádí `session.disable_cpu_ep_fallback=1`
a HTP profilování `profiling_level=basic` nebo `detailed` s
`profiling_file_path`. U nativního QNN/QAIRT spusť stejný model také s HTP
profilováním v odpovídající verzi SDK. Ověř exekuční událost modelu, ne pouze
inicializaci backendu. Zopakuj stejný prompt a parametry jednou na CPU a
jednou s vynuceným HTP. Grafy telefonu pak pomůžou srovnat dopad, důkaz o
běhu konkrétní operace ber z runtime nebo z profilu.

**GPU a CPU:** Perfetto umí ukázat běh vláken na CPU a podle dostupných
zdrojů i GPU čítače či render/compute události. Pro GPU profilování ověř
kompatibilitu konkrétního ovladače na telefonu. Teplota a zátěž Adrena
samy o sobě nerozliší Vulkan, OpenCL a OpenGL.

**Aplikace při měření:** Pro první srovnání nech debug Activity v popředí.
Když vadí zhasnutí displeje, zapni v ní po dobu testu
`window.addFlags(android.view.WindowManager.LayoutParams.FLAG_KEEP_SCREEN_ON)`.
Zruš jej po skončení. Pro běh se zhasnutým displejem řeš životní cyklus
inference samostatně přes příslušný Android foreground service a pravidla dané
verze Androidu. Stejné nastavení displeje a nabíjení při obou bězích usnadní
srovnání teplot.

## Dokumentace

- [Android ADB](https://developer.android.com/tools/adb) a [logcat](https://developer.android.com/tools/logcat)
- [Android Thermal API](https://developer.android.com/reference/android/os/PowerManager#THERMAL_STATUS_NONE)
- [Qualcomm QAIRT HTP backend](https://docs.qualcomm.com/bundle/publicresource/topics/80-63442-10/htp_backend.html)
- [ONNX Runtime QNN EP](https://onnxruntime.ai/docs/execution-providers/QNN-ExecutionProvider.html)
- [Perfetto GPU](https://perfetto.dev/docs/data-sources/gpu) a [GPU validace zařízení](https://developer.android.com/agi/start)
- [Android – udržení zapnutého displeje](https://developer.android.com/develop/background-work/background-tasks/awake/screen-on)
