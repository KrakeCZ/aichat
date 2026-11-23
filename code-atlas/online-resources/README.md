# Offline prostředky Atlasu — lokální úložiště

Výchozí analýza nikdy sama nekontaktuje internet. Tento adresář patří Atlasu, nikoli analyzovanému projektu. Žádné procházení systémových disků.

- `manifest.json`: seznam importovaných artefaktů s SHA-256.
- `blobs/<první dva hex znaky>/<SHA256>`: jedna trvalá obsahově adresovaná kopie.
- `inbox/`: uživatel sem může uložit veřejné .js/.css před explicitním importem; automaticky se neskýtuje ani nespouští.
- `python atlas_resources.py catalog` / `instructions`: oficiální odkazy, přesná verze a kam zkopírovat.
- `pack-create --output USB.zip`: ověřený export lokálních souborů.
- `pack-install --file USB.zip`: ověřený import bez internetu, bez extrakce cizích cest.
- `fetch-graphics --name FILENAME --allow-network`: **výhradně vědomé** stažení jednoho veřejného verzovaného JS/CSS; nikdy se nevolá automaticky při analýze.

Nemáme žádný remote registry pro soukromé repozitáře. Kontrolní součet není kryptografický podpis vydavatele. Po importu musí uživatel respektovat upstream licence. Pro hlubší sémantiku lze časem přidat samostatné kompatibilní offline packs.

## Stage 19: ručně stažené npm balíčky

Nově lze přímo importovat původní `.tgz` distribuce bez spouštění npm či projektových skriptů:

`python atlas_resources.py npm-packages` — devět oficiálních verzovaných archivů pro deset grafických souborů.

`python atlas_resources.py import-npm --file online-resources/inbox/BALIK.tgz --integrity sha512-DUVERYHODNY_HASH` — ověřený import; nebo výslovné `--accept-unverified-origin` s neověřeným původem.

Původní licenční texty jsou v `online-resources/licenses/`; přenos přes `pack-create`/`pack-install` je zachová. Žádné automatické hledání archivek ani stažení ze sítě neprobíhá.

## Stage 20: jedno kliknutí z místního správce

Z rozbalené složky Atlasu dvojklikem spustit `SPRAVCE_GRAFIKY_WINDOWS.bat` nebo `python atlas_resources.py manager`. Prohlížeč zobrazí „Stáhnout vše a nainstalovat“. Získává pouze devět pevně připnutých veřejných npm archivů, a to až po potvrzení. Vlastní archivy se mohou uchovat v `packages/`, potřebné `.js/.css` zůstávají v obsahově adresovaném `blobs/` a jejich SHA-256 v `manifest.json`. Ani tento příkaz nikdy nečte soukromé projektové soubory.

Pozor: tato distribuce neobsahuje originální npm knihovny; jejich získání na cílovém počítači dosud nebylo ověřeno v tomto prostředí. SHA-512 z HTTPS npm metadat není nezávislý kryptografický podpis autorů.
