#!/usr/bin/env python3
"""Open an EXISTING generated Atlas with its source tree, without analysis."""
from pathlib import Path
from launch_atlas import choose_project
from atlas import serve


def main() -> int:
    base=Path(__file__).resolve().parent
    site=base/'.code-atlas'
    if not (site/'index.html').is_file():
        print('Nebyl nalezen hotový výstup .code-atlas/index.html. Nejprve dokonči analýzu projektu.')
        return 1
    project=choose_project()
    if not project:
        print('Nebyl vybrán projekt; zdrojový kód se neprohledával.')
        return 1
    root=Path(project).expanduser()
    if not root.is_dir():
        print('Složka projektu neexistuje.')
        return 1
    try:
        # The user is explicitly opening an already generated site. Repair only
        # the known AVSDF JS loader, without reindexing private source files.
        from atlas_avsdf_hotfix import repair as repair_avsdf
        try:
            changed = repair_avsdf(site)
            if changed:
                print('Opraveno bez nové analýzy:',', '.join(changed))
        except (OSError, ValueError) as issue:
            # Unknown customized JS must not prevent the rest of Atlas opening.
            print('AVSDF automatická oprava nebyla použita:', issue)
        serve(root,site,port=0,open_browser=True)
        return 0
    except (OSError,ValueError) as exc:
        print('Nebylo možné spustit již hotový Atlas:',exc)
        return 1


if __name__=='__main__':
    raise SystemExit(main())
