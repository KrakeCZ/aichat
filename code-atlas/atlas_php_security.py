#!/usr/bin/env python3
"""Explicit, offline PHP runtime security status. Never reads project sources."""
from __future__ import annotations

import argparse
from pathlib import Path
import subprocess
from analyzer.semantic_adapters.php_tokens import _php_binary, _safe_php_environment, _version_is_approved, SECURITY_ADVISORY


def main(argv=None):
    parser = argparse.ArgumentParser(description='Atlas: explicit PHP runtime security check (no project scan, no network)')
    parser.add_argument('command', choices=('status',))
    parser.add_argument('--php', help='Explicit absolute path to a PHP CLI executable')
    options = parser.parse_args(argv)
    if options.php:
        path = Path(options.php).expanduser()
        binary = str(path) if path.is_absolute() and path.is_file() and not path.is_symlink() else None
    else:
        binary = _php_binary()
    if not binary:
        print('PHP CLI: nebylo nalezeno nebo není povolena zadaná cesta. Atlas PHP adaptér nepotřebuje k základní indexaci.')
        return 2
    try:
        probe = subprocess.run([binary, '-n', '-v'], capture_output=True, text=True,
                               timeout=6, env=_safe_php_environment())
        if probe.returncode:
            print('PHP CLI: selhalo bezpečné zjištění verze.')
            return 2
        approved, version = _version_is_approved(probe.stdout)
    except (OSError, subprocess.TimeoutExpired):
        print('PHP CLI: nejde ověřit verzi nebo proces překročil časový limit.')
        return 2
    print('PHP CLI:',version)
    if approved:
        print('PHP2Shell: splňuje známé minimální upstream verze k 4. 11. 2026.')
        print('Pozor: verze sama nepotvrzuje stav distribučních oprav, externích rozšíření ani OS sandbox.')
        return 0
    print('PHP2Shell: NEPOVOLENO. Použij opravené PHP 8.2.35 / 8.3.36 / 8.4.27 / 8.5.12 nebo vyšší podporované vydání.')
    print('Oficiální zdroj:',SECURITY_ADVISORY)
    return 2


if __name__ == '__main__':
    raise SystemExit(main())
