"""Offline read-only provider availability report; never executes analyzed projects.

This is a diagnostic, NOT a certification or an install/version guard.
"""
from __future__ import annotations

import json
import shutil
from pathlib import Path

from analyzer.semantic_adapters.kotlin_psi import _locate_toolchain, _locate_bundled_psi, _locate_java
from analyzer.semantic_adapters.clang_ast import _clang_executable
from analyzer.typescript_ast import _node_executable, ADAPTER, COMPILER


def inspect() -> dict:
    bundle_classes, bundle_libraries, bundle_issue = _locate_bundled_psi()
    kotlin, external_java, kotlin_libraries, kotlin_issue = _locate_toolchain()
    java = _locate_java()
    node = _node_executable()
    clang = _clang_executable()
    js_ready = bool(node and ADAPTER.is_file() and COMPILER.is_file())
    kotlin_ready = bool(java and ((bundle_classes and bundle_libraries and not bundle_issue)
                                  or (kotlin and kotlin_libraries and not kotlin_issue)))
    return {
        'schema': 'atlas-offline-analysis-providers/1',
        'source_code_executed': False,
        'network_requested': False,
        'certification': False,
        'adapters': {
            'python_ast': {'available': True, 'provider': 'current Python interpreter'},
            'xml_expat': {'available': True, 'provider': 'Python stdlib Expat'},
            'typescript_ast': {'available': js_ready,
                               'provider': node, 'bundled_compiler': COMPILER.is_file(),
                               'remediation': None if js_ready else 'Set ATLAS_NODE to a local Node.js binary. TypeScript compiler is shipped in vendor.'},
            'kotlin_psi': {'available': kotlin_ready,
                           'provider': ('bundled Kotlin PSI 1.9.0 (Java-only runtime)' if bundle_classes else kotlin), 'java': java,
                           'bundled_psi_verified': bool(bundle_classes),
                           'remediation': (None if kotlin_ready else ('Bundled PSI integrity error: ' + bundle_issue if bundle_issue and 'invalid bundled' in bundle_issue else 'Java runtime not found: set ATLAS_JAVA / ATLAS_JAVA_HOME or use Android Studio JBR.') if bundle_classes else kotlin_issue)},
            'clang_ast': {'available': bool(clang), 'provider': clang,
                          'remediation': None if clang else 'Set ATLAS_CLANG to an installed clang++/clang executable.'},
            'go_ast': {'available': bool(shutil.which('go')),
                       'provider': shutil.which('go')},
            'ruby_ripper': {'available': bool(shutil.which('ruby')),
                            'provider': shutil.which('ruby')},
        },
        'note': 'Provider presence is not proof that a particular source project parsed, or that an adapter is production-complete.'
    }


if __name__ == '__main__':
    print(json.dumps(inspect(), ensure_ascii=False, indent=2))
