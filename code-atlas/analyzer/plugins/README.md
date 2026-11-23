# Language plugins (v0.2)

Every analyzer is an isolated Python module. Atlas detects file languages in
`analyzer/inventory.py`, registers applicable plugins in `__init__.py`, then
passes a read-only project path and the mutable graph to the plugin.

To support another language:

1. Register its file extensions and language name in `inventory.py`.
2. Create `<language>_plugin.py`, exporting `PLUGIN` with a unique `name`, a
   `languages` tuple and `analyze(ctx)` method.
3. Use `ctx.files_by_language` to select files, `ctx.read(path)` to read, and
   `ctx.add(source, target, line, snippet, plugin_name, edge_type)` for an
   evidence-backed edge. `target` must exist in `ctx.paths`.
4. Register the plugin in `run_plugins()` and write a tiny fixture test.
5. Keep dependencies completely offline and **do not execute scanned code**.

For pattern matching, prefer conservative resolution. An import of an external
library should **not** create a fake project edge. Mark ambiguous relationships
`ConfidenceLevel.CANDIDATE` to require the user's "inferred" filter.

The core Android/Kotlin/Java/C++/JNI/CMake/Gradle analyzers are intentionally
preserved as built-in passes while the non-Android parsers are now modular.
