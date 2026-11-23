<?php
/**
 * Universal Code Atlas: PHP's built-in token parser, NOT a PHP code runner.
 * Only file_get_contents and token_get_all(TOKEN_PARSE) touch project sources.
 * The target source is never include'd, require'd, eval'd, or executed.
 * No Composer autoloading, no HTTP, no plugins.
 */

declare(strict_types=1);

function atlas_significant(array $tokens, int $start): ?int {
    for ($i = $start, $n = count($tokens); $i < $n; ++$i) {
        if (!in_array($tokens[$i]['id'], [T_WHITESPACE, T_COMMENT, T_DOC_COMMENT], true)) {
            return $i;
        }
    }
    return null;
}

function atlas_quoted(string $text): ?string {
    if (strlen($text) < 2) return null;
    $quote = $text[0];
    if (($quote !== '"' && $quote !== "'") || substr($text, -1) !== $quote) return null;
    $body = substr($text, 1, -1);
    // Reject interpolation and escapes; parse only unambiguous literal paths.
    if (strpbrk($body, "\\\\$") !== false || str_contains($body, "\0")) return null;
    return $body;
}

function atlas_include(array $tokens, int $index): ?array {
    $start = $index;
    $parts = [];
    $count = count($tokens);
    $parentheses = 0;
    for ($i = $index + 1; $i < $count; ++$i) {
        $tok = $tokens[$i];
        if (in_array($tok['id'], [T_WHITESPACE,T_COMMENT,T_DOC_COMMENT], true)) continue;
        if ($tok['text'] === '(') { ++$parentheses; continue; }
        if ($tok['text'] === ')') { if ($parentheses === 0) return null; --$parentheses; continue; }
        if ($tok['text'] === ';' && $parentheses === 0) break;
        if ($tok['id'] === T_DIR) { $parts[] = ['dir', '']; continue; }
        if ($tok['id'] === T_CONSTANT_ENCAPSED_STRING) {
            $value = atlas_quoted($tok['text']);
            if ($value === null) return null;
            $parts[] = ['text', $value]; continue;
        }
        if ($tok['text'] === '.') { $parts[] = ['dot', '']; continue; }
        return null; // variables, function calls and complex expressions are unknown
    }
    if ($parentheses !== 0 || !$parts || count($parts) % 2 === 0) return null;
    foreach ($parts as $position => $piece) {
        if ($position % 2 && $piece[0] !== 'dot') return null;
        if ($position % 2 === 0 && $piece[0] === 'dot') return null;
    }
    $isDir = $parts[0][0] === 'dir';
    if (!$isDir && count(array_filter($parts, fn($x) => $x[0] === 'dir')) > 0) return null;
    $joined = '';
    foreach ($parts as $piece) if ($piece[0] === 'text') $joined .= $piece[1];
    if ($joined === '' || strlen($joined) > 400) return null;
    return ['value' => $joined, 'base' => $isDir ? '__DIR__' : 'php_include_path',
            'line' => $tokens[$start]['line'], 'column' => $tokens[$start]['column']];
}

function atlas_parse(string $source): array {
    $tokens = [];
    $line = 1;
    $column = 1;
    $raw = token_get_all($source, TOKEN_PARSE); // syntax validation only
    if (count($raw) > 250000) throw new RuntimeException('token_limit');
    foreach ($raw as $item) {
        $id = is_array($item) ? $item[0] : null;
        $text = is_array($item) ? $item[1] : $item;
        $tokens[] = ['id' => $id, 'text' => $text, 'line' => $line, 'column' => $column];
        $breaks = substr_count($text, "\n");
        if ($breaks) { $line += $breaks; $column = strlen($text) - (strrpos($text, "\n") + 1) + 1; }
        else { $column += strlen($text); }
    }
    $nodes = []; $includes = [];
    $namespace = '';
    $classScopes = []; $namespaceScopes = []; $pending = null; $depth = 0;
    foreach ($tokens as $i => $tok) {
        $id = $tok['id']; $value = $tok['text'];
        if ($id === T_NAMESPACE) {
            $next = atlas_significant($tokens, $i + 1);
            $name = '';
            if ($next !== null) {
                for ($j = $next; $j < count($tokens); ++$j) {
                    $t = $tokens[$j];
                    if ($t['text'] === '{' || $t['text'] === ';') break;
                    if (in_array($t['id'], [T_STRING, T_NAME_QUALIFIED, T_NS_SEPARATOR, T_NAME_FULLY_QUALIFIED], true)) $name .= $t['text'];
                    elseif ($t['id'] !== T_WHITESPACE && $t['id'] !== T_COMMENT) { $name = ''; break; }
                }
                $pending = ['namespace', trim($name,'\\')];
                if (isset($tokens[$j]) && $tokens[$j]['text'] === ';') { $namespace = trim($name,'\\'); $pending = null; }
            }
        }
        if (in_array($id, [T_CLASS,T_INTERFACE,T_TRAIT,T_ENUM],true)) {
            $next = atlas_significant($tokens, $i + 1);
            if ($next !== null && $tokens[$next]['id'] === T_STRING) {
                $name = $tokens[$next]['text'];
                $qualified = ($namespace === '' ? '' : $namespace.'\\').$name;
                $kind = match ($id) { T_CLASS => 'class',T_INTERFACE => 'interface',T_TRAIT => 'trait',default => 'enum' };
                $nodes[] = ['kind'=>$kind,'name'=>$name,'qual'=>$qualified,'parent'=>'',
                            'line'=>$tokens[$next]['line'],'column'=>$tokens[$next]['column']];
                $pending = ['class', $qualified];
            }
        }
        if ($id === T_FUNCTION) {
            $next = atlas_significant($tokens, $i + 1);
            if ($next !== null && trim($tokens[$next]['text']) === '&') $next = atlas_significant($tokens, $next + 1);
            if ($next !== null && $tokens[$next]['id'] === T_STRING) {
                $name = $tokens[$next]['text'];
                $class = $classScopes ? end($classScopes)['name'] : '';
                $kind = $class === '' ? 'function' : 'method';
                $qualified = $class !== '' ? $class.'::'.$name : (($namespace === '' ? '' : $namespace.'\\').$name);
                $nodes[] = ['kind'=>$kind,'name'=>$name,'qual'=>$qualified,'parent'=>$class,
                            'line'=>$tokens[$next]['line'],'column'=>$tokens[$next]['column']];
            }
        }
        if (in_array($id,[T_REQUIRE,T_REQUIRE_ONCE,T_INCLUDE,T_INCLUDE_ONCE],true)) {
            $inc = atlas_include($tokens, $i);
            if ($inc) $includes[] = $inc;
        }
        if ($value === '{') {
            ++$depth;
            if ($pending) {
                if ($pending[0] === 'class') $classScopes[] = ['name'=>$pending[1],'depth'=>$depth];
                if ($pending[0] === 'namespace') {
                    $namespaceScopes[] = ['old'=>$namespace,'depth'=>$depth];
                    $namespace = $pending[1];
                }
                $pending = null;
            }
        } elseif ($value === '}') {
            --$depth;
            while ($classScopes && end($classScopes)['depth'] > $depth) array_pop($classScopes);
            while ($namespaceScopes && end($namespaceScopes)['depth'] > $depth) $namespace = array_pop($namespaceScopes)['old'];
        } elseif ($value === ';' && $pending && $pending[0] === 'class') {
            $pending = null;
        }
    }
    return ['nodes'=>$nodes,'includes'=>$includes];
}

try {
    // Strict JSON over stdin: source is already read and bounded by trusted
    // Python. PHP never accesses repository files and no project code runs.
    $input = stream_get_contents(STDIN, 24000000);
    if ($input === false || strlen($input) > 23999999) throw new RuntimeException('input_limit');
    $manifest = json_decode($input, true, 512, JSON_THROW_ON_ERROR);
    if (!is_array($manifest) || !is_array($manifest['files'] ?? null) || count($manifest['files']) > 300) {
        throw new RuntimeException('invalid_input_manifest');
    }
    $results = [];
    foreach ($manifest['files'] as $entry) {
        if (!is_array($entry) || !is_string($entry['path'] ?? null)
            || !is_string($entry['content_b64'] ?? null)
            || strlen($entry['content_b64']) > 1400000) {
            throw new RuntimeException('invalid_source_record');
        }
        $relative = $entry['path'];
        $source = base64_decode($entry['content_b64'], true);
        if ($source === false || strlen($source) > 1000000) throw new RuntimeException('invalid_source_bytes');
        try {
            $results[] = ['path'=>$relative] + atlas_parse($source);
        } catch (Throwable $e) {
            $results[] = ['path'=>$relative,'error'=>'php_token_parse_failed','detail'=>substr($e->getMessage(),0,160)];
        }
    }
    $json = json_encode($results,JSON_THROW_ON_ERROR|JSON_INVALID_UTF8_SUBSTITUTE|JSON_UNESCAPED_UNICODE);
    if (strlen($json) > 15000000) throw new RuntimeException('token_output_limit');
    echo $json;
} catch (Throwable $e) {
    fwrite(STDERR, 'Atlas PHP tokenizer bridge refused input: '.substr($e->getMessage(),0,140)."\n");
    exit(2);
}
