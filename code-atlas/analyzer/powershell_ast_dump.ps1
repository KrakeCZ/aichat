param([Parameter(Mandatory=$true)][string] $InputPath)
# This helper parses PowerShell syntax, but NEVER executes the target file.
$ErrorActionPreference = 'Stop'
$tok = $null
$parseErrors = $null
$tree = [System.Management.Automation.Language.Parser]::ParseFile($InputPath,[ref]$tok,[ref]$parseErrors)
$functions = @($tree.FindAll({param($n) $n -is [System.Management.Automation.Language.FunctionDefinitionAst]},$true) | ForEach-Object {
    $fn = $_
    $params = @()
    if ($fn.Body -and $fn.Body.ParamBlock) { $params = @($fn.Body.ParamBlock.Parameters | ForEach-Object { $_.Name.VariablePath.UserPath }) }
    [pscustomobject]@{name=$fn.Name;line=$fn.Extent.StartLineNumber;end_line=$fn.Extent.EndLineNumber;parameters=@($params)}
})
$commands = @($tree.FindAll({param($n) $n -is [System.Management.Automation.Language.CommandAst]},$true) | ForEach-Object {
    $cmd = $_
    $name = $cmd.GetCommandName()
    if ($null -ne $name) {
        $cur = $cmd.Parent
        $owner = $null
        while ($cur) {
            if ($cur -is [System.Management.Automation.Language.FunctionDefinitionAst]) { $owner = $cur.Name; break }
            $cur = $cur.Parent
        }
        [pscustomobject]@{name=$name;line=$cmd.Extent.StartLineNumber;owner=$owner;snippet=$cmd.Extent.Text.Substring(0,[Math]::Min(200,$cmd.Extent.Text.Length))}
    }
})
$assignments = @($tree.FindAll({param($n) $n -is [System.Management.Automation.Language.AssignmentStatementAst]},$true) | ForEach-Object {
    $a = $_
    if ($a.Left -is [System.Management.Automation.Language.VariableExpressionAst]) {
        $cur = $a.Parent
        $owner = $null
        while ($cur) {
            if ($cur -is [System.Management.Automation.Language.FunctionDefinitionAst]) { $owner = $cur.Name; break }
            $cur = $cur.Parent
        }
        [pscustomobject]@{name=$a.Left.VariablePath.UserPath;line=$a.Extent.StartLineNumber;owner=$owner}
    }
})
[pscustomobject]@{
  parser='powershell_ast'
  errors=@($parseErrors | ForEach-Object { [pscustomobject]@{line=$_.Extent.StartLineNumber;message=$_.Message} })
  functions=@($functions)
  commands=@($commands)
  assignments=@($assignments)
} | ConvertTo-Json -Depth 9 -Compress | ForEach-Object {
  # An ASCII envelope avoids Windows PowerShell's locale-specific code page
  # entirely. The payload is UTF-8 JSON; only Atlas-owned parse metadata is
  # emitted, never executed code or evaluated target script statements.
  $bytes = [System.Text.Encoding]::UTF8.GetBytes([string]$_)
  'ATLAS_PS_AST_UTF8_B64:' + [System.Convert]::ToBase64String($bytes)
}
