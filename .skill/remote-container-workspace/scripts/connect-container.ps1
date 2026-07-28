[CmdletBinding(DefaultParameterSetName = 'Verify')]
param(
    [Parameter(Mandatory)]
    [string]$HostName,

    [Parameter(Mandatory)]
    [string]$UserName,

    [ValidateRange(1, 65535)]
    [int]$Port = 22,

    [Parameter(Mandatory)]
    [string]$Container,

    [Parameter(Mandatory)]
    [string]$WorkDir,

    [Parameter(ParameterSetName = 'Command')]
    [string]$Command,

    [Parameter(ParameterSetName = 'Interactive')]
    [switch]$Interactive,

    [Parameter(ParameterSetName = 'Interactive')]
    [string]$Shell = 'bash',

    [string]$IdentityFile,

    [string]$PasswordEnvironmentVariable,

    [switch]$DryRun
)

$ErrorActionPreference = 'Stop'

function ConvertTo-ShSingleQuoted {
    param([Parameter(Mandatory)][string]$Value)
    return "'" + $Value.Replace("'", "'`"'`"'") + "'"
}

$ssh = (Get-Command ssh -ErrorAction Stop).Source
$sshArgs = @(
    '-p', $Port.ToString(),
    '-o', 'StrictHostKeyChecking=accept-new',
    '-o', 'NumberOfPasswordPrompts=1'
)

if ($IdentityFile) {
    $resolvedIdentity = (Resolve-Path -LiteralPath $IdentityFile).Path
    $sshArgs += @('-i', $resolvedIdentity)
}

$quotedWorkDir = ConvertTo-ShSingleQuoted $WorkDir
$quotedContainer = ConvertTo-ShSingleQuoted $Container

if ($Interactive) {
    $quotedShell = ConvertTo-ShSingleQuoted $Shell
    $remoteCommand = "docker exec -it -w $quotedWorkDir $quotedContainer $quotedShell"
    $sshArgs += '-tt'
}
elseif ($Command) {
    $quotedCommand = ConvertTo-ShSingleQuoted $Command
    $remoteCommand = "docker exec -w $quotedWorkDir $quotedContainer sh -lc $quotedCommand"
}
else {
    $remoteCommand = "docker exec -w $quotedWorkDir $quotedContainer pwd"
}

$sshArgs += "$UserName@$HostName"
$sshArgs += $remoteCommand

if ($DryRun) {
    [pscustomobject]@{
        Executable = $ssh
        Arguments = $sshArgs
        UsesPasswordEnvironmentVariable = [bool]$PasswordEnvironmentVariable
    }
    return
}

$askPassPath = $null
$savedEnvironment = @{}
try {
    if ($PasswordEnvironmentVariable) {
        $password = [Environment]::GetEnvironmentVariable($PasswordEnvironmentVariable, 'Process')
        if ([string]::IsNullOrEmpty($password)) {
            throw "The process environment variable '$PasswordEnvironmentVariable' is empty or undefined."
        }

        foreach ($name in 'RCW_ASKPASS_PASSWORD', 'SSH_ASKPASS', 'SSH_ASKPASS_REQUIRE', 'DISPLAY') {
            $savedEnvironment[$name] = [Environment]::GetEnvironmentVariable($name, 'Process')
        }

        $askPassPath = Join-Path ([IO.Path]::GetTempPath()) ("rcw-askpass-{0}.cmd" -f [guid]::NewGuid())
        Set-Content -LiteralPath $askPassPath -Value '@echo off', 'echo %RCW_ASKPASS_PASSWORD%' -Encoding Ascii
        $env:RCW_ASKPASS_PASSWORD = $password
        $env:SSH_ASKPASS = $askPassPath
        $env:SSH_ASKPASS_REQUIRE = 'force'
        $env:DISPLAY = 'remote-container-workspace'
    }

    & $ssh @sshArgs
    if ($LASTEXITCODE -ne 0) {
        throw "SSH exited with code $LASTEXITCODE."
    }
}
finally {
    if ($askPassPath -and (Test-Path -LiteralPath $askPassPath)) {
        Remove-Item -LiteralPath $askPassPath -Force
    }
    foreach ($entry in $savedEnvironment.GetEnumerator()) {
        if ($null -eq $entry.Value) {
            Remove-Item -Path "Env:$($entry.Key)" -ErrorAction SilentlyContinue
        }
        else {
            [Environment]::SetEnvironmentVariable($entry.Key, $entry.Value, 'Process')
        }
    }
}
