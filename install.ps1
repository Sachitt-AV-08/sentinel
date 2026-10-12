<#
.SYNOPSIS
    Sentinel installer - installs Sentinel CLI with Ollama support
.DESCRIPTION
    Downloads the latest Sentinel release wheel from GitHub and installs it.
    Detects uv/pipx/pip and uses the best available installer.
#>

param(
    [string]$Version = "latest",
    [switch]$Force,
    [switch]$NoOllama
)

$ErrorActionPreference = "Stop"

# Colors
$Green  = [ConsoleColor]::Green
$Yellow = [ConsoleColor]::Yellow
$Red    = [ConsoleColor]::Red
$Cyan   = [ConsoleColor]::Cyan

function Write-Color($msg, $color) {
    $orig = $Host.UI.RawUI.ForegroundColor
    $Host.UI.RawUI.ForegroundColor = $color
    Write-Host $msg
    $Host.UI.RawUI.ForegroundColor = $orig
}

function Get-LatestRelease {
    $api = "https://api.github.com/repos/Sachitt-AV-08/sentinel/releases/latest"
    try {
        $resp = Invoke-RestMethod -Uri $api -Headers @{ "Accept" = "application/vnd.github.v3+json" }
        return $resp.tag_name
    } catch {
        Write-Color "Failed to fetch latest release: $($_.Exception.Message)" $Red
        return "v0.1.0"
    }
}

function Get-Installer {
    if (Get-Command uv -ErrorAction SilentlyContinue) {
        return "uv"
    }
    if (Get-Command pipx -ErrorAction SilentlyContinue) {
        return "pipx"
    }
    if (Get-Command pip -ErrorAction SilentlyContinue) {
        return "pip"
    }
    return $null
}

# Get version
if ($Version -eq "latest") {
    Write-Color "Fetching latest release..." $Yellow
    $Version = Get-LatestRelease
    Write-Color "Latest version: $Version" $Green
}

$installer = Get-Installer
if (-not $installer) {
    Write-Color "No installer found. Please install uv, pipx, or pip." $Red
    exit 1
}

Write-Color "Using installer: $installer" $Cyan

$extra = if ($NoOllama) { "" } else { "[ollama]" }
$pkg = "sentinel$extra"

if ($installer -eq "uv") {
    $args = "pip install $pkg"
    if ($Version -ne "latest") {
        $args += " --index-url https://github.com/Sachitt-AV-08/sentinel/releases/download/$Version"
    }
    Write-Color "Running: uv $args" $Cyan
    & uv $args.Split()
}
elseif ($installer -eq "pipx") {
    $args = "install $pkg"
    if ($Force) { $args += " --force" }
    Write-Color "Running: pipx $args" $Cyan
    & pipx $args.Split()
}
else {
    $args = "install $pkg"
    if ($Force) { $args += " --force-reinstall" }
    Write-Color "Running: pip $args" $Cyan
    & pip $args.Split()
}

# Verify
try {
    sentinel --version
    sentinel doctor
    Write-Color "`nSentinel installed successfully!" $Green
} catch {
    Write-Color "Install verification failed: $($_.Exception.Message)" $Red
    exit 1
}