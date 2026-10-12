#!/usr/bin/env bash
# Sentinel installer - installs Sentinel CLI with Ollama support
# Detects uv/pipx/pip and uses the best available installer

set -euo pipefail

# Colors
GREEN='\033[0;32m'
YELLOW='\033[1;33m'
RED='\033[0;31m'
CYAN='\033[0;36m'
NC='\033[0m' # No Color

# Parse flags FIRST
FORCE=false
NO_OLLAMA=false
POSITIONAL=()

while [[ $# -gt 0 ]]; do
    case $1 in
        --force) FORCE=true; shift ;;
        --no-ollama) NO_OLLAMA=true; shift ;;
        -*) echo "Unknown option: $1"; exit 1 ;;
        *) POSITIONAL+=("$1"); shift ;;
    esac
done

# Now get version from positional args or default
VERSION="${POSITIONAL[0]:-latest}"

echo -e "${CYAN}╔══════════════════════════════════════════════╗${NC}"
echo -e "${CYAN}║           Sentinel Installer                ║${NC}"
echo -e "${CYAN}║   Local classification for agent gating     ║${NC}"
echo -e "${CYAN}╚══════════════════════════════════════════════╝${NC}"
echo ""

# Get latest version if not specified
if [[ "$VERSION" == "latest" ]]; then
    echo -e "${YELLOW}Fetching latest release...${NC}"
    VERSION=$(curl -s https://api.github.com/repos/Sachitt-AV-08/sentinel/releases/latest 2>/dev/null | grep '"tag_name"' | sed -E 's/.*"([^"]+)".*/\1/') || true
    if [[ -z "$VERSION" ]]; then
        echo -e "${RED}Failed to fetch latest release${NC}"
        VERSION="v0.1.0"
    fi
fi

echo -e "${GREEN}Version: $VERSION${NC}"

# Detect installer
if command -v uv >/dev/null 2>&1; then
    INSTALLER="uv"
elif command -v pipx >/dev/null 2>&1; then
    INSTALLER="pipx"
elif command -v pip >/dev/null 2>&1; then
    INSTALLER="pip"
else
    echo -e "${RED}No installer found. Please install uv, pipx, or pip.${NC}"
    exit 1
fi

echo -e "${CYAN}Using installer: $INSTALLER${NC}"

EXTRA=""
if [[ "$NO_OLLAMA" != true ]]; then
    EXTRA="[ollama]"
fi

PKG="sentinel$EXTRA"

case $INSTALLER in
    uv)
        ARGS=("pip" "install" "$PKG")
        if [[ "$VERSION" != "latest" ]]; then
            ARGS+=("--index-url" "https://github.com/Sachitt-AV-08/sentinel/releases/download/$VERSION")
        fi
        echo -e "${CYAN}Running: uv ${ARGS[*]}${NC}"
        uv "${ARGS[@]}"
        ;;
    pipx)
        ARGS=("install" "$PKG")
        if [[ "$FORCE" == true ]]; then
            ARGS+=("--force")
        fi
        echo -e "${CYAN}Running: pipx ${ARGS[*]}${NC}"
        pipx "${ARGS[@]}"
        ;;
    pip)
        ARGS=("install" "$PKG")
        if [[ "$FORCE" == true ]]; then
            ARGS+=("--force-reinstall")
        fi
        echo -e "${CYAN}Running: pip ${ARGS[*]}${NC}"
        pip "${ARGS[@]}"
        ;;
esac

# Verify
echo ""
if command -v sentinel >/dev/null 2>&1; then
    sentinel --version
    sentinel doctor
    echo -e "${GREEN}Sentinel installed successfully!${NC}"
else
    echo -e "${RED}Install verification failed: sentinel not in PATH${NC}"
    exit 1
fi