#!/usr/bin/env bash
# scripts/podman-setup.sh
#
# One-shot setup script: installs Podman + podman-compose and
# creates the persistent Neurolink container network.
#
# Usage:
#   bash scripts/podman-setup.sh
#
# Tested on: Ubuntu 22.04 / 24.04, Fedora 38+, macOS 14 (Homebrew)

set -euo pipefail

BLUE='\033[0;34m'
GREEN='\033[0;32m'
YELLOW='\033[1;33m'
NC='\033[0m'

log()  { echo -e "${BLUE}[neurolink]${NC} $*"; }
ok()   { echo -e "${GREEN}[ok]${NC} $*"; }
warn() { echo -e "${YELLOW}[warn]${NC} $*"; }

# ---------------------------------------------------------------------------
# 1. Install Podman
# ---------------------------------------------------------------------------
if command -v podman &>/dev/null; then
  ok "Podman already installed: $(podman --version)"
else
  log "Installing Podman..."
  if [[ "$(uname)" == "Darwin" ]]; then
    brew install podman
    podman machine init || true
    podman machine start || true
  elif command -v apt-get &>/dev/null; then
    sudo apt-get update -qq
    sudo apt-get install -y --no-install-recommends podman
  elif command -v dnf &>/dev/null; then
    sudo dnf install -y podman
  elif command -v pacman &>/dev/null; then
    sudo pacman -Sy --noconfirm podman
  else
    warn "Unknown package manager. Install Podman manually: https://podman.io/getting-started/installation"
    exit 1
  fi
  ok "Podman installed: $(podman --version)"
fi

# ---------------------------------------------------------------------------
# 2. Install podman-compose
# ---------------------------------------------------------------------------
if command -v podman-compose &>/dev/null; then
  ok "podman-compose already installed: $(podman-compose --version)"
else
  log "Installing podman-compose via pip..."
  pip install --quiet podman-compose
  ok "podman-compose installed: $(podman-compose --version)"
fi

# ---------------------------------------------------------------------------
# 3. Create a persistent Podman network for Neurolink
#    (podman-compose also creates one automatically, but having a named
#     network makes manual `podman run` commands easier)
# ---------------------------------------------------------------------------
NETWORK_NAME="neurolink-net"
if podman network inspect "${NETWORK_NAME}" &>/dev/null; then
  ok "Podman network '${NETWORK_NAME}' already exists"
else
  log "Creating Podman network '${NETWORK_NAME}'..."
  podman network create "${NETWORK_NAME}"
  ok "Network '${NETWORK_NAME}' created"
fi

# ---------------------------------------------------------------------------
# 4. Print next steps
# ---------------------------------------------------------------------------
echo
echo -e "${GREEN}Setup complete.${NC} Run the development stack with:"
echo
echo "  podman-compose -f compose.dev.yml up --build"
echo
echo "Or with the frontend profile:"
echo
echo "  podman-compose -f compose.dev.yml --profile frontend up --build"
echo
warn "BLE note: for rootless Podman to access BlueZ, ensure"
warn "/run/dbus/system_bus_socket is readable by your user, OR run:"
warn "  sudo setcap 'cap_net_admin,cap_net_raw+eip' \$(which python3)"
