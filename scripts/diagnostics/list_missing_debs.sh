#!/bin/bash
# List every .deb URI still needed to complete the broken CUDA toolkit install.
apt-get install -y --fix-broken --print-uris 2>/dev/null | grep -oE "'https?://[^']+'" | tr -d "'" | sort -u
