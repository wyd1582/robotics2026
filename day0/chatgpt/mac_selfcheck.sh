#!/bin/bash
# Read-only checks. No install, network, sudo, cleanup, or configuration writes.
# Does not print serial numbers, environment variables, credentials or process arguments.
set -u

check() {
  printf '\n[%s]\n' "$1"
  shift
  "$@" 2>&1 || printf '(Unavailable or needs a manual check; nothing changed.)\n'
}

if [ "$(uname -s)" != Darwin ]; then
  printf 'This checklist is for macOS. Nothing changed.\n'
  exit 1
fi

check 'macOS version' /usr/bin/sw_vers
check 'Current process architecture: expect arm64 on native Apple Silicon' /usr/bin/uname -m
check 'Chip' /usr/sbin/sysctl -n machdep.cpu.brand_string
check 'Physical memory in bytes: divide by 1073741824 for GiB' /usr/sbin/sysctl -n hw.memsize
check 'Rosetta translation: 1 means translated; missing/0 can be native' /usr/sbin/sysctl -n sysctl.proc_translated
check 'Free space on the data volume' /bin/df -h /System/Volumes/Data
check 'Swap usage: interpret with Activity Monitor memory pressure' /usr/sbin/sysctl vm.swapusage
check 'FileVault status' /usr/bin/fdesetup status
check 'System Integrity Protection status' /usr/bin/csrutil status
check 'Gatekeeper status' /usr/sbin/spctl --status
check 'Selected developer tools directory' /usr/bin/xcode-select -p

printf '\n[Executable locations; missing is normal on a new Mac]\n'
for tool in git brew uv python3 code ffmpeg docker aws; do
  if command -v "$tool" >/dev/null 2>&1; then
    printf '%s: %s\n' "$tool" "$(command -v "$tool")"
  else
    printf '%s: not on PATH\n' "$tool"
  fi
done
printf '\nManual: Activity Monitor > Memory; System Settings > Login Items; encrypted backup.\n'
printf 'A missing tool is not a system failure. Install only the next experiment needs.\n'
