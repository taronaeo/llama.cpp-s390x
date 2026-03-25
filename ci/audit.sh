#!/usr/bin/env bash

#
# Llama.cpp GitHub Self-Hosted Runners Audit Script
#
# This script performs a security audit of the GitHub self-hosted runners used
# in the Llama.cpp repository. It checks for potential vulnerabilities,
# misconfigurations, and ensures that the runners are secure. If any of the
# self-hosted runners are found to be vulnerable or misconfigured, the script
# will exit with a non-zero status code.
#
# Note: Llama.cpp maintainers reserve the right to take appropriate actions,
#       including but not limited to disabling or removing any self-hosted
#       runners that are found to be vulnerable or misconfigured.
# Note: Llama.cpp maintainers will run audits on a periodic basis to ensure
#       the security of the self-hosted runners. We do not guarantee that
#       all vulnerabilities will be detected, but we will make every effort
#       to identify and address any issues that are found.
#
# Usage: ./audit.sh
#

FAIL=0
COUNT=0

assert_fail() {
  if output=$(eval "$1" 2>&1); then
    printf "| %3d: %-89s |\n" "$COUNT" "FAIL: ${1//\$/\\$} should have failed"
    COUNT=$((COUNT + 1))
    FAIL=$((FAIL + 1))
  else
    printf "| %3d: %-89s |\n" "$COUNT" "PASS: ${1//\$/\\$}: $output"
    COUNT=$((COUNT + 1))
  fi
}

printf ""
printf "+$(printf '%0.s-' {1..89})+\n"
printf "| GitHub Self-Hosted Actions Audit  %-20s (%-6s)   %-10s |\n" "${{ runner.name }}" "$(uname -m)" "$(date +'%Y-%m-%d')"
printf "+$(printf '%0.s=' {1..89})+\n"

# 1. Check non-root
if [ "$(id -u)" -eq 0 ]; then
  printf "| %3d: %-89s |\n" "$COUNT" "FAIL: Runner should not run as root"
  FAIL=$((FAIL + 1))
  COUNT=$((COUNT + 1))
fi

# 2. Sensitive files
for file in /etc/passwd /etc/shadow /etc/sudoers /etc/ssh/sshd_config; do
  assert_fail "ls $file"
done

# 3. SSH private keys
if find /root/.ssh /Users/*/.ssh -name "id_*" ! -name "*.pub" 2>/dev/null | grep -q .; then
  printf "| %3d: %-89s |\n" "$COUNT" "FAIL: SSH private keys should not be findable"
  FAIL=$((FAIL + 1))
  COUNT=$((COUNT + 1))
else
  printf "| %3d: %-89s |\n" "$COUNT" "PASS: No SSH private keys found"
  COUNT=$((COUNT + 1))
fi

# 4. Sudo without password
assert_fail "sudo -n true"

# 5. Docker socket
assert_fail "ls /var/run/docker.sock"

# 6. World-writable files
WORLD_WRITABLE_IN_PATH=""
IFS=: read -ra PATH_DIRS <<< "$PATH"
for dir in "${PATH_DIRS[@]}"; do
  if [ -d "$dir" ] && [ -w "$dir" ]; then
    WORLD_WRITABLE_IN_PATH="$dir $WORLD_WRITABLE_IN_PATH"
  fi
done

assert_fail "[ -n \"$WORLD_WRITABLE_IN_PATH\" ]"


if [ "$FAIL" -gt 0 ]; then
  printf "| Some checks failed. Please review the above output and take appropriate actions.           |\n"
  exit 1
fi
