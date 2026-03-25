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

assert_fail() {
  if output=$(eval "$1" 2>&1); then
    echo "FAIL: $1 should have failed" >&2
    FAIL=$((FAIL + 1))
  else
    echo "OK: $1: $output"
  fi
}

# 1. Check non-root
if [ "$(id -u)" -eq 0 ]; then
  echo "FAIL: Runner should not run as root" >&2
  FAIL=$((FAIL + 1))
fi

# 2. Sensitive files
for file in /etc/passwd /etc/shadow /etc/sudoers /etc/ssh/sshd_config; do
  assert_fail "cat $file"
done

# Generate report
printf ""
printf "+-----------------------------------------------------------------------------------------+"
printf "| GitHub Self-Hosted Runners Security Audit Report                        FAILURES: %-3s |" $FAIL
printf "|=========================================================================================|"

if [ "$FAIL" -gt 0 ]; then
  echo "| Some checks failed. Please review the above output and take appropriate actions.           |"
  exit 1
fi
