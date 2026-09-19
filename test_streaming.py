#!/usr/bin/env python3
"""
Test client for Ghost Agent backend streaming support.
Tests the three streaming scenarios from the spec.
"""

import subprocess
import sys
import time
import json
import os

BASE_URL = os.getenv("BASE_URL", "https://ghost-agent-backend.onrender.com")

# Test colors
GREEN = '\033[92m'
RED = '\033[91m'
YELLOW = '\033[93m'
RESET = '\033[0m'

def run_test(name, test_func):
    """Run a test and report results."""
    print(f"\n{'='*60}")
    print(f"Test: {name}")
    print('='*60)
    try:
        result = test_func()
        if result:
            print(f"{GREEN}✓ PASSED{RESET}")
            return True
        else:
            print(f"{RED}✗ FAILED{RESET}")
            return False
    except Exception as e:
        print(f"{RED}✗ FAILED with exception: {e}{RESET}")
        return False

def get_token():
    """Get a valid session token by registering and logging in."""
    email = f"test{int(time.time())}@streamtest.dev"
    password = "StreamTest123!"

    # Register
    reg_cmd = [
        "curl", "-s", "-X", "POST", f"{BASE_URL}/auth/register",
        "-H", "Content-Type: application/json",
        "-d", f'{{"email":"{email}","password":"{password}","displayName":"StreamTest"}}'
    ]
    reg_result = subprocess.run(reg_cmd, capture_output=True, text=True)
    if reg_result.returncode != 0:
        print(f"Registration failed: {reg_result.stderr}")
        return None

    # Login
    login_cmd = [
        "curl", "-s", "-X", "POST", f"{BASE_URL}/auth/login",
        "-H", "Content-Type: application/json",
        "-d", f'{{"email":"{email}","password":"{password}"}}'
    ]
    login_result = subprocess.run(login_cmd, capture_output=True, text=True)
    if login_result.returncode != 0:
        print(f"Login failed: {login_result.stderr}")
        return None

    try:
        data = json.loads(login_result.stdout)
        return data.get("session")
    except:
        print(f"Failed to parse login response: {login_result.stdout}")
        return None

def test_non_streaming():
    """Test 1: Non-streaming still works."""
    token = get_token()
    if not token:
        print("Failed to get token")
        return False

    cmd = [
        "curl", "-s", "-X", "POST", f"{BASE_URL}/chat",
        "-H", f"Authorization: Bearer {token}",
        "-H", "Content-Type: application/json",
        "-d", '{"messages":[{"role":"user","content":"say hi"}],"task":"text","stream":false}'
    ]
    result = subprocess.run(cmd, capture_output=True, text=True)

    print(f"Status: {result.returncode}")
    print(f"Response: {result.stdout[:200]}")

    if result.returncode != 0:
        print(f"Error: {result.stderr}")
        return False

    try:
        data = json.loads(result.stdout)
        if "message" in data and "content" in data["message"]:
            print(f"Response message: {data['message']['content'][:100]}")
            return True
        else:
            print("Missing expected fields in response")
            return False
    except json.JSONDecodeError as e:
        print(f"Invalid JSON: {e}")
        return False

def test_streaming():
    """Test 2: Streaming returns chunks."""
    token = get_token()
    if not token:
        print("Failed to get token")
        return False

    cmd = [
        "curl", "-N", "-X", "POST", f"{BASE_URL}/chat",
        "-H", f"Authorization: Bearer {token}",
        "-H", "Content-Type: application/json",
        "-d", '{"messages":[{"role":"user","content":"count from 1 to 5"}],"task":"text","stream":true}'
    ]
    result = subprocess.run(cmd, capture_output=True, text=True, timeout=30)

    print(f"Status: {result.returncode}")
    print(f"Raw output (first 500 chars):\n{result.stdout[:500]}")

    if result.returncode != 0:
        print(f"Error: {result.stderr}")
        return False

    # Check for SSE format
    lines = result.stdout.strip().split('\n')
    has_data_prefix = any(line.startswith('data:') for line in lines)
    has_done = 'data: [DONE]' in result.stdout

    print(f"Has data: prefix: {has_data_prefix}")
    print(f"Has [DONE]: {has_done}")
    print(f"Total lines: {len(lines)}")

    return has_data_prefix and has_done and len(lines) > 2

def test_streaming_auth():
    """Test 3: Auth still enforced for streaming."""
    cmd = [
        "curl", "-N", "-X", "POST", f"{BASE_URL}/chat",
        "-H", "Content-Type: application/json",
        "-d", '{"messages":[{"role":"user","content":"hi"}],"task":"text","stream":true}'
    ]
    result = subprocess.run(cmd, capture_output=True, text=True, timeout=10)

    print(f"Status: {result.returncode}")
    print(f"Response: {result.stdout[:200]}")

    # Should get 401
    if result.returncode != 0:
        # curl returns non-zero for HTTP errors
        try:
            data = json.loads(result.stdout)
            if data.get("detail") and "authorization" in data["detail"].lower():
                print("Got expected 401 error")
                return True
        except:
            pass

    # Check if it's actually a 401 response
    if "401" in result.stderr or "unauthorized" in result.stdout.lower():
        return True

    print("Did not get expected 401")
    return False

def main():
    print(f"\n{'#'*60}")
    print(f"# Ghost Agent Backend - Streaming Support Tests")
    print(f"# Base URL: {BASE_URL}")
    print(f"{'#'*60}")

    results = []

    # Test 1: Non-streaming still works
    results.append(run_test("Non-streaming chat (Test 1)", test_non_streaming))

    # Test 2: Streaming returns chunks
    results.append(run_test("Streaming chat (Test 2)", test_streaming))

    # Test 3: Auth still enforced
    results.append(run_test("Streaming auth enforcement (Test 3)", test_streaming_auth))

    # Summary
    print(f"\n{'='*60}")
    print("SUMMARY")
    print('='*60)
    passed = sum(results)
    total = len(results)
    failed = total - passed

    print(f"Total tests: {total}")
    print(f"Passed:      {passed}")
    print(f"Failed:      {failed}")

    if failed == 0:
        print(f"\n{GREEN}✓ All tests passed!{RESET}")
        return 0
    else:
        print(f"\n{RED}✗ Some tests failed{RESET}")
        return 1

if __name__ == "__main__":
    sys.exit(main())