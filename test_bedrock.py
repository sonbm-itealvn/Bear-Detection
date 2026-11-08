"""
test_bedrock.py

Test Claude Bedrock connection and diagnose why ERROR count keeps growing.
Checks: credentials, model ID, rate limits, image encoding.

Usage:
    python test_bedrock.py
    python test_bedrock.py --image path/to/test.jpg
    python test_bedrock.py --verbose  # show full error traceback
"""

import os
import io
import sys
import time
import argparse
import traceback
from pathlib import Path


def print_ok(msg):   print(f"  \033[92m[OK]\033[0m {msg}")
def print_fail(msg): print(f"  \033[91m[FAIL]\033[0m {msg}")
def print_warn(msg): print(f"  \033[93m[WARN]\033[0m {msg}")
def print_info(msg): print(f"  \033[94m[INFO]\033[0m {msg}")


# ============================================================
# TEST 1: Credentials
# ============================================================
def test_credentials():
    print("\n[1] Checking AWS credentials...")
    key = os.environ.get("AWS_ACCESS_KEY_ID", "")
    secret = os.environ.get("AWS_SECRET_ACCESS_KEY", "")
    region = os.environ.get("AWS_REGION", "us-east-1")
    model_id = os.environ.get("BEDROCK_MODEL_ID", "global.anthropic.claude-sonnet-4-5-20250929-v1:0")

    if not key:
        print_fail("AWS_ACCESS_KEY_ID not set")
        return False
    if not secret:
        print_fail("AWS_SECRET_ACCESS_KEY not set")
        return False

    print_ok(f"AWS_ACCESS_KEY_ID: {key[:8]}...{key[-4:]}")
    print_ok(f"AWS_REGION: {region}")
    if model_id:
        print_ok(f"BEDROCK_MODEL_ID: {model_id[:60]}...")
    else:
        print_warn("BEDROCK_MODEL_ID not set, will use default")
        model_id = "global.anthropic.claude-sonnet-4-5-20250929-v1:0"
        print_info(f"Default: {model_id}")
    return True


# ============================================================
# TEST 2: boto3 import
# ============================================================
def test_boto3():
    print("\n[2] Checking boto3...")
    try:
        import boto3
        print_ok(f"boto3 version: {boto3.__version__}")
        return True
    except ImportError as e:
        print_fail(f"boto3 not installed: {e}")
        print_info("Fix: pip install boto3 --break-system-packages")
        return False


# ============================================================
# TEST 3: Create client
# ============================================================
def test_client():
    print("\n[3] Creating Bedrock client...")
    try:
        import boto3
        client = boto3.client(
            service_name="bedrock-runtime",
            region_name=os.environ.get("AWS_REGION", "us-east-1"),
            aws_access_key_id=os.environ.get("AWS_ACCESS_KEY_ID", ""),
            aws_secret_access_key=os.environ.get("AWS_SECRET_ACCESS_KEY", ""),
        )
        print_ok("Client created successfully")
        return client
    except Exception as e:
        print_fail(f"Failed to create client: {e}")
        return None


# ============================================================
# TEST 4: Text-only call (no image)
# ============================================================
def test_text_call(client, verbose=False):
    print("\n[4] Testing text-only API call...")
    model_id = os.environ.get(
        "BEDROCK_MODEL_ID",
        "global.anthropic.claude-sonnet-4-5-20250929-v1:0"
    )
    try:
        t0 = time.time()
        resp = client.converse(
            modelId=model_id,
            messages=[{"role": "user", "content": [{"text": "Reply with one word: Hello"}]}],
            inferenceConfig={"maxTokens": 10, "temperature": 0.0}
        )
        elapsed = time.time() - t0
        ans = resp['output']['message']['content'][0]['text']
        print_ok(f"Response: '{ans}' ({elapsed:.2f}s)")
        print_info(f"Input tokens: {resp['usage']['inputTokens']}, "
                   f"Output tokens: {resp['usage']['outputTokens']}")
        return True
    except Exception as e:
        err = str(e)
        print_fail(f"Text call failed: {err}")
        if verbose:
            traceback.print_exc()
        _diagnose_error(err)
        return False


# ============================================================
# TEST 5: Image call
# ============================================================
def test_image_call(client, image_path=None, verbose=False):
    print("\n[5] Testing image API call...")
    from PIL import Image as PILImage

    model_id = os.environ.get(
        "BEDROCK_MODEL_ID",
        "global.anthropic.claude-sonnet-4-5-20250929-v1:0"
    )

    # Create or load test image
    if image_path and Path(image_path).exists():
        img = PILImage.open(image_path).convert("RGB")
        print_info(f"Using image: {image_path} ({img.size})")
    else:
        import numpy as np
        # Create a simple green 100x100 image
        arr = np.zeros((100, 100, 3), dtype=np.uint8)
        arr[:, :, 1] = 100  # green
        img = PILImage.fromarray(arr)
        print_info("Using synthetic green 100x100 test image")

    img.thumbnail((512, 512))
    buf = io.BytesIO()
    img.save(buf, format="JPEG", quality=85)
    img_bytes = buf.getvalue()
    print_info(f"Image size: {len(img_bytes) / 1024:.1f} KB")

    try:
        t0 = time.time()
        resp = client.converse(
            modelId=model_id,
            messages=[{"role": "user", "content": [
                {"image": {"format": "jpeg", "source": {"bytes": img_bytes}}},
                {"text": "What color is dominant in this image? Reply one word."}
            ]}],
            inferenceConfig={"maxTokens": 20, "temperature": 0.0}
        )
        elapsed = time.time() - t0
        ans = resp['output']['message']['content'][0]['text']
        print_ok(f"Response: '{ans}' ({elapsed:.2f}s)")
        print_info(f"Tokens: in={resp['usage']['inputTokens']}, "
                   f"out={resp['usage']['outputTokens']}")
        return True
    except Exception as e:
        err = str(e)
        print_fail(f"Image call failed: {err}")
        if verbose:
            traceback.print_exc()
        _diagnose_error(err)
        return False


# ============================================================
# TEST 6: Concurrent calls (simulate HEM)
# ============================================================
def test_concurrent(client, n=5, verbose=False):
    print(f"\n[6] Testing {n} concurrent calls (simulate HEM batch)...")
    from concurrent.futures import ThreadPoolExecutor, as_completed
    import numpy as np
    from PIL import Image as PILImage

    model_id = os.environ.get(
        "BEDROCK_MODEL_ID",
        "global.anthropic.claude-sonnet-4-5-20250929-v1:0"
    )

    def make_test_image(i):
        arr = np.zeros((100, 100, 3), dtype=np.uint8)
        arr[:, :, i % 3] = 128
        img = PILImage.fromarray(arr)
        buf = io.BytesIO()
        img.save(buf, format="JPEG", quality=85)
        return buf.getvalue()

    def call_one(i):
        img_bytes = make_test_image(i)
        try:
            resp = client.converse(
                modelId=model_id,
                messages=[{"role": "user", "content": [
                    {"image": {"format": "jpeg", "source": {"bytes": img_bytes}}},
                    {"text": "Reply ONE WORD: YES or NO or UNSURE"}
                ]}],
                inferenceConfig={"maxTokens": 10, "temperature": 0.0}
            )
            ans = resp['output']['message']['content'][0]['text'].strip()
            return i, ans, None
        except Exception as e:
            return i, None, str(e)

    results = {"ok": 0, "error": 0, "errors": []}
    t0 = time.time()

    with ThreadPoolExecutor(max_workers=n) as ex:
        futures = {ex.submit(call_one, i): i for i in range(n)}
        for fut in as_completed(futures):
            i, ans, err = fut.result()
            if err:
                results["error"] += 1
                results["errors"].append(err)
                print_fail(f"  Call {i}: {err[:100]}")
            else:
                results["ok"] += 1
                print_ok(f"  Call {i}: '{ans}'")

    elapsed = time.time() - t0
    print_info(f"  {results['ok']}/{n} OK, {results['error']} errors in {elapsed:.1f}s")

    if results["errors"]:
        print("\n  Error breakdown:")
        from collections import Counter
        categories = Counter()
        for err in results["errors"]:
            err_lower = err.lower()
            if "throttling" in err_lower or "rate" in err_lower:
                categories["throttling/rate_limit"] += 1
            elif "auth" in err_lower or "credentials" in err_lower or "access" in err_lower:
                categories["auth/credentials"] += 1
            elif "model" in err_lower and ("not found" in err_lower or "invalid" in err_lower):
                categories["invalid_model_id"] += 1
            elif "timeout" in err_lower or "connect" in err_lower:
                categories["network/timeout"] += 1
            else:
                categories[f"other: {err[:60]}"] += 1
        for cat, cnt in categories.items():
            print_warn(f"    {cat}: {cnt}")

    return results["ok"] == n


# ============================================================
# DIAGNOSE ERROR
# ============================================================
def _diagnose_error(err):
    err_lower = err.lower()
    print("\n  Diagnosis:")
    if "invalidclienttokenid" in err_lower or "invalid" in err_lower and "token" in err_lower:
        print_warn("  -> AWS_ACCESS_KEY_ID is invalid or expired")
        print_info("  Fix: Check/refresh your AWS credentials")
    elif "signaturedoesnotmatch" in err_lower or "signature" in err_lower:
        print_warn("  -> AWS_SECRET_ACCESS_KEY is wrong")
        print_info("  Fix: Check your secret key for typos")
    elif "accessdenied" in err_lower or "not authorized" in err_lower:
        print_warn("  -> IAM user doesn't have Bedrock access")
        print_info("  Fix: Add policy 'AmazonBedrockFullAccess' to IAM user")
    elif "resourcenotfound" in err_lower or "model" in err_lower and "not found" in err_lower:
        print_warn("  -> Model ID is invalid or not enabled in your region")
        print_info("  Fix: Check BEDROCK_MODEL_ID and enable model in AWS Console > Bedrock > Model access")
    elif "throttling" in err_lower or "toomanyrequests" in err_lower:
        print_warn("  -> Rate limit hit (too many concurrent requests)")
        print_info("  Fix: Reduce MAX_WORKERS from 10 to 5")
    elif "connect" in err_lower or "timeout" in err_lower:
        print_warn("  -> Network/connection issue")
        print_info("  Fix: Check internet access from server, try VPN if needed")
    elif "endpointresolutionerror" in err_lower:
        print_warn("  -> Wrong region or endpoint")
        print_info("  Fix: Check AWS_REGION matches where Bedrock is enabled")
    else:
        print_warn(f"  -> Unknown error: {err[:200]}")


# ============================================================
# MAIN
# ============================================================
def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--image", default=None, help="Path to test image")
    parser.add_argument("--verbose", action="store_true", help="Show full tracebacks")
    parser.add_argument("--concurrent", type=int, default=5,
                        help="Number of concurrent calls to test (default: 5)")
    args = parser.parse_args()

    print("="*60)
    print("CLAUDE BEDROCK CONNECTION TEST")
    print("="*60)

    # Check PIL
    try:
        from PIL import Image
        print_ok(f"Pillow available")
    except ImportError:
        print_fail("Pillow not installed")
        print_info("Fix: pip install Pillow --break-system-packages")

    # Run tests
    creds_ok = test_credentials()
    if not creds_ok:
        print("\n[STOP] Fix credentials first.")
        sys.exit(1)

    boto_ok = test_boto3()
    if not boto_ok:
        sys.exit(1)

    client = test_client()
    if not client:
        sys.exit(1)

    text_ok = test_text_call(client, verbose=args.verbose)
    if not text_ok:
        print("\n[STOP] Text call failed — fix before testing image calls.")
        sys.exit(1)

    img_ok = test_image_call(client, image_path=args.image, verbose=args.verbose)
    conc_ok = test_concurrent(client, n=args.concurrent, verbose=args.verbose)

    # Summary
    print("\n" + "="*60)
    print("SUMMARY")
    print("="*60)
    all_ok = text_ok and img_ok and conc_ok
    status = "\033[92mALL PASS\033[0m" if all_ok else "\033[91mSOME FAILED\033[0m"
    print(f"  Status: {status}")

    if all_ok:
        print_ok("Bedrock is working correctly")
        print_info("If ERROR count still grows during training, check:")
        print_info("  1. Images that can't be opened (corrupted files)")
        print_info("  2. Bedrock rate limits under high load")
        print_info("  3. Network interruptions on the server")
    else:
        print_warn("Fix the failed tests before running training")

    print()


if __name__ == "__main__":
    main()
