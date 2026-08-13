"""Post-deploy smoke tests – health check and prediction endpoint (M4)."""

import argparse
import io
import logging
import sys
import time

import numpy as np

logger = logging.getLogger(__name__)
logging.basicConfig(level=logging.INFO, format="%(asctime)s - %(levelname)s - %(message)s")


def _wait(base_url: str, retries: int = 12, delay: float = 5.0) -> bool:
    import requests

    for attempt in range(1, retries + 1):
        try:
            r = requests.get(f"{base_url}/health", timeout=5)
            if r.status_code == 200:
                logger.info("Service ready after %d attempt(s)", attempt)
                return True
        except requests.exceptions.ConnectionError:
            pass
        logger.info("Attempt %d/%d – waiting %.0fs…", attempt, retries, delay)
        time.sleep(delay)
    logger.error("Service not reachable after %d attempts", retries)
    return False


def test_health(base_url: str) -> bool:
    import requests

    logger.info("→ Testing /health …")
    try:
        r = requests.get(f"{base_url}/health", timeout=10)
        if r.status_code != 200:
            logger.error("FAIL: status=%d", r.status_code)
            return False
        data = r.json()
        if data.get("status") != "healthy":
            logger.error("FAIL: unexpected body: %s", data)
            return False
        logger.info("PASS: %s", data)
        return True
    except Exception as exc:
        logger.error("FAIL: %s", exc)
        return False


def test_predict(base_url: str) -> bool:
    import requests
    from PIL import Image

    logger.info("→ Testing /predict …")
    try:
        arr = np.random.randint(0, 255, (224, 224, 3), dtype=np.uint8)
        img = Image.fromarray(arr)
        buf = io.BytesIO()
        img.save(buf, format="JPEG")
        buf.seek(0)

        r = requests.post(
            f"{base_url}/predict",
            files={"file": ("smoke_test.jpg", buf, "image/jpeg")},
            timeout=30,
        )
        if r.status_code != 200:
            logger.error("FAIL: status=%d body=%s", r.status_code, r.text)
            return False

        data = r.json()
        assert "prediction" in data and data["prediction"] in ("cat", "dog"), "bad prediction"
        assert "confidence" in data and 0.0 <= data["confidence"] <= 1.0, "bad confidence"
        logger.info("PASS: prediction=%s confidence=%.4f latency_ms=%.1f",
                    data["prediction"], data["confidence"], data.get("latency_ms", -1))
        return True
    except Exception as exc:
        logger.error("FAIL: %s", exc)
        return False


def run(base_url: str) -> bool:
    logger.info("=== Smoke tests against %s ===", base_url)
    if not _wait(base_url):
        return False

    results = {
        "health": test_health(base_url),
        "predict": test_predict(base_url),
    }

    logger.info("=== Results ===")
    all_passed = True
    for name, passed in results.items():
        logger.info("  %-10s %s", name, "PASSED" if passed else "FAILED")
        if not passed:
            all_passed = False

    if all_passed:
        logger.info("All smoke tests PASSED ✓")
    else:
        logger.error("Smoke tests FAILED ✗")
    return all_passed


def main():
    parser = argparse.ArgumentParser(description="Smoke tests for the deployed API")
    parser.add_argument("--base-url", default="http://localhost:8000")
    args = parser.parse_args()
    sys.exit(0 if run(args.base_url) else 1)


if __name__ == "__main__":
    main()
