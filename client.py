"""
Task 5: Python Client for Testing FastAPI Model Serving
-------------------------------------------------------
Demonstrates testing of the deployed REST API:
  1. GET /health - Validates API health and model readiness.
  2. POST /predict - Valid Request 1 (1st class female passenger).
  3. POST /predict - Valid Request 2 (3rd class male passenger).
  4. POST /predict - Invalid Request (testing Pydantic validation: invalid class, gender, negative age).
"""

import json
import time
import requests

BASE_URL = "http://127.0.0.1:8000"


def print_section(title: str):
    print("\n" + "=" * 70)
    print(f"  {title}")
    print("=" * 70)


def wait_for_server(max_retries: int = 10, delay: float = 1.0):
    print(f"[*] Checking connection to FastAPI server at {BASE_URL}...")
    for attempt in range(1, max_retries + 1):
        try:
            r = requests.get(f"{BASE_URL}/health", timeout=2)
            if r.status_code == 200:
                print(f"[+] Server is UP and healthy on attempt {attempt}.")
                return True
        except requests.exceptions.RequestException:
            time.sleep(delay)
    print("[-] Server is not responding. Please ensure 'uvicorn app:app' is running.")
    return False


def test_health():
    print_section("TEST 1: GET /health (API Health Status)")
    url = f"{BASE_URL}/health"
    start_time = time.perf_counter()
    response = requests.get(url)
    elapsed = (time.perf_counter() - start_time) * 1000

    print(f"Request URL   : {url}")
    print(f"Status Code   : {response.status_code} {response.reason}")
    print(f"Round-trip    : {elapsed:.2f} ms")
    print("Response Body :")
    print(json.dumps(response.json(), indent=2))
    assert response.status_code == 200, "Health check failed!"


def test_valid_prediction_1():
    print_section("TEST 2: POST /predict (Valid Request - 1st Class Female)")
    url = f"{BASE_URL}/predict"
    payload = {
        "Pclass": 1,
        "Sex": "female",
        "Age": 28.0,
        "SibSp": 0,
        "Parch": 0,
        "Fare": 85.50,
        "Embarked": "C"
    }

    start_time = time.perf_counter()
    response = requests.post(url, json=payload)
    elapsed = (time.perf_counter() - start_time) * 1000

    print(f"Request URL   : {url}")
    print(f"Status Code   : {response.status_code} {response.reason}")
    print(f"Round-trip    : {elapsed:.2f} ms")
    print("Request Body  :")
    print(json.dumps(payload, indent=2))
    print("Response Body :")
    print(json.dumps(response.json(), indent=2))

    data = response.json()
    assert response.status_code == 200
    print(f"[+] Prediction: {data['label']} (Probability: {data['probability_survived'] * 100:.1f}%)")


def test_valid_prediction_2():
    print_section("TEST 3: POST /predict (Valid Request - 3rd Class Male)")
    url = f"{BASE_URL}/predict"
    payload = {
        "Pclass": 3,
        "Sex": "male",
        "Age": 22.0,
        "SibSp": 1,
        "Parch": 0,
        "Fare": 7.25,
        "Embarked": "S"
    }

    start_time = time.perf_counter()
    response = requests.post(url, json=payload)
    elapsed = (time.perf_counter() - start_time) * 1000

    print(f"Request URL   : {url}")
    print(f"Status Code   : {response.status_code} {response.reason}")
    print(f"Round-trip    : {elapsed:.2f} ms")
    print("Request Body  :")
    print(json.dumps(payload, indent=2))
    print("Response Body :")
    print(json.dumps(response.json(), indent=2))

    data = response.json()
    assert response.status_code == 200
    print(f"[+] Prediction: {data['label']} (Probability: {data['probability_not_survived'] * 100:.1f}%)")


def test_invalid_prediction():
    print_section("TEST 4: POST /predict (Invalid Request - Pydantic Validation Test)")
    url = f"{BASE_URL}/predict"
    # Invalid fields: Pclass 5 (only 1-3 allowed), Sex 'robot', Age -15 (negative age)
    invalid_payload = {
        "Pclass": 5,
        "Sex": "robot",
        "Age": -15.0,
        "SibSp": -2,
        "Parch": 0,
        "Fare": -50.0,
        "Embarked": "X"
    }

    start_time = time.perf_counter()
    response = requests.post(url, json=invalid_payload)
    elapsed = (time.perf_counter() - start_time) * 1000

    print(f"Request URL   : {url}")
    print(f"Status Code   : {response.status_code} {response.reason} (Expected: 422 Unprocessable Entity)")
    print(f"Round-trip    : {elapsed:.2f} ms")
    print("Request Body (Intentionally Invalid) :")
    print(json.dumps(invalid_payload, indent=2))
    print("FastAPI / Pydantic Validation Error Response :")
    print(json.dumps(response.json(), indent=2))

    assert response.status_code == 422, "Expected HTTP 422 for invalid request!"
    print("[+] Pydantic input validation successfully caught and blocked all invalid fields!")


def main():
    if not wait_for_server():
        return
    test_health()
    test_valid_prediction_1()
    test_valid_prediction_2()
    test_invalid_prediction()
    print_section("ALL CLIENT TESTS PASSED SUCCESSFULLY!")


if __name__ == "__main__":
    main()
