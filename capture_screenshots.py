"""
Script to automatically generate high-resolution screenshots for:
1. Online Inference Terminal Output
2. Batch Inference Terminal Output
3. Streaming Inference Terminal Output
4. FastAPI Client Test Output
5. FastAPI Swagger UI (/docs) - Health Check
6. FastAPI Swagger UI (/docs) - Valid Prediction
7. FastAPI Swagger UI (/docs) - Invalid Prediction (422)
"""

import os
import subprocess
import time
from PIL import Image, ImageDraw, ImageFont
from playwright.sync_api import sync_playwright

SCREENSHOT_DIR = "screenshots"
os.makedirs(SCREENSHOT_DIR, exist_ok=True)


def get_font(size=15):
    font_paths = [
        "C:\\Windows\\Fonts\\consola.ttf",
        "C:\\Windows\\Fonts\\cour.ttf",
        "C:\\Windows\\Fonts\\arial.ttf"
    ]
    for p in font_paths:
        if os.path.exists(p):
            return ImageFont.truetype(p, size)
    return ImageFont.load_default()


def render_terminal_image(title: str, text: str, output_path: str, width=1100):
    font = get_font(15)
    header_font = get_font(13)

    lines = text.split("\n")
    line_height = 22
    header_height = 42
    padding = 24
    height = header_height + padding * 2 + len(lines) * line_height

    img = Image.new("RGB", (width, height), color="#11111b")
    draw = ImageDraw.Draw(img)

    # Window header background
    draw.rectangle([(0, 0), (width, header_height)], fill="#181825")
    draw.line([(0, header_height), (width, header_height)], fill="#313244", width=1)

    # Window buttons
    draw.ellipse([(16, 14), (28, 26)], fill="#f38ba8")  # Red
    draw.ellipse([(36, 14), (48, 26)], fill="#f9e2af")  # Yellow
    draw.ellipse([(56, 14), (68, 26)], fill="#a6e3a1")  # Green

    # Window Title
    draw.text((85, 12), title, font=header_font, fill="#cdd6f4")

    # Content lines
    y = header_height + padding
    for line in lines:
        fill_color = "#cdd6f4"
        if line.startswith("=") or line.startswith("-"):
            fill_color = "#585b70"
        elif "TASK" in line or "TEST" in line:
            fill_color = "#89b4fa"
        elif "[+]" in line or "SUCCESS" in line or "PASSED" in line or "Survived (1)" in line or "status: 200" in line:
            fill_color = "#a6e3a1"
        elif "[-]" in line or "Error" in line or "422" in line or "Did Not Survive" in line:
            fill_color = "#f38ba8"
        elif "[*]" in line or "Latenc" in line or "Throughput" in line:
            fill_color = "#f9e2af"
        elif "EVENT" in line or "Passenger" in line or "Features" in line:
            fill_color = "#b4befe"

        draw.text((padding, y), line, font=font, fill=fill_color)
        y += line_height

    img.save(output_path)
    print(f"[+] Saved terminal screenshot: '{output_path}'")


def capture_cli_screenshots():
    tasks = [
        ("Terminal - Task 2: Online Inference", ["python", "online_inference.py"], os.path.join(SCREENSHOT_DIR, "online_inference.png")),
        ("Terminal - Task 3: Batch Inference", ["python", "batch_inference.py"], os.path.join(SCREENSHOT_DIR, "batch_inference.png")),
        ("Terminal - Task 4: Streaming Inference", ["python", "streaming_inference.py"], os.path.join(SCREENSHOT_DIR, "streaming_inference.png")),
        ("Terminal - Task 5: Python Client API Tests", ["python", "client.py"], os.path.join(SCREENSHOT_DIR, "python_client_test.png")),
    ]

    for title, cmd, out_path in tasks:
        print(f"[*] Running command '{' '.join(cmd)}' for screenshot...")
        res = subprocess.run(cmd, capture_output=True, text=True, cwd=".")
        output_text = res.stdout if res.stdout else res.stderr
        render_terminal_image(title, output_text.strip(), out_path)


def capture_swagger_screenshots():
    print("[*] Launching Chromium to capture Swagger UI interactions...")
    with sync_playwright() as p:
        browser = p.chromium.launch(headless=True)
        page = browser.new_page(viewport={"width": 1280, "height": 950})
        page.goto("http://127.0.0.1:8000/docs", wait_until="networkidle")
        time.sleep(1)

        # Overview Screenshot
        overview_path = os.path.join(SCREENSHOT_DIR, "fastapi_swagger_overview.png")
        page.screenshot(path=overview_path)
        print(f"[+] Saved Swagger overview: '{overview_path}'")

        # 1. Health endpoint test in Swagger
        print("[*] Testing /health in Swagger...")
        health_op = page.locator(".opblock-summary-path[data-path='/health']").first
        if not health_op.is_visible():
            health_op = page.locator("text='/health'").first
        health_op.click()
        time.sleep(0.5)

        page.locator(".try-out__btn").first.click()
        time.sleep(0.5)
        page.locator(".execute").first.click()
        time.sleep(1)

        health_path = os.path.join(SCREENSHOT_DIR, "fastapi_health_swagger.png")
        page.screenshot(path=health_path)
        print(f"[+] Saved Swagger /health test: '{health_path}'")

        # 2. Predict endpoint valid in Swagger
        print("[*] Testing /predict valid in Swagger...")
        page.goto("http://127.0.0.1:8000/docs", wait_until="networkidle")
        time.sleep(1)

        predict_op = page.locator(".opblock-summary-path[data-path='/predict']").first
        if not predict_op.is_visible():
            predict_op = page.locator("text='/predict'").first
        predict_op.click()
        time.sleep(0.5)

        page.locator(".try-out__btn").first.click()
        time.sleep(0.5)

        valid_payload = '{\n  "Pclass": 1,\n  "Sex": "female",\n  "Age": 28,\n  "SibSp": 0,\n  "Parch": 0,\n  "Fare": 85.5,\n  "Embarked": "C"\n}'
        page.locator("textarea.body-param__text").fill(valid_payload)
        time.sleep(0.5)
        page.locator(".execute").first.click()
        time.sleep(1)

        predict_valid_path = os.path.join(SCREENSHOT_DIR, "fastapi_predict_valid_swagger.png")
        page.screenshot(path=predict_valid_path)
        print(f"[+] Saved Swagger /predict (valid) test: '{predict_valid_path}'")

        # 3. Predict endpoint invalid (422) in Swagger
        print("[*] Testing /predict invalid in Swagger...")
        invalid_payload = '{\n  "Pclass": 5,\n  "Sex": "alien",\n  "Age": -10,\n  "SibSp": -1,\n  "Parch": 0,\n  "Fare": -25,\n  "Embarked": "Z"\n}'
        page.locator("textarea.body-param__text").fill(invalid_payload)
        time.sleep(0.5)
        page.locator(".execute").first.click()
        time.sleep(1)

        predict_invalid_path = os.path.join(SCREENSHOT_DIR, "fastapi_predict_invalid_swagger.png")
        page.screenshot(path=predict_invalid_path)
        print(f"[+] Saved Swagger /predict (invalid) test: '{predict_invalid_path}'")

        browser.close()


if __name__ == "__main__":
    capture_cli_screenshots()
    capture_swagger_screenshots()
    print("\n[+] ALL SCREENSHOTS GENERATED SUCCESSFULLY IN 'screenshots/' DIRECTORY!")
