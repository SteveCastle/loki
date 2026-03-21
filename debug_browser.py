"""Test thumbnail generation - check items without thumbnails get them generated."""
import os
import time
from playwright.sync_api import sync_playwright

os.makedirs("debug_screenshots", exist_ok=True)

def main():
    with sync_playwright() as p:
        browser = p.chromium.launch(headless=False)
        context = browser.new_context(viewport={"width": 1400, "height": 900})
        page = context.new_page()

        preview_results = []
        def on_response(response):
            if "/api/media/preview" in response.url and response.request.method == "POST":
                body = ""
                try: body = response.text()[:100]
                except: pass
                preview_results.append(f"[{response.status}] {body}")
        page.on("response", on_response)

        page.goto("http://localhost:8090/app/")
        page.wait_for_timeout(2000)
        if "login" in page.url.lower():
            form = page.locator("#loginForm")
            form.locator("#username").fill("steve")
            form.locator("#password").fill("t2ng0d0wn")
            form.locator("button[type='submit']").click()
            page.wait_for_timeout(3000)
            if "/app" not in page.url:
                page.goto("http://localhost:8090/app/")
                page.wait_for_timeout(3000)
        page.wait_for_timeout(8000)

        print(f"Preview API results: {len(preview_results)}")
        null_count = sum(1 for r in preview_results if "null" in r)
        path_count = sum(1 for r in preview_results if "thumbnail" in r.lower() or "\\\\" in r or "/" in r)
        print(f"  null (no thumbnail): {null_count}")
        print(f"  with path (has thumbnail): {path_count}")
        for r in preview_results[:5]:
            print(f"  {r}")

        # Wait a bit more for async thumbnail generation
        page.wait_for_timeout(5000)

        # Now reload to see if thumbnails were generated
        preview_results.clear()
        page.reload()
        page.wait_for_timeout(10000)

        print(f"\nAfter reload - Preview API results: {len(preview_results)}")
        null_count2 = sum(1 for r in preview_results if "null" in r)
        path_count2 = sum(1 for r in preview_results if "thumbnail" in r.lower() or "\\\\" in r or "/" in r)
        print(f"  null: {null_count2}")
        print(f"  with path: {path_count2}")
        for r in preview_results[:5]:
            print(f"  {r}")

        page.screenshot(path="debug_screenshots/thumbnail_test.png")
        browser.close()

if __name__ == "__main__":
    main()
