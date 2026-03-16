"""Final state persistence test with reload."""
import os
from playwright.sync_api import sync_playwright

os.makedirs("debug_screenshots", exist_ok=True)

def main():
    with sync_playwright() as p:
        browser = p.chromium.launch(headless=False)
        context = browser.new_context(viewport={"width": 1400, "height": 900})
        page = context.new_page()

        page.on("console", lambda msg: print(f"  [{msg.type}] {msg.text[:120]}") if any(k in msg.text for k in ["loading from DB", "loadedFromDB", "restoringWeb"]) else None)

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
        page.wait_for_timeout(6000)
        print("1. Loaded")

        # Select Anime category + Beserk tag
        page.locator("text=Anime").first.click()
        page.wait_for_timeout(1000)
        page.locator(".tags .tag").first.click(force=True)
        page.wait_for_timeout(5000)

        imgs_before = len(page.locator("img.Image").all())
        settings = page.evaluate("JSON.parse(localStorage.getItem('loki-settings') || '{}')")
        session = page.evaluate("JSON.parse(localStorage.getItem('loki-session') || '{}')")
        print(f"\n2. State before reload:")
        print(f"   activeCategory: {settings.get('activeCategory', 'NOT SET')}")
        print(f"   tags: {session.get('query', {}).get('dbQuery', {}).get('tags', [])}")
        print(f"   images: {imgs_before}")

        page.screenshot(path="debug_screenshots/final_before.png")

        # RELOAD
        print("\n--- RELOADING ---")
        page.reload()
        page.wait_for_timeout(10000)

        imgs_after = len(page.locator("img.Image").all())
        active_cat = page.evaluate("document.querySelector('.category.active .category-label')?.textContent || 'none'")
        active_tags = page.evaluate("Array.from(document.querySelectorAll('.tags .tag.active .label')).map(t => t.textContent)")

        print(f"\n3. State after reload:")
        print(f"   active category (visual): {active_cat}")
        print(f"   active tags (visual): {active_tags}")
        print(f"   images: {imgs_after}")

        page.screenshot(path="debug_screenshots/final_after.png")

        match = imgs_before == imgs_after and imgs_before > 0
        print(f"\n{'PASS' if match else 'FAIL'}: Images before={imgs_before}, after={imgs_after}")

        browser.close()

if __name__ == "__main__":
    main()
