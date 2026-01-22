import os
from playwright.sync_api import sync_playwright

SITE_URL = os.environ["SITE_URL"]
USER = os.environ["TEST_USER"]
PASS = os.environ["TEST_PASS"]

def test_login_streamlit():
    with sync_playwright() as p:
        browser = p.chromium.launch(headless=True)
        page = browser.new_page()

        page.goto(SITE_URL, wait_until="domcontentloaded")

        # Preenche o login (pelos rótulos que aparecem no seu app)
        page.get_by_label("Nome do Treinador").fill(USER)
        page.get_by_label("Senha").fill(PASS)
        page.get_by_role("button", name="Entrar").click()

        # Espera um sinal de que logou:
        # no seu app, depois do login aparece o menu lateral com "Treinador:"
        page.wait_for_timeout(1500)
        assert page.get_by_text("Treinador:").is_visible()

        browser.close()

if __name__ == "__main__":
    test_login_streamlit()
    print("OK")
