"""Run every Streamlit page with processed fixture data and assert nothing raises."""
import pytest
from streamlit.testing.v1 import AppTest

APP = str(__import__("pathlib").Path(__file__).resolve().parents[1] / "app.py")
PAGES = ["1_Overview", "2_Demand_Forecasting", "3_Inventory_Intelligence", "4_Replenishment", "5_Sales_Analytics",
         "6_Reports_Exports", "7_Data_Quality", "8_Alerts", "9_Settings"]


def _app(ws, settings, run_forecast=True):
    if run_forecast:
        ws.run_forecast(settings)
    at = AppTest.from_file(APP, default_timeout=120)
    at.session_state["ws"] = ws
    at.session_state["settings"] = settings
    return at.run()


@pytest.mark.parametrize("page", PAGES)
def test_page_renders_with_data(ws, settings, page):
    at = _app(ws, settings)
    at.switch_page(f"pages/{page}.py").run()
    assert not at.exception, [e.value for e in at.exception]


def test_forecast_pages_prompt_instead_of_inventing_results(ws, settings):
    at = _app(ws, settings, run_forecast=False)
    at.switch_page("pages/2_Demand_Forecasting.py").run()
    assert not at.exception
    assert any("Run forecast" in b.label for b in at.button)
    assert not any("WAPE" in m.value for m in at.markdown)


def test_no_data_shows_welcome_everywhere():
    at = AppTest.from_file(APP, default_timeout=60).run()
    for p in PAGES[:6] + ["8_Alerts"]:
        at.switch_page(f"pages/{p}.py").run()
        assert not at.exception
        assert any("Welcome to FORESIGHT — Upload your datasets to begin." in m.value for m in at.markdown), p
