"""Run after installing requirements and downloading historical data."""
import unittest
from pathlib import Path
from streamlit.testing.v1 import AppTest

class DashboardTests(unittest.TestCase):
    @unittest.skipUnless(Path('data/raw/2627.csv').exists(),'Download season data first')
    def test_forecast_match_explorer_and_backtest(self):
        app=AppTest.from_file(str(Path(__file__).resolve().parents[1]/'app.py'),default_timeout=90).run()
        self.assertEqual(len(app.exception),0)
        self.assertEqual(len(app.error),0)
        self.assertGreaterEqual(len(app.metric),6)
        self.assertIn('Points by gameweek',[tab.label for tab in app.tabs])
        self.assertFalse(any(select.label=='Season' for select in app.selectbox))
        app.slider[0].set_value(5).run(timeout=90)
        self.assertEqual(len(app.exception),0)
        app.button[0].click().run(timeout=90)
        self.assertEqual(len(app.exception),0)
        self.assertEqual(len(app.error),0)
        self.assertGreaterEqual(len(app.metric),10)

if __name__=='__main__':
    unittest.main()
