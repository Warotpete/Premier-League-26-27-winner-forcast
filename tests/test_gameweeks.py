import unittest
import numpy as np
import pandas as pd
from engine import gameweek_points

class FixedModel:
    def rates(self,home,away):
        return 1.5,1.
    def probabilities(self,home,away):
        return np.array([.5,.25,.25])

class GameweekTests(unittest.TestCase):
    def fixtures(self):
        return pd.DataFrame([
            {'id':1,'gameweek':1,'kickoff':pd.Timestamp('2026-08-22',tz='UTC'),'home':'A','away':'B','hg':2,'ag':0,'finished':True},
            {'id':2,'gameweek':3,'kickoff':pd.Timestamp('2026-09-05',tz='UTC'),'home':'B','away':'A','hg':0,'ag':1,'finished':True},
            {'id':3,'gameweek':3,'kickoff':pd.Timestamp('2026-09-06',tz='UTC'),'home':'A','away':'B','hg':np.nan,'ag':np.nan,'finished':False}])
    def test_actual_blank_double_and_expectation(self):
        result=gameweek_points(FixedModel(),self.fixtures(),'2026-08-23',200)
        a=result[result.team=='A'].set_index('gameweek')
        self.assertEqual(a.loc[0,'expected_points'],0)
        self.assertEqual(a.loc[1,'actual_points'],3)
        self.assertEqual(a.loc[1,'points_low'],3)
        self.assertEqual(a.loc[2,'expected_points'],3)
        self.assertEqual(a.loc[3,'expected_points'],3+1+1.75)
        self.assertTrue(np.isnan(a.loc[3,'actual_points']))
        self.assertTrue((a.expected_points.diff().dropna()>=0).all())
    def test_future_score_does_not_affect_forecast(self):
        fixtures=self.fixtures()
        before=gameweek_points(FixedModel(),fixtures,'2026-08-23',200)
        fixtures.loc[1,['hg','ag']]=[20,0]
        after=gameweek_points(FixedModel(),fixtures,'2026-08-23',200)
        pd.testing.assert_frame_equal(before,after)
    def test_unassigned_fixture_blocks_curve(self):
        fixtures=self.fixtures()
        fixtures.loc[2,'gameweek']=np.nan
        with self.assertRaisesRegex(ValueError,'no assigned gameweek'):
            gameweek_points(FixedModel(),fixtures,'2026-08-23')
    def test_known_double_week_is_exact(self):
        fixtures=self.fixtures()
        fixtures.loc[2,['hg','ag','finished']]=[1,1,True]
        result=gameweek_points(FixedModel(),fixtures,'2026-09-07',100)
        a=result[(result.team=='A') & (result.gameweek==3)].iloc[0]
        self.assertEqual(a.actual_points,7)
        self.assertEqual(a.expected_points,7)
        self.assertEqual(a.points_low,a.points_high)

if __name__=='__main__':
    unittest.main()
