import unittest
import numpy as np
import pandas as pd
from engine import GoalModel, simulate

class EngineTests(unittest.TestCase):
    def model(self):
        model=GoalModel()
        model.teams=[f'T{i:02}' for i in range(20)]
        model.index={t:i for i,t in enumerate(model.teams)}
        model.params=np.zeros(42)
        return model
    def test_probability_orientation(self):
        model=self.model()
        model.params[1]=1
        p=model.probabilities('T00','T01')
        self.assertAlmostEqual(p.sum(),1)
        self.assertGreater(p[0],p[2])
    def test_completed_season_is_fixed(self):
        model=self.model()
        rows=[{'home':h,'away':a,'hg':int(h<a),'ag':int(a<h)} for h in model.teams for a in model.teams if h!=a]
        result,dist,remaining=simulate(model,pd.DataFrame(rows),model.teams,200)
        self.assertEqual(remaining,0)
        self.assertEqual(result.iloc[0].team,'T00')
        self.assertEqual(result.iloc[0].title_probability,1)
        np.testing.assert_allclose(dist.sum(axis=0),1)
    def test_reproducibility_and_conservation(self):
        model=self.model()
        empty=pd.DataFrame(columns=['home','away','hg','ag'])
        first,dist,remaining=simulate(model,empty,model.teams,300)
        second,_,_=simulate(model,empty,model.teams,300)
        pd.testing.assert_frame_equal(first,second)
        self.assertEqual(remaining,380)
        self.assertAlmostEqual(first.title_probability.sum(),1)
        self.assertAlmostEqual(first.top_four.sum(),4)
        self.assertAlmostEqual(first.relegation.sum(),3)
    def test_fit_excludes_future_results(self):
        rng=np.random.default_rng(2)
        rows=pd.DataFrame({'date':pd.date_range('2020-01-01',periods=120),'home':['A']*120,'away':['B']*120,'hg':rng.poisson(1.5,120),'ag':rng.poisson(1,120)})
        first=GoalModel().fit(rows,'2020-05-01')
        future=rows.iloc[:1].copy()
        future['date']=pd.Timestamp('2030-01-01')
        future['hg']=100
        second=GoalModel().fit(pd.concat([rows,future]),'2020-05-01')
        np.testing.assert_allclose(first.params,second.params)

if __name__=='__main__':
    unittest.main()
