"""Tests for selection boundaries, tied scores, and fitting-only preprocessing."""
import sys
from pathlib import Path
import unittest
import numpy as np
import pandas as pd
from sklearn.metrics import precision_recall_curve

sys.path.insert(0,str(Path(__file__).resolve().parents[1]))
from run_python_models import cutoff, score, transform, weights, validate_roles


class ProtocolTests(unittest.TestCase):
    def test_cutoff_matches_independent_curve_with_ties_and_saturated_scores(self):
        rng=np.random.default_rng(807)
        cases=[([0,1,0,1],[0,0,1,1]),([0,1,1,0],[0,0,0,0]),([0,1,1,0],[1,1,1,1])]
        cases += [(rng.integers(0,2,100),np.round(rng.random(100),1)) for _ in range(20)]
        for labels,probability in cases:
            threshold,policy=cutoff(labels,probability)
            precision,recall,_=precision_recall_curve(labels,probability)
            den=.25*precision+recall
            curve=np.divide(1.25*precision*recall,den,out=np.zeros_like(den),where=den>0)
            self.assertAlmostEqual(score(labels,probability,threshold)['customer_f0_5'],float(curve.max()),places=12)
            self.assertFalse(policy['test_used'])

    def test_scoring_data_cannot_change_fitted_medians(self):
        settings=dict(numeric_features=['duration_sec','temperature'],categorical_features=['rider_locality'])
        fit=pd.DataFrame(dict(duration_sec=[99.,999.,9999.],temperature=[10.,np.nan,30.],rider_locality=['local',None,'outside']))
        _,medians=transform(fit,settings)
        scoring=pd.DataFrame(dict(duration_sec=[np.nan],temperature=[np.nan],rider_locality=['new-category']))
        result,after=transform(scoring,settings,medians)
        self.assertEqual(after,medians)
        self.assertEqual(result.temperature.iloc[0],20.)
        self.assertAlmostEqual(result.duration_sec.iloc[0],np.log1p(999.))
        self.assertEqual(result.rider_locality.iloc[0],'new-category')

    def test_roles_must_cover_exact_training_ids_once(self):
        train=pd.DataFrame({'id':range(5)})
        valid=pd.DataFrame(dict(id=range(5),training_role=['model_selection']*3+['calibration']*2,cv_fold=[0,1,2,np.nan,np.nan]))
        validate_roles(valid,train)
        for invalid in [valid.iloc[:-1],pd.concat([valid,valid.iloc[:1]]),valid.assign(id=[0,1,2,3,999])]:
            with self.assertRaises(AssertionError):validate_roles(invalid,train)

    def test_balanced_weight_totals_and_no_prediction_weight_dependency(self):
        labels=np.array([0,0,0,0,1,1])
        values,_=weights(labels)
        self.assertAlmostEqual(values[labels==0].sum(),values[labels==1].sum())
        self.assertEqual(score(labels,[.1,.2,.3,.4,.8,.9],.5)['customer_f0_5'],1.)


if __name__=='__main__':unittest.main()
