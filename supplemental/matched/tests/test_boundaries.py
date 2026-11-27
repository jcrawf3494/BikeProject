"""Verify disjoint selection roles and categorical fitting vocabularies."""
import sys
from pathlib import Path
import unittest
import numpy as np
import pandas as pd

sys.path.insert(0,str(Path(__file__).resolve().parents[1]))
from train_matched_models import make_roles,validate_phases,meta_matrix,fit_python,predict_python,technical_grids


class MatchedStudyTests(unittest.TestCase):
    def test_stack_validation_and_calibration_are_outside_every_base_cv_fit(self):
        n=600
        train=pd.DataFrame(dict(id=np.arange(n),label=(np.arange(n)%5==0).astype(float)))
        source=pd.DataFrame(dict(id=np.arange(n),training_role=np.where(np.arange(n)%7==0,'calibration','model_selection'),cv_fold=np.arange(n)%3))
        source.loc[source.training_role=='calibration','cv_fold']=np.nan
        roles=make_roles(train,source);validate_phases(roles,train)
        self.assertEqual(set(roles.loc[roles.phase=='calibration','id']),set(source.loc[source.training_role=='calibration','id']))
        self.assertEqual(len(roles),n)

    def test_logit_features_are_finite_at_extreme_probabilities(self):
        result=meta_matrix(pd.DataFrame([[0.,.5,1.],[1.,0.,.5]]))
        self.assertTrue(np.isfinite(result).all())
        self.assertEqual(result[0,1],0.)

    def test_lightgbm_unseen_category_is_not_learned_from_scoring_data(self):
        settings=dict(numeric_features=['duration_sec','temperature'],categorical_features=['rider_locality'])
        train=pd.DataFrame(dict(duration_sec=np.arange(40)+100.,temperature=np.arange(40)+10.,rider_locality=['local','outside']*20,label=[0,1]*20))
        payload=fit_python('P8',technical_grids()['P8'][0],train,settings,42,1)
        scoring=pd.DataFrame(dict(duration_sec=[200.],temperature=[30.],rider_locality=['new-city']))
        before=dict(payload['categories']);probability=predict_python(payload,scoring)
        self.assertEqual(payload['categories'],before)
        self.assertEqual(len(probability),1)
        self.assertNotIn('new-city',payload['categories']['rider_locality'])


if __name__=='__main__':unittest.main()
