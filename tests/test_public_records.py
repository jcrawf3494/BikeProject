import importlib.util
import json
from pathlib import Path
import unittest

ROOT = Path(__file__).resolve().parents[1]
spec = importlib.util.spec_from_file_location('public_checks', ROOT / 'tools/check_public_artifacts.py')
checks = importlib.util.module_from_spec(spec)
spec.loader.exec_module(checks)


class PublicRecordsTests(unittest.TestCase):
    def test_public_tree_and_all_aggregate_arithmetic(self):
        report = checks.inspect_tree()
        self.assertEqual(report['findings'], [], '\n'.join(report['findings']))
        self.assertEqual(report['run_folders'], 29)
        self.assertEqual(report['model_results'], 96)
        self.assertGreater(report['aggregate_metric_records_checked'], 100)

    def test_reconstruction_protocols_keep_distinct_populations(self):
        results = checks.load(ROOT / 'validation/reconstruction_10354_fixed_test_results.json')
        compressed = checks.load(ROOT / 'validation/reconstruction_10354_compressed_results.json')
        self.assertEqual(len(results), 12)
        self.assertEqual(len(compressed), 6)
        self.assertTrue(all(r['training_rows'] == 10354 and r['evaluation_rows'] == 133553 for r in results))
        self.assertTrue(all(r['training_rows'] == 8283 and r['evaluation_rows'] == 2071 for r in compressed))
        self.assertTrue(all(r['compressed_total_rows'] == 10354 for r in compressed))

    def test_failures_and_fixtures_cannot_be_called_final_results(self):
        runs = checks.load(ROOT / 'experiments/registry.json')['runs']
        failed = [r for r in runs if r['status'] == 'failed']
        self.assertEqual(len(failed), 1)
        self.assertEqual(failed[0]['measured_result_count'], 0)
        for run in runs:
            if run['kind'] == 'technical_verification':
                self.assertTrue(all(r['technical_verification'] for r in run['measured_results']))


if __name__ == '__main__':
    unittest.main()
