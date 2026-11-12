import unittest
import tempfile
from pathlib import Path
from unittest.mock import patch
from profile_observer import HTTPObserver, parse_metrics


class MetricsTests(unittest.TestCase):
    def base(self):
        return '\n'.join(k+' 0' for k in ('vllm:num_requests_running',
            'vllm:num_requests_waiting','vllm:generation_tokens_total','vllm:prompt_tokens_total'))

    def test_aggregate_distinct_engine_labels_and_skip_histogram_buckets(self):
        data=self.base()+'\nvllm:generation_tokens_total{engine="0"} 10\nvllm:generation_tokens_total{engine="1"} 12\nvllm:queue_bucket{le="1"} 800'
        values,raw=parse_metrics(data)
        self.assertEqual(values['vllm:generation_tokens_total'],22)
        self.assertNotIn('vllm:queue_bucket',values)

    def test_missing_required_metric_invalidates_sample(self):
        with self.assertRaises(ValueError):parse_metrics('vllm:generation_tokens_total 10')

    def test_ignore_invalid_nonfinite_optional_values(self):
        values,_=parse_metrics(self.base()+'\nvllm:optional NaN\nvllm:optional2 bad')
        self.assertNotIn('vllm:optional',values)
        self.assertNotIn('vllm:optional2',values)

    def test_clean_thread_exit_does_not_imply_successful_polls(self):
        with tempfile.TemporaryDirectory() as tmp:
            observer=HTTPObserver(Path(tmp)/'metrics.jsonl',{},interval=.001,max_seconds=.005)
            with patch('profile_observer.urlopen',side_effect=OSError('unavailable')):
                observer._loop()
            self.assertTrue(observer.summary['complete'])
            self.assertGreater(observer.summary['errors'],0)
            self.assertEqual(observer.summary['successful_samples'],0)
            self.assertNotIn('first_success_epoch',observer.summary)

if __name__=='__main__':unittest.main()
