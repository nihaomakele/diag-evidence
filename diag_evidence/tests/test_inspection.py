from dataclasses import replace
import unittest

from diag_evidence.demo import run_scenario
from diag_evidence.patrol import ExecutionConfig, simulate
from diag_evidence.patrol.inspection import Alert, Asset, InspectionPlan, alert_from_result
from diag_evidence.patrol.models import Observation, TaskSpec
from diag_evidence.tests.test_patrol import tiny


class InspectionTests(unittest.TestCase):
    def test_real_diagnosis_result_schema_drives_gate(self):
        _, high, _ = run_scenario('consistent')
        _, low, _ = run_scenario('low_confidence')
        _, unavailable, _ = run_scenario('service_down')

        def adapt(result):
            return alert_from_result(asset='pump', observed_at=1, available_at=2, result=result)

        self.assertEqual((adapt(high).classification, adapt(high).supported), ('fault', True))
        self.assertEqual((adapt(low).classification, adapt(low).supported), ('fault', False))
        self.assertEqual(adapt(unavailable).classification, 'unavailable')
        plan = InspectionPlan((Asset('pump', (1, 0), 'fault'),), (adapt(high),))
        run = simulate(tiny(), execution=ExecutionConfig(), inspection=plan)
        self.assertEqual(run['metrics']['inspection_fault_findings'], 1)

    def test_alert_release_deduplication_and_confirmed_finding(self):
        scenario = tiny(report_delay=2, tasks=(TaskSpec('T0', (2, 0), 7, 15, 2),),
                        observations=(Observation('mission', 'recheck-1', 1, 'completed',
                                                  3, 3, 'R0', 'spoof'),))
        alert = Alert('pump', 'first', 1, 2, 'fault', True)
        plan = InspectionPlan((Asset('pump', (1, 0), 'normal'),),
                              (alert, replace(alert, available_at=3),
                               replace(alert, run_id='second', available_at=4)))
        run = simulate(scenario, execution=ExecutionConfig(), inspection=plan)
        self.assertEqual(run['metrics']['inspection_tasks'], 1)
        self.assertEqual(run['metrics']['inspection_confirmed'], 1)
        self.assertEqual(run['metrics']['inspection_ignored_alerts'], 2)
        self.assertEqual(run['metrics']['message_rejections'], {'invalid_message': 1})
        self.assertEqual([t['id'] for t in run['frames'][1]['tasks']], ['T0'])
        self.assertEqual(run['frames'][5]['inspection_assets'][0]['state'], 'awaiting_recheck')
        self.assertEqual(run['frames'][7]['inspection_assets'][0]['state'], 'normal')
        self.assertEqual(run['inspection_assets'][0]['confirmed_at'], 7)

    def test_weak_alarm_gate_and_stale_report(self):
        asset = Asset('pump', (1, 0), 'normal')
        weak = Alert('pump', 'weak', 0, 1, 'fault', False)
        stale = Alert('pump', 'stale', 0, 13, 'fault', True)
        plan = InspectionPlan((asset,), (weak, stale))
        config = ExecutionConfig()
        verified = simulate(tiny(), execution=config, inspection=plan)
        all_faults = simulate(tiny(), execution=config, inspection=plan, alarm_policy='all')
        self.assertEqual(verified['metrics']['inspection_tasks'], 0)
        self.assertEqual(all_faults['metrics']['inspection_tasks'], 1)
        self.assertEqual(all_faults['metrics']['inspection_normal_findings'], 1)
        ignored = [e['reason'] for e in verified['events'] if e['kind'] == 'inspection_alert_ignored']
        self.assertEqual(ignored, ['insufficient_support', 'stale'])
