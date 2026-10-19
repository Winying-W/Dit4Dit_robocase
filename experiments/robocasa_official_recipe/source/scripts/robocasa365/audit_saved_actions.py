"""Independently reconstruct simulator commands from saved policy predictions.

Uses only NumPy and saved training statistics, not the adapter's decoder or the
runtime action converter. Can audit completed trials while an evaluation runs.
"""
import argparse
import json
from pathlib import Path

import numpy as np


def audit_trial(folder, statistics, execute_horizon):
    result = json.loads((folder / 'result.json').read_text())
    with np.load(folder / 'trajectory.npz', allow_pickle=False) as trace:
        prediction = trace['normalized_chunks'].copy()
        query_steps = trace['query_steps'].copy()
        sent = trace['actions_dataset_order'].copy()
        received = trace['actions_actually_received_by_simulator'].copy()
    steps = result['steps']
    np.testing.assert_array_equal(query_steps, np.arange(0, steps, execute_horizon))
    assert prediction.shape == (len(query_steps), 16, 32)
    assert sent.shape == received.shape == (steps, 12)
    assert np.isfinite(prediction).all() and np.isfinite(sent).all()
    assert np.all(np.abs(sent) <= 1)

    # Independent formula in float64; allow float32 rounding in the producer.
    low = np.asarray(statistics['action']['min'], dtype=np.float64)
    high = np.asarray(statistics['action']['max'], dtype=np.float64)
    decoded = prediction[..., :12].astype(np.float64).copy()
    continuous = [0, 1, 2, 3, 5, 6, 7, 8, 9, 10]
    decoded[..., continuous] = (
        (decoded[..., continuous] + 1) * 0.5
        * (high[continuous] - low[continuous]) + low[continuous]
    )
    executed = decoded[:, :execute_horizon, :].reshape(-1, 12)[:steps]
    expected_sent = np.clip(executed, -1, 1)
    np.testing.assert_allclose(sent, expected_sent, rtol=0, atol=2e-7)

    expected_sim = expected_sent.copy()
    for index in (4, 11):
        expected_sim[:, index] = np.where(expected_sim[:, index] < 0.5, -1, 1)
    expected_sim = expected_sim[:, [5, 6, 7, 8, 9, 10, 11, 0, 1, 2, 3, 4]]
    np.testing.assert_allclose(received, expected_sim, rtol=0, atol=2e-7)
    contract = json.loads((folder / 'controller_contract.json').read_text())
    assert contract['input_type'] == 'delta' and contract['reference_frame'] == 'base'
    np.testing.assert_allclose(contract['output_max'], [.05, .05, .05, .5, .5, .5], rtol=0, atol=1e-12)
    return dict(
        trial=result['trial'], seed=result['seed'], steps=steps,
        prediction_to_sent_max_error=float(np.max(np.abs(expected_sent - sent))),
        prediction_to_simulator_max_error=float(np.max(np.abs(expected_sim - received))),
        executed_clipped_components_by_dataset_dimension=np.sum(np.abs(executed) > 1, axis=0).tolist(),
        executed_arm_clipped_component_fraction=float(np.mean(np.abs(executed[:, 5:11]) > 1)),
        executed_gripper_closed_fraction=float(np.mean(sent[:, 11] >= .5)),
        executed_base_mode_fraction=float(np.mean(sent[:, 4] >= .5)),
        passed=True,
    )


def audit_phase(directory, statistics):
    evaluation = json.loads((directory / 'evaluation.json').read_text())
    trials = [audit_trial(directory / f"trial_{trial['trial']:03d}", statistics,
                          evaluation['execution_horizon']) for trial in evaluation['episodes']]
    return dict(
        evaluation_status=evaluation['status'], completed_trials=len(trials),
        audited_steps=sum(row['steps'] for row in trials), trials=trials,
        passed=bool(trials) and all(row['passed'] for row in trials),
        scope='Saved predictions -> training inverse normalization -> raw clipping -> official binary thresholds and reordering -> actual simulator input. No adapter decoding code is used.',
    )


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--phase', type=Path, required=True)
    parser.add_argument('--normalization', type=Path, required=True)
    parser.add_argument('--output', type=Path, required=True)
    args = parser.parse_args()
    report = audit_phase(args.phase, json.loads(args.normalization.read_text()))
    args.output.parent.mkdir(parents=True, exist_ok=True)
    args.output.write_text(json.dumps(report, indent=2) + '\n')
    print(json.dumps({key: value for key, value in report.items() if key != 'trials'}, indent=2))


if __name__ == '__main__':
    main()
