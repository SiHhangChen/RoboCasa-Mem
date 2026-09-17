from types import SimpleNamespace

import numpy as np
import pytest

from bench.execution.manipulation import ManipulationRuntimeConfig
from bench.tasks.tm02.execution import door


def make_controller(monkeypatch, fractions, inward_pass=True):
    values = iter(fractions)
    env = SimpleNamespace(
        cabinet_open_fraction=lambda: next(values),
        physical_observer=SimpleNamespace(config=SimpleNamespace(cabinet_open_threshold=0.7)),
        semantic_state=SimpleNamespace(absorbing_failure=False),
    )
    current = np.eye(4)
    current[:3, 3] = [0.6, 0.8, 1.2]
    executor = SimpleNamespace(
        config=ManipulationRuntimeConfig(),
        current_eef_site_world=lambda: current,
    )
    calls = []
    def run(target, **kwargs):
        calls.append((target.copy(), kwargs))
        assert kwargs['stage'] in executor.config.planner_args['preserve_ik_branch_stages']
        return {'pass': inward_pass}
    controller = SimpleNamespace(
        env=env, obs={}, stage_executor=executor,
        refresh_obs=lambda: None, run_planned_stage=run,
    )
    monkeypatch.setattr(door, 'matrix_from_base_obs', lambda obs: np.eye(4))
    monkeypatch.setattr(door, 'target_in_current_base', lambda c, target: target)
    return controller, calls


def test_inward_precedes_return_and_preserves_height(monkeypatch):
    controller, calls = make_controller(monkeypatch, [0.75, 0.75, 0.74])
    original = controller.stage_executor.config
    def return_arm():
        assert len(calls) == 1
        assert controller.stage_executor.config is original
        return {'pass': True}
    result = door.return_arm_after_cabinet_open(controller, return_arm)
    assert result['pass']
    target = calls[0][0]
    np.testing.assert_allclose(target[:3, 3], [0.54, 0.72, 1.2])
    np.testing.assert_array_equal(target[:3, :3], np.eye(3))
    assert result['cabinet_open_fraction_after_return'] == 0.74


@pytest.mark.parametrize('inward_pass,fraction', [(False, 0.75), (True, 0.1)])
def test_failed_disengagement_does_not_return_arm(monkeypatch, inward_pass, fraction):
    controller, calls = make_controller(monkeypatch, [0.75, fraction], inward_pass)
    original = controller.stage_executor.config
    def forbidden_return():
        pytest.fail('must not return arm after failed disengagement')
    result = door.return_arm_after_cabinet_open(controller, forbidden_return)
    assert not result['pass']
    assert result['reason'] == 'tm02_cabinet_post_open_inward_failed'
    assert controller.stage_executor.config is original


def test_door_closing_during_return_is_failure(monkeypatch):
    controller, calls = make_controller(monkeypatch, [0.75, 0.75, 0.1])
    result = door.return_arm_after_cabinet_open(controller, lambda: {'pass': True})
    assert not result['pass']
    assert not result['checks']['cabinet_remains_open']


def test_planner_exception_restores_config(monkeypatch):
    controller, calls = make_controller(monkeypatch, [0.75])
    original = controller.stage_executor.config
    def fail(*args, **kwargs):
        raise RuntimeError('planner error')
    controller.run_planned_stage = fail
    with pytest.raises(RuntimeError):
        door.return_arm_after_cabinet_open(controller, lambda: {'pass': True})
    assert controller.stage_executor.config is original
