# RoboCasa365 — two reproducibility issues with the released demonstrations

Two independent issues found while reconstructing the released RoboCasa365
demonstrations and re-running them in-sim. Each block is a standalone issue
(first line = title, rest = body); post separately if preferred.

### Data source
- **Dataset:** the released **RoboCasa365 (v1.0)** demonstration dataset (the RC365
  benchmark; PandaOmron robot, 12-D actions, atomic tasks), obtained via the official
  release described at
  https://robocasa.ai/docs/build/html/datasets/using_datasets.html.
- Each episode retains the per-episode `extras/`: `states.npz` (flattened MuJoCo
  states), `model.xml.gz` (scene XML), `ep_meta.json`.
- The dataset's `env_args` report it was collected with **`env_version 0.5.1`,
  `robosuite 1.5.2`, `mujoco 3.3.1`**.

### Versions we used to reproduce
`robocasa 1.0.1` (main; also verified `1.0.0` — checks identical) · `robosuite 1.5.2`
· `mujoco 3.3.1` · `numpy 2.2.5` · `python 3.12` · robot `PandaOmron`, 12-D actions.

### Scene reconstruction (shared by both issues)
For each demo we rebuild the exact recorded scene from `extras/` — this is bit-exact:
```
reset_to(env, {"model": model.xml, "ep_meta": json.dumps(ep_meta), "states": states})
# = set_ep_meta → reset → reset_from_xml_string(edit_model_xml(model))
#   → sim.reset → set_state_from_flattened(states[0]) → forward
```

---

## Issue 1 — Released demos disagree with the released success check: `next.reward=1` in the data, but `_check_success()` returns False on the same recorded states (CloseBlenderLid, NavigateKitchen)

**Symptom.** Setting each demo's own recorded MuJoCo states back into the env and
calling the released check, two atomic tasks fail their own recorded success. The
other **16/18** atomic tasks we tested reproduce **100%** (state-replay), so the
harness is correct; the disagreement is task-specific.

| task | demos with `next.reward→1` | reward=1 & re-check PASS | reward=1 & re-check FAIL |
|---|---|---|---|
| CloseBlenderLid | 50/50 | 6 | **44** |
| NavigateKitchen | 50/50 | 38 | **12** |
| OpenDrawer (control) | 50/50 | **50** | 0 |
| *(other 15 atomic tasks)* | — | all 100% | 0 |

**The check method is upstream's own.** The dataset's `next.reward` is itself produced
at LeRobot conversion by `convert_hdf5_lerobot.py::get_traj`:
`reset_to({"states": states[t]})` → `env.get_reward()` = `float(_check_success())`
with `reward_shaping=False`. Our re-check uses the identical path
(`set_state_from_flattened(states[t])` → `forward()` → `update_state()` →
`_check_success()`). So "reward=1 in data but re-check False" means the
conversion-time robocasa returned True on those states while released 1.0.x returns
False on the same states via the same method.

For **CloseBlenderLid**, at reward=1 frames the reconstructed lid sits ~15 cm (Y) from
the check's computed "closed" position (`closed ≈ [4.72, -3.78, 1.37]`,
`lid ≈ [4.73, -3.63, 1.36]`); recomputing the threshold live gives the same value and
the lid moves correctly across frames — not a stale-cache/stuck-lid bug.

**Ruled out:** *(a)* not code version within 1.0.x (1.0.0 == 1.0.1, byte-identical);
*(b)* not assets — the "closed" anchor is a `site` in the blender fixture's model XML
(an asset); we downloaded the official **v1.0.1** `fixtures_lightwheel.zip` and
`diff -rq` vs the assets we tested → **empty** (byte-identical, 25/25 blender models;
`Blender014` `anchor_site` pos identical). Reproduces under the literal official
install.

**Questions:** (1) Which version scored the **released** demos? `env_version 0.5.1`
isn't in the released history — can its `_check_success` for these tasks be shared, or
the demos re-scored with 1.0.1? (2) NavigateKitchen: is the success target
deterministic given `(model.xml, ep_meta, states)`? Its target recompute varies across
reconstructions within one version. (3) Should consumers trust shipped `next.reward`
or re-check with 1.0.1? They disagree here.

---

## Issue 2 — Open-loop replay of the recorded commanded actions does not reproduce the demonstrated success

**Symptom.** Replaying each demo's **recorded commanded actions** (the 12-D OSC-pose +
gripper commands, in robosuite order, stepped as-is) open-loop in its reconstructed
scene does **not** reproduce the demonstrated ~100% success. On the 16 tasks with valid
checks, per-task success is ~**92% mean, not 100%**, concentrated on contact/grasp
tasks. By contrast, **state-teleport replay** (setting `states[t]` each frame instead of
stepping actions) reproduces **100%** — so scene, physics, and checker are all fine;
only *action execution* diverges. Two controller configs give the same result: the
dataset's own recording controller (`env_args`) and a base-frame OSC-pose eval
controller.

Per-task open-loop action-replay success (n=10/task):

| task | success | | task | success |
|---|---|---|---|---|
| SlideDishwasherRack | 100% | | OpenStandMixerHead | 90% |
| TurnOnElectricKettle | 100% | | TurnOffStove | 90% |
| OpenDrawer | 100% | | TurnOnSinkFaucet | 90% |
| OpenCabinet | 100% | | CoffeeSetupMug | 80% |
| CloseFridge | 100% | | CloseToasterOvenDoor | 80% |
| PickPlaceSinkToCounter | 100% | | TurnOnMicrowave | 80% |
| PickPlaceCounterToStove | 100% | | **PickPlaceDrawerToCounter** | **60%** |
| PickPlaceCounterToCabinet | 100% | | *CloseBlenderLid* † | 10% |
| PickPlaceToasterToCounter | 100% | | *NavigateKitchen* † | 80% |

† confounded by Issue 1 (their checks are broken regardless of actions) — exclude when
reading the replay ceiling.

**End-effector tracking.** We logged per-step eef deviation ‖ sim
`robot0_base_to_eef_pos` − recorded `state[7:10]` ‖. On cleanly-replayed tasks it stays
~1–3 mm; on the persistently-failing contact task **PickPlaceDrawerToCounter** it grows
to ~3 cm. Free-motion segments track tightly; divergence appears **at contact** and
compounds into a missed/late grasp → downstream failure — consistent with contact
solver nondeterminism rather than a scaling/frame error.

**Questions:** (1) Is open-loop replay of the recorded commanded actions expected to
reproduce success, or does demo generation/validation rely on closed-loop state
feedback that pure action replay can't reproduce? (2) What controller config + solver
settings were the demos recorded and validated with, for byte-reproducible replay?
(3) Are recorded actions absolute or delta targets, and in which reference frame? (We
consume them as-is, robosuite order.) (4) Any recommended MuJoCo determinism settings
(solver / `ls_iterations`, cone type, `njmax`) to reduce contact-timing
nondeterminism?

---

## Minimal repro (CPU-only; imports only robosuite/robocasa/mujoco/numpy)

```python
import glob, gzip, json
from pathlib import Path
import numpy as np, robosuite
from robosuite.controllers import load_composite_controller_config

DEMOS = Path("<path-to>/robocasa_lerobot_demos/pretrain/atomic")  # released RC365 demos

def demo_dir(task):
    return Path(sorted(glob.glob(str(DEMOS / task / "*" / "lerobot")))[0])

def load_extras(d, ep):
    ed = d / "extras" / f"episode_{ep:06d}"
    states = np.load(ed / "states.npz")["states"]
    xml = gzip.open(ed / "model.xml.gz", "rb").read().decode()
    ep_meta = json.load(open(ed / "ep_meta.json"))
    return states, xml, ep_meta

def load_actions(d, ep):  # LeRobot column order -> robosuite step() order
    import pyarrow.parquet as pq
    f = sorted(glob.glob(str(d / "data" / "*" / f"episode_{ep:06d}.parquet")))[0]
    le = np.array(pq.read_table(f, columns=["action"]).column("action").to_pylist(), float)
    ro = np.empty_like(le)
    ro[:, 0:3], ro[:, 3:6], ro[:, 6] = le[:, 5:8], le[:, 8:11], le[:, 11]   # eef dpos, eef daa, gripper
    ro[:, 7:11], ro[:, 11] = le[:, 0:4], le[:, 4]                            # base motion, control mode
    return ro

def make_env(task):
    cc = load_composite_controller_config(controller=None, robot="PandaOmron")
    if cc.get("body_parts", {}).get("right"):
        cc["body_parts"]["right"].update(input_ref_frame="base", ramp_ratio=0.2)
    return robosuite.make(task, robots="PandaOmron", controller_configs=cc,
                          has_renderer=False, has_offscreen_renderer=False,
                          use_camera_obs=False, use_object_obs=True, ignore_done=True,
                          obj_instance_split="pretrain", layout_ids=-2, style_ids=-2)

def reset_to(env, xml, ep_meta, s0):
    env.set_ep_meta(ep_meta); env.reset()
    env.reset_from_xml_string(env.edit_model_xml(xml))
    env.sim.reset(); env.sim.set_state_from_flattened(s0); env.sim.forward()
    env.update_state()

def run(task, ep, mode):  # mode: "states" (teleport) or "actions" (open-loop replay)
    d = demo_dir(task); states, xml, ep_meta = load_extras(d, ep)
    env = make_env(task); reset_to(env, xml, ep_meta, states[0])
    if mode == "states":
        for t in range(len(states)):
            env.sim.set_state_from_flattened(states[t]); env.sim.forward(); env.update_state()
            if env._check_success(): env.close(); return True
    else:
        for a in load_actions(d, ep):
            env.step(a)
            if env._check_success(): env.close(); return True
    env.close(); return False

# CloseBlenderLid: run(..., "states") is False though the demo ships next.reward=1  (Issue 1)
# PickPlaceDrawerToCounter: run(..., "actions") often False though "states" is True (Issue 2)
```

Happy to share the full harness or per-episode logs for either issue.
