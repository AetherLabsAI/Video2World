"""Task success and progress: phi predicates, candidate canonicalization, receiver matching, task geometry."""
from __future__ import annotations

import json
import re
from pathlib import Path

import numpy as np
import trimesh
from scipy.spatial import ConvexHull
from scipy.spatial.transform import Rotation as R

from v2w import paths
from v2w.metrics import geometry as M
from v2w.metrics.geometry import entity_mesh, support_planes
from video2sim.bench.evaluate import eval_phi as _eval_phi_v1

# ---------------------------------------------------------------- success predicates
# Success predicates v2 for the FurnitureBench families.
#
# The upstream evaluator (video2sim/bench/evaluate.py::eval_phi, pinned) stays untouched; this module adds two terminal-state
# predicates and delegates every other type to it.
#
#   seated             the part's terminal pose RELATIVE TO THE RECEIVING PART (lamp_base) is within pos_tol / rot_tol of the
#                      demonstrated terminal relative pose (rotation = angle between the part's symmetry axis and the demonstrated
#                      one; spin about the axis is free), the gripper has released it, and the scene is quiescent.
#                      bulb-insert: pos_tol 3 cm, rot_tol 30 deg  ("seated in the socket"; a threaded fit is not reproducible in the
#                      simulator, see the audit).
#   in_region_upright  in_region (3-D, radius) AND the part's symmetry axis within rot_tol of the demonstrated terminal axis AND
#                      released AND quiescent.  hood-place: radius 5 cm, rot_tol 30 deg.
#
# `extra` carries what the terminal-state json knows: grasping_final {name: bool} (release check), quiescent (bool | None = not
# recorded -> completion cannot be certified), target_pose (pose7 of the receiving part IN THE EVALUATED SCENE --
# the candidate's own lamp_base when it built one; falls back to phi['target_pose'], the GT one).

V2_TYPES = ('seated', 'in_region_upright')


def T_of(p):
    p = np.asarray(p, float); M = np.eye(4); M[:3, :3] = R.from_quat([p[4], p[5], p[6], p[3]]).as_matrix(); M[:3, 3] = p[:3]; return M


def axis_angle_deg(a, b):
    return float(np.degrees(np.arccos(np.clip(float(np.dot(a, b)), -1.0, 1.0))))


def rot_angle_deg(Ra, Rb):
    """Geodesic angle between two rotation matrices (full rotation error; used when the part has no symmetry axis)."""
    return float(np.degrees(np.linalg.norm(R.from_matrix(np.asarray(Ra) @ np.asarray(Rb).T).as_rotvec())))


def eval_phi2(phi: dict, final: dict, init: dict, names_map: dict, extra: dict | None = None) -> dict:
    t = phi['type']
    if t not in V2_TYPES:
        out = _eval_phi_v1(phi, final, init, names_map, (extra or {}).get('contacts', {}), (extra or {}).get('scene', {}))
        out['geometry_ok'] = bool(out.get('success'))
        # Legacy region/lift tasks keep their declared semantics. Explicitly
        # required evidence must not be silently discarded by delegation.
        gf = (extra or {}).get('grasping_final') or {}; n = names_map.get(phi.get('source'))
        released = not gf[n] if type(gf.get(n)) is bool else None
        quiet = (extra or {}).get('quiescent')
        out.update(released=released, quiescent=quiet)
        if phi.get('require_released') is True and released is not True:
            out.update(success=False, reason='release evidence missing or still held')
        elif phi.get('require_quiescent') is True and quiet is not True:
            out.update(success=False, reason='quiescence evidence missing or moving')
        return out
    extra = extra or {}
    n = names_map.get(phi['source'])
    if n is None or n not in final:
        return dict(type=t, success=False, reason='source object not matched')
    fin = np.asarray(final[n], float); ax = phi.get('axis', 'y'); i = 'xyz'.index(ax or 'y')   # ax None (ext furniture, non-symmetric part): seated uses the FULL rotation error
    gf = extra.get('grasping_final') or {}
    released = (not gf[n]) if type(gf.get(n)) is bool else None          # None = not recorded
    quiescent = extra.get('quiescent')
    out = dict(type=t, released=released, quiescent=quiescent)
    if t == 'seated':
        tgt = extra.get('target_pose'); out['target_source'] = 'scene' if tgt is not None else 'gt'
        if tgt is None: tgt = phi['target_pose']
        rel = np.linalg.inv(T_of(tgt)) @ T_of(fin); ref = np.asarray(phi['gt_rel'], float)
        dp = float(np.linalg.norm(rel[:3, 3] - ref[:3, 3])); ang = axis_angle_deg(rel[:3, i], ref[:3, i]) if ax else rot_angle_deg(rel[:3, :3], ref[:3, :3])
        out.update(rel_pos_m=dp, rel_axis_deg=ang, pos_tol=phi['pos_tol'], rot_tol_deg=phi['rot_tol_deg'])
        if not ax: out['rot_metric'] = 'full'
        geo_ok = dp <= float(phi['pos_tol']) and ang <= float(phi['rot_tol_deg'])
    else:
        c = np.asarray(phi['center'], float); d = float(np.linalg.norm(fin[:len(c)] - c))
        ang = axis_angle_deg(T_of(fin)[:3, i], np.asarray(phi['gt_axis'], float))
        out.update(dist_m=d, axis_deg=ang, radius=phi['radius'], rot_tol_deg=phi['rot_tol_deg'])
        geo_ok = d <= float(phi['radius']) and ang <= float(phi['rot_tol_deg'])
    rel_ok = not phi.get('require_released', True) or released is True
    q_ok = not phi.get('require_quiescent', True) or quiescent is True
    out.update(geometry_ok=bool(geo_ok), success=bool(geo_ok and rel_ok and q_ok))
    if not out['success']:
        out['reason'] = 'geometry' if not geo_ok else (('release state not recorded' if released is None else 'still held') if not rel_ok else ('quiescence not recorded' if quiescent is None else 'not quiescent'))
    return out


# ---------------------------------------------------------------- scene access
def jload(p):
    p = Path(p); return json.loads(p.read_text()) if p.exists() else None



def gt_scene_objects(sd: Path):
    g = jload(sd / 'hidden/scene_gt.json'); z = np.load(sd / 'hidden/objects_6d.npz')
    names = [n for n in z.files if n != 'source']; return g, {n: np.asarray(z[n], float) for n in names}


def gt_clock(sd: Path) -> float:
    """The demonstration's sample interval (s). Every sample records its own fps; a missing clock is an error, never a guess."""
    z = np.load(sd / 'hidden/trajectory.npz', allow_pickle=True)
    if 'fps' in z.files and float(z['fps']) > 0: return 1.0 / float(z['fps'])
    for key in ('time_s', 'video_time_s'):   # twin / robodojo sources record timestamps instead of a frame rate
        if key in z.files and len(z[key]) > 1: return float(np.median(np.diff(np.asarray(z[key], float))))
    raise ValueError('sample declares no demonstration clock (fps / time_s / video_time_s)')


def cand_scene(pkg: Path):
    """The package's scene in the base frame (candidates are already aligned into <name>_baseframe; gt_pkg is base-frame)."""
    from video2sim.bench.protocol import load_manifest, scene_in_base_frame
    m = load_manifest(pkg); return scene_in_base_frame(m, pkg), m


LEGACY_TARGETS = ('tray', 'lamp_base', 'base', 'socket')   # receiving parts the name rules below already know; any other phi['target'] is matched by name (ext furniture)


def _static_pose(entity):
    # Some GT entities carry their poses only in objects_6d.npz. Never invent
    # an origin for geometric receiver matching when the static pose is absent.
    try:
        pos = np.asarray(entity.get('pos'), float)
        quat = np.asarray(entity.get('quat', (1, 0, 0, 0)), float)
    except (TypeError, ValueError):
        return None
    if pos.shape != (3,) or quat.shape != (4,) or not np.isfinite(pos).all() or not np.isfinite(quat).all():
        return None
    return np.r_[pos, quat]


def gt_target(gt_scene: dict, phi: dict, gt_traj: dict):
    """(entity, pose7) of the GT task target: on_top target object; else the tray / lamp_base / base prop; else the socket
    (four walls -> a virtual entity at their mean position, wall0's orientation)."""
    if phi.get('type') == 'on_top' and phi.get('target') in gt_traj:
        e = next((o for o in gt_scene['objects'] if o['name'] == phi['target']), None); return (e, _static_pose(e)) if e else (None, None)
    if phi.get('target') and phi['target'] not in LEGACY_TARGETS:   # phi names its receiving prop explicitly (square_table_top / drawer_box / cabinet_body), or a tracked GT object
        e = next((p for p in gt_scene.get('props', []) + [o for o in gt_scene.get('objects', []) if o['name'] != phi.get('source')] if p['name'] == phi['target']), None)
        if e: return e, _static_pose(e)
    for nm in ('tray', 'lamp_base', 'base'):
        e = next((p for p in gt_scene.get('props', []) if p['name'] == nm), None)
        if e: return e, _static_pose(e)
    walls = [p for p in gt_scene.get('props', []) if p['name'].startswith('socket')]
    if walls:
        c = np.mean([w['pos'] for w in walls], 0); e = dict(walls[0], name='socket', pos=c.tolist()); return e, np.r_[c, walls[0].get('quat', (1, 0, 0, 0))]
    return None, None


def find_target_parts(scene: dict, phi: dict, gt_scene: dict, gt_pos) -> list:
    """All candidate entities that stand in for the GT receiving part (lamp_base). Agents name their parts informatively but often
    split the base ('lamp_base_plate' + 'lamp_base_socket', 'lamp_base_body' + 'lamp_base_top_*'), so: (1) every prop / object whose
    name contains 'base' and that is not the manipulated object; (2) otherwise the nearest non-table prop within 15 cm of the GT
    receiving part (never the manipulated object, never an entity wider than 40 cm). Empty list when nothing qualifies."""
    if phi.get('type') == 'on_top' and phi.get('target'):
        named = [o for o in scene.get('objects', []) + scene.get('props', []) if phi['target'].lower() in o['name'].lower()][:1]
        if named: return named
    gt_t, _ = gt_target(gt_scene, phi, {})
    if gt_t is None: return []
    src = phi.get('source', '').lower()
    ents = [o for o in scene.get('props', []) + scene.get('objects', []) if o['name'].lower() != src]
    if phi.get('target_match') and len(scene.get('objects', [])) == 1: ents = [o for o in ents if o is not scene['objects'][0]]   # DROID: the candidate's single manipulated object is never its own receiver
    if gt_t['name'] == 'socket':
        walls = [p for p in scene.get('props', []) if p['name'].startswith('socket')]
        return walls
    def width(o):
        hs = o.get('half_size')
        return 2 * max(float(v) for v in hs) if hs else 0.0
    # the receiving ASSEMBLY: everything the agent named as part of the lamp (base plate / socket / stem / neck / bulb), minus the
    # manipulated part, minus the hood when the bulb is manipulated (the hood lies elsewhere), never the robot / table
    ban = ('robot', 'table', 'bench', 'tag', 'marker', 'board', 'sheet', 'plate_tag') + (('hood',) if 'bulb' in src else ())   # 'base_tag' is the AprilTag board, not the lamp
    tn = str(phi.get('target') or '').lower()
    if tn and tn not in LEGACY_TARGETS:
        # the receiving part is whatever the candidate named like phi['target'] (or any of phi['target_match'],
        # e.g. 'table_top' / 'tabletop' for square_table_top); never the robot. The lamp name rule below does not apply to these.
        keys = [str(k).lower() for k in (phi.get('target_match') or [tn])]
        named = [o for o in ents if any(k in o['name'].lower() for k in keys) and 'robot' not in o['name'].lower() and width(o) <= 0.6]
        if named: return named
        return unique_geometric_receiver(scene, phi, gt_scene, gt_t, ents)
    else:
        named = [o for o in ents if ('base' in o['name'].lower() or 'lamp' in o['name'].lower()) and not any(b in o['name'].lower() for b in ban) and width(o) <= 0.4]
    if named: return named
    best, bd = None, 0.15
    for o in ents:
        if width(o) > 0.4 or 'table' in o['name'].lower() or 'bench' in o['name'].lower(): continue
        d = float(np.linalg.norm(np.asarray(o['pos'][:2]) - np.asarray(gt_t['pos'][:2])))
        if d < bd: best, bd = o, d
    return [best] if best else []


def find_target(scene: dict, phi: dict, gt_scene: dict, gt_pos):
    """The primary candidate entity standing in for the receiving part (largest of find_target_parts; a virtual 'socket' entity for
    socket walls)."""
    parts = find_target_parts(scene, phi, gt_scene, gt_pos)
    if not parts: return None
    gt_t, _ = gt_target(gt_scene, phi, {})
    if gt_t is not None and gt_t['name'] == 'socket': return dict(parts[0], name='socket', pos=np.mean([w['pos'] for w in parts], 0).tolist())
    def vol(o):
        hs = o.get('half_size'); return float(np.prod([2 * float(v) for v in hs])) if hs else 1e-3
    return max(parts, key=vol)


# ---------------------------------------------------------------- reviewed clip goals
# Reviewed per-clip goals that override an annotated goal the clip does not show (furniture-clip-goal/1).

CLIP_GOALS = json.loads((Path(__file__).parent / 'rules/furniture_clip_goals.json').read_text())
CLIP_GOAL_PROTOCOL = CLIP_GOALS['protocol']


def clip_goal(sample: str, phi: dict) -> dict:
    """phi with the reviewed clip goal applied; unchanged for samples outside the registry."""
    clip = CLIP_GOALS['clips'].get(sample)
    if clip is None or phi is None:
        return phi
    if phi.get('type') != 'seated' or phi.get('goal_kind') != clip['previous_goal_kind']:
        raise ValueError('clip goal registry does not match the annotated phi of ' + sample)
    return dict(phi, goal_kind=CLIP_GOALS['goal_kind'], require_released=clip['require_released'],
                require_quiescent=clip['require_quiescent'], clip_goal=dict(protocol=CLIP_GOAL_PROTOCOL, **clip))


def load_phi(sd) -> dict:
    """None when the sample has no phi.json."""
    p = Path(sd) / 'hidden/phi.json'
    return clip_goal(Path(sd).name, json.loads(p.read_text())) if p.exists() else None


# ---------------------------------------------------------------- object roles
# Task-object selection independent of dict order; no pose or trajectory editing.

OBJECT_ROLE_VERSION = 'ego-object-role/1.0'


def tokens(name):
    aliases = {'wooden': 'wood', 'vegetables': 'vegetable'}
    return frozenset(aliases.get(t, t) for t in re.findall(r'[a-z]+', name.lower())
                     if len(t) > 1)


def match_objects(agent_init, ref_init, *, source, legacy_match):
    """Establish unique name evidence before distance matching.

    A sole delivered dynamic object is the declared manipulated part (public
    scene contract). Multiple parts never get a nearest/first-object fallback
    without evidence. Initial-position errors remain in the geometric metrics.
    """
    agents = sorted(agent_init); refs = sorted(ref_init)
    out = {r: None for r in refs}; used = set(); evidence = {}
    if len(agents) == 1 and not any(tokens(agents[0]) == tokens(r) for r in refs if r != source):
        out[source] = agents[0];used.add(agents[0]);evidence[source] = 'normalized_name' if tokens(agents[0]) == tokens(source) else 'sole_delivered_manipulated_object'
    # Primary role first, then other roles. Avoid incidental scene order.
    order = [source] + [r for r in refs if r != source]
    for r in order:
        if out.get(r) is not None:continue
        if r not in out:
            raise ValueError('task source absent from reference objects')
        exact = [a for a in agents if a not in used and a.lower() == r.lower()]
        equal = [a for a in agents if a not in used and tokens(a) == tokens(r) and tokens(r)]
        # Names like plate_b and white_plate have one explicit object noun.
        nouns = {'plate', 'spoon', 'spatula', 'bottle', 'bowl', 'mug', 'cup',
                 'apple', 'pear', 'lemon', 'banana', 'plum', 'tomato', 'box', 'can'}
        rn = tokens(r) & nouns
        named = [a for a in agents if a not in used and rn and rn <= tokens(a)]
        candidates, rule = (exact, 'exact_name') if exact else ((equal, 'normalized_name') if equal else (named, 'unique_object_noun'))
        # Repeated nouns (two bottles, body/cap) need exact evidence or geometry.
        competing = [rr for rr in refs if rr != r and rn and rn <= tokens(rr)]
        if rule == 'unique_object_noun' and competing:
            candidates = []
        if len(candidates) == 1:
            a = candidates[0];out[r] = a;used.add(a);evidence[r] = rule
    if out[source] is None and len(agents) == 1 and agents[0] not in used:
        a = agents[0]
        # Do not relabel an explicitly named different GT part as the task part.
        contradictory = any(tokens(a) == tokens(r) for r in refs if r != source)
        if not contradictory:
            out[source] = a;used.add(a);evidence[source] = 'sole_delivered_manipulated_object'
    # Preserve established legacy matches not contradicted by stronger name/role evidence.
    for r, a in legacy_match(agent_init, ref_init).items():
        if out.get(r) is None and a is not None and a not in used:
            out[r] = a;used.add(a);evidence[r] = 'legacy_uncontested'
    rest_a = {a: agent_init[a] for a in agents if a not in used}
    rest_r = {r: ref_init[r] for r in refs if out[r] is None}
    if rest_a and rest_r:
        # Prevent legacy single-reference fallback after filtering a multi-part scene.
        old = legacy_match(rest_a, rest_r)
        for r, a in old.items():
            if a is not None and a not in used and np.linalg.norm(np.asarray(agent_init[a])[:3]-np.asarray(ref_init[r])[:3]) <= .25:
                out[r] = a;used.add(a);evidence[r] = 'initial_distance_le_25cm'
    return out, dict(version=OBJECT_ROLE_VERSION, source=source, rules=evidence,
                    unmatched=[r for r in refs if out[r] is None])


def trajectory_object(record, context, scene, source):
    """Fail closed on conflicting identities, absent trajectories, or ambiguity."""
    names = set((record or {}).get('traj') or {})
    if not names:
        raise ValueError('missing actual object trajectory')
    matched = ((scene or {}).get('object_match') or {}).get(source)
    executed = record.get('target')
    if not isinstance(executed, str):
        executed = None
    if matched and executed and matched != executed:
        raise ValueError('scene mapping and executed target disagree')
    name = matched or executed or (source if source in names else None)
    if name is None and len(names) == 1:
        name = next(iter(names))
    if name not in names:
        raise ValueError('task object has no recorded trajectory')
    if context.get('cname') and context['cname'] != name:
        raise ValueError('canonical transform belongs to another object')
    for field in ('init_poses', 'final_poses'):
        if name not in record.get(field, {}):
            raise ValueError('task object missing ' + field)
    return name


# ---------------------------------------------------------------- receiver matching
# Conservative initial-scene receiver correspondence; never fit metric poses.

RECEIVER_MATCHING_VERSION = 'receiver-matching/1.0'
POSITION_TOLERANCE_M = 0.15
EXTENT_RATIO_LIMIT = 3.0
CONTEXT_WORDS = {'robot', 'table', 'bench', 'board', 'sheet', 'tag', 'floor', 'wall'}


def unique_geometric_receiver(scene, phi, gt_scene, target, entities):
    """Name-free fallback. All gates must pass for exactly one candidate.

    Compare receiver-to-source INITIAL vectors (translation invariant, no rotation
    or scale fitting) and sorted intrinsic geometry extents. A missing/ambiguous
    source, contextual surface, unsupported shape, or multiple receivers rejects
    fallback. Bad mesh files raise: infrastructure errors are not missing objects.
    """
    from v2w.metrics.geometry import entity_mesh
    objects = scene.get('objects', [])
    sources = [o for o in objects if o['name'].lower() == phi.get('source', '').lower()]
    if not sources and len(objects) == 1:
        sources = objects
    gt_sources = [o for o in gt_scene.get('objects', []) if o['name'] == phi.get('source')]
    if len(sources) != 1 or len(gt_sources) != 1:
        return []
    source = sources[0]
    def pos(e):
        try:
            p = np.asarray(e.get('pos'), float)
        except (TypeError, ValueError):
            return None
        return p if p.shape == (3,) and np.isfinite(p).all() else None
    def extents(e):
        if e.get('kind') not in {'box', 'cylinder', 'sphere', 'container', 'mesh', 'library'}:
            return None
        # Mesh paths were resolved by scene_in_base_frame; don't guess a package.
        if e.get('kind') == 'mesh':
            path = Path(e.get('mesh_path') or e['collision_path'])
            if not path.is_absolute():
                raise ValueError('Receiver matching requires a resolved mesh path: ' + str(path))
        v = np.sort(np.asarray(entity_mesh(e, Path('/')).extents, float))
        return v if v.shape == (3,) and np.isfinite(v).all() and np.all(v > 1e-6) else None
    gsize = extents(target)
    gp, gs, cs = pos(target), pos(gt_sources[0]), pos(source)
    if gsize is None or any(p is None for p in (gp, gs, cs)):
        return []
    matches = []
    for entity in entities:
        if entity is source or entity['name'].lower() == source['name'].lower():
            continue
        words = set(re.findall(r'[a-z]+', entity['name'].lower()))
        if words & CONTEXT_WORDS:
            continue
        cp = pos(entity)
        if cp is None or np.linalg.norm((cp - cs) - (gp - gs)) > POSITION_TOLERANCE_M:
            continue
        size = extents(entity)
        if size is None:
            continue
        ratios = size / gsize
        if np.any(ratios < 1 / EXTENT_RATIO_LIMIT) or np.any(ratios > EXTENT_RATIO_LIMIT):
            continue
        matches.append(entity)
    return matches if len(matches) == 1 else []


# ---------------------------------------------------------------- completion evidence
# Evidence gate for tasks whose initial geometry already satisfies the endpoint.
#
# An endpoint alone cannot distinguish doing the task from constructing its result.
# Use the task's own geometric predicate, not a universal displacement threshold.
# Pure geometry stays available to snap-fit and diagnostics.

COMPLETION_VERSION = 'task-completion/1'


def _frames(events, name, rec, count):
    frames = []
    objects = rec.get('objects') or list(rec.get('init_poses', {}))
    unscoped_ok = objects == [name] or rec.get('target') == name
    for event in events or []:
        if isinstance(event, dict):
            owner = event.get('object', event.get('name', event.get('target')))
            if owner != name and not (owner is None and unscoped_ok):
                continue
            frame = event.get('frame')
        else:
            if not unscoped_ok:
                continue
            frame = event
        if type(frame) is int and 0 <= frame < count:
            frames.append(frame)
    return sorted(set(frames))


def gate_initial_goal(result, phi, rec, name, geometry, geometry_at_frame=None):
    result['completion_protocol'] = COMPLETION_VERSION
    # Lift/move predicates already compare against the actual initial state;
    # hand/native tasks have their own task-specific trajectory state machines.
    if phi.get('type') not in ('seated', 'in_region_upright', 'in_region', 'on_top', 'in_container', 'hoi4d_place'):
        return result
    if not result.get('success'):
        return result
    initial = rec.get('init_poses', {}).get(name)
    if initial is None:
        result.update(success=False, reason='initial source pose not recorded')
        return result
    initial_goal = bool(geometry_at_frame(initial, 0) if geometry_at_frame else geometry(initial))
    evidence = dict(initial_geometry_ok=initial_goal, rule='leave goal and complete a new placement when initially at goal')
    result['completion_evidence'] = evidence
    if not initial_goal:
        return result
    trajectory = np.asarray((rec.get('traj') or {}).get(name, []), float)
    if trajectory.ndim != 2 or trajectory.shape[1] != 7 or not np.isfinite(trajectory).all():
        result.update(success=False, degenerate_start=True, reason='initially at goal; no trajectory evidence of a new completion')
        return result
    release_required = phi.get('require_released', True) and phi.get('goal_kind') != 'push_t'
    intervals = [(0, len(trajectory))]
    if release_required:
        grasps = _frames(rec.get('attach_events'), name, rec, len(trajectory))
        releases = _frames(rec.get('release_events'), name, rec, len(trajectory))
        intervals = []
        # Do not join an excursion from one attempt to a later release.
        active = None
        for frame, held in sorted([(k, 1) for k in grasps] + [(k, 0) for k in releases]):
            if held and active is None:
                active = frame
            elif not held and active is not None:
                if frame > active:
                    intervals.append((active, frame))
                active = None
        evidence.update(attach_frames=grasps, release_frames=releases)
    for start, end in intervals:
        # Events index action intervals; poses include frame zero. Examine the
        # states through the release boundary; the terminal predicate checks rest.
        for k in range(start + 1, min(end + 1, len(trajectory))):
            if not (geometry_at_frame(trajectory[k].tolist(), k) if geometry_at_frame else geometry(trajectory[k].tolist())):
                evidence.update(new_completion=True, left_goal_frame=k, attempt_start=start, release_frame=end if release_required else None)
                return result
    evidence['new_completion'] = False
    result.update(success=False, degenerate_start=True,
                  reason='already at goal initially; no goal exit during a completed manipulation attempt')
    return result


# ---------------------------------------------------------------- task progress
# Task-local progress from complete manipulation attempts and terminal outcome.

PROGRESS_VERSION = 'task-progress/2'


def rigid_kind(phi, hint=None):
    if phi.get('goal_kind') == 'push_t' or hint == 'rigid_push':
        return 'rigid_push'
    if phi.get('type') == 'lifted' or hint == 'rigid_lift':
        return 'rigid_lift'
    if phi.get('type') == 'seated' and phi.get('goal_kind') not in ('cap_over_lamp', 'set_aside', 'clip_relative_pose'):
        return 'rigid_insert'
    return 'rigid_place'


def definition(kind, phi=None):
    phi = phi or {}
    spec = dict(protocol=PROGRESS_VERSION, kind=kind, initially_held=bool(phi.get('initially_held', False)),
                movement_m=.01, near_goal_m=.10, lift_m=min(.03, float(phi.get('min_dz', .06)) * .5),
                hold_s=0., release_required=bool(phi.get('require_released', True)),
                rule='maximum ordered intermediate prefix within one attempt; validated terminal success is completion')
    overrides = phi.get('progress_spec') or {}
    for k in ('initially_held', 'movement_m', 'near_goal_m', 'lift_m', 'hold_s'):
        if k in overrides: spec[k] = overrides[k]
    for k in ('movement_m', 'near_goal_m', 'lift_m', 'hold_s'):
        if not np.isfinite(spec[k]) or spec[k] < 0:
            raise ValueError('invalid GT progress threshold: ' + k)
    return spec


def _clock(d, n):
    if 'time_s' in d:
        t = np.asarray(d['time_s'], float)
    else:
        dt = d.get('dt')
        if not isinstance(dt, (int, float)) or isinstance(dt, bool) or not np.isfinite(dt) or dt <= 0:
            raise ValueError('progress requires the execution sample clock')
        t = np.arange(n) * dt
    if t.shape != (n,) or not np.isfinite(t).all() or (np.diff(t) <= 0).any():
        raise ValueError('invalid progress clock')
    return t


def _events(values, n):
    out = []
    for value in values or []:
        k = value.get('frame') if isinstance(value, dict) else value
        if isinstance(k, bool) or not isinstance(k, (int, np.integer)) or not 0 <= k < n:
            raise ValueError('invalid manipulation event frame')
        out.append(int(k))
    return sorted(set(out))


def task_progress(kind, d):
    tr = np.asarray(d['traj'], float)
    if tr.ndim != 2 or tr.shape[1] < 3 or not len(tr) or not np.isfinite(tr).all():
        raise ValueError('progress requires finite nonempty object poses')
    n = len(tr); times = _clock(d, n); spec = d.get('progress_spec') or definition(kind, d.get('phi'))
    if spec['kind'] != kind: raise ValueError('progress definition/task kind mismatch')
    success = d.get('phi_success')
    if type(success) is not bool: raise ValueError('progress requires a boolean terminal verdict')
    def first(mask, start, stop):
        # Search after the prerequisite, including a state already true at that time.
        for k in range(start, stop):
            if not mask[k]: continue
            end = int(np.searchsorted(times, times[k] + spec['hold_s'], side='left'))
            if end < stop and np.all(mask[k:end + 1]): return k
        return None
    def stage(name, frame):
        return dict(name=name, done=frame is not None, frame=frame,
                    time_s=float(times[frame]) if frame is not None else None)
    attempts = []
    if kind == 'rigid_push':
        moved = np.linalg.norm(tr[:, :2] - tr[0, :2], axis=1) >= .02
        mv = first(moved, 0, n); stages = [stage('contact_move', mv)]
        goal = d.get('goal_xy')
        if goal is not None:
            near = np.linalg.norm(tr[:, :2] - np.asarray(goal)[:2], axis=1) <= 2 * float(d.get('goal_radius') or .03)
            stages.append(stage('near_goal', first(near, mv, n) if mv is not None else None))
        stages.append(stage('placed', n - 1 if success else None))
    else:
        if kind not in ('rigid_lift', 'rigid_place', 'rigid_insert'): raise ValueError(kind)
        grasps, releases = _events(d.get('attach_events'), n), _events(d.get('release_events'), n)
        events = sorted([(k, 1) for k in grasps] + [(k, 0) for k in releases])
        active = 0 if spec['initially_held'] else None; intervals = []
        for k, held in events:
            if held and active is None: active = k
            elif not held and active is not None:
                intervals.append((active, k)); active = None
        if active is not None: intervals.append((active, None))
        names = (['held_initial'] if spec['initially_held'] else ['grasp'])
        names += ['lift', 'held_high'] if kind == 'rigid_lift' else (['approach' if kind == 'rigid_insert' else 'transport'] + (['release'] if spec['release_required'] else []) + ['seated' if kind == 'rigid_insert' else ('placed' if spec['release_required'] else 'held_at_goal')])
        stages = [stage(k, None) for k in names]
        for start, release in intervals:
            end = release if release is not None else n
            ss = [stage(names[0], start)]
            if kind == 'rigid_lift':
                mask = tr[:, 2] - tr[0, 2] >= spec['lift_m']
            else:
                mask = np.linalg.norm(tr[:, :3] - tr[start, :3], axis=1) >= spec['movement_m']
                if d.get('goal_xy') is not None:
                    mask &= np.linalg.norm(tr[:, :2] - np.asarray(d['goal_xy'])[:2], axis=1) <= spec['near_goal_m']
            move = first(mask, start, end); ss.append(stage(names[1], move))
            if kind != 'rigid_lift' and spec['release_required']: ss.append(stage('release', release if move is not None else None))
            ss.append(stage(names[-1], n - 1 if success else None))
            count = 0
            for s in ss[:-1]:
                if not s['done']: break
                count += 1
            attempts.append(dict(start_frame=start, release_frame=release, achieved_intermediate=count, stages=ss))
        if attempts:
            best = max(attempts, key=lambda a: (a['achieved_intermediate'], a['start_frame']))
            stages = best['stages']
        else: stages[-1] = stage(names[-1], n - 1 if success else None)
    intermediate = 0
    for s in stages[:-1]:
        if not s['done']: break
        intermediate += 1
    total = len(stages)
    return dict(protocol=PROGRESS_VERSION, kind=kind, definition=spec, stages=stages, attempts=attempts,
                achieved=total if success else intermediate, total=total,
                progress=1. if success else intermediate / total, order_ok=True,
                completion_source='terminal_task_success' if success else 'ordered_intermediate_evidence',
                intermediate_evidence_complete=intermediate == total - 1,
                missing_stage_evidence=[s['name'] for s in stages[:-1] if not s['done']],
                terminal_success=success)


# ---------------------------------------------------------------- lid requirement
# Lid requirement: a shape-sorting box must be delivered WITH its top face (the holes are the
# task); a candidate that models it as an open container gets the block dropped in from above and is judged FAILED even though the
# block ends inside. Test: rays cast downward over the receiving entity's footprint from above its top; `coverage` = fraction of rays
# whose first hit lies within `depth_tol` of the entity's top height (lid with holes ~0.8, open box ~0.2-0.3, solid block 1.0).


def lid_coverage(entity: dict, pkg_dir=None, step=0.005, depth_tol=0.015) -> dict:
    m = entity_mesh(entity, pkg_dir)
    if m is None: return dict(coverage=None, reason='no geometry')
    q = entity.get('quat', [1, 0, 0, 0]); T = np.eye(4); T[:3, :3] = R.from_quat([q[1], q[2], q[3], q[0]]).as_matrix(); T[:3, 3] = entity['pos']
    m = m.copy(); m.apply_transform(T); lo, hi = m.bounds; top = float(hi[2])
    xs = np.arange(lo[0] + step / 2, hi[0], step); ys = np.arange(lo[1] + step / 2, hi[1], step); X, Y = np.meshgrid(xs, ys)
    O = np.c_[X.ravel(), Y.ravel(), np.full(X.size, top + 0.02)]; D = np.tile([[0.0, 0.0, -1.0]], (len(O), 1))
    loc, idx, _ = m.ray.intersects_location(O, D, multiple_hits=False)
    hit = np.zeros(len(O), bool); z = np.full(len(O), -np.inf); hit[idx] = True; z[idx] = loc[:, 2]
    inside = hit   # rays that meet the entity at all = its footprint
    n = int(inside.sum())
    if n < 10: return dict(coverage=None, reason='footprint too small')
    cov = float(np.mean(z[inside] >= top - depth_tol))
    return dict(coverage=round(cov, 3), top_z=round(top, 4), n_rays=n, kind=entity.get('kind'))


def lid_ok(entity: dict, pkg_dir, req: dict) -> dict:
    r = lid_coverage(entity, pkg_dir); thr = float((req or {}).get('min_coverage', 0.5))
    r['min_coverage'] = thr; r['ok'] = bool(r.get('coverage') is not None and r['coverage'] >= thr); return r


# ---------------------------------------------------------------- task geometry relations
# Candidate-geometry task relations, isolated review revision.
# No threshold, axis or selection depends on model identity or success labels.
TASK_GEOMETRY_VERSION='task-geometry-review/1'
def world_vertices(entity,pkg,pose=None):
 T=M.T_of(pose if pose is not None else np.r_[entity['pos'],entity.get('quat',[1,0,0,0])]);v=np.asarray(M.entity_mesh(entity,Path(pkg)).vertices)
 return v@T[:3,:3].T+T[:3,3]
def body_axis(entity,pkg):
 """Axis and directed end from actual geometry; symmetric primitives remain lines.
 Bottle-neck sign compares the top/bottom 15% bands; geometry differences
 are resolved above mesh precision (0.1 mm + 1% relative), not world orientation.
 It never uses the desired world orientation to pick the sign.
 """
 mesh=M.entity_mesh(entity,Path(pkg));ax,res=M.symmetry_axis(mesh)
 if ax is None:return dict(status='unresolved',reason='no supported revolution axis')
 e=np.eye(3)['xyz'.index(ax)];v=np.asarray(mesh.vertices);center=mesh.bounds.mean(0);s=(v-center)@e;lo,hi=s.min(),s.max();rad=np.linalg.norm(v-center-np.outer(s,e),axis=1)
 widths=[float(np.max(rad[s<=lo+.15*(hi-lo)])),float(np.max(rad[s>=hi-.15*(hi-lo)]))]
 directed=not np.isclose(widths[0],widths[1],atol=1e-4,rtol=.01)
 if directed and widths[0]<widths[1]:e=-e
 if not directed:
  # Only prove an unsigned axis when both ends actually have the same surface.
  from scipy.spatial import cKDTree
  u=np.eye(3)[("xyz".index(ax)+1)%3];H=2*np.outer(u,u)-np.eye(3);V=v-center;Vh=V@H.T
  residual=max(cKDTree(V).query(Vh)[0].max(),cKDTree(Vh).query(V)[0].max())
  if residual>1e-4:return dict(status='unresolved',reason='asymmetric ends without resolved bottle neck',half_turn_residual_m=float(residual))
 return dict(status='ok',axis=e.tolist(),directed=bool(directed),end_radii_m=widths,reason='narrow neck end' if directed else 'ends geometrically indistinguishable')

def resting_on_support(entity,pkg,pose,scene,clearance_tol=.01,footprint_margin=.005):
 W=world_vertices(entity,pkg,pose);out=[]
 for s in support_planes(scene,pkg):
  T=s['T'];Q=(W-T[:3,3])@T[:3,:3];gap=float(Q[:,2].min());xy=Q[:,:2];inside=bool(((xy>=s['bounds'][0]-footprint_margin)&(xy<=s['bounds'][1]+footprint_margin)).all());inside=inside and (s['hull'] is None or bool((xy@s['hull'][:,:2].T+s['hull'][:,2]<=footprint_margin).all()));out.append(dict(support=s['name'],gap_m=gap,inside=inside,ok=bool(inside and abs(gap)<=clearance_tol),normal=T[:3,2].tolist()))
 return dict(ok=any(x['ok'] for x in out),planes=out,clearance_tol_m=clearance_tol,footprint_margin_m=footprint_margin)

def upright_supported(entity,pkg,pose,scene,tilt_tol_deg=30,clearance_tol=.01):
 axis=body_axis(entity,pkg);support=resting_on_support(entity,pkg,pose,scene,clearance_tol);out=dict(protocol=TASK_GEOMETRY_VERSION,axis=axis,support=support,geometry_ok=False)
 if axis['status']!='ok':return dict(out,reason='semantic axis unresolved')
 d=M.T_of(pose)[:3,:3]@np.asarray(axis['axis']);angles=[]
 for p in support['planes']:
  dot=float(d@p['normal']);dot=dot if axis['directed'] else abs(dot);ang=float(np.degrees(np.arccos(np.clip(dot,-1,1))));angles.append(dict(support=p['support'],angle_deg=ang,supported=p['ok']))
 ok=any(a['supported'] and a['angle_deg']<=tilt_tol_deg for a in angles)
 return dict(out,geometry_ok=ok,angles=angles,tilt_tol_deg=tilt_tol_deg,reason='ok' if ok else 'not upright on delivered support')

def upright_verdict(entity,pkg,scene,rec,name,tilt_tol_deg=30,clearance_tol=.01):
 geo=lambda pose:upright_supported(entity,pkg,pose,scene,tilt_tol_deg,clearance_tol)
 final=rec.get('final_poses',{}).get(name)
 if final is None:return dict(success=False,reason='missing final pose',protocol=TASK_GEOMETRY_VERSION)
 r=geo(final);held=rec.get('grasping_final');held=held.get(name) if isinstance(held,dict) else None;release=held is False;stable=rec.get('quiescent') is True
 r.update(success=bool(r['geometry_ok'] and release and stable),released=release,quiescent=stable)
 if not release:r['reason']='still held or release unrecorded'
 elif not stable:r['reason']='not quiescent or quiescence unrecorded'
 return gate_initial_goal(r,dict(type='in_region_upright',goal_kind='upright_supported',require_released=True),rec,name,lambda pose:geo(pose)['geometry_ok'])

def center_world(entity,pkg,pose):
 mesh=M.entity_mesh(entity,Path(pkg));T=M.T_of(pose);return T[:3,:3]@mesh.centroid+T[:3,3]

def receiver_vertices(parts,pkg,rec=None):
 return np.concatenate([world_vertices(e,pkg,(rec or {}).get('final_poses',{}).get(e['name'])) for e in parts]) if parts else np.empty((0,3))

def receiver_relation(entity,pkg,pose,parts,kind,rec=None,margin=.005):
 W=world_vertices(entity,pkg,pose);V=receiver_vertices(parts,pkg,rec)
 if len(V)<4:return dict(geometry_ok=False,reason='receiving geometry missing')
 c=center_world(entity,pkg,pose);hull=ConvexHull(V[:,:2]);inside_xy=bool(np.max(hull.equations[:,:2]@c[:2]+hull.equations[:,2])<=margin)
 bottom=float(W[:,2].min());top=float(V[:,2].max());low=float(V[:,2].min())
 if kind=='on_top':
  # Query the actual support surface below the source centre.  A plate rim is
  # higher than its interior and must not substitute for the contact surface.
  heights=[]
  for e in parts:
   mesh=M.entity_mesh(e,Path(pkg)).copy();tp=(rec or {}).get('final_poses',{}).get(e['name'],np.r_[e['pos'],e.get('quat',[1,0,0,0])]);mesh.apply_transform(M.T_of(tp))
   hits,_,_=mesh.ray.intersects_location(np.array([[c[0],c[1],top+.1]]),np.array([[0.,0.,-1.]]))
   heights.extend(hits[:,2].tolist())
  surface=max(heights) if heights else None;gap=bottom-surface if surface is not None else None;ok=inside_xy and gap is not None and abs(gap)<=.01
  return dict(geometry_ok=bool(ok),reason='ok' if ok else 'not supported on receiver surface',inside_receiver_footprint=inside_xy,bottom_to_surface_m=gap,support_surface_z_m=surface,contact_tol_m=.01)
 # Convex envelope describes the annotated receiver interior, unlike a sphere
 # around one demonstrated object origin. Partial insertion is permitted because
 # pens/spoons and nested cups can protrude above a rim.
 mesh=M.entity_mesh(entity,Path(pkg));P=M.surface_points(mesh,pose,3000);H=ConvexHull(V);dist=P@H.equations[:,:3].T+H.equations[:,3];inside=(dist<=.003).all(axis=1)
 depth=.002  # require measurable insertion below rim (2 mm), independent of vessel height
 inside &= P[:,2]<=top-depth
 # An envelope alone also contains solid walls and handle holes. Verify that
 # inserted surface points occupy free cavity space enclosed in four horizontal
 # directions; a solid box cannot masquerade as an open container.
 meshes=[]
 for e in parts:
  mesh=M.entity_mesh(e,Path(pkg)).copy();tp=(rec or {}).get('final_poses',{}).get(e['name'],np.r_[e['pos'],e.get('quat',[1,0,0,0])]);mesh.apply_transform(M.T_of(tp));meshes.append(mesh)
 vessel=trimesh.util.concatenate(meshes);ids=np.flatnonzero(inside);cavity=np.zeros(len(P),bool)
 if len(ids):
  Q=P[ids];free=np.ones(len(Q),bool)
  if vessel.is_watertight:free &= ~vessel.contains(Q)
  enclosed=np.ones(len(Q),bool)
  for direction in ([1.,0.,0.],[-1.,0.,0.],[0.,1.,0.],[0.,-1.,0.]):
   enclosed &= vessel.ray.intersects_any(Q,np.tile(direction,(len(Q),1)))
  cavity[ids]=free & enclosed
 fraction=float(np.mean(cavity));ok=fraction>=.05 and bottom>=low-.01
 return dict(geometry_ok=bool(ok),reason='ok' if ok else 'no meaningful insertion into free receiver cavity',inside_fraction=fraction,envelope_fraction=float(np.mean(inside)),minimum_inside_fraction=.05,below_rim_depth_m=depth,source_bottom_m=bottom,receiver_bottom_m=low,receiver_top_m=top,cavity_rule='outside closed vessel material and four horizontal rays intersect receiver walls')

def relation_geometry(entity,pkg,pose,scene,rec,rule,parts):
 kind=rule['kind']
 if kind=='upright_supported':return upright_supported(entity,pkg,pose,scene)
 if kind in ['on_top','in_container']:return receiver_relation(entity,pkg,pose,parts,kind,rec)
 support=resting_on_support(entity,pkg,pose,scene);c=center_world(entity,pkg,pose);out=dict(support=support,geometry_ok=False)
 if kind=='remove_to_support':
  V=receiver_vertices(parts,pkg,rec)
  if len(V)<4:return dict(out,reason='initial container missing')
  H=ConvexHull(V[:,:2]);W=world_vertices(entity,pkg,pose);outside=np.any(W[:,:2]@H.equations[:,:2].T+H.equations[:,2]>.005,axis=1);frac=float(outside.mean());ok=support['ok'] and frac>=.95
  return dict(out,geometry_ok=bool(ok),outside_receiver_fraction=frac,reason='ok' if ok else 'not released outside container onto support')
 direction=np.asarray(rule['direction_world'],float);direction[2]=0;direction/=np.linalg.norm(direction)
 if kind=='move_direction':
  initial=rec['init_poses'][entity['name']];origin=center_world(entity,pkg,initial)
 elif kind=='side_of':
  V=receiver_vertices(parts,pkg,rec)
  if len(V)<4:return dict(out,reason='reference object missing')
  origin=V.mean(0)
 else:raise ValueError(kind)
 delta=float((c-origin)@direction);ok=support['ok'] and delta>=.03
 return dict(out,geometry_ok=bool(ok),direction_displacement_m=delta,min_displacement_m=.03,reason='ok' if ok else 'direction or support not satisfied')

def relation_verdict(entity,pkg,scene,rec,name,rule,parts):
 geo=lambda p:relation_geometry(entity,pkg,p,scene,rec,rule,parts)
 if name not in rec.get('final_poses',{}) or name not in rec.get('init_poses',{}):return dict(success=False,geometry_ok=False,reason='source states missing')
 r=geo(rec['final_poses'][name]);held=(rec.get('grasping_final') or {}).get(name);released=held is False;stable=rec.get('quiescent') is True
 r.update(protocol=TASK_GEOMETRY_VERSION,kind=rule['kind'],success=bool(r['geometry_ok'] and released and stable),released=released,quiescent=stable)
 if not released:r['reason']='still held or release unrecorded'
 elif not stable:r['reason']='not quiescent or quiescence unrecorded'
 def at_frame(p,k):
  states=rec.get('init_poses',{}) if k==0 else {n:tr[k] for n,tr in rec.get('traj',{}).items() if k<len(tr)}
  moving=[e['name'] for e in parts if e['name'] in rec.get('init_poses',{})]
  if any(n not in states for n in moving):raise ValueError('receiver trajectory missing at completion frame')
  return relation_geometry(entity,pkg,p,scene,dict(rec,final_poses=states),rule,parts)['geometry_ok']
 return gate_initial_goal(r,dict(type='in_region',require_released=True),rec,name,lambda p:geo(p)['geometry_ok'],geometry_at_frame=at_frame)


# ---------------------------------------------------------------- semantic axis
# Registered upright-bottle axis: cap-directed when observable, otherwise a line.
# No desired world pose chooses a candidate mesh-axis sign. Task success separately
# requires a resolved neck-up direction or proven end symmetry plus support.
SEMANTIC_AXIS_VERSION='bottle-semantic-axis/1'
def register(ctx,phi):
 if phi.get('type')!='in_region_upright' or phi.get('source')!='bottle':return ctx
 a=body_axis(ctx['ce'],paths.resolve(ctx['pkg']));ctx['semantic_axis']=dict(a,protocol=SEMANTIC_AXIS_VERSION,gt_axis=phi['axis'])
 return ctx

def errors(ctx,cposes,gposes):
 a=ctx['semantic_axis']
 if a['status']!='ok':raise ValueError('unresolved candidate bottle axis')
 c=np.array([M.T_of(p)[:3,:3]@np.array(a['axis']) for p in cposes]);g=np.array([M.T_of(p)[:3,'xyz'.index(a['gt_axis'])] for p in gposes]);dots=np.einsum('ij,ij->i',c,g)
 if not a['directed']:dots=np.abs(dots)
 return np.degrees(np.arccos(np.clip(dots,-1,1)))

def apply_metrics(out,ctx,rec,name,gt,dt,gdt):
 if 'semantic_axis' not in ctx:return
 a=np.array(rec['traj'][name]);g=np.asarray(gt);own=out.get('own_trajectory',{});ix=np.clip(np.rint(np.arange(len(g))*gdt/dt).astype(int),0,len(a)-1);vals=errors(ctx,a[ix],g)
 if own.get('ape') is not None:own['ape'].update(rot_deg=float(vals.mean()),rot_per_frame_deg=vals.tolist(),rotation_protocol=SEMANTIC_AXIS_VERSION,axis_direction_observable=ctx['semantic_axis']['directed'])
 dur=own.get('duration_normalized',{});ids=np.rint(np.linspace(0,len(a)-1,len(g))).astype(int);vs=errors(ctx,a[ids],g)
 if dur.get('ape') is not None:dur['ape'].update(rot_deg=float(vs.mean()),rot_per_frame_deg=vs.tolist(),rotation_protocol=SEMANTIC_AXIS_VERSION,axis_direction_observable=ctx['semantic_axis']['directed'])
 endpoint=float(errors(ctx,[rec['final_poses'][name]],[g[-1]])[0]);out['semantic_terminal_rotation']=dict(protocol=SEMANTIC_AXIS_VERSION,deg=endpoint,directed=ctx['semantic_axis']['directed'],reason=ctx['semantic_axis']['reason'],metric='directed neck axis' if ctx['semantic_axis']['directed'] else 'unsigned body axis; end identity absent from delivered geometry')
 # Existing canonical endpoint positions remain motion diagnostics. For this
 # initial-object anchor they do not contain a declared relative rotation.
 out['canonical']['axis_ambiguity']=ctx['semantic_axis']


# ---------------------------------------------------------------- annotated scene scope
# Resolve annotated physical instances and assemblies before Scene sampling.
#
# Aliases are ordered from specific to permissive; they are NOT a union of every
# matching instance. Nearby auxiliary geometry is collected in the host frame.
# The task evaluator retains its separate task predicate/receiver contract.
SCENE_SCOPE_VERSION='annotated-assembly-scene/2'

def _vertices(e,pkg):
 m=M.entity_mesh(e,pkg);t=M.T_of(e['pos']+e['quat'])
 return np.asarray(m.vertices)@t[:3,:3].T+t[:3,3]

def _attached(primary,part,pkg):
 """Conservative thin-component test in the primary's own coordinate frame.

 Neither a part's name nor GT fit enters this test. A separate large container
 cannot be attached merely because it shares a generic receiver alias.
 """
 T=M.T_of(primary['pos']+primary['quat']);a=np.asarray(M.entity_mesh(primary,pkg).vertices)
 b=(_vertices(part,pkg)-T[:3,3])@T[:3,:3]
 lo,hi=a.min(0),a.max(0);bl,bh=b.min(0),b.max(0);ext=hi-lo
 # The tolerance admits fitted mating surfaces/overhanging lids, not arbitrary
 # objects resting nearby. Parts must be thin relative to the host.
 pe=np.sort(np.asarray(M.entity_mesh(part,pkg).extents))
 if pe[0] > min(.015,.3*max(ext)):return False
 margin=np.maximum(.012,.2*ext)
 if np.any(bl<lo-margin) or np.any(bh>hi+margin):return False
 gap=np.maximum(np.maximum(lo-bh,bl-hi),0)
 return np.linalg.norm(gap)<=.012

def scoped_scene(scene,pkg,gt,sd,phi,source_name):
 meta=json.loads((Path(sd)/'meta.json').read_text());scope=meta.get('gt_scope')
 if not scope:return scene,dict(protocol='full-delivered-scene',applied=False)
 if not isinstance(scope.get('manipulated'),list) or not isinstance(scope.get('interaction'),list):raise ValueError('invalid partial GT scope')
 allg={e['name']:e for e in gt.get('objects',[])+gt.get('props',[])}
 declared=set(scope['manipulated']+scope['interaction'])
 if declared != set(allg):raise ValueError(f'partial GT annotation inventory mismatch: {declared ^ set(allg)}')
 entities=scene.get('objects',[])+scene.get('props',[]);byname={e['name']:e for e in entities}
 if source_name not in byname:raise ValueError('manipulated candidate absent')
 matches={phi['source']:[source_name]};used={source_name};primaries={phi['source']:byname[source_name]}
 from v2w.metrics.geometry import support_planes
 supports=[(p['area'],p['name']) for p in support_planes(scene,pkg) if p['name']!=source_name and p['name']!='<support>']
 support=max(supports)[1] if supports else None
 decisions={}
 for name in sorted(declared-{phi['source']}):
  aliases=phi.get('target_match',[name]) if name==phi.get('target') else [name]
  available=[e for e in entities if e['name'] not in used and e['name']!=support]
  ranked=[]
  for e in available:
   hits=[j for j,a in enumerate(aliases) if a.lower() in e['name'].lower()]
   if hits:
    extent=M.entity_mesh(e,pkg).extents
    ranked.append((min(hits),-float(np.prod(extent)),e['name'],e))
  if ranked:
   primary=min(ranked,key=lambda v:v[:3])[-1]
   decisions[name]=dict(rule='ordered semantic alias, then primary volume',primary=primary['name'],aliases=aliases)
  else:
   available_scene=dict(scene,objects=[e for e in scene.get('objects',[]) if e in available or e['name']==source_name],props=[e for e in scene.get('props',[]) if e in available])
   p=find_target_parts(available_scene,dict(phi,source=source_name,target=name,target_match=aliases),gt,np.asarray(allg[phi['source']]['pos']))
   primary=max(p,key=lambda e:float(np.prod(M.entity_mesh(e,pkg).extents))) if p else None
   decisions[name]=dict(rule='existing conservative geometry fallback',primary=primary['name'] if primary else None)
  matches[name]=[primary['name']] if primary else []
  if primary:primaries[name]=primary;used.add(primary['name'])
 # Resolve auxiliary components only after reserving every independent GT role.
 for name,primary in primaries.items():
  for e in entities:
   if e['name'] not in used and e['name']!=support and _attached(primary,e,pkg):
    matches[name].append(e['name']);used.add(e['name'])
 # Exported compound GT must be matched to the same physical assembly. Rider
 # names are declared by the GT exporter, not inferred from the candidate score.
 rider_aliases={'pen':['pen','marker','utensil']}
 for rider,host in meta.get('merged_into_host',{}).items():
  if host not in primaries:continue
  primary=primaries[host];pv=_vertices(primary,pkg);lo,hi=pv.min(0)-.12,pv.max(0)+.12
  for e in entities:
   if e['name'] in used or e['name']==support:continue
   if any(a in e['name'].lower() for a in rider_aliases.get(rider,[rider])):
    center=_vertices(e,pkg).mean(0)
    if np.all(center>=lo) and np.all(center<=hi):matches[host].append(e['name']);used.add(e['name'])
 assemblies=[]
 for name in sorted(matches):
  parts=[byname[n] for n in matches[name]]
  if parts:assemblies.append(dict(name=name,kind='_scene_assembly',pos=[0.,0.,0.],quat=[1.,0.,0.,0.],_scene_parts=parts))
 if support:used.add(support)
 result=dict(scene,objects=assemblies,props=[byname[support]] if support else [],_scene_table_names=[support] if support else [],_scene_explicit_assemblies=True)
 ignored=[e['name'] for e in entities if e['name'] not in used]
 return result,dict(protocol=SCENE_SCOPE_VERSION,applied=True,matches=matches,decisions=decisions,support_entity=support,ignored_context=ignored,scope=scope,merged_into_host=meta.get('merged_into_host',{}))


def task_receiver(scene,pkg,gt,sd,phi,source_name):
 """Use the same annotated assembly for task receiver geometry and Scene CD.
 Return original physical entities for trajectory predicates, never a fake body.
 """
 target=phi.get('target')
 if not target:return None,[],None
 scoped,scope=scoped_scene(scene,pkg,gt,sd,phi,source_name)
 names=scope.get('matches',{}).get(target,[])
 byname={e['name']:e for e in scene.get('objects',[])+scene.get('props',[])}
 parts=[byname[n] for n in names]
 primary=scope.get('decisions',{}).get(target,{}).get('primary')
 entity=byname.get(primary) or (parts[0] if parts else None)
 assembly=next((e for e in scoped['objects'] if e['name']==target),None)
 return entity,parts,assembly


# ---------------------------------------------------------------- candidate canonicalization
# Candidate-part canonicalization: every terminal-state / trajectory quantity of a CANDIDATE part is
# evaluated with the part re-expressed in the GT mesh's frame convention, via the rigid shape alignment (geometry.canonical_transform).
# Why: agents deliver their own primitives / meshes, often symmetric about local z while the GT
# meshes are symmetric about local y, and primitive origins sit at the centre while the GT origin is the CAD origin -- so the
# symmetry-axis tilt (78-178 deg for parts standing upright on the base) and the relative position were measured in the wrong frame.
# `context()` computes the alignment once per candidate package; `verdict()` evaluates phi v2 on canonical poses.
CAP_XY_TOL, CAP_TILT_TOL = 0.07, 40.0   # hood leaning on the bulb: lamp top within 7 cm of the hood axis, axis within 40 deg of vertical (GT terminals: max 6.4 cm / 36 deg)
CAP_WALL_MARGIN = 0.01   # capped also requires radial <= hood radius at that height + this margin


def context(sd: Path, pkg: Path, scene_json=None) -> dict:
    sd, pkg = Path(sd), Path(pkg); phi = clip_goal(sd.name, jload(sd / 'hidden/phi.json')); g, gt_traj = gt_scene_objects(sd); part = phi['source']
    sc = jload(scene_json) if scene_json else None
    scene, m = cand_scene(pkg); cname = (sc or {}).get('object_match', {}).get(part) or part
    ce = next((o for o in scene['objects'] if o['name'] == cname), None) or (scene['objects'][0] if scene['objects'] else None)
    ge = next(o for o in g['objects'] if o['name'] == part)
    A_obj = M.canonical_transform(ce, pkg, ge, sd / 'gt_pkg') if ce is not None else np.eye(4)
    rule = DROID_RULES.get(sd.name)
    if rule: phi=dict(phi,target=rule.get('target'),target_match=rule.get('target_match'))
    parts_c = find_target_parts(scene, phi, g, gt_traj[part][0]); tgt_c = find_target(scene, phi, g, gt_traj[part][0]); tgt_g, tg_pose = gt_target(g, phi, gt_traj)
    if rule and ce is not None:
        tgt_c, parts_c, _ = task_receiver(scene,pkg,g,sd,phi,ce['name'])
    A_tgt, tgt_pose = np.eye(4), None
    if parts_c and tgt_g is not None and tgt_g['name'] != 'socket':
        # anchor = (xy centroid, top z) of the receiving part's surface: the socket opening the part seats on / the lamp the hood
        # is placed against. The GT receiving mesh is placed so that its anchor coincides with the candidate's (union of base-named
        # parts), with the GT orientation: bases stand upright in both scenes, only their placement differs.
        def anchor(ents, pkg_dir):
            P = []
            for e in ents:
                try:
                    mesh = M.entity_mesh(e, pkg_dir); P.append(M.surface_points(mesh, np.r_[e['pos'], e.get('quat', (1, 0, 0, 0))], 2000))
                except Exception: pass
            P = np.concatenate(P) if P else None
            return None if P is None else np.r_[P[:, :2].mean(0), np.percentile(P[:, 2], 98)]
        gt_parts = [tgt_g] + ([p_ for p_ in g['props'] if p_['name'] == 'lamp_bulb'] if part == 'lamp_hood' else [])   # hood: the lamp = base + bulb
        ac = anchor(parts_c, pkg); ag = anchor(gt_parts, sd / 'gt_pkg')
        if ac is not None and ag is not None:
            A_tgt[:3, 3] = ag - ac; tgt_pose = np.r_[np.asarray(tg_pose[:3], float) + (ac - ag), np.asarray(tg_pose[3:], float)]
    ctx = dict(sample=sd.name, scene=scene, pkg=str(pkg), ce=ce, parts_c=parts_c, part=part, cname=(ce or {}).get('name', part), A_obj=A_obj, A_tgt=A_tgt, target_pose=(tgt_pose.tolist() if tgt_pose is not None else None), target_name=(tgt_c or {}).get('name'), target_parts=[p['name'] for p in parts_c],
                identity=bool(np.allclose(A_obj, np.eye(4), atol=1e-6)), axis_rot_deg=float(np.degrees(np.linalg.norm(__import__('scipy.spatial.transform', fromlist=['Rotation']).Rotation.from_matrix(A_obj[:3, :3]).as_rotvec()))), origin_shift_cm=float(np.linalg.norm(A_obj[:3, 3]) * 100))
    return register(ctx,phi)



def canon_final(ctx: dict, poses: dict, name: str):
    return {name: M.canon_pose7(poses[name], ctx['A_obj']).tolist()}


def _pose_verdict(ctx: dict, phi: dict, rec: dict, name: str) -> dict:
    """phi v2 on the canonicalized terminal pose of `name` in a rollout / replay record (final_poses, init_poses, grasping_final, quiescent)."""
    phi = dict(phi, **FURNITURE_RULES.get(ctx.get('sample', ''), {}).get('parameters', {}))
    n = name
    if n not in rec.get('final_poses', {}) or n not in rec.get('init_poses', {}):
        return dict(success=False, geometry_ok=False, reason='source pose not recorded')
    fin = canon_final(ctx, rec['final_poses'], n); ini = canon_final(ctx, rec['init_poses'], n)
    gf = rec.get('grasping_final')
    r = eval_phi2(phi, fin, ini, {phi['source']: n}, dict(grasping_final=gf, quiescent=rec.get('quiescent'), target_pose=ctx['target_pose']))
    r['canonical'] = dict(A_identity=ctx['identity'], axis_rot_deg=ctx['axis_rot_deg'], origin_shift_cm=ctx['origin_shift_cm'], target_source='scene' if ctx['target_pose'] is not None else 'gt'); r['final_pose_canonical'] = fin[n]
    if phi.get('goal_kind') == 'cap_over_lamp' and ctx.get('ce') is not None and ctx.get('parts_c'):
        # cap_over_lamp: the outcome test is geometric (hood encloses the lamp top), the relative pose stays as a diagnostic
        c = capped(ctx['ce'], Path(ctx['pkg']), rec['final_poses'][n], ctx['parts_c'], Path(ctx['pkg']), xy_tol=CAP_XY_TOL, tilt_tol_deg=CAP_TILT_TOL)   # tolerances chosen so every demonstrated cap passes (GT terminal 30/30)
        rel_ok = not phi.get('require_released', True) or r.get('released') is True; q_ok = not phi.get('require_quiescent', True) or r.get('quiescent') is True
        r['capped'] = c; r['geometry_ok'] = bool(c['success']); r['success'] = bool(c['success'] and rel_ok and q_ok)
        r['reason'] = None if r['success'] else (c.get('reason') if not c['success'] else ('still held' if not rel_ok else 'not quiescent'))
        if r['success']: r['reason'] = 'ok'
    if phi.get('goal_kind') == 'door_on_body' and ctx.get('ce') is not None and ctx.get('parts_c'):
        # cabinet-door: outcome test on the candidate's own cabinet body — the door hangs on the demonstrated
        # hinge side of the body: its long axis is parallel to the demonstrated one, its centre lies near the demonstrated line mapped
        # onto the candidate body (OBB-normalised offset, axes matched by world direction), and it touches the body; the hinge angle
        # (closed flat / open) is free. The demonstrated relative pose is not required (bodies differ in size / orientation, a flat
        # door's flips are not observable).
        db = door_on_body(ctx, phi, rec['final_poses'][n])
        rel_ok = not phi.get('require_released', True) or r.get('released') is True; q_ok = not phi.get('require_quiescent', True) or r.get('quiescent') is True
        r['door_on_body'] = db; r['geometry_ok'] = bool(db['success']); r['success'] = bool(db['success'] and rel_ok and q_ok)
        r['reason'] = 'ok' if r['success'] else (db.get('reason') if not db['success'] else ('still held' if not rel_ok else 'not quiescent'))
        r['rel_axis_deg'] = float(db['long_axis_deg']); r['rel_axis_note'] = 'door: angle between the long axes (flat plate, flips not observable)'
    if phi.get('goal_kind') == 'leg_in_top' and ctx.get('ce') is not None and ctx.get('parts_c'):
        # leg-insert: outcome test on the candidate's own table top — the leg stands upright with its
        # bottom end at the plate's top surface (inserted or resting on it) at the demonstrated corner; agents model the plate as a solid
        # box without holes, so "1.3 cm inside the plate" (the demonstrated relative pose) is not representable and must not be required.
        li = leg_in_top(ctx, phi, rec['final_poses'][n])
        rel_ok = not phi.get('require_released', True) or r.get('released') is True; q_ok = not phi.get('require_quiescent', True) or r.get('quiescent') is True
        r['leg_in_top'] = li; r['geometry_ok'] = bool(li['success']); r['success'] = bool(li['success'] and rel_ok and q_ok)
        r['reason'] = 'ok' if r['success'] else (li.get('reason') if not li['success'] else ('still held' if not rel_ok else 'not quiescent'))
        if ctx['ce'].get('kind') in ('box', 'cylinder') and r.get('rel_axis_deg') is not None:   # a primitive has no distinguishable ends: axis angle between LINES
            r['rel_axis_deg'] = float(min(r['rel_axis_deg'], 180.0 - r['rel_axis_deg'])); r['rel_axis_note'] = 'primitive part: ends indistinguishable, angle between lines'
    if phi.get('goal_kind') == 'container_in_box' and ctx.get('ce') is not None and ctx.get('parts_c'):
        # drawer-insert: outcome test on the candidate's own drawer box — the tray is inside the box (xy within the
        # footprint + margin, centroid between box bottom and top), level, and in the demonstrated slot (height above the box bottom) when the
        # candidate box is tall enough to have two slots; the demonstrated relative pose is not required (agents' boxes differ in size / slots
        # and a near-square tray's spin is not observable).
        cb = container_in_box(ctx, phi, rec['final_poses'][n])
        rel_ok = not phi.get('require_released', True) or r.get('released') is True; q_ok = not phi.get('require_quiescent', True) or r.get('quiescent') is True
        r['container_in_box'] = cb; r['geometry_ok'] = bool(cb['success']); r['success'] = bool(cb['success'] and rel_ok and q_ok)
        r['reason'] = 'ok' if r['success'] else (cb.get('reason') if not cb['success'] else ('still held' if not rel_ok else 'not quiescent'))
        r['rel_axis_deg'] = float(cb['tilt_deg']); r['rel_axis_note'] = 'tray: tilt of the up axis only (spin of a near-square tray is not observable)'
    if phi.get('goal_kind') == 'set_aside' and ctx.get('ce') is not None:
        # set_aside: the outcome test is geometric (off the lamp, on the table, same standing / lying as the demo); rel pose = diagnostic
        scene_c, _ = cand_scene(Path(ctx['pkg'])); tz = table_top_z(scene_c, Path(ctx['pkg']))
        a = aside(ctx['ce'], Path(ctx['pkg']), rec['final_poses'][n], ctx.get('parts_c') or [], Path(ctx['pkg']), tz, bool(phi.get('demo_upright', True)))
        rel_ok = not phi.get('require_released', True) or r.get('released') is True; q_ok = not phi.get('require_quiescent', True) or r.get('quiescent') is True
        r['aside'] = a; r['geometry_ok'] = bool(a['success']); r['success'] = bool(a['success'] and rel_ok and q_ok)
        r['reason'] = 'ok' if r['success'] else (a.get('reason') if not a['success'] else ('still held' if not rel_ok else 'not quiescent'))
    if (phi.get('type') == 'seated' or (phi.get('goal_kind') in ('cap_over_lamp', 'set_aside') and (phi.get('target') or phi.get('target_match')))) and not ctx.get('parts_c'):
        r.update(success=False, geometry_ok=False, reason='receiving part not found')
    return r


def verdict(ctx: dict, phi: dict, rec: dict, name: str) -> dict:
    """Terminal geometry plus required evidence that this execution completed the task."""
    rule = DROID_RULES.get(ctx.get('sample'))
    if rule:
        return relation_verdict(ctx['ce'],Path(ctx['pkg']),ctx['scene'],rec,name,rule,ctx['parts_c'])
    r = _pose_verdict(ctx, phi, rec, name)
    # The lower-level predicate also serves snap-fit; this gate never changes geometry_ok.
    def geometry(pose):
        rr = _pose_verdict(ctx, phi, dict(rec, final_poses={name: pose}), name)
        return bool(rr.get('geometry_ok', rr.get('success')))
    if r.get('geometry_ok') and phi.get('type') in ('seated', 'in_region_upright'):
        if phi.get('require_released', True) and r.get('released') is None:
            r.update(success=False, reason='release state not recorded')
        elif phi.get('require_quiescent', True) and r.get('quiescent') is None:
            r.update(success=False, reason='quiescence not recorded')
    return gate_initial_goal(r, phi, rec, name, geometry)


# ---------------------------------------------------------------- cap_over_lamp: a geometric "the hood encloses the lamp top" test
def _axis_dir(mesh, pose7, n=2000):
    """World-frame unit symmetry axis of a body of revolution at pose7, pointing from its WIDE end to its NARROW end (a hood's
    opening is the wide end), plus the axial extent (bottom, top) along that direction and the surface centroid."""
    import trimesh
    ax, _ = M.symmetry_axis(mesh); ax = ax or 'y'
    P, _ = trimesh.sample.sample_surface(mesh, n, seed=0); c = P.mean(0); e = np.eye(3)['xyz'.index(ax)]; s = (P - c) @ e
    lo, hi = np.percentile(s, 2), np.percentile(s, 98); rad = lambda m: np.linalg.norm((P[m] - c) - np.outer(s[m], e), axis=1).mean()
    if rad(s < lo + 0.15 * (hi - lo)) < rad(s > hi - 0.15 * (hi - lo)): e, lo, hi = -e, -hi, -lo        # make +e point to the narrow end
    T = M.T_of(pose7); d = T[:3, :3] @ e; cw = T[:3, :3] @ c + T[:3, 3]
    _axis_dir.profile = (P, c, e, lo, hi)   # mesh-frame samples for hood_radius_at()
    return d, cw + d * lo, cw + d * hi, cw


def hood_radius_at(a, slab=0.008):
    """Radius of the hood shell (mesh frame, from the last _axis_dir call) at axial coordinate `a` measured from the opening (bottom):
    median radial distance of the surface samples within +-slab of that height; None if no samples there."""
    P, c, e, lo, hi = _axis_dir.profile; s = (P - c) @ e; m = np.abs(s - (lo + a)) <= slab
    if m.sum() < 8: return None
    return float(np.median(np.linalg.norm((P[m] - c) - np.outer(s[m], e), axis=1)))


def capped(hood_e, hood_pkg, hood_pose7, lamp_parts, lamp_pkg, xy_tol=None, tilt_tol_deg=None):
    """The hood sits over the lamp: its symmetry axis is within tilt_tol of vertical, the lamp's topmost point lies between the
    hood's opening and its top along the axis, and within xy_tol of the axis. Independent of how tall or wide the reconstructed
    lamp and hood are, so it judges the outcome rather than the proportions of the reconstruction."""
    xy_tol = CAP_XY_TOL if xy_tol is None else xy_tol; tilt_tol_deg = CAP_TILT_TOL if tilt_tol_deg is None else tilt_tol_deg
    mh = M.entity_mesh(hood_e, hood_pkg); d, bottom, top, _ = _axis_dir(mh, hood_pose7)
    P = []
    for e in lamp_parts:
        try: P.append(M.surface_points(M.entity_mesh(e, lamp_pkg), np.r_[e['pos'], e.get('quat', (1, 0, 0, 0))], 2000))
        except Exception: pass
    if not P: return dict(success=False, reason='no lamp parts')
    P = np.concatenate(P); p_top = P[np.argmax(P[:, 2])]
    tilt = float(np.degrees(np.arccos(np.clip(abs(d[2]), -1, 1)))); L = float(np.linalg.norm(top - bottom)); a = float((p_top - bottom) @ d); radial = float(np.linalg.norm((p_top - bottom) - a * d))
    # "capped" means the lamp top is INSIDE the hood, so the radial tolerance is the hood's own
    # radius at that height (from its mesh), never more than xy_tol; a lamp top beside the hood wall is not capped.
    r_hood = hood_radius_at(min(max(a, 0.0), L)) if 0.0 <= a <= L else None
    r_tol = min(xy_tol, r_hood + CAP_WALL_MARGIN) if r_hood is not None else xy_tol   # CAP_WALL_MARGIN = the GT hood pose noise (tag PnP, ~1 cm); beyond it the lamp top is beside the wall
    ok = tilt <= tilt_tol_deg and 0.0 <= a <= L and radial <= r_tol
    return dict(success=bool(ok), tilt_deg=tilt, lamp_top_along_axis_m=a, hood_length_m=L, radial_m=radial, hood_radius_m=r_hood, radial_tol_m=r_tol, reason=None if ok else ('tilt' if tilt > tilt_tol_deg else ('lamp top not inside the hood' if not (0.0 <= a <= L) else ('outside the hood wall' if r_hood is not None and radial > r_hood + CAP_WALL_MARGIN else 'off axis'))))


# ---------------------------------------------------------------- leg_in_top: "the leg stands in the demonstrated corner of the table top"
LEG_TILT_TOL, LEG_CORNER_TOL, LEG_BOTTOM_LO, LEG_BOTTOM_HI, LEG_FOOT_MARGIN = 30.0, 0.07, -0.05, 0.015, 0.02   # upright within 30 deg; the NEAREST plate corner is the demonstrated one and within 7 cm (the hole centre sits ~4 cm inside the corner; the adjacent corner is >= 11 cm away even on an agent's 11.5 cm plate); bottom between 5 cm inside and 1.5 cm above the plate top; bottom xy inside the plate footprint + 2 cm


def _part_axis_world(e, pkg, pose7):
    """Unit symmetry axis (or longest axis for a primitive) of a part in the world at pose7, and its two axial end points."""
    import trimesh
    mesh = M.entity_mesh(e, pkg); ax, _ = M.symmetry_axis(mesh)
    if not ax: ax = 'xyz'[int(np.argmax(mesh.bounding_box.extents))]
    P, _ = trimesh.sample.sample_surface(mesh, 2000, seed=0); c = P.mean(0); eloc = np.eye(3)['xyz'.index(ax)]; sax = (P - c) @ eloc
    T = M.T_of(pose7); d = T[:3, :3] @ eloc; cw = T[:3, :3] @ c + T[:3, 3]
    return d, cw + d * float(sax.min()), cw + d * float(sax.max())


def _plate_geometry(parts, pkg):
    """Top-surface height and the 4 corners (xy, world) of the receiving plate = union of the matched target parts."""
    P = np.concatenate([M.surface_points(M.entity_mesh(e, pkg), np.r_[e['pos'], e.get('quat', (1, 0, 0, 0))], 3000) for e in parts])
    import trimesh
    top_z = float(np.percentile(P[:, 2], 98)); top = P[P[:, 2] >= top_z - 0.01]; xy = P[:, :2]  # full footprint: top-height slicing biases tilted/circular plates
    T2, ext = trimesh.bounds.oriented_bounds_2D(xy)   # minimum-area rectangle of the top face (a square has no principal axes)
    Ti = np.linalg.inv(T2); h = np.asarray(ext) / 2
    corners = np.array([(Ti @ np.array([a, b, 1.0]))[:2] for a in (-h[0], h[0]) for b in (-h[1], h[1])]); c = corners.mean(0)
    return top_z, corners, c


def leg_in_top(ctx, phi, pose7):
    """Candidate leg (its own mesh / primitive) vs the candidate's own table top: upright, bottom end at the plate top (inserted or
    resting), at the corner that corresponds to the demonstrated one (matched by world-frame direction from the plate centre)."""
    pkg = Path(ctx['pkg']); d, e0, e1 = _part_axis_world(ctx['ce'], pkg, pose7)
    bottom = e0 if e0[2] < e1[2] else e1; up = d if d[2] > 0 else -d; tilt = float(np.degrees(np.arccos(np.clip(up[2], -1, 1))))
    top_z, corners, centre = _plate_geometry(ctx['parts_c'], pkg)
    if phi.get('placement_site') == 'center':
        dxy=float(np.linalg.norm(bottom[:2]-centre));dz=float(bottom[2]-top_z);ok=tilt<=LEG_TILT_TOL and dxy<=float(phi['center_xy_tol_m']) and LEG_BOTTOM_LO<=dz<=LEG_BOTTOM_HI
        return dict(success=bool(ok),tilt_deg=tilt,center_dist_m=dxy,center_tol_m=phi['center_xy_tol_m'],bottom_above_top_m=dz,site='center',reason=None if ok else 'central pedestal not upright/seated at receiver centre')
    demo = np.asarray(phi.get('demo_corner_dir') or (0, 0), float)   # world-frame direction of the demonstrated corner from the plate centre (written by the sample builder)
    dists = np.linalg.norm(corners[:, :2] - bottom[:2], axis=1); k_near = int(np.argmin(dists))
    k_demo = int(np.argmax((corners[:, :2] - centre) @ demo / (np.linalg.norm(corners[:, :2] - centre, axis=1) + 1e-9))) if np.linalg.norm(demo) > 0 else k_near
    corner = corners[k_demo]; dxy = float(dists[k_demo]); dz = float(bottom[2] - top_z)
    # inside the plate footprint (+ margin): distance from the bottom xy to the rectangle
    v0 = corners[1] - corners[0]; v1 = corners[2] - corners[0]; q = bottom[:2] - corners[0]
    u0 = q @ v0 / (v0 @ v0); u1 = q @ v1 / (v1 @ v1); out = np.array([max(-u0, 0, u0 - 1) * np.linalg.norm(v0), max(-u1, 0, u1 - 1) * np.linalg.norm(v1)]); d_out = float(np.linalg.norm(out))
    ok_tilt = tilt <= LEG_TILT_TOL; ok_xy = (k_near == k_demo) and dxy <= LEG_CORNER_TOL and d_out <= LEG_FOOT_MARGIN; ok_z = LEG_BOTTOM_LO <= dz <= LEG_BOTTOM_HI
    ok = ok_tilt and ok_xy and ok_z
    return dict(success=bool(ok), tilt_deg=tilt, corner_dist_m=dxy, nearest_corner_is_demo=bool(k_near == k_demo), outside_footprint_m=d_out, bottom_above_top_m=dz, plate_top_z=top_z, corner_xy=corner[:2].tolist(), demo_corner_dir=demo.tolist(),
                reason=None if ok else ('tilt' if not ok_tilt else (('wrong corner' if k_near != k_demo else ('off the plate' if d_out > LEG_FOOT_MARGIN else 'not at the demonstrated corner')) if not ok_xy else ('above the plate' if dz > LEG_BOTTOM_HI else 'below the plate'))))


# ---------------------------------------------------------------- container_in_box: "the tray sits inside the drawer box, in the demonstrated slot"
BOX_TILT_TOL, BOX_OUT_MARGIN, BOX_Z_MARGIN, BOX_SLOT_TOL, BOX_TWO_SLOT_MIN_H = 30.0, 0.03, 0.01, 0.03, 0.09   # level within 30 deg; centroid within 3 cm of the footprint (the demos leave the tray ~1 cm out); centroid between box bottom - 1 cm and top; slot height within 3 cm of the demonstrated one (slots ~4 cm apart) when the box is >= 9 cm tall


def _box_geometry(parts, pkg):
    import trimesh
    P = np.concatenate([M.surface_points(M.entity_mesh(e, pkg), np.r_[e['pos'], e.get('quat', (1, 0, 0, 0))], 3000) for e in parts])
    T2, ext = trimesh.bounds.oriented_bounds_2D(P[:, :2]); return float(P[:, 2].min()), float(P[:, 2].max()), T2, np.asarray(ext, float)


def _tray_geometry(e, pkg, pose7):
    """Tray surface centroid, tilt of its up axis (= its thinnest dimension) from vertical."""
    mesh = M.entity_mesh(e, pkg); ax = int(np.argmin(mesh.bounding_box.extents)); T = M.T_of(pose7); up = T[:3, :3] @ np.eye(3)[ax]; up = up if up[2] > 0 else -up
    return M.surface_points(mesh, pose7, 3000).mean(0), float(np.degrees(np.arccos(np.clip(up[2], -1, 1))))


def container_in_box(ctx, phi, pose7):
    pkg = Path(ctx['pkg']); zb, zt, T2, ext = _box_geometry(ctx['parts_c'], pkg); c, tilt = _tray_geometry(ctx['ce'], pkg, pose7)
    q = (T2 @ np.array([c[0], c[1], 1.0]))[:2]; d_out = float(np.linalg.norm(np.maximum(np.abs(q) - ext / 2, 0)))
    h = float(c[2] - zb); demo_h = phi.get('demo_slot_height_m'); two_slot = (zt - zb) >= BOX_TWO_SLOT_MIN_H
    ok_tilt = tilt <= BOX_TILT_TOL; ok_xy = d_out <= BOX_OUT_MARGIN; ok_z = (zb - BOX_Z_MARGIN) <= c[2] <= zt
    ok_slot = True if (demo_h is None or not two_slot) else abs(h - float(demo_h)) <= BOX_SLOT_TOL
    ok = ok_tilt and ok_xy and ok_z and ok_slot
    return dict(success=bool(ok), tilt_deg=tilt, outside_footprint_m=d_out, height_above_box_bottom_m=h, demo_slot_height_m=demo_h, box_height_m=float(zt - zb), slot_checked=bool(demo_h is not None and two_slot),
                reason=None if ok else ('tilt' if not ok_tilt else ('not inside the box' if not (ok_xy and ok_z) else 'wrong slot')))


# ---------------------------------------------------------------- door_on_body: "the door hangs on the demonstrated hinge side of the cabinet body"
DOOR_AXIS_TOL, DOOR_LINE_TOL, DOOR_GAP_TOL = 30.0, 0.06, 0.03   # long axis within 30 deg of the demonstrated one; centre within 6 cm of the mapped demonstrated line (door width 5.4 cm, body pose noise); door surface within 3 cm of the body


def _body_obb(parts, pkg):
    import trimesh
    P = np.concatenate([M.surface_points(M.entity_mesh(e, pkg), np.r_[e['pos'], e.get('quat', (1, 0, 0, 0))], 3000) for e in parts]); T, ext = trimesh.bounds.oriented_bounds(P)
    return np.linalg.inv(T), np.asarray(ext, float), P   # T_world<-obb, extents, surface samples


def _door_geometry(e, pkg, pose7):
    mesh = M.entity_mesh(e, pkg); ext = mesh.bounding_box.extents; T = M.T_of(pose7)
    return M.surface_points(mesh, pose7, 3000), T[:3, :3] @ np.eye(3)[int(np.argmax(ext))], T[:3, :3] @ np.eye(3)[int(np.argmin(ext))]


def door_on_body(ctx, phi, pose7):
    from scipy.spatial import cKDTree
    pkg = Path(ctx['pkg']); Tb, ext, Pb = _body_obb(ctx['parts_c'], pkg); P, lon, nrm = _door_geometry(ctx['ce'], pkg, pose7); c = P.mean(0)
    demo_dir = np.asarray(phi.get('demo_door_axis_world', (0, 0, 1)), float); demo_off = np.asarray(phi.get('demo_door_offset_norm', (0, 0, 0)), float); Rg = np.asarray(phi.get('demo_body_axes_world', np.eye(3).tolist()), float)
    Rc = Tb[:3, :3]; q = np.zeros(3)
    for i in range(3):   # GT body axis i -> the candidate body axis most parallel to it (world direction), with sign
        d = Rc.T @ Rg[:, i]; k = int(np.argmax(np.abs(d))); q[k] = float(np.sign(d[k]) or 1) * demo_off[i] * ext[k] / 2
    p0 = Rc @ q + Tb[:3, 3]; v = c - p0; dist_line = float(np.linalg.norm(v - (v @ demo_dir) * demo_dir)); ang = float(np.degrees(np.arccos(np.clip(abs(lon @ demo_dir), -1, 1))))
    gap = float(cKDTree(Pb).query(P)[0].min()); hinge = float(np.degrees(np.arccos(np.clip(np.max(np.abs(Rc.T @ nrm)), -1, 1))))
    ok_axis = ang <= DOOR_AXIS_TOL; ok_line = dist_line <= DOOR_LINE_TOL; ok_gap = gap <= DOOR_GAP_TOL; ok = ok_axis and ok_line and ok_gap
    return dict(success=bool(ok), long_axis_deg=ang, centre_to_demo_line_m=dist_line, door_body_gap_m=gap, hinge_angle_deg=hinge, body_extents_m=ext.tolist(),
                reason=None if ok else ('door not along the hinge' if not ok_axis else ('not at the demonstrated door opening' if not ok_line else 'not touching the body')))


# ---------------------------------------------------------------- set_aside: "the hood is off the lamp, resting on the table"
ASIDE_TABLE_TOL, ASIDE_MIN_DXY, UPRIGHT_MAX_TILT = 0.04, 0.08, 45.0   # table tol 4 cm: the demonstrated set-aside terminals sit within -3.5..+2.9 cm of the table plane (tag-pose noise)


def table_top_z(scene: dict, pkg_dir, default=-0.015):
    """Top of the scene's table: the support plane if declared, else the highest top face among table / bench props, else default."""
    sp = scene.get('support')
    if sp and 'z' in sp: return float(sp['z'])
    tops = []
    for p in scene.get('props', []):
        if any(k in p['name'].lower() for k in ('table', 'bench', 'workbench')) and p.get('half_size'):
            tops.append(float(p['pos'][2]) + float(p['half_size'][2]))
    return max(tops) if tops else default


def aside(hood_e, hood_pkg, hood_pose7, lamp_parts, lamp_pkg, table_z, demo_upright: bool):
    """set_aside outcome: not capped over the lamp; the hood's lowest point within ASIDE_TABLE_TOL of the table top (resting on it);
    its axis-centre at least ASIDE_MIN_DXY from the lamp in xy (not leaning on it); standing / lying like the demonstration
    (axis within UPRIGHT_MAX_TILT of vertical = standing). Where the demonstration put it on the table is NOT scored."""
    mh = M.entity_mesh(hood_e, hood_pkg); d, bottom, top, _ = _axis_dir(mh, hood_pose7)
    P = M.surface_points(mh, hood_pose7, 2000); zmin = float(P[:, 2].min()); tilt = float(np.degrees(np.arccos(np.clip(abs(d[2]), -1, 1))))
    cap = capped(hood_e, hood_pkg, hood_pose7, lamp_parts, lamp_pkg)['success'] if lamp_parts else False
    L = []
    for e in lamp_parts:
        try: L.append(M.surface_points(M.entity_mesh(e, lamp_pkg), np.r_[e['pos'], e.get('quat', (1, 0, 0, 0))], 1000))
        except Exception: pass
    dxy = float(np.linalg.norm(((bottom + top) / 2)[:2] - np.concatenate(L)[:, :2].mean(0))) if L else float('inf')
    on_table = abs(zmin - table_z) <= ASIDE_TABLE_TOL; upright = tilt <= UPRIGHT_MAX_TILT
    ok = (not cap) and on_table and dxy >= ASIDE_MIN_DXY and (upright == demo_upright)
    reason = None if ok else ('still on the lamp' if cap else ('not on the table' if not on_table else ('leaning on the lamp' if dxy < ASIDE_MIN_DXY else ('lying, demo standing' if demo_upright else 'standing, demo lying'))))
    return dict(success=bool(ok), tilt_deg=tilt, zmin_minus_table_cm=100 * (zmin - table_z), dxy_lamp_cm=100 * dxy, capped=bool(cap), upright=bool(upright), demo_upright=bool(demo_upright), reason=reason)

FURNITURE_RULES = json.loads((Path(__file__).parent / 'rules/furniture_task_rules.json').read_text())['rules']

DROID_RULES = json.loads((Path(__file__).parent / 'rules/droid_task_rules.json').read_text())['rules']


# ---------------------------------------------------------------- snap-fit
# Snap-fit at release.
#
# The real lamp parts hold where the demonstrator leaves them: the bulb is screwed into the socket, the hood rests on the bulb.
# The simulator has no threads and its convex-decomposed contacts do not hold a cone on a sphere, so a part released in the right
# place still tips over (bulb: 43/49 Reference replays are within tolerance at the release frame, only 16 remain after settling).
# Rule: at the frame the gripper RELEASES the part, if the part already satisfies the family's placement geometry (bulb `seated`
# 3 cm / 30 deg relative to the lamp; hood `capped`), the part is locked in place (made kinematic) -- a thread / tight-fit proxy.
# set_aside and any part that is NOT in place at release get the ordinary free release. Applies identically to the Reference
# (GT package) and to candidates (their own parts, their own lamp, canonicalized).


def snap_decision(ctx: dict, phi: dict, pose7, name: str):
    """(snap?, details). Geometry only: release / quiescence are not part of the decision."""
    if phi.get('snap_fit') is False: return False, dict(reason='snap-fit disabled for this sample (phi.snap_fit=false: DROID place tasks have no thread / tight fit; the release is judged by physics)')
    if phi.get('goal_kind') == 'set_aside' or (phi.get('type') != 'seated' and phi.get('goal_kind') != 'cap_over_lamp'):   # set_aside never snaps; cap_over_lamp is judged by capped() whatever the phi type
        return False, dict(reason='no snap for this predicate', goal_kind=phi.get('goal_kind'), type=phi.get('type'))
    rec = dict(final_poses={name: [float(x) for x in pose7]}, init_poses={name: [float(x) for x in pose7]})
    r = verdict(ctx, phi, rec, name)
    keep = {k: r.get(k) for k in ('geometry_ok', 'rel_pos_m', 'rel_axis_deg', 'capped', 'reason')}
    return bool(r.get('geometry_ok')), keep
