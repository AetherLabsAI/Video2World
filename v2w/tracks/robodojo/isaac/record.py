"""Execute a submitted RoboDojo scene with the native task in Isaac Sim and record states, masks and video.

Runs in the Isaac Sim interpreter. The submitted geometry replaces the task layout; robots, controllers, physics
and the native task predicates are RoboDojo's. Object states are never written after initialization. Outputs in
--out: trajectory.npz, objects.json, geometry/, camera.json, video.mp4, instance masks, task_result.json, run.json.
"""
import argparse
import json
import os
import sys
import tempfile
import time
import traceback
from copy import deepcopy
from pathlib import Path

from v2w import paths
from . import source_root

ARM_URDF = paths.asset('twins', 'upstream/real2sim-eval/assets/robots/xarm/xarm7.urdf')


def jsonable(x):
    if hasattr(x, 'detach'):
        return x.detach().cpu().numpy().tolist()
    if hasattr(x, 'tolist'):
        return x.tolist()
    return str(x)


def write(path, value):
    Path(path).parent.mkdir(parents=True, exist_ok=True)
    Path(path).write_text(json.dumps(value, indent=2, default=jsonable, allow_nan=False) + '\n')


def kit_config(output):
    """Per-process writable Kit caches and a bounded CPU worker count (no physics settings)."""
    threads = int(os.environ.get('V2W_ISAAC_THREADS', '8'))
    if threads < 1 or threads > 256:
        raise ValueError('V2W_ISAAC_THREADS must be in [1,256]')
    available = len(os.sched_getaffinity(0)) if hasattr(os, 'sched_getaffinity') else (os.cpu_count() or 1)
    threads = min(threads, available)
    parent = Path(output) / 'kit_runtime'; parent.mkdir(parents=True, exist_ok=True)
    root = Path(tempfile.mkdtemp(prefix='kit-', dir=parent))
    dirs = {name: str(root / name) for name in ['cache', 'data', 'logs', 'DerivedDataCache', 'shadercache', 'nv_shadercache']}
    for path in dirs.values():
        Path(path).mkdir()
    settings = {'/app/cachePath': dirs['cache'], '/app/dataPath': dirs['data'], '/app/logPath': dirs['logs'],
                '/UJITSO/datastore/localCachePath': dirs['DerivedDataCache'], '/app/tokens/omni_cache': dirs['cache'],
                '/rtx/shaderDb/shaderCachePath': dirs['shadercache'], '/rtx/shaderDb/driverShaderCachePath': dirs['nv_shadercache']}
    config = dict(headless=True, enable_cameras=True, multi_gpu=False, limit_cpu_threads=threads,
                  extra_args=['--' + key + '=' + value for key, value in settings.items()])
    return config, settings, threads


def verify_kit(settings, expected, threads):
    observed = {key: settings.get(key) for key in expected}
    ok = all(observed[key] == value for key, value in expected.items()) and all(
        settings.get(key) == threads for key in ['/plugins/carb.tasking.plugin/threadCount', '/plugins/omni.tbb.globalcontrol/maxThreadCount'])
    if not ok:
        raise RuntimeError('Kit ignored cache/thread isolation settings')


def main():
    p = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    p.add_argument('--task', required=True)
    p.add_argument('--layout', type=Path, required=True, help='submitted world-frame scene.json')
    p.add_argument('--actions', type=Path, required=True)
    p.add_argument('--candidate-task', action='store_true', help='score with the native task predicates on the submitted geometry')
    p.add_argument('--embodiment', choices=['x5', 'xarm7'], default='x5')
    p.add_argument('--robot-meshes', type=Path, required=True, help='official xArm gripper/visual meshes')
    p.add_argument('--out', type=Path, required=True)
    p.add_argument('--max-capture-attempts', type=int, default=8)
    p.add_argument('--capture-product', choices=['single', 'tiled'], default='single')
    p.add_argument('--render-flush-steps', type=int, default=12, help='render-only updates per physics state before reading annotations')
    p.add_argument('--device', default='auto')
    args = p.parse_args()
    args.out.mkdir(parents=True, exist_ok=True)
    if (args.out / 'trajectory.npz').exists():
        raise FileExistsError('Do not overwrite a completed recording')
    source = source_root()
    started = time.monotonic(); app = None
    import faulthandler
    faulthandler.enable(all_threads=False)
    run_status = {'status': 'starting', 'task': args.task, 'embodiment': args.embodiment, 'scope': 'agent candidate scene execution',
               'fps': 25, 'physics_dt': .004, 'cuda_visible_devices': os.environ.get('CUDA_VISIBLE_DEVICES')}

    def mark(s):
        print(f'RECORD {time.monotonic() - started:.1f}s {s}', flush=True)
    write(args.out / 'run.json', run_status)
    try:
        sys.path.insert(0, str(source))
        config, expected_settings, threads = kit_config(args.out)
        from isaacsim import SimulationApp
        app = SimulationApp(config); mark('app started')
        import carb
        settings = carb.settings.get_settings()
        verify_kit(settings, expected_settings, threads)
        settings.set_bool('/isaaclab/render/offscreen', True)
        settings.set_bool('/isaaclab/render/active_viewport', False)
        settings.set_bool('/isaaclab/render/rtx_sensors', False)
        settings.set_bool('/isaaclab/cameras_enabled', True)
        settings.set_bool('/physics/fabricUpdateTransformations', True)
        # Synthetic-data capture requires real, synchronous rendered frames.
        for key in ['/app/asyncRendering', '/app/asyncRenderingLowLatency', '/omni/replicator/asyncRendering', '/rtx-transient/dlssg/enabled']:
            settings.set_bool(key, False)
        from isaacsim.core.utils.extensions import enable_extension
        enable_extension('isaacsim.sensors.camera')
        enable_extension('isaacsim.replicator.behavior')
        app.update()
        import numpy as np
        import cv2
        from omegaconf import OmegaConf
        from pxr import Usd
        from isaacsim.core.utils.semantics import add_update_semantics

        def array(value):
            return value.detach().cpu().numpy() if hasattr(value, 'detach') else np.asarray(value)
        # Joint-target execution never plans; stub the optional cuRobo import when it is not installed.
        import importlib.util
        import types
        if importlib.util.find_spec('curobo') is None:
            m = types.ModuleType('env.planner_manager.curobo_planner')

            class DisabledPlanner:
                def __init__(self, *a, **kw):
                    raise RuntimeError('cuRobo disabled: only precomputed joint targets supported')
            m.CuroboPlanner = DisabledPlanner
            sys.modules[m.__name__] = m
        from utils.load_file import load_yaml
        from utils.pipeline_utils import process_config
        base = source / 'env_cfg'
        entry = load_yaml(str(base / 'arx_x5.yml'))
        cfg = OmegaConf.create({k: load_yaml(str(base / k / (entry['config'][k] + '.yml'))) for k in ['sim', 'scene', 'camera', 'robot']})
        cfg.task_env = load_yaml(str(source / 'task/RoboDojo/config' / f'{args.task}.yml'))
        cfg.eval_cfg = entry
        cfg, _ = process_config(cfg, args.task)
        cfg.sim.scene.num_envs = 1; cfg.sim.seed = [7]; cfg.sim.use_fabric = True
        cfg.sim.device = 'cpu' if args.device == 'auto' else args.device
        if args.embodiment == 'xarm7':
            from .xarm import install_adapter
            run_status['robot_adapter'] = install_adapter(args.out / 'robot_assets', ARM_URDF, mesh_root=args.robot_meshes)
            for r in cfg.robot.robots[:2]:
                r.robot_name = 'v2w_xarm7'
        for r in cfg.robot.robots:
            r.need_planner = False
        for key in list(cfg.camera.annotator):
            cfg.camera.annotator[key].enabled = False
        cfg.camera.default_frequency = 25
        from .scene import install
        Task = install(cfg, args.layout, args.out / 'candidate_assets', task_name=args.task if args.candidate_task else None)
        write(args.out / 'resolved_config.json', OmegaConf.to_container(cfg, resolve=True))
        env = Task(cfg, app)
        env.success = [True]   # active-environment mask for reward helpers, not task success
        from .runtime import TaskState, install_pose_reader
        install_pose_reader(env)
        env.scene_manager.layout_manager.set_saved_layout(0, json.loads(args.layout.read_text()))

        def check_initialization(phase):
            from v2w.tracks.robodojo.audit import check_initial_state
            import xml.etree.ElementTree as ET
            checks = []
            for art, robot in zip(env.robot_manager.robot_key, env.robot_manager.robot_list):
                relations = []
                for joint in ET.parse(robot.urdf_path).getroot().findall('joint'):
                    mimic = joint.find('mimic')
                    if mimic is not None:
                        relations.append({'follower': joint.get('name'), 'reference': mimic.get('joint'),
                                          'multiplier': float(mimic.get('multiplier', 1)), 'offset': float(mimic.get('offset', 0))})
                checks.append(check_initial_state(list(art.joint_names), array(art.data.joint_pos[0]), array(art.data.default_joint_pos[0]), relations))
            write(args.out / ('initialization_gate_' + phase + '.json'), checks)
            if not all(check['passed'] for check in checks):
                raise RuntimeError('Invalid candidate robot initialization at ' + phase + '; do not score this rollout')
        env.reset(seed=[7]); mark('task reset')
        check_initialization('before_scene_restore')
        # TaskEnv.reset parks table/objects offscreen; restore the submitted poses and settle.
        env.scene_manager.apply_saved_poses(env_idx_list=[0])
        for _ in range(50):
            env.sim_step(render=False)
        env.render(); mark('saved scene restored and settled')
        check_initialization('after_scene_restore')
        env.robot_manager.set_origin_endpose(); env.robot_manager.set_robot_init_state()
        if args.candidate_task:
            env.reward_manager.init_state()
        lm = env.scene_manager.layout_manager
        objects = []
        for typ in ['Rigid', 'Dynamic', 'Geometry', 'Articulation', 'Garment', 'Fluid']:
            for record in lm.get_layout_records(0, typ):
                ob = dict(record); ob['object_type'] = typ; ob['label'] = ob.get('label') or ob['inst_name']
                ob['metadata'] = lm.get_instance_metadata(0, inst_name=record['inst_name'])
                ob['instance_id'] = len(objects) + 1
                root_pos, root_quat = lm.get_instance_pose(0, inst_name=record['inst_name'], relative=False)
                ob['root_pose'] = np.r_[array(root_pos).reshape(3), array(root_quat).reshape(4)].tolist()
                prim = env.stage.GetPrimAtPath(record['prim_path'])
                add_update_semantics(prim, ob['label'])
                properties = []
                for child in Usd.PrimRange(prim):
                    attrs = {a.GetName(): a.Get() for a in child.GetAttributes() if a.GetName().startswith(('physics:', 'physx')) and a.HasAuthoredValueOpinion()}
                    if attrs:
                        properties.append({'path': str(child.GetPath()), 'attributes': attrs})
                ob['authored_runtime_physics'] = properties
                objects.append(ob)
        write(args.out / 'objects.json', objects)
        from .geometry import export_scene_geometry
        robot_paths = [str(prim.GetPath()) for prim in env.stage.GetPrimAtPath('/World/envs/env_0').GetChildren() if prim.GetName().startswith('robot')]
        geometry_manifest = export_scene_geometry(env.stage, objects, args.out / 'geometry', robot_paths=robot_paths)
        inventory = []
        for ob in objects:
            inventory.append({'prim_path': ob['prim_path'], 'label': ob.get('label', ob['inst_name']),
                              'category': ob.get('category', ob['metadata'].get('model_name', 'unknown')),
                              'entity_type': 'static_fixture' if ob['object_type'] == 'Geometry' else 'task_object'})
        for ob in geometry_manifest['objects']:
            if ob.get('kind') != 'static' or not ob.get('vertices', 0):
                continue
            category = next((name.lower() for name in ['Table', 'Ground', 'Rooms'] if '/' + name + '/' in ob['prim_path']), 'fixture')
            inventory.append({'prim_path': ob['prim_path'], 'label': ob['label'], 'category': category, 'entity_type': 'static_fixture'})
        for index, path in enumerate(robot_paths):
            inventory.append({'prim_path': path, 'label': 'robot_left' if index == 0 else 'robot_right', 'category': args.embodiment, 'entity_type': 'robot'})
        for entity in inventory:
            add_update_semantics(env.stage.GetPrimAtPath(entity['prim_path']), entity['label'])
        write(args.out / 'scene_inventory.json', {'schema_version': 1, 'entities': inventory})
        mark('scene geometry exported and scene labels assigned')
        head = env.camera_manager.camera_names[0].index('cam_head')
        camera = {'name': 'cam_head', 'resolution': list(env.camera_manager.cameras[0][head].get_resolution()),
                  'clipping_range': list(env.camera_manager.cameras[0][head].get_clipping_range()),
                  'intrinsics': env.camera_manager.get_camera_intrinsics(head, 0), 'extrinsics': env.camera_manager.get_camera_extrinsics(head, 0)}
        # Optical camera frame (ROS axes: +x right, +y down, +z forward).
        import transforms3d as t3d
        cp, cq = env.camera_manager.cameras[0][head].get_world_pose(camera_axes='ros')
        camera_to_world = np.eye(4); camera_to_world[:3, :3] = t3d.quaternions.quat2mat(array(cq)); camera_to_world[:3, 3] = array(cp)
        camera['camera_to_world_ros'] = camera_to_world
        camera['world_to_camera_ros'] = np.linalg.inv(camera_to_world)
        write(args.out / 'camera.json', camera)
        import omni.replicator.core as rep
        if args.capture_product == 'single':
            native_camera = env.camera_manager.cameras[0][head]
            render_product = rep.create.render_product(native_camera.prim_path, tuple(native_camera.get_resolution()), name='V2WHeadCapture').path
        else:
            render_product = env.capture_manager.tiled_cameras[head]._render_product_path
        rgb_capture = rep.AnnotatorRegistry.get_annotator('rgb', device='cpu', do_array_copy=True)
        mask_capture = rep.AnnotatorRegistry.get_annotator('instance_segmentation_fast', init_params={'colorize': False}, device='cpu', do_array_copy=True)
        rgb_capture.attach([render_product]); mask_capture.attach([render_product])
        actors = env.robot_manager.robot_key[:2]
        names = [list(art.joint_names) for art in actors]
        write(args.out / 'robot_joints.json', names)
        robot_limits = []
        for art in actors:
            record = {'joint_names': list(art.joint_names), 'initial_joint_position': array(art.data.joint_pos[0]), 'initial_joint_velocity': array(art.data.joint_vel[0])}
            for key in ['joint_pos_limits', 'soft_joint_pos_limits', 'joint_vel_limits', 'soft_joint_vel_limits', 'joint_effort_limits']:
                value = getattr(art.data, key, None)
                if value is not None:
                    record[key] = array(value[0])
            robot_limits.append(record)
        write(args.out / 'robot_limits.json', robot_limits)
        arm_dof = len(env.robot_manager.robot_list[0].arm_joints_name)
        stride = arm_dof + 1
        home = np.zeros(stride * 2, dtype=np.float32); home[[stride - 1, 2 * stride - 1]] = 1
        if args.embodiment == 'xarm7':
            from .xarm import REST
            home[:arm_dof] = REST; home[stride:stride + arm_dof] = REST
        actions = np.load(args.actions)
        if actions.ndim != 2 or actions.shape[1] != stride * 2 or not np.isfinite(actions).all():
            raise ValueError(f'Expected finite [T,{stride * 2}] {args.embodiment} actions')

        def gripper_targets(robot, normalized):
            if callable(getattr(robot, 'gripper_joint_targets', None)):
                targets = robot.gripper_joint_targets(normalized)
            else:   # native X5 scale/sign/mimic
                lo, hi = robot.gripper_scale
                val = lo + normalized * (hi - lo) if robot.gripper_move['sign'] == 1 else hi - normalized * (hi - lo)
                mimic = robot.gripper_move['mimic']
                targets = {robot.gripper_joints_name[0]: val, robot.gripper_joints_name[1]: val * mimic[1] + mimic[2]}
            if not targets or any(not np.isfinite(v) for v in targets.values()):
                raise ValueError('Robot must return finite gripper joint targets')
            return {joint: float(value) for joint, value in targets.items()}
        import xml.etree.ElementTree as ET
        control_contract = []
        for side, robot in enumerate(env.robot_manager.robot_list[:2]):
            opened = gripper_targets(robot, 1.); closed = gripper_targets(robot, 0.)
            try:
                urdf_joints = ET.parse(Path(robot.urdf_path)).getroot().findall('joint')
                joint_types = {j.attrib['name']: j.attrib.get('type', 'unknown') for j in urdf_joints}
                mimic_relations = []
                for joint in urdf_joints:
                    mimic = joint.find('mimic')
                    if mimic is not None:
                        mimic_relations.append({'follower': joint.attrib['name'], 'reference': mimic.attrib['joint'],
                                                'multiplier': float(mimic.attrib.get('multiplier', '1')), 'offset': float(mimic.attrib.get('offset', '0'))})
            except (OSError, ET.ParseError):
                joint_types = {}; mimic_relations = []
            gripper_joint_names = set(opened) | set(closed) | set(robot.gripper_joints_name)
            # Every URDF mimic follower connected to this gripper, including chained followers.
            for _ in range(len(mimic_relations)):
                for relation in mimic_relations:
                    if relation['reference'] in gripper_joint_names or relation['follower'] in gripper_joint_names:
                        gripper_joint_names.update([relation['reference'], relation['follower']])
            control_contract.append({'side': side, 'normalized_grip_open': 1., 'normalized_grip_closed': 0.,
                                     'gripper_open_joint_positions': opened, 'gripper_closed_joint_positions': closed,
                                     'gripper_joint_types': {joint: joint_types.get(joint, 'unknown') for joint in sorted(gripper_joint_names)},
                                     'mimic_relations': [r for r in mimic_relations if r['follower'] in gripper_joint_names]})
        write(args.out / 'robot_control_contract.json', {'schema_version': 1, 'sides': control_contract})

        def set_targets(row):
            for side, art in enumerate(actors):
                q = art.data.joint_pos.clone()
                robot = env.robot_manager.robot_list[side]
                for j, joint in enumerate(robot.arm_joints_name):
                    q[:, names[side].index(joint)] = float(row[side * stride + j])
                normalized = float(np.clip(row[side * stride + arm_dof], 0, 1))
                for joint, value in gripper_targets(robot, normalized).items():
                    q[:, names[side].index(joint)] = value
                art.set_joint_position_target(q)
        set_targets(home)
        for _ in range(10):
            env.render()
        records = []; mask_frames = []; labels_all = []; writer = None; capture_retries = []

        def object_poses():
            pose = []
            for ob in objects:
                pos, quat = lm.get_instance_pose(0, inst_name=ob['inst_name'], relative=False)
                pose.append(np.r_[array(pos).reshape(3), array(quat).reshape(4)])
            return np.asarray(pose)

        def capture_frame(frame_index):
            step_before = int(env.sim._sim_step_counter)
            pose_before = object_poses()
            for attempt in range(args.max_capture_attempts):
                # Render-only updates: never advance physics to repair a bad capture.
                for _ in range(args.render_flush_steps):
                    env.render()
                rgb = np.asarray(rgb_capture.get_data()).copy()[..., :3]
                instance = mask_capture.get_data()
                mask = np.asarray(instance['data']).squeeze().copy()
                info = deepcopy(instance.get('info', {}))
                known = np.asarray([int(key) for key in info.get('idToLabels', {})], dtype=np.uint64)
                unknown = np.unique(mask)[~np.isin(np.unique(mask), known)]
                valid = (rgb.ndim == 3 and rgb.shape[2] == 3 and mask.shape == rgb.shape[:2] and rgb.dtype == np.uint8
                         and mask.dtype == np.uint32 and len(known) > 0 and len(unknown) == 0)
                if int(env.sim._sim_step_counter) != step_before or not np.array_equal(pose_before, object_poses()):
                    raise RuntimeError('Physics changed during render-only capture/retry')
                if valid:
                    return rgb, mask, info, pose_before
                capture_retries.append({'frame': frame_index, 'attempt': attempt + 1, 'unknown_id_count': len(unknown)})
            raise RuntimeError(f'Invalid mask/RGB after {args.max_capture_attempts} render-only attempts at frame {frame_index}')
        task_state = TaskState(env, args.out, args.embodiment)
        for t, row in enumerate(actions):
            set_targets(row)
            for _ in range(10):
                task_state.before_physics(); env.sim_step(render=False)
            task_state.after_action()
            rgb, mask, instance_info, pose = capture_frame(t)
            if writer is None:
                writer = cv2.VideoWriter(str(args.out / 'video.mp4'), cv2.VideoWriter_fourcc(*'mp4v'), 25, (rgb.shape[1], rgb.shape[0]))
                if not writer.isOpened():
                    raise RuntimeError('Video writer failed')
            writer.write(cv2.cvtColor(rgb, cv2.COLOR_RGB2BGR))
            states = [art.data.joint_pos[0].detach().cpu().numpy().copy() for art in actors]
            velocities = [array(art.data.joint_vel[0]).copy() for art in actors]
            torques = [array(art.data.applied_torque[0]).copy() for art in actors]
            task_state.capture()
            records.append({'pose': pose, 'joints': states, 'velocities': velocities, 'torques': torques})
            mask_frames.append(mask); labels_all.append(instance_info)
            if t % 25 == 0:
                mark(f'frame {t}/{len(actions)}')
        writer.release()
        write(args.out / 'capture_retries.json', capture_retries)
        objpose = np.asarray([r['pose'] for r in records]); joints = np.asarray([r['joints'] for r in records])
        if not np.isfinite(objpose).all():
            raise RuntimeError('Nonfinite object states')
        np.savez_compressed(args.out / 'trajectory.npz', time_s=(np.arange(len(actions)) + 1) / 25, object_pose_wxyz=objpose,
                            robot_joint_positions=joints, robot_actions=actions, video_time_s=np.arange(len(actions)) / 25,
                            robot_joint_velocities=np.asarray([r['velocities'] for r in records]),
                            robot_applied_torques=np.asarray([r['torques'] for r in records]),
                            object_ids=np.asarray([o['instance_id'] for o in objects]),
                            object_labels=np.asarray([o.get('label', o['inst_name']) for o in objects]))
        np.savez_compressed(args.out / 'instance_masks.npz', masks=np.asarray(mask_frames))
        write(args.out / 'instance_maps.json', labels_all)
        task_result = task_state.finish()
        run_status['task_result'] = dict(task_success=task_result['task_success'], environment_valid=task_result['environment_valid'])
        rm = env.reward_manager if args.candidate_task else None
        predicates = []

        def iter_predicates(group):
            if isinstance(group, (list, tuple)) and len(group) == 2 and isinstance(group[0], str) and isinstance(group[1], dict):
                yield group
            elif isinstance(group, (list, tuple)):
                for child in group:
                    yield from iter_predicates(child)
        for item in iter_predicates([] if rm is None else rm.check_list[0]):
            try:
                predicates.append({'predicate': item, 'value': rm.call_func_parser(item, 0)})
            except Exception as e:
                predicates.append({'predicate': item, 'error': repr(e)})
        write(args.out / 'native_predicates.json', predicates)
        run_status.update(status='recorded', frames=len(actions), objects=len(objects), seconds=time.monotonic() - started, native_predicates=predicates)
        write(args.out / 'run.json', run_status); mark('recorded')
    except BaseException as exc:
        run_status.update(status='error', error=repr(exc), traceback=traceback.format_exc(), seconds=time.monotonic() - started)
        write(args.out / 'run.json', run_status); traceback.print_exc(); return 1
    finally:
        faulthandler.cancel_dump_traceback_later()
        if app is not None:
            app.close(wait_for_replicator=False, skip_cleanup=True)
    return 0


if __name__ == '__main__':
    raise SystemExit(main())
