"""Load the submitted scene into the native RoboDojo layout manager and point the head camera at the submitted view.

Call inside the recorder after SimulationApp starts. All object state writes are the native environment's
initialization/reset writes.
"""
from copy import deepcopy
from pathlib import Path
import json
import numpy as np


def install(cfg, scene_path, artifact_dir, task_name=None):
    from omegaconf import OmegaConf
    from pxr import Usd, UsdGeom, UsdPhysics, UsdShade, Sdf, Gf
    from env.scene_manager.layout_manager import LayoutManager
    from env.camera_manager import camera_manager as cm
    from env.environment.task_env import TaskEnv
    import transforms3d as t3d
    scene_path = Path(scene_path).resolve()
    scene = json.loads(scene_path.read_text())
    artifact_dir = Path(artifact_dir).resolve()
    artifact_dir.mkdir(parents=True, exist_ok=True)
    entities = [(obj, 'Rigid') for obj in scene['objects']]
    entities += [(scene['table'], 'Geometry')]
    entities += [(obj, 'Geometry') for obj in scene.get('props', [])]
    boxes = []
    vertices_unit = np.array([[-1,-1,-1],[1,-1,-1],[1,1,-1],[-1,1,-1],[-1,-1,1],[1,-1,1],[1,1,1],[-1,1,1]], float)
    faces = [[0,3,2,1],[4,5,6,7],[0,1,5,4],[1,2,6,5],[2,3,7,6],[3,0,4,7]]
    for index, (obj, typ) in enumerate(entities):
        size = np.asarray(obj['size'], float)
        q = np.asarray(obj['quat_wxyz'], float)
        if size.shape != (3,) or min(size) <= 0 or not np.isfinite(size).all(): raise ValueError('invalid box size')
        if q.shape != (4,) or not np.isfinite(q).all() or abs(np.linalg.norm(q)-1) > .01: raise ValueError('invalid box quaternion')
        mass = float(obj.get('mass', .1))
        if not np.isfinite(mass) or (not task_name and mass > .5) or mass < 0 or (typ == 'Rigid' and mass == 0 and not (task_name and obj.get('asset_path'))): raise ValueError('mass outside declared schema')
        if obj.get('asset_path'):
            asset = (scene_path.parent / obj['asset_path']).resolve()
            if not asset.is_relative_to(scene_path.parent) or not asset.is_file():
                raise ValueError('Asset must be a file inside the submitted package')
            vertices = vertices_unit * size / 2
            boxes.append((deepcopy(obj), typ, str(asset), vertices.tolist()))
            continue
        file = artifact_dir / f'box_{index:03d}.usda'
        stage = Usd.Stage.CreateNew(str(file))
        UsdGeom.SetStageUpAxis(stage, UsdGeom.Tokens.z); UsdGeom.SetStageMetersPerUnit(stage, 1.)
        root = UsdGeom.Xform.Define(stage, '/box').GetPrim(); stage.SetDefaultPrim(root)
        mesh = UsdGeom.Mesh.Define(stage, '/box/mesh')
        vertices = vertices_unit * size / 2
        mesh_faces = faces
        if obj.get('geometry_npz'):
            geometry_path = (scene_path.parent / obj['geometry_npz']).resolve()
            if not geometry_path.is_relative_to(scene_path.parent):
                raise ValueError('Geometry must remain inside package')
            geometry = np.load(geometry_path, allow_pickle=False)
            vertices = geometry['vertices']; counts = geometry['face_vertex_counts']; indices = geometry['face_vertex_indices']
            mesh_faces = []; offset = 0
            for count in counts:
                mesh_faces.append(indices[offset:offset+count].tolist()); offset += count
            if offset != len(indices) or not np.isfinite(vertices).all(): raise ValueError('Invalid mesh')
        mesh.CreatePointsAttr(vertices.tolist())
        mesh.CreateFaceVertexCountsAttr([len(face) for face in mesh_faces])
        mesh.CreateFaceVertexIndicesAttr(sum(mesh_faces, []))
        mesh.CreateSubdivisionSchemeAttr('none')
        mesh.CreateDisplayColorAttr([Gf.Vec3f(*obj['color'])])
        UsdPhysics.CollisionAPI.Apply(mesh.GetPrim()).CreateCollisionEnabledAttr(True)
        UsdPhysics.MeshCollisionAPI.Apply(mesh.GetPrim()).CreateApproximationAttr('convexHull' if typ == 'Rigid' else 'none')
        if typ == 'Rigid':
            UsdPhysics.RigidBodyAPI.Apply(root).CreateRigidBodyEnabledAttr(True)
            UsdPhysics.MassAPI.Apply(root).CreateMassAttr(mass)
        material = UsdShade.Material.Define(stage, '/box/material')
        shader = UsdShade.Shader.Define(stage, '/box/material/shader')
        shader.CreateIdAttr('UsdPreviewSurface')
        shader.CreateInput('diffuseColor', Sdf.ValueTypeNames.Color3f).Set(Gf.Vec3f(*obj['color']))
        shader.CreateInput('roughness', Sdf.ValueTypeNames.Float).Set(.5)
        material.CreateSurfaceOutput().ConnectToSource(shader.ConnectableAPI(), 'surface')
        UsdShade.MaterialBindingAPI.Apply(mesh.GetPrim()).Bind(material)
        stage.GetRootLayer().Save()
        boxes.append((deepcopy(obj), typ, str(file), vertices.tolist()))

    def load_candidate(self, env_idx):
        if self.saved_layouts[env_idx] is None: return None
        self.clear_layout_state([env_idx])
        result = {'Rigid': {'candidate': []}, 'Geometry': {'candidate': []}}
        for index, (obj, typ, file, vertices) in enumerate(boxes):
            prim_path, inst_name = self._generate_object_paths(env_idx, 'candidate', index, type=typ.lower())
            record = dict(category=obj.get('category', obj['name']), category_idx=index, label=obj['name'],
                          inst_name=inst_name, prim_path=prim_path, usd_path=file,
                          default_pos=obj['pos'], default_ori=obj['quat_wxyz'], scale=obj.get('scale', [1.,1.,1.]),
                          physics={'type':typ.lower(), 'mass':obj.get('mass',.1),
                                   'collision':obj.get('collision', True), 'static_friction':obj.get('static_friction',.6),
                                   'dynamic_friction':obj.get('dynamic_friction',.6), 'restitution':obj.get('restitution',0.)}, visual={})
            metadata = {'model_name':obj.get('category',obj['name']), 'model_id':index,
                        'geometry':{'oriented_bbox':{'vertices':vertices}}, 'provenance':'package-authored geometry; shared GT/candidate loader'}
            # Submitted semantic frames belong to the candidate, never hidden GT.
            if task_name and obj.get('metadata'):
                metadata.update(deepcopy(obj['metadata']))
                metadata['provenance'] = 'submitted candidate metadata; independently scored geometry'
            self.object_records_by_type[typ].add_instance(env_idx,record,metadata)
            self.instance_type_by_env[env_idx][inst_name] = typ.lower()
            result[typ]['candidate'].append(record)
        table = scene['table']; sx,sy,sz = table['size']
        self.table_info[env_idx] = {'pos':table['pos'], 'size':[-sx/2,-sy/2,sx/2,sy/2], 'height':table['pos'][2]+sz/2}
        self.cluttered_generator_init(env_idx)
        return result

    LayoutManager.load_saved_layout = load_candidate
    # Neutral constant illumination belongs to the renderer, not episode geometry.
    def setup_background(self):
        from pxr import UsdLux
        UsdLux.DomeLight.Define(self.stage, '/World/candidate_light').CreateIntensityAttr(1000.)
    from env.scene_manager.scene_manager import SceneManager
    # Imported articulations initially exist at URDF zero before TaskEnv.reset
    # drives the public home posture. The xArm zero-pose fingers intersect a
    # table at mounting height and can destroy passive mimic constraints on
    # the very first scene-manager physics step. Keep candidate objects away
    # during this initialization, then native apply_saved_poses restores the
    # exact agent poses after robot reset. No object writes occur in rollout.
    original_create = SceneManager.create_scene_object
    staging_counter = [0]
    def create_staged(self, env_id, asset_to_spawn, prim_path, inst_cfg,
                      default_pos, default_ori, scale):
        staging_counter[0] += 1
        parked = (1000. + staging_counter[0] * 10., 1000., 1000.)
        obj = original_create(self, env_id, asset_to_spawn, prim_path,
                              inst_cfg, parked, default_ori, scale)
        if obj is not None:
            obj.default_pos = deepcopy(default_pos)
        return obj
    SceneManager.create_scene_object = create_staged
    SceneManager._setup_background = setup_background
    SceneManager.reload_background = lambda self: None
    cfg.scene = OmegaConf.create({}); cfg.task_env = OmegaConf.create({})
    camera = scene['camera']; position = np.asarray(camera['pos'], float)
    forward = np.asarray(camera['look_at'], float)-position; forward /= np.linalg.norm(forward)
    right = np.cross(forward,[0.,0.,1.] if abs(forward[2])<.999 else [0.,1.,0.]); right /= np.linalg.norm(right)
    up = np.cross(right,forward)
    rot = np.column_stack([right,up,-forward])
    if camera.get('T_world_camera') is not None:
        optical=np.asarray(camera['T_world_camera'],float)
        position=optical[:3,3]
        rot=optical[:3,:3]@np.diag([1.,-1.,-1.])
    orientation = t3d.quaternions.mat2quat(rot).tolist()
    width,height = int(camera['width']),int(camera['height']); focal = float(camera['focal_px'])
    if not 64 <= width <= 1920 or not 64 <= height <= 1080 or focal <= 0: raise ValueError('invalid camera')
    cm.REAL_MAP['candidate_view'] = {'resolution':(width,height),'focal_length':10.,
        'horizontal_aperture':10.*width/focal,'vertical_aperture':10.*height/focal,'clipping_range':(.005,100.)}
    cfg.camera.cam_head.camera = OmegaConf.create({'type':'candidate_view','mesh':'pinhole','pos':position.tolist(),'ori':orientation})
    (artifact_dir/'scene_source.json').write_text(json.dumps(scene,indent=2)+'\n')
    if task_name:
        from task.RoboDojo.task_registry import load_task_class
        _, task_class = load_task_class(task_name)
        return task_class
    return TaskEnv
