"""Dual xArm7 robot for the RoboDojo tasks: the UFACTORY gripper with one motor and five PhysX mimic constraints.

Official visual/collision meshes, original linkage and inertias. Registered as robot 'v2w_xarm7' in the RoboDojo
robot manager; call ``install_adapter`` after SimulationApp starts, before constructing the task environment.
"""
import hashlib
import json
from pathlib import Path
import sys
import types
import xml.etree.ElementTree as ET
import numpy as np

REST=np.deg2rad([0.,-45.,0.,30.,0.,75.,0.])


GRIPPER_JOINTS = ['drive_joint', 'left_finger_joint', 'left_inner_knuckle_joint',
                  'right_outer_knuckle_joint', 'right_finger_joint', 'right_inner_knuckle_joint']
GRIPPER_MOTOR_TORQUE_NM = 1.2


def build_urdf(source: Path, out: Path, official_mesh_dir: Path, visual_mesh_dir: Path):
    """Preserve physics; use the manufacturer DAE visual option with embedded materials."""
    root = ET.parse(source).getroot()
    root.set('name', 'v2w_xarm7_ufactory_gripper')
    joint_names = {j.get('name') for j in root.findall('joint')}
    if not set(GRIPPER_JOINTS).issubset(joint_names):
        raise ValueError('Expected full xarm7_with_gripper.urdf with real six-joint gripper')
    for item in list(root):
        if item.tag in ('gazebo', 'transmission'):
            root.remove(item)
    for link in root.findall('link'):
        for geometry_type in ('visual', 'collision'):
            for geometry in link.findall(geometry_type):
                for mesh in geometry.iter('mesh'):
                    name = mesh.get('filename', '').replace('package://', '')
                    if geometry_type == 'visual':
                        folder = 'gripper/xarm' if name.startswith('xarm_gripper/meshes/') else 'xarm7_1305/visual'
                        path = (visual_mesh_dir / folder / (Path(name).stem + '.dae')).resolve()
                        # Match common.link.xacro: DAE owns its material assignments.
                        for material in list(geometry.findall('material')):
                            geometry.remove(material)
                    elif name.startswith('xarm_gripper/meshes/'):
                        filename = Path(name).name
                        if link.get('name') == 'xarm_gripper_base_link' and geometry_type == 'collision':
                            filename = 'base_link_collision.stl'
                        path = (official_mesh_dir / filename).resolve()
                    else:
                        path = (source.parent / name).resolve()
                    if not path.exists():
                        raise FileNotFoundError(path)
                    mesh.set('filename', str(path))
    out.parent.mkdir(parents=True, exist_ok=True)
    ET.indent(root)
    ET.ElementTree(root).write(out, encoding='unicode', xml_declaration=True)
    return out


def install_adapter(artifact_dir: Path, arm_urdf: Path, mesh_root: Path):
    """Call after SimulationApp, before constructing a native TaskEnv."""
    from isaaclab.sim.converters import UrdfConverter, UrdfConverterCfg
    from isaaclab.assets import ArticulationCfg
    from isaaclab.actuators import ImplicitActuatorCfg
    import isaaclab.sim as su
    from env.robot_manager.robot_class.x5 import X5
    from env.robot_manager import robot_manager as rm
    artifact_dir=artifact_dir.resolve();artifact_dir.mkdir(parents=True,exist_ok=True)
    if arm_urdf.name == 'xarm7.urdf':
        arm_urdf = arm_urdf.with_name('xarm7_with_gripper.urdf')
    mesh_root = Path(mesh_root)
    official_mesh_dir = mesh_root / 'official_xarm_gripper/xarm_gripper/meshes'
    visual_mesh_dir = mesh_root / 'official_visual_inspection/xarm_description/meshes'
    urdf=build_urdf(arm_urdf,artifact_dir/'xarm7_ufactory_gripper.urdf',official_mesh_dir,visual_mesh_dir)
    # Local compatibility for installed Isaac Sim5.1 importer. This checkout's
    # IsaacLab converter pins unavailable2.4.31 and calls a merge-ignore-inertia
    # API absent from the installed importer. We preserve fixed joints, so that
    # optional merge behavior is irrelevant; do not patch the shared install.
    from isaaclab.sim.converters.asset_converter_base import AssetConverterBase
    import omni.kit.commands
    class InstalledUrdfConverter(UrdfConverter):
        def __init__(self,cfg):
            from isaacsim.asset.importer.urdf._urdf import acquire_urdf_interface
            self._urdf_interface=acquire_urdf_interface()
            AssetConverterBase.__init__(self,cfg=cfg)
        def _get_urdf_import_config(self):
            _,config=omni.kit.commands.execute('URDFCreateImportConfig')
            settings={'distance_scale':1.,'make_default_prim':True,'create_physics_scene':False,
                'density':self.cfg.link_density,'convex_decomp':self.cfg.collider_type=='convex_decomposition',
                'collision_from_visuals':self.cfg.collision_from_visuals,
                'merge_fixed_joints':self.cfg.merge_fixed_joints,'fix_base':self.cfg.fix_base,
                'self_collision':self.cfg.self_collision,
                'parse_mimic':True,
                'replace_cylinders_with_capsules':self.cfg.replace_cylinders_with_capsules}
            for key,value in settings.items():getattr(config,'set_'+key)(value)
            if hasattr(config,'set_merge_fixed_ignore_inertia'):
                config.set_merge_fixed_ignore_inertia(self.cfg.merge_fixed_joints)
            return config
        def _update_joint_parameters(self):
            super()._update_joint_parameters()
            from isaacsim.asset.importer.urdf._urdf import UrdfJointTargetType
            for name in GRIPPER_JOINTS[1:]:
                self._robot_model.joints[name].drive.set_target_type(UrdfJointTargetType.JOINT_DRIVE_NONE)
    converted=InstalledUrdfConverter(UrdfConverterCfg(asset_path=str(urdf),usd_dir=str(artifact_dir/'usd'),
        usd_file_name='xarm7_ufactory_gripper.usd',fix_base=True,merge_fixed_joints=False,
        make_instanceable=False,self_collision=True,collider_type='convex_decomposition',
        joint_drive=UrdfConverterCfg.JointDriveCfg(gains=UrdfConverterCfg.JointDriveCfg.PDGainsCfg(stiffness=1500.,damping=80.))))
    usd=converted.usd_path
    from pxr import Usd, UsdPhysics
    usd_stage = Usd.Stage.Open(usd)
    mimic_schema_paths = [str(prim.GetPath()) for prim in usd_stage.Traverse()
                          if any('PhysxMimicJointAPI' in value for value in prim.GetAppliedSchemas())]
    if len(mimic_schema_paths) != 5:
        raise RuntimeError(f'Expected five native mimic constraints, found {mimic_schema_paths}')
    # Finite compliant native linkage. At 250/1 the 1.2Nm contact load
    # deflected followers by 0.22rad; 1000/1 reached 0.028rad during placement.
    # Use 2000/1 and audit the full loaded trajectory against the same 0.02rad bound.
    # Native hard sentinels 0/0 and -1/-1 both failed in this USD environment.
    # Followers remain passive; no independent follower motor effort is added.
    for path in mimic_schema_paths:
        prim = usd_stage.GetPrimAtPath(path)
        prim.GetAttribute('physxMimicJoint:rotX:naturalFrequency').Set(2000.)
        prim.GetAttribute('physxMimicJoint:rotX:dampingRatio').Set(1.)
    # Preserve the OEM SRDF's complete internal robot collision exclusions. Object
    # contacts, table contacts and contacts with the other arm remain enabled.
    srdf = mesh_root / 'official_xarm_gripper/xarm7_gripper_moveit_config/config/xarm7_with_gripper.srdf'
    gripper_links = {'xarm_gripper_base_link', 'left_outer_knuckle', 'left_finger',
                     'left_inner_knuckle', 'right_outer_knuckle', 'right_finger', 'right_inner_knuckle'}
    rigid = {prim.GetName():prim for prim in usd_stage.Traverse()
             if prim.HasAPI(UsdPhysics.RigidBodyAPI)}
    filtered_pairs = []
    for pair in ET.parse(srdf).getroot().findall('disable_collisions'):
        left, right = pair.get('link1'), pair.get('link2')
        if left in rigid and right in rigid:
            UsdPhysics.FilteredPairsAPI.Apply(rigid[left]).CreateFilteredPairsRel().AddTarget(rigid[right].GetPath())
            filtered_pairs.append({'link1':left,'link2':right,'reason':pair.get('reason')})
    visual_repair=repair(usd_stage)
    (artifact_dir/'visual_subset_repair.json').write_text(json.dumps(visual_repair,indent=2)+'\n')
    usd_stage.GetRootLayer().Save()
    class XArm7(X5):
        def __init__(self,cfg):
            # Reuse interface fields while replacing all robot-specific values.
            seed=dict(cfg);seed['robot_name']='x5';super().__init__(seed)
            self.robot_name='v2w_xarm7';self.urdf_path=str(urdf)
            self.arm_joints_name=[f'joint{i}' for i in range(1,8)]
            self.ee_joint_name='joint7';self.ee_link_name='link7';self.base_link='link_base'
            self.gripper_joints_name=GRIPPER_JOINTS.copy()
            self.save_gripper_joints_name=self.arm_joints_name+self.gripper_joints_name
            self.gripper_move={'base':'drive_joint','sign':-1.,'mimic':['right_outer_knuckle_joint',1.,0.]}
            self.gripper_scale=[0.,.85];self.gripper_bias=.172
            self.camera=None
        def gripper_joint_targets(self, normalized_open):
            # Only one powered DOF. Five followers are native PhysX constraints.
            return {'drive_joint': .85 * (1. - float(np.clip(normalized_open, 0., 1.)))}
    def get_config():
        return ArticulationCfg(spawn=su.UsdFileCfg(usd_path=usd,
                rigid_props=su.RigidBodyPropertiesCfg(disable_gravity=True,max_depenetration_velocity=5.),
                articulation_props=su.ArticulationRootPropertiesCfg(enabled_self_collisions=True,
                    solver_position_iteration_count=64,solver_velocity_iteration_count=8,fix_root_link=True)),
            init_state=ArticulationCfg.InitialStateCfg(joint_pos={**{f'joint{i+1}':float(q) for i,q in enumerate(REST)},**{name:0. for name in GRIPPER_JOINTS}}),
            actuators={'arm':ImplicitActuatorCfg(joint_names_expr=['joint[1-7]'],effort_limit_sim={f'joint{i+1}':limit for i,limit in enumerate([50.,50.,30.,30.,30.,20.,20.])},velocity_limit_sim=3.,stiffness=1500.,damping=80.,armature=.01),
                       'gripper':ImplicitActuatorCfg(joint_names_expr=['drive_joint'],effort_limit_sim=GRIPPER_MOTOR_TORQUE_NM,velocity_limit_sim=2.,stiffness=20.,damping=.8),
                       'passive_gripper':ImplicitActuatorCfg(joint_names_expr=GRIPPER_JOINTS[1:],effort_limit_sim=0.,velocity_limit_sim=2.,stiffness=0.,damping=0.)})
    m=types.ModuleType('env.robot_manager.robot_class.v2w_xarm7');m.XArm7=XArm7;sys.modules[m.__name__]=m
    c=types.ModuleType('env.robot_manager.robot_config.v2w_xarm7');c.get_robot_config=get_config;sys.modules[c.__name__]=c
    rm.ROBOT_CLASS_REGISTRY['v2w_xarm7']={'module':'v2w_xarm7','classes':('XArm7',)}
    rm.ROBOT_CONFIG_REGISTRY['v2w_xarm7']='v2w_xarm7'
    return {'robot_name':'v2w_xarm7','urdf':str(urdf),'usd':usd,
            'gripper':'UFACTORY xArm Gripper: pinned official CAD, original six-joint linkage, one motor + five PhysX mimic constraints',
            'official_gripper_mesh_dir':str(official_mesh_dir),
            'visual_mesh_dir':str(visual_mesh_dir),
            'visual_source':'Official xarm7 mesh_suffix=dae option selects xarm7_1305/visual; original gripper DAE; embedded material assignments retained',
            'visual_subset_repair':str(artifact_dir/'visual_subset_repair.json'),
            'visual_physics_scope':'Only URDF visual meshes/material assignments changed. Existing collision proxies, inertias, joints and drives retained.',
            'native_mimic_joint_paths':mimic_schema_paths,
            'native_mimic_properties':[{a.GetName():str(a.Get()) for a in usd_stage.GetPrimAtPath(path).GetAttributes() if 'Mimic' in a.GetName()} for path in mimic_schema_paths],
            'solver_position_iterations':64,'solver_velocity_iterations':8,
            'mimic_natural_frequency_parameter':2000.,'mimic_frequency_units':'native angular frequency s^-1 (not cycle-frequency Hz)','mimic_constraint_damping_ratio':1.,'mimic_constraint_mode':'finite compliant',
            'oem_internal_collision_exclusions':filtered_pairs,
            'oem_collision_exclusions_source':str(srdf),
            'gripper_motor_torque_limit_nm':GRIPPER_MOTOR_TORQUE_NM,
            'gripper_motor_force_basis':'tau/abs(dy_dq) <=28.6N total pinch over URDF stroke; conservative vs30N hardware rating',
            'arm_source':str(arm_urdf.resolve()),'home_q':REST.tolist()}

def repair(stage):
    """Represent imported DAE material subsets as directly bound visual meshes (some GeomSubset-only meshes do not render).

    Every triangle, normal and material is kept; collision and physics are untouched.
    """
    from pxr import UsdGeom, UsdShade, Vt
    deinstanced=[]
    while True:
        instances=[p for p in stage.Traverse() if p.IsInstance() and '/visuals' in str(p.GetPath())]
        if not instances:break
        for prim in instances:
            deinstanced.append(str(prim.GetPath()));prim.SetInstanceable(False)
    records=[]
    meshes=[p for p in stage.Traverse() if p.GetTypeName()=='Mesh' and '/visuals/' in str(p.GetPath())]
    for prim in meshes:
        subsets=[p for p in prim.GetChildren() if p.GetTypeName()=='GeomSubset' and p.GetAttribute('elementType').Get()=='face']
        if not subsets:continue
        mesh=UsdGeom.Mesh(prim)
        counts=np.array(mesh.GetFaceVertexCountsAttr().Get());assert np.all(counts==3)
        indices=np.array(mesh.GetFaceVertexIndicesAttr().Get()).reshape(-1,3)
        points=np.array(mesh.GetPointsAttr().Get(),dtype=np.float32)
        normals=np.array(mesh.GetNormalsAttr().Get(),dtype=np.float32)
        assert mesh.GetNormalsInterpolation()=='faceVarying' and normals.shape==(len(indices)*3,3)
        assert not UsdGeom.PrimvarsAPI(prim).GetPrimvarsWithAuthoredValues(), 'Unexpected primvars require explicit preservation'
        source_triangles=points[indices];rebuilt=np.empty_like(source_triangles);coverage=np.zeros(len(indices),dtype=np.int32);parts=[]
        transform=UsdGeom.Xformable(prim).GetLocalTransformation()
        for number,subset in enumerate(subsets):
            faces=np.array(subset.GetAttribute('indices').Get(),dtype=np.int64)
            material,_=UsdShade.MaterialBindingAPI(subset).ComputeBoundMaterial();assert material
            destination=prim.GetParent().GetPath().AppendChild(prim.GetName()+f'_material_{number}')
            part=UsdGeom.Mesh.Define(stage,destination)
            vertices=source_triangles[faces].reshape(-1,3)
            part.CreatePointsAttr(Vt.Vec3fArray.FromNumpy(vertices))
            part.CreateFaceVertexCountsAttr(Vt.IntArray([3]*len(faces)))
            part.CreateFaceVertexIndicesAttr(Vt.IntArray.FromNumpy(np.arange(len(vertices),dtype=np.int32)))
            part.CreateNormalsAttr(Vt.Vec3fArray.FromNumpy(normals.reshape(-1,3,3)[faces].reshape(-1,3)))
            part.SetNormalsInterpolation(UsdGeom.Tokens.faceVarying)
            part.CreateSubdivisionSchemeAttr(UsdGeom.Tokens.none)
            part.CreateOrientationAttr(mesh.GetOrientationAttr().Get())
            part.CreateDoubleSidedAttr(mesh.GetDoubleSidedAttr().Get())
            part.CreateExtentAttr(Vt.Vec3fArray.FromNumpy(np.stack([vertices.min(0),vertices.max(0)])))
            part.AddTransformOp().Set(transform)
            UsdShade.MaterialBindingAPI.Apply(part.GetPrim()).Bind(material)
            coverage[faces]+=1;rebuilt[faces]=vertices.reshape(-1,3,3)
            parts.append({'path':str(destination),'material':str(material.GetPath()),'faces':len(faces)})
        assert np.all(coverage==1) and np.array_equal(rebuilt,source_triangles)
        # Keep the imported source for inspection but never draw it twice.
        UsdGeom.Imageable(prim).CreateVisibilityAttr(UsdGeom.Tokens.invisible)
        records.append({'original_mesh':str(prim.GetPath()),'triangles':len(indices),
                        'geometry_exact_equal':True,'triangle_coordinates_sha256':hashlib.sha256(source_triangles.tobytes()).hexdigest(),'parts':parts})
    return {'scope':'Visual representation compatibility fix only. Original triangle coordinates, winding, normals and factory material assignments preserved. No collision or joint edits.',
            'deinstanced_visual_roots':deinstanced,'repaired_meshes':records,'all_triangle_coordinates_exact':bool(records) and all(r['geometry_exact_equal'] for r in records)}
