"""Read-only USD geometry export in each object's physical root frame.

Call after Isaac initialization. ``root_pose`` is xyz + wxyz from the physics
pose getter, never inferred from possibly stale Fabric-to-USD world poses.
Descendant transforms come from USD; the root's stretch/scale is baked once.
OBJ and NPZ preserve every polygon, including n-gons, rather than discarding
all but the first mesh. Analytic primitives are tessellated and marked as such.
"""
from __future__ import annotations

import json
from pathlib import Path
import re

import numpy as np


def _array(value, dtype=float):
    if hasattr(value, "detach"):
        value = value.detach().cpu().numpy()
    return np.asarray(value, dtype=dtype)


def _matrix(value):
    """USD/Gf uses row vectors; our arrays use column-vector transforms."""
    return np.asarray(value, dtype=float).T


def _rotation_and_stretch(matrix):
    linear = matrix[:3, :3]
    u, singular, vt = np.linalg.svd(linear)
    if np.min(singular) < 1e-12:
        raise ValueError("Cannot export a root with singular scale")
    rotation = u @ vt
    if np.linalg.det(rotation) < 0:
        u[:, -1] *= -1
        rotation = u @ vt
    return rotation, rotation.T @ linear


def _rotation_from_quaternion(q):
    q = _array(q)
    if q.shape != (4,) or not np.isfinite(q).all() or np.linalg.norm(q) < 1e-12:
        raise ValueError("Expected a finite, nonzero wxyz quaternion")
    w, x, y, z = q / np.linalg.norm(q)
    return np.array([[1-2*(y*y+z*z), 2*(x*y-z*w), 2*(x*z+y*w)],
                     [2*(x*y+z*w), 1-2*(x*x+z*z), 2*(y*z-x*w)],
                     [2*(x*z-y*w), 2*(y*z+x*w), 1-2*(x*x+y*y)]])


def _primitive_geometry(prim, time_code, segments):
    from pxr import UsdGeom
    if prim.IsA(UsdGeom.Cube):
        s = float(UsdGeom.Cube(prim).GetSizeAttr().Get(time_code)) / 2
        vertices = np.array([[-1,-1,-1],[1,-1,-1],[1,1,-1],[-1,1,-1],
                             [-1,-1,1],[1,-1,1],[1,1,1],[-1,1,1]], float) * s
        return vertices, [[0,3,2,1],[4,5,6,7],[0,1,5,4],[1,2,6,5],[2,3,7,6],[3,0,4,7]], "cube_exact"
    if prim.IsA(UsdGeom.Sphere):
        radius = float(UsdGeom.Sphere(prim).GetRadiusAttr().Get(time_code))
        rings = segments // 2
        vertices = [[0,0,radius]]
        for i in range(1, rings):
            phi = np.pi * i / rings
            for j in range(segments):
                theta = 2*np.pi*j/segments
                vertices.append([radius*np.sin(phi)*np.cos(theta), radius*np.sin(phi)*np.sin(theta), radius*np.cos(phi)])
        bottom = len(vertices)
        vertices.append([0,0,-radius])
        faces = [[0, 1+j, 1+(j+1)%segments] for j in range(segments)]
        for i in range(rings-2):
            a, b = 1+i*segments, 1+(i+1)*segments
            faces.extend([[a+j,b+j,b+(j+1)%segments,a+(j+1)%segments] for j in range(segments)])
        a = 1+(rings-2)*segments
        faces.extend([[a+j,bottom,a+(j+1)%segments] for j in range(segments)])
        return np.asarray(vertices), faces, "sphere_tessellated"
    if prim.IsA(UsdGeom.Cylinder):
        shape = UsdGeom.Cylinder(prim)
        radius = float(shape.GetRadiusAttr().Get(time_code))
        height = float(shape.GetHeightAttr().Get(time_code))
        vertices = [[radius*np.cos(2*np.pi*j/segments),radius*np.sin(2*np.pi*j/segments),z]
                    for z in [-height/2,height/2] for j in range(segments)]
        faces = [list(range(segments-1,-1,-1)),list(range(segments,2*segments))]
        faces.extend([[j,(j+1)%segments,(j+1)%segments+segments,j+segments] for j in range(segments)])
        vertices = np.asarray(vertices)
        axis = str(shape.GetAxisAttr().Get(time_code))
        if axis == "X":
            vertices = vertices[:, [2,0,1]]
        elif axis == "Y":
            vertices = vertices[:, [1,2,0]]
        elif axis != "Z":
            raise ValueError(f"Unknown cylinder axis {axis}")
        return vertices, faces, "cylinder_tessellated"
    return None


def _mesh_geometry(prim, time_code):
    from pxr import UsdGeom
    mesh = UsdGeom.Mesh(prim)
    vertices = _array(mesh.GetPointsAttr().Get(time_code)).reshape(-1, 3)
    counts = _array(mesh.GetFaceVertexCountsAttr().Get(time_code), np.int64)
    indices = _array(mesh.GetFaceVertexIndicesAttr().Get(time_code), np.int64)
    if counts.sum() != indices.size or np.any(counts < 3):
        raise ValueError(f"Malformed polygon topology at {prim.GetPath()}")
    if indices.size and (indices.min() < 0 or indices.max() >= len(vertices)):
        raise ValueError(f"Out-of-range vertex index at {prim.GetPath()}")
    holes = set(mesh.GetHoleIndicesAttr().Get(time_code) or [])
    faces = []
    offset = 0
    for i, count in enumerate(counts):
        if i not in holes:
            faces.append(indices[offset:offset+count].tolist())
        offset += count
    return vertices, faces, "authored_mesh_control_cage", {
        "subdivision_scheme": str(mesh.GetSubdivisionSchemeAttr().Get(time_code)),
        "excluded_hole_face_indices": sorted(holes),
        "authored_face_count": len(counts),
    }


def _write_geometry(directory, vertices, faces, part_index):
    counts = np.asarray([len(face) for face in faces], dtype=np.int32)
    indices = np.asarray([index for face in faces for index in face], dtype=np.int64)
    np.savez_compressed(directory / "geometry.npz", vertices=vertices,
                        face_vertex_counts=counts, face_vertex_indices=indices,
                        face_part_index=np.asarray(part_index, dtype=np.int32))
    with (directory / "geometry.obj").open("w") as f:
        f.write("# Geometry only; meters in object rigid root frame; root scale already baked.\n")
        for vertex in vertices:
            f.write("v " + " ".join(format(float(v), ".17g") for v in vertex) + "\n")
        previous = None
        for face, part in zip(faces, part_index):
            if part != previous:
                f.write(f"g part_{part}\n")
                previous = part
            f.write("f " + " ".join(str(i+1) for i in face) + "\n")


def discover_static_fixtures(stage):
    """Recognize only RoboDojo's explicit Table/Ground/Rooms instance groups."""
    fixtures = {}
    for prim in stage.Traverse():
        components = str(prim.GetPath()).strip("/").split("/")
        for i, component in enumerate(components):
            if component in {"Table", "Ground", "Rooms"} and len(components) == i+3:
                fixtures[f"{component.lower()}_{components[-1]}"] = str(prim.GetPath())
    return fixtures


def _under(path, root):
    return path == root or path.startswith(root.rstrip("/")+"/")


def export_scene_geometry(stage, objects, output_dir, *, static_paths=None,
                          robot_paths=None, time_code=None, primitive_segments=32):
    """Export all visible descendant meshes and supported primitives.

    ``objects`` are recorder object records: prim_path, label/inst_name, root_pose
    (xyz+wxyz), instance_id, scale. For static paths pass {label: prim_path},
    or omit to discover RoboDojo Table/Ground/Rooms. ``robot_paths`` identifies
    explicitly excluded canonical robot asset roots. Other visible geometry
    is reported as uncovered, never silently treated as static or complete.
    Dynamic root_pose is required: it defines how these local vertices should
    be placed, while potentially stale USD root translation is only diagnostic.
    Shape changes (cloth, skinning, blend shapes, subdivision displacement) are
    not evaluated here; the manifest records this limitation explicitly.
    """
    from pxr import Usd, UsdGeom
    if primitive_segments < 8 or primitive_segments % 2:
        raise ValueError("primitive_segments must be even and at least eight")
    time_code = Usd.TimeCode.Default() if time_code is None else time_code
    if isinstance(time_code, (float, int)):
        time_code = Usd.TimeCode(time_code)
    output_dir = Path(output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)
    cache = UsdGeom.XformCache(time_code)
    meters_per_unit = float(UsdGeom.GetStageMetersPerUnit(stage))
    if not np.isclose(meters_per_unit, 1.0):
        raise ValueError(f"Recorder physics poses assume meters, but stage metersPerUnit={meters_per_unit}")
    entries = [dict(obj, export_kind="object") for obj in objects]
    static_paths = discover_static_fixtures(stage) if static_paths is None else static_paths
    robot_paths = [str(path) for path in (robot_paths or [])]
    entries += [{"label": name, "prim_path": path, "export_kind": "static"}
                for name, path in (static_paths or {}).items()]
    manifest = {"schema_version": 1, "frame_convention": "root xyz+wxyz; right-handed meters",
                "root_scale_baked": True, "stage_meters_per_unit": meters_per_unit,
                "primitive_segments": primitive_segments, "objects": [],
                "limitations": ["Geometry only: no material textures or UV coordinates copied; canonical USD assets remain authoritative.",
                                "Analytic sphere/cylinder are tessellated; authored mesh polygons are preserved.",
                                "Subdivision/skin/deformation/displacement are not evaluated; authored control meshes are exported.",
                                "Fabric root world transforms may lag physics; descendant local transforms are used, physics root_pose remains authoritative."]}
    for index, obj in enumerate(entries):
        root = stage.GetPrimAtPath(str(obj["prim_path"]))
        if not root.IsValid():
            raise ValueError(f"Missing root prim {obj['prim_path']}")
        world = _matrix(cache.GetLocalToWorldTransform(root))
        stage_rotation, stretch = _rotation_and_stretch(world)
        # Descendants have root transform removed. Reapply only root stretch;
        # never bake its (potentially stale) rigid world pose into local geometry.
        stretch4 = np.eye(4)
        stretch4[:3,:3] = stretch
        label = str(obj.get("label") or obj.get("inst_name") or f"object_{index}")
        slug = f"{index:03d}_" + re.sub(r"[^A-Za-z0-9_.-]", "_", label)
        directory = output_dir / slug
        directory.mkdir(exist_ok=True)
        record = {"label": label, "instance_id": obj.get("instance_id"),
                  "prim_path": str(root.GetPath()), "kind": obj["export_kind"],
                  "root_stretch_baked": stretch.tolist(), "source_stage_root_matrix": world.tolist(),
                  "parts": [], "skipped": [], "directory": slug}
        if obj["export_kind"] == "object":
            pose = _array(obj.get("root_pose"))
            if pose.shape != (7,) or not np.isfinite(pose).all():
                raise ValueError(f"{label}: expected authoritative root_pose xyz+wxyz (7,)")
            actual_rotation = _rotation_from_quaternion(pose[3:])
            record["root_pose"] = pose.tolist()
            record["root_pose_source"] = "physics_getter"
            record["stage_physics_translation_delta_m"] = float(np.linalg.norm(world[:3,3]-pose[:3]))
            record["stage_physics_rotation_delta_deg"] = float(np.degrees(np.arccos(np.clip((np.trace(stage_rotation.T@actual_rotation)-1)/2,-1,1))))
        else:
            root_matrix = np.eye(4)
            root_matrix[:3,:3], root_matrix[:3,3] = stage_rotation, world[:3,3]
            record["root_matrix"] = root_matrix.tolist()
            record["root_pose_source"] = "static_stage_transform"
        vertex_chunks, faces, face_parts = [], [], []
        vertex_count = 0
        for prim in Usd.PrimRange(root, Usd.TraverseInstanceProxies()):
            mesh = prim.IsA(UsdGeom.Mesh)
            primitive = _primitive_geometry(prim, time_code, primitive_segments) if not mesh else None
            if not mesh and primitive is None:
                if prim.IsA(UsdGeom.Gprim):
                    record["skipped"].append({"path": str(prim.GetPath()), "reason": f"unsupported_primitive:{prim.GetTypeName()}"})
                continue
            imageable = UsdGeom.Imageable(prim)
            if imageable.ComputeVisibility(time_code) == UsdGeom.Tokens.invisible:
                record["skipped"].append({"path": str(prim.GetPath()), "reason": "invisible"})
                continue
            if mesh:
                vertices, local_faces, representation, extra = _mesh_geometry(prim, time_code)
            else:
                vertices, local_faces, representation = primitive
                extra = {}
            relative, resets = cache.ComputeRelativeTransform(prim, root)
            if resets:
                raise ValueError(f"{prim.GetPath()}: resetXformStack breaks rigid-root attachment")
            transform = stretch4 @ _matrix(relative)
            vertices = vertices @ transform[:3,:3].T + transform[:3,3]
            if not np.isfinite(vertices).all():
                raise ValueError(f"Nonfinite geometry at {prim.GetPath()}")
            orientation = UsdGeom.Gprim(prim).GetOrientationAttr().Get(time_code)
            flip = (np.linalg.det(transform[:3,:3]) < 0) ^ (orientation == UsdGeom.Tokens.leftHanded)
            if flip:
                local_faces = [face[::-1] for face in local_faces]
            part = len(record["parts"])
            record["parts"].append({"path": str(prim.GetPath()), "type": str(prim.GetTypeName()),
                                    "purpose": str(imageable.ComputePurpose()), "representation": representation,
                                    "vertex_offset": vertex_count, "vertices": len(vertices),
                                    "face_offset": len(faces), "faces": len(local_faces),
                                    "transform_to_root": transform.tolist(), **extra})
            vertex_chunks.append(vertices)
            faces.extend([[i+vertex_count for i in face] for face in local_faces])
            face_parts.extend([part]*len(local_faces))
            vertex_count += len(vertices)
        vertices = np.concatenate(vertex_chunks) if vertex_chunks else np.empty((0,3))
        _write_geometry(directory, vertices, faces, face_parts)
        record.update(vertices=len(vertices), faces=len(faces),
                      complete_supported_visible_geometry=not any(s["reason"].startswith("unsupported") for s in record["skipped"]))
        if len(vertices):
            record["bounds_root"] = [vertices.min(axis=0).tolist(), vertices.max(axis=0).tolist()]
        manifest["objects"].append(record)
    ownership = {str(obj["prim_path"]): obj["label"] for obj in manifest["objects"]}
    coverage = {"known_robot_roots_excluded": robot_paths, "covered_visible_prims": [],
                "excluded_robot_visible_prims": [], "uncovered_visible_prims": []}
    exported_paths = {part["path"] for obj in manifest["objects"] for part in obj["parts"]}
    for prim in Usd.PrimRange(stage.GetPseudoRoot(), Usd.TraverseInstanceProxies()):
        if not prim.IsA(UsdGeom.Gprim):
            continue
        if UsdGeom.Imageable(prim).ComputeVisibility(time_code) == UsdGeom.Tokens.invisible:
            continue
        path = str(prim.GetPath())
        if path in exported_paths:
            coverage["covered_visible_prims"].append(path)
        elif any(_under(path, robot) for robot in robot_paths):
            coverage["excluded_robot_visible_prims"].append(path)
        else:
            coverage["uncovered_visible_prims"].append({"path": path, "type": str(prim.GetTypeName()),
                "declared_owner": next((label for root,label in ownership.items() if _under(path,root)), None)})
    coverage["all_visible_gprims_accounted_for"] = not coverage["uncovered_visible_prims"]
    coverage["scope"] = "USD visible geometry inventory, not camera-frustum visibility or pixel-to-instance mapping verification"
    manifest["coverage"] = coverage
    (output_dir / "manifest.json").write_text(json.dumps(manifest, indent=2, allow_nan=False)+"\n")
    return manifest
