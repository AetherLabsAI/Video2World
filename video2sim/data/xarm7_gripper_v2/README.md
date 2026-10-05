# xArm7 with source revolute gripper (urdf-v2)

Derived from the upstream xarm7_with_gripper_collision.urdf (SHA-256 in robot_spec.json).
This collision variant includes the author's extended fingers. It is a declared
collision variant, not a claim that it is the shorter visual-URDF fingertip.

Use robot.xml as a standalone mechanical test. For a scene, use compiler
angle="radian" meshdir="robot", include robot/robot_assets.xml at the top level,
robot/robot_body.xml inside worldbody, robot/robot_actuators.xml inside actuator,
robot/robot_equalities.xml inside equality and robot/robot_contacts.xml inside
contact. Keep the original body placement except the candidate-owned xarm_base
root pos/quat. The seven joints' reference offsets are already baked correctly.

Controls: 8 position targets = joint1..7 (rad), drive_joint (0 open to 0.85 closed).
Five other revolute joints follow drive_joint via the original mimic relations.
Native execution slews the drive target at at most 2 rad/s. Use implicitfast,
0.00005 s timestep and 100 solver iterations. Keep all limits/inertias/gains,
the five mimic equalities and exactly four linkage-adjacency contact exclusions.
No robot/object or receiver contact may be excluded. No object weld/actuator.
TCP site link_tcp is 0.172 m along link7's local z. Open inner gap ~87.73 mm;
closed gap ~0.446 mm. These are mesh-derived geometric gaps, not slide distances.

Canonical includes are robot_{assets,body,actuators,equalities,contacts}.xml.
import_meshes and mechanical_contact_trial.xml are conversion/diagnostic artifacts,
not scene inputs. Original mesh shape is retained in binary STL for MuJoCo.

Native execution checks all five mimic relations at EVERY physics step and rejects error >0.001 rad. Use equality solref=".0001 1" at this timestep. Earlier 0.0002 s templates were insufficient under some Toy contacts.
