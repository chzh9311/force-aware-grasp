import os
import time
import numpy as np
import sys
sys.path.append('.')
import trimesh
from lxml import etree
import coacd
import mujoco
import mujoco.viewer
from collections import defaultdict
from matplotlib import pyplot as plt
import quaternion
import copy
import open3d as o3d
from math import asin, sqrt
from tqdm import tqdm
import open3d.visualization as vis
from mujoco import rollout
from easydict import EasyDict as edict
from common.utils.vis import o3dmesh_from_trimesh, o3d_arrow
from common.utils.manolayer import get_part_meshes

kine_tree = {'Index': [0, 1, 2, 3], 'Mid': [0, 4, 5, 6], 'Little': [0, 10, 11, 12],
             'Ring': [0, 7, 8, 9], 'Thumb': [0, 13, 14, 15]}

kine_tree_w_tips = {'Index': [0, 1, 2, 3, 17], 'Mid': [0, 4, 5, 6, 18], 'Little': [0, 10, 11, 12, 20],
                    'Ring': [0, 7, 8, 9, 19], 'Thumb': [0, 13, 14, 15, 16]}

## 800N for thumb & palm; 150N for the rest fingers.
force_threshold = [800,] + [150,] * 12 + [800,] * 3

def array2str(array):
    return " ".join([f"{a:.4f}" for a in array])

def run_mujoco_simulate(label_cfg, model_path, hand_mesh, hand_joints, obj_model, obj_hulls, part_ids, part_normals=None, part_dists=None):
    mesh2mjcf(label_cfg.default_mjcf_settings, model_path, hand_mesh, hand_joints, obj_model, obj_hulls, part_ids, part_normals, part_dists)
    contact_info, obj_qposes, opt_tr = run_adaptive_sim(model_path, label_cfg.adaptive_sim)
    if opt_tr != 1:
        new_model = etree.parse(os.path.join(model_path, 'hand_model.xml'))
        element = new_model.xpath("//mujoco/default/default[@class='hand_model']/geom")[0]
        solref = element.get('solref')
        new_solref = f"{solref.split(' ')[0]} {opt_tr:.4f}"
        element.set('solref', new_solref)
        with open(os.path.join(model_path, "hand_model.xml"), "wb") as file:
            new_model.write(file)

    return contact_info, obj_qposes

def mesh2mjcf(default_cfg, model_path, hand_mesh, hand_joints, obj_model, obj_hulls, part_ids, part_normals, part_dists):
    """
    :param hand_params: dict with rot_aa, trans, pose, shape.
    :param mano_layer: mano layer.
    :param obj_mesh: object mesh in trimesh format.
    Saves the current state to MJCF file.
    """
    timestep = default_cfg.timestep
    joint_type = default_cfg.joint_type
    os.makedirs(model_path, exist_ok=True)

    hand_parts = get_part_meshes(hand_mesh.vertices, hand_mesh.faces, part_ids)
    hand_parts = [trimesh.convex.convex_hull(m) for m in hand_parts]
    # hand_parts = [hand_parts[0]] + [trimesh.convex.convex_hull(m) for m in hand_parts[1:]]

    hand_mesh_name = 'hand_model'
    obj_mesh_name = 'obj_model'
    os.makedirs(os.path.join(model_path, hand_mesh_name), exist_ok=True)
    os.makedirs(os.path.join(model_path, obj_mesh_name), exist_ok=True)

    root = etree.Element('mujoco')
    root.set('model', hand_mesh_name)
    compiler = etree.SubElement(root, 'compiler')
    compiler.set('meshdir', '.')
    compiler.set('angle', 'radian')
    compiler.set('autolimits', 'true')

    default_top = etree.SubElement(root, 'default')
    default_hand_class = etree.SubElement(default_top, 'default')

    default_hand_class.set('class', hand_mesh_name)
    default_hand_geom = etree.SubElement(default_hand_class, 'geom')
    default_hand_geom.set('type', 'mesh')
    default_hand_geom.set('rgba', '0.8 0.9 0.6 1')
    default_hand_geom.set('contype', '1')
    default_hand_geom.set('conaffinity', '1')
    for attr in ['solref', 'margin', 'solimp', 'friction']:
        if hasattr(default_cfg, attr):
            default_hand_geom.set(attr, default_cfg[attr])
    # default_hand_geom.set('margin', default_cfg.margin)
    # default_hand_geom.set('solimp', default_cfg.solimp)
    default_joint = etree.SubElement(default_hand_class, 'joint')
    if joint_type != 'rigid':
        default_joint.set('type', joint_type)
    default_joint.set('stiffness', '0')
    default_joint.set('limited', 'true')

    default_obj_class = etree.SubElement(default_top, 'default')
    default_obj_class.set('class', obj_mesh_name)
    default_obj_geom = etree.SubElement(default_obj_class, 'geom')
    default_obj_geom.set('type', 'mesh')
    default_obj_geom.set('rgba', '0.7 0.7 0.95 1')
    default_obj_geom.set('contype', '1')
    default_obj_geom.set('conaffinity', '1')
    default_obj_geom.set('friction', '1. 0.5 0.01')
    # default_joint.set('range' '-0.2, 0.2') ## Approximately 12 degrees
    option = etree.SubElement(root, 'option')
    option.set('timestep', str(timestep))
    flag = etree.SubElement(option, 'flag')
    flag.set('multiccd', 'enable')

    asset = etree.SubElement(root, "asset")

    # actuator = etree.SubElement(root, 'actuator')
    #
    # contact = etree.SubElement(root, 'contact')

    keyframe = etree.SubElement(root, 'keyframe')
    key = etree.SubElement(keyframe, 'key')

    ## Material
    texture = etree.Element('texture', attrib={"name": "grid", "type": "2d", "builtin": "checker",
                                               "rgb1": ".1 .2 .3", "rgb2": ".2 .3 .4", "width": "300",
                                               "height": "300", "mark": "edge", "markrgb": ".2 .3 .4"})
    asset.append(texture)
    material = etree.Element('material', attrib={"name": "grid", "texture": "grid", "texrepeat": "2 2",
                                                 "texuniform": "true", "reflectance": ".2",})
    asset.append(material)

    ## hand link to the world coordinate system.
    worldbody = etree.SubElement(root, "worldbody")
    camera = etree.SubElement(worldbody, "camera", attrib={"name": 'cam', "pos": "-.1 -.6 .3", "xyaxes": "1 0 0 0 1 2"})
    light = etree.SubElement(worldbody, "light", attrib={"pos": "0 0 1"})

    body = etree.SubElement(worldbody, "body")
    body.set('name', 'hand_world')
    body.set('pos', array2str([0, 0, 0]))
    body.set('euler', array2str([0, 0, 0]))

    ## palm
    palm_body = etree.SubElement(body, "body")
    palm_body.set('name', 'Palm_link')

    hand_root_pos = hand_joints[0]
    palm_body.set('pos', array2str(hand_root_pos))
    palm_body.set('euler', array2str([0, 0, 0]))
    # rot_angle = np.linalg.norm(hand_params['rot_aa'].squeeze())
    # rot_axis = hand_params['rot_aa'].squeeze() / (rot_angle + 0.00001)
    # body.set('axisangle', array2str(list(rot_axis) + [rot_angle]))

    # o3d.visualization.draw_geometries([o3dmesh_from_trimesh(hand_parts[0])])
    palm_decompose = False
    if palm_decompose:
        palm_coacd_mesh = coacd.Mesh(hand_parts[0].vertices, hand_parts[0].faces)
        palm_hulls = coacd.run_coacd(palm_coacd_mesh, threshold=0.1)
        for i, hull in enumerate(palm_hulls):
            palm_geom = etree.Element("geom")
            palm_part_mesh = trimesh.Trimesh(hull[0], hull[1])
            palm_part_mesh.apply_translation(-hand_root_pos)
            palm_geom.set('class', hand_mesh_name)
            palm_geom.set('name', f'palm_geom_{i}')

            palm_mesh_leaf = etree.Element("mesh")
            palm_mesh_leaf.set('name', f'palm_hull_{i}')
            palm_mesh_leaf.set('file', os.path.join(hand_mesh_name, f'palm_hull_{i}.stl'))
            asset.append(palm_mesh_leaf)

            palm_geom.set('mesh', f'palm_hull_{i}')
            palm_part_mesh.export(os.path.join(model_path, hand_mesh_name, f'palm_hull_{i}.stl'))
            palm_body.append(palm_geom)
    else:
        palm_part_mesh = hand_parts[0]
        palm_geom = etree.Element("geom")
        palm_part_mesh.apply_translation(-hand_root_pos)
        palm_geom.set('class', hand_mesh_name)
        palm_geom.set('name', f'palm_geom')

        palm_mesh_leaf = etree.Element("mesh")
        palm_mesh_leaf.set('name', f'palm_hull')
        palm_mesh_leaf.set('file', os.path.join(hand_mesh_name, f'palm_hull.stl'))
        asset.append(palm_mesh_leaf)

        palm_geom.set('mesh', f'palm_hull')
        palm_part_mesh.export(os.path.join(model_path, hand_mesh_name, f'palm_hull.stl'))
        palm_body.append(palm_geom)

    joint_k = []
    for fg_name, ktree in kine_tree.items():
        parent_body = palm_body
        root_coord = hand_root_pos
        for i, jt in enumerate(ktree):
            if i == 0:
                continue
            finger_mesh = hand_parts[jt]
            finger_mesh.apply_translation(- hand_joints[jt])
            finger_mesh_leaf = etree.Element('mesh')
            finger_mesh_leaf.set('name', fg_name + f"_hull_{i}")
            finger_mesh_leaf.set('file', os.path.join(hand_mesh_name, f"{fg_name}_hull_{i}.stl"))
            asset.append(finger_mesh_leaf)

            finger_body = etree.SubElement(parent_body, "body")
            finger_body.set('name', fg_name + f"_link_{i}")
            finger_body.set('pos', array2str(hand_joints[jt] - root_coord))
            # finger_body.set('pos', array2str([0, 0, 0]))
            finger_body.set('euler', array2str([0, 0, 0]))
            ## gravity compensation set to 1 to avoid the effect of self-gravity.
            finger_body.set('gravcomp', '1')

            finger_geom = etree.Element("geom")
            finger_geom.set('class', hand_mesh_name)
            finger_geom.set('name', fg_name + f"_geom_{i}")
            finger_geom.set('mesh', fg_name + f"_hull_{i}")
            finger_body.append(finger_geom)

            # Assume only one joint for the thumb.
            # finger_obj_dist = part_dists[jt].item()

            # if finger_obj_dist < -0.001:
            #     ## Solve penetrations
            #     obj_normal = part_normals[jt]
            #     finger_joint = etree.Element("joint")
            #
            #     if joint_type == 'rigid':
            #         ## Make slight shifts to confirm contact
            #         finger_mesh.apply_translation(- obj_normal * finger_obj_dist)
            #     else:
            #         if joint_type == 'hinge':
            #             joint2tip = hand_joints[jt] - hand_joints[ktree[i-1]]
            #             hinge_axis = np.cross(obj_normal, joint2tip)
            #             hinge_axis /= np.linalg.norm(hinge_axis, axis=-1, keepdims=True) + 1.0e-6
            #             finger_joint.set('axis', array2str(hinge_axis))
            #             error_angle = np.arcsin(finger_obj_dist / np.linalg.norm(joint2tip)).item()
            #             joint_k.append(error_angle * 2 / 3)
            #             finger_joint.set('range', f'{error_angle:.4f} 0.00')
            #
            #         elif joint_type == 'slide':
            #             finger_joint.set('axis', array2str(-obj_normal))
            #             finger_joint.set('range', f'{finger_obj_dist*0.9:.4f} 0.0000')
            #             joint_k.append(finger_obj_dist * 0.9)
            #
            #         finger_joint.set("class", hand_mesh_name)
            #         finger_joint.set('name', f'{fg_name}_joint_{i}')
            #         finger_joint.set('pos', array2str([0, 0, 0]))
            #         finger_joint.set('stiffness', '5')
            #         finger_body.append(finger_joint)
            #
            #         motor = etree.Element("motor")
            #         motor.set('name', f'{fg_name}_motor_{i}')
            #         motor.set('joint', f'{fg_name}_joint_{i}')
            #         motor.set('gear', '1 0 0 0 0 0')
            #         actuator.append(motor)
            #         exclude_contact = etree.Element("exclude")
            #         exclude_contact.set('name', f'{fg_name}_exclude_{i}')
            #         exclude_contact.set('body1', f'{fg_name}_link_{i}')
            #         if i == 1:
            #             exclude_contact.set('body2', 'Palm_link')
            #         else:
            #             exclude_contact.set('body2', f'{fg_name}_link_{i - 1}')
            #         contact.append(exclude_contact)
            #
            finger_mesh.export(os.path.join(model_path, hand_mesh_name, f'{fg_name}_hull_{i}.stl'))
            parent_body = finger_body
            root_coord = hand_joints[jt]

    joint_k.extend([0, 0, 0, 1, 0, 0, 0])
    key.set('name', 'Fixed H-O pose')
    key.set('qpos', array2str(joint_k))

    ## Object mesh
    # obj_coacd_mesh = coacd.Mesh(obj_mesh.vertices, obj_mesh.faces)
    # obj_hulls = coacd.run_coacd(obj_coacd_mesh)
    body = etree.SubElement(worldbody, "body")
    body.set('name', 'object')

    ## The object is floating
    obj_free_joint = etree.Element('joint')
    obj_free_joint.set('type', 'free')
    obj_free_joint.set('name', 'object2world')
    # obj_free_joint.set('stiffness', '20')
    obj_free_joint.set('damping', '1')
    body.append(obj_free_joint)

    # inertial = etree.Element('inertial')
    # inertial.set('pos', array2str(obj_model.center_mass))
    # inertial.set('mass', '1.0') # object masses are fixed to 1kg.
    # inertial.set('quat', '1.0 0 0 0')

    ## Constructing scene with plane
    # planebody = etree.SubElement(worldbody, "body")
    # plane = etree.SubElement(planebody, "geom")
    # plane.set('type', 'plane')
    # plane.set('size', '2 2 .1')
    # plane.set('pos', '0 0 -0.5')
    # plane.set('friction', '1. 0.5 0.01')
    # plane.set('material', 'grid')
    total_vol = sum([hull.volume for hull in obj_hulls])
    density = 1 / total_vol # Ensure the object is 1kg and of even density.

    for i in range(len(obj_hulls)):
        # convex_hull = trimesh.Trimesh(obj_hulls[i][0], obj_hulls[i][1])
        convex_hull = obj_hulls[i]
        convex_hull_name = f"obj_hull_{i}"
        convex_hull_mesh = etree.Element("mesh")
        convex_hull_mesh.set("name", convex_hull_name)
        convex_hull_mesh.set(
            "file", f"{obj_mesh_name}/{convex_hull_name}.stl")
        asset.append(convex_hull_mesh)
        convex_hull.export(os.path.join(model_path, obj_mesh_name, f'{convex_hull_name}.stl'))

        convex_hull_geom = etree.Element("geom")
        convex_hull_geom.set("type", "mesh")
        convex_hull_geom.set("mesh", convex_hull_name)
        convex_hull_geom.set("class", obj_mesh_name)
        convex_hull_geom.set('density', str(density))
        body.append(convex_hull_geom)

    tree = etree.ElementTree(root)
    etree.indent(tree, space="  ", level=0)
    with open(os.path.join(model_path, "hand_model.xml"), "wb") as files:
        tree.write(files)

def run_sim(model_path, visualize=False):
    ## Options
    dist_stable_time = 1 # second
    dist_stable_th = 0.001
    force_sample_steps = 100

    model = mujoco.MjModel.from_xml_path(os.path.join(model_path, "hand_model.xml"))
    timestep = model.opt.timestep
    data = mujoco.MjData(model)
    # data.qpos = 0.01
    jt = model.joint('object2world')
    # jt.stiffness = 50
    obj_body = data.body('object')
    # data.ctrl = 5
    # data.ctrl = 1
    # data.actuator_force = 10
    # visualize contact frames and forces, make body transparent
    # tweak scales of contact visualization elements

    # mujoco.mj_resetDataKeyframe(model, data, 0)
    slow_rate = 1
    if visualize:
        model.vis.scale.contactwidth = 0.1
        model.vis.scale.contactheight = 0.03
        model.vis.scale.forcewidth = 0.02
        model.vis.map.force = 0.01
        viewer = mujoco.viewer.launch_passive(model, data)
        viewer.opt.flags[mujoco.mjtVisFlag.mjVIS_CONTACTFORCE] = True
        viewer.opt.flags[mujoco.mjtVisFlag.mjVIS_TRANSPARENT] = True

    disps = []
    obj_qpos = []
    forcetorque = np.zeros(6)
    stable = False
    dynamic_conatct_info = []
    step_cnt = 0
    n_stable_step = dist_stable_time / timestep
    # print('start simulation...')

    while not visualize or viewer.is_running():
        step_start = time.time()
        mujoco.mj_step(model, data)
        disp = obj_body.xpos
        # print(data.joint('object2world').qpos)
        if len(disps) >= n_stable_step and not stable:
            if np.linalg.norm(disps[0] - disps[-1]) < dist_stable_th:
                ## Stable
                stable = True
                # print("Pose stable")
        else:
            disps.append(disp)
        if visualize:
            viewer.sync()

        if stable and step_cnt < force_sample_steps:
            contact_info = []
            for j, c in enumerate(data.contact):
                mujoco.mj_contactForce(model, data, j, forcetorque)
                if c.geom1 < 16 <= c.geom2:
                    hpart = c.geom1
                elif c.geom2 < 16 <= c.geom1:
                    hpart = c.geom2
                else:
                    continue
                contact_info.append({ 'frame': c.frame,
                    'hand_part_id': hpart, 'force': forcetorque[0:3], 'pos': c.pos
                })
            dynamic_conatct_info.append(contact_info)
            obj_qpos.append(data.joint('object2world').qpos.copy())
            step_cnt += 1

        time_until_next_step = slow_rate * model.opt.timestep - (time.time() - step_start)
        if visualize and time_until_next_step > 0:
            time.sleep(time_until_next_step)

        if step_cnt > force_sample_steps - 1 and not visualize:
            # print(np.mean(force, axis=0))
            break

    # force_abs = np.linalg.norm(force, axis=-1)
    # plt.plot(np.arange(self.force_sample_steps), force_abs)
    # plt.show()
    return dynamic_conatct_info, np.stack(obj_qpos, axis=0)


def run_adaptive_sim(model_path, ada_cfg, visualize=False):
    """
    Search for the best stiffness for each contact part separately.
    """
    sim_time = 1 # second

    model = mujoco.MjModel.from_xml_path(os.path.join(model_path, "hand_model.xml"))
    total_steps = int(sim_time / model.opt.timestep)
    data = mujoco.MjData(model)

    # start_ratio2 = time_ratio2[np.argmin(np.array(disps))]
    contact_geom_ids = []
    mujoco.mj_step(model, data)
    for j, c in enumerate(data.contact):
        if c.geom1 < 16 <= c.geom2 and c.geom1 not in contact_geom_ids:
            contact_geom_ids.append(c.geom1)
        elif c.geom2 < 16 <= c.geom1 and c.geom1 not in contact_geom_ids:
            contact_geom_ids.append(c.geom2)
        opt_tr = 1
    if ada_cfg.use_ada_solref:
        time_ratios = np.exp(np.linspace(ada_cfg.coarse_search_range[0], ada_cfg.coarse_search_range[1], ada_cfg.coarse_nbins))
        disps = run_simulations_w_param(model, data, contact_geom_ids, time_ratios, total_steps)
        opt_idx = np.argmin(disps)
        if opt_idx == 0:
            new_trs = np.linspace(time_ratios[opt_idx] * 0.1, time_ratios[opt_idx + 1], ada_cfg.fine_nbins)
        elif opt_idx == len(time_ratios) - 1:
            new_trs = np.linspace(time_ratios[opt_idx-1], time_ratios[opt_idx] * 1.2, ada_cfg.fine_nbins)
        else:
            new_trs = np.linspace(time_ratios[opt_idx-1], time_ratios[opt_idx + 1], ada_cfg.fine_nbins)

        disps = run_simulations_w_param(model, data, contact_geom_ids, new_trs, total_steps)
        opt_idx = np.argmin(disps)
        for gid in contact_geom_ids:
            model.geom(gid).solref[1] = new_trs[opt_idx]
        mujoco.mj_resetDataKeyframe(model, data, 0)
        opt_tr = new_trs[opt_idx]

    dynamic_contact_info = []
    forcetorque = np.zeros(6)
    obj_qpos = []

    if visualize:
        model.vis.scale.contactwidth = 0.1
        model.vis.scale.contactheight = 0.03
        model.vis.scale.forcewidth = 0.02
        model.vis.map.force = 0.01
        viewer = mujoco.viewer.launch_passive(model, data)
        viewer.opt.flags[mujoco.mjtVisFlag.mjVIS_CONTACTFORCE] = True
        viewer.opt.flags[mujoco.mjtVisFlag.mjVIS_TRANSPARENT] = True
        slow_rate = 10
        while viewer.is_running():
            step_start = time.time()
            mujoco.mj_step(model, data)
            viewer.sync()
            time_until_next_step = slow_rate * model.opt.timestep - (time.time() - step_start)
            if visualize and time_until_next_step > 0:
                time.sleep(time_until_next_step)
            print(data.joint('object2world').qpos.copy())

    for i in range(total_steps):
        contact_info = []
        mujoco.mj_step(model, data)
        for j, c in enumerate(data.contact):
            mujoco.mj_contactForce(model, data, j, forcetorque)
            if c.geom1 < 16 <= c.geom2:
                hpart = c.geom1
            elif c.geom2 < 16 <= c.geom1:
                hpart = c.geom2
            else:
                continue
            contact_info.append({'frame': c.frame.copy(),
                                 'hand_part_id': hpart, 'force': forcetorque[0:3].copy(), 'pos': c.pos.copy()
                                 })
        dynamic_contact_info.append(contact_info)
        obj_qpos.append(data.joint('object2world').qpos.copy())

    # plt.plot(new_trs, disps)
    # plt.xscale('log')
    # plt.show()

    return dynamic_contact_info, np.stack(obj_qpos, axis=0), opt_tr
    # nr = len(rel_ratio2)
    # ng = len(contact_geom_ids)
    # n_batches = nr ** ng
    # mujoco.mj_resetDataKeyframe(model, data, 0)
    # init_states = get_state(model, data, n_batches)
    # mjdatas = [copy.copy(data) for _ in range(20)]
    # models = []
    # disps = []
    # spec = mujoco.MjSpec.from_file(os.path.join(model_path, "hand_model.xml"))
    # for b in tqdm(range(n_batches)):
    #     for i, gid in enumerate(contact_geom_ids):
    #         model.geom(gid).solref[1] = sqrt(start_ratio2 * rel_ratio2[int(b // nr ** i) % nr])
    #     mujoco.mj_resetDataKeyframe(model, data, 0)
    #     for _ in range(total_steps):
    #         mujoco.mj_step(model, data)
    #     disps.append(np.linalg.norm(data.joint('object2world').qpos[:3]))
    #     print(disps[-1])
    # print(min(disps))
    # state, _ = rollout.rollout(models, mjdatas, init_states, nstep=total_steps)
    # print(state.shape)


def run_simulations_w_param(model, data, contact_geom_ids, solref_list, total_steps):
    disps = []
    # rel_ratio2 = [0.6, 0.8, 1, 1.2, 1.4]
    for t in solref_list:
        # for gid in contact_geom_ids:
        for gid in range(16):
            model.geom(gid).solref[1] = t
        mujoco.mj_resetDataKeyframe(model, data, 0)
        for i in range(total_steps):
            mujoco.mj_step(model, data)
        disps.append(np.linalg.norm(data.joint('object2world').qpos[:3]))
    return np.array(disps)


def get_state(model, data, nbatch=1):
  full_physics = mujoco.mjtState.mjSTATE_FULLPHYSICS
  state = np.zeros((mujoco.mj_stateSize(model, full_physics),))
  mujoco.mj_getState(model, data, state, full_physics)
  return np.tile(state, (nbatch, 1))


if __name__ == '__main__':
    # ada_cfg = edict({'use_ada_solref': True, 'coarse_search_range': (-3, 3), 'coarse_nbins': 30, 'fine_nbins': 30})
    # frame_info, qpos = run_adaptive_sim('tmp/grab/0000', ada_cfg, visualize=True)
    # print(qpos[-1])
    # print(label_info)
    # print(rollout.rollout.__doc__)
    run_sim('tmp/grab/0004', True)
    # simulator.simulate(None, None, None, None, None, None)
