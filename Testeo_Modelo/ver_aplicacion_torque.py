# muscle_torque_probe.py
import json, os, time
import sys
import argparse
import numpy as np
import pybullet as p
import pybullet_data

ROBOT_NAME = "2_legged_human_like_robot12DOF"

CANDIDATE_PATHS = [
    f"Robots_Versiones/{ROBOT_NAME}.urdf",
    os.path.abspath(f"/mnt/data/{ROBOT_NAME}.urdf"),
    os.path.abspath(f"Robots_Versiones/cassie_description/urdf/{ROBOT_NAME}.urdf"),
]

def joint_index_by_name(robot, joint_name):
    """Devuelve el índice del joint por nombre."""
    for j in range(p.getNumJoints(robot)):
        if p.getJointInfo(robot, j)[1].decode("utf-8") == joint_name:
            return j
    raise KeyError(f"Joint '{joint_name}' no encontrado")

def find_urdf():
    for path in CANDIDATE_PATHS:
        if os.path.exists(path):
            return path
    raise FileNotFoundError("No encuentro el URDF. Ajusta CANDIDATE_PATHS.")

def tau_of_joint(robot, actuator, q, force_mag, joint_name):
    """Conveniencia: torque escalar en una junta concreta."""
    tau = calc_tau_from_actuator(robot, actuator, q, force_mag)
    j = joint_index_by_name(robot, joint_name)
    return float(tau[j])

# ======== DEFINE TUS MÚSCULOS (PAMs) ========
# Cada PAM: nombre, anclaje A (link_a, pos_local_a), anclaje B (link_b, pos_local_b)
# pos_local_* en coordenadas del link (metros).
# EJEMPLOS PLACEHOLDER: cámbialos por tus anclajes reales

ACTUATORS = [
    dict(name="left_hip_pitch_flexor",  A=("base_link", [ 0.05,  0.03, 0.02]),
                                       B=("left_hip_pitch_link",    [-0.02,  0.02, -0.18])),
    dict(name="left_hip_pitch_extensor",A=("base_link", [-0.05,  0.02, 0.02]),
                                       B=("left_hip_pitch_link",    [ 0.02, -0.02, -0.18])),
    dict(name="left_knee_flexor",       A=("left_thigh_link",   [ 0.00,  0.02, -0.20]),
                                       B=("left_shin_link",    [ 0.00, -0.02,  0.20])),
    dict(name="left_knee_extensor",     A=("left_thigh_link",   [ 0.00, -0.02, -0.20]),
                                       B=("left_shin_link",    [ 0.00,  0.02,  0.20])),
    dict(name="left_anckle_flexor",       A=("left_shin_link",   [ 0.00,  0.02, -0.20]),
                                       B=("left_ankle_pitch_link",    [ 0.00, -0.02,  0.20])),
    dict(name="left_anckle_extensor",     A=("left_shin_link",   [ 0.00, -0.02, -0.20]),
                                       B=("left_ankle_pitch_link",    [ 0.00,  0.02,  0.20])),
    dict(name="right_hip_pitch_flexor",  A=("base_link", [ 0.05,  0.03, 0.02]),
                                       B=("right_hip_pitch_link",    [-0.02,  0.02, -0.18])),
    dict(name="right_hip_pitch_extensor",A=("base_link", [-0.05,  0.02, 0.02]),
                                       B=("right_hip_pitch_link",    [ 0.02, -0.02, -0.18])),
    dict(name="right_knee_flexor",       A=("right_thigh_link",   [ 0.00,  0.02, -0.20]),
                                       B=("right_shin_link",    [ 0.00, -0.02,  0.20])),
    dict(name="right_knee_extensor",     A=("right_thigh_link",   [ 0.00, -0.02, -0.20]),
                                       B=("right_shin_link",    [ 0.00,  0.02,  0.20])),
    dict(name="right_anckle_flexor",       A=("right_shin_link",   [ 0.00,  0.02, -0.20]),
                                       B=("right_ankle_pitch_link",    [ 0.00, -0.02,  0.20])),
    dict(name="right_anckle_extensor",     A=("right_shin_link",   [ 0.00, -0.02, -0.20]),
                                       B=("right_ankle_pitch_link",    [ 0.00,  0.02,  0.20])),
    # ... añade tobillo y lado derecho
]

# Ejes canónicos por tipo de articulación en tu URDF (ajusta si tu eje no es Y)
# Para "pitch" normalmente el eje es Y. Usamos esto para etiquetar flexor/extensor por signo.
REV_AXIS_WORLD = np.array([0.0, 1.0, 0.0])  # eje Y



def name_to_link(robot, link_name):
    for j in range(p.getNumJoints(robot)):
        info = p.getJointInfo(robot, j)
        if info[12].decode("utf-8") == link_name:  # linkName
            return j
    # base link:
    if link_name in ("base", "pelvis", "torso", "base_link"):
        return -1
    raise KeyError(f"No encuentro el link '{link_name}'")

def joint_names(robot):
    names = []
    for j in range(p.getNumJoints(robot)):
        names.append(p.getJointInfo(robot, j)[1].decode("utf-8"))
    return names

def calc_tau_from_actuator(robot, actuator, q_all, active_joints, dof2j, 
                           joints_to_dof, dof_to_joints, force_mag=1.0):
    """
    τ por joint index, usando SOLO DoF activos en calculateJacobian.
    Si un anclaje está en la base (-1), su contribución de Jacobiano se toma como 0.
    """
    num_j = p.getNumJoints(robot)

    # --- Parse anclajes
    linkA_name, posA_loc = actuator["A"]
    linkB_name, posB_loc = actuator["B"]
    linkA = name_to_link(robot, linkA_name)
    linkB = name_to_link(robot, linkB_name)

    # Sanitizar local positions (size 3)
    posA_loc = list(posA_loc[:3])
    posB_loc = list(posB_loc[:3])

    # --- Estados para jacobiano (solo DoF activos)
    q_dof   = joints_to_dof(q_all, active_joints)
    qdot_d  = [0.0]*len(q_dof)
    qddot_d = [0.0]*len(q_dof)

    # --- Posiciones mundo de A y B
    if linkA == -1:
        wposA, _ = p.getBasePositionAndOrientation(robot)
        wposA = np.array(wposA)
    else:
        lsA = p.getLinkState(robot, linkA, computeForwardKinematics=True)
        wposA = np.array(lsA[0])
    if linkB == -1:
        wposB, _ = p.getBasePositionAndOrientation(robot)
        wposB = np.array(wposB)
    else:
        lsB = p.getLinkState(robot, linkB, computeForwardKinematics=True)
        wposB = np.array(lsB[0])

    d = (wposB - wposA)
    L = np.linalg.norm(d) + 1e-9
    u = d / L  # dirección A->B en mundo

    F_A =  force_mag * u
    F_B = -force_mag * u

    # --- Jacobianos lineales (3 x nDoF). Si link = base, usa 0.
    if linkA == -1:
        Jlin_A = np.zeros((3, len(q_dof)))
    else:
        Jlin_A = np.array(p.calculateJacobian(robot, linkA, posA_loc, q_dof, qdot_d, qddot_d)[0])
    if linkB == -1:
        Jlin_B = np.zeros((3, len(q_dof)))
    else:
        Jlin_B = np.array(p.calculateJacobian(robot, linkB, posB_loc, q_dof, qdot_d, qddot_d)[0])

    tau_dof = Jlin_A.T @ F_A + Jlin_B.T @ F_B  # (nDoF,)

    # Expandir a todos los joints (0 en FIXED)
    tau_full = dof_to_joints(tau_dof, num_j, dof2j)
    return tau_full

def main():
    # ===== CLI args para modo ligero/headless =====
    cid = p.connect(p.GUI)
    p.setAdditionalSearchPath(pybullet_data.getDataPath())
    p.resetSimulation()
    p.setGravity(0, 0, 0)
    p.setTimeStep(1.0 / 240.0)
    p.setRealTimeSimulation(0)
    plane_id=p.loadURDF("plane.urdf", [0,0,0])
    urdf_path = find_urdf()
    start_pos = [0, 0, 1.5]
    start_orn = p.getQuaternionFromEuler([0, 0, 0])
    robot = p.loadURDF(urdf_path, start_pos, start_orn, useFixedBase=True,
                       flags=p.URDF_USE_SELF_COLLISION)

    num_j = p.getNumJoints(robot)
    # Solo cuentan REVOLUTE/PRISMATIC/SPHERICAL (no FIXED)
    active_joints = [j for j in range(num_j) if p.getJointInfo(robot, j)[2] != p.JOINT_FIXED]
    # Mapas de índice DoF -> joint_index y viceversa
    dof2j = dict(enumerate(active_joints))
    j2dof = {j: i for i, j in dof2j.items()}

    def joints_to_dof(q_all, active_joints):
        """Convierte lista q por 'joint index' -> vector q_dof (solo activos)."""
        return [q_all[j] for j in active_joints]

    def dof_to_joints(vec_dof, size_all, dof2j):
        """Coloca vec_dof en un vector tamaño 'size_all', ceros en FIXED."""
        full = np.zeros(size_all, dtype=float)
        for i, j in dof2j.items():
            full[j] = vec_dof[i]
        return full

    

    # Desactiva motores
    for j in range(p.getNumJoints(robot)):
        p.setJointMotorControl2(robot, j, p.VELOCITY_CONTROL, force=0)

    # Sliders de articulaciones
    sliders_q = []
    for j in range(p.getNumJoints(robot)):
        info = p.getJointInfo(robot, j)
        name = info[1].decode("utf-8")
        lo, hi = float(info[8]), float(info[9])
        if lo < -3.0 or lo == -1e10: lo = -1.5
        if hi >  3.0 or hi ==  1e10: hi =  1.5
        sliders_q.append((j, name, p.addUserDebugParameter(name, lo, hi, 0.0)))

    # Selector de músculo y magnitud de fuerza
    musc_idx_slider = p.addUserDebugParameter("actuator_index", 0, max(0, len(ACTUATORS)-1), 0)
    force_slider    = p.addUserDebugParameter("force_mag(N)", 0.0, 1000.0, 100.0)

    joint_list = joint_names(robot)

    print("Mueve ángulos, elige 'actuator_index' y 'force_mag'. Mira consola para τ.")
    while p.isConnected():
        # Lee pose
        q = []
        for j, _, sid in sliders_q:
            qj = p.readUserDebugParameter(sid)
            q.append(qj)
            p.resetJointState(robot, j, qj)

        # Lee selección
        i_act = int(round(p.readUserDebugParameter(musc_idx_slider)))
        i_act = np.clip(i_act, 0, len(ACTUATORS)-1)
        Fmag = float(p.readUserDebugParameter(force_slider))
        actuator = ACTUATORS[i_act]
        # calc_tau_from_actuator(robot, actuator, q_all, Fmag, active_joints, dof2j)
        # Calcula τ
        tau = calc_tau_from_actuator(robot, actuator, q, active_joints, dof2j, joints_to_dof, dof_to_joints, force_mag=Fmag)

        # Imprime resumen con signo para “pitch” (asumimos eje Y)
        os.system('cls' if os.name == 'nt' else 'clear')
        if Fmag!=0:
            print(f"Actuador: {actuator['name']} | Fuerza = {Fmag:.1f} N (A->B)")
            print("Joint\t\t tau [N·m]\t sentido_pitch(flex/-ext)")
            for j, name in enumerate(joint_list):
                sign_pitch = np.sign(tau[j])  # si eje es Y y convención  = flexión
                print(f"{name:24s} {tau[j]: .3f}\t\t {'' if sign_pitch>0 else '-' if sign_pitch<0 else '0'}")

        p.stepSimulation()
        time.sleep(1.0/240.0)

    p.disconnect()

if __name__ == "__main__":
    main()