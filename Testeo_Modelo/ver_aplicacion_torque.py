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
    dict(name="left_knee_flexor",       A=("left_upper_leg",   [ 0.00,  0.02, -0.20]),
                                       B=("left_lower_leg",    [ 0.00, -0.02,  0.20])),
    dict(name="left_knee_extensor",     A=("left_upper_leg",   [ 0.00, -0.02, -0.20]),
                                       B=("left_lower_leg",    [ 0.00,  0.02,  0.20])),
    dict(name="left_anckle_flexor",       A=("left_upper_leg",   [ 0.00,  0.02, -0.20]),
                                       B=("left_lower_leg",    [ 0.00, -0.02,  0.20])),
    dict(name="left_anckle_extensor",     A=("left_upper_leg",   [ 0.00, -0.02, -0.20]),
                                       B=("left_lower_leg",    [ 0.00,  0.02,  0.20])),
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

def calc_tau_from_actuator(robot, actuator, q, force_mag=1.0):
    """
    Calcula vector de pares articulares τ (size = num_joints) que produce el PAM
    aplicando una tracción ideal de magnitud 'force_mag' a lo largo de A->B.
    """
    num_j = p.getNumJoints(robot)
    q = np.asarray(q, dtype=float)

    # Parse anclajes
    linkA_name, posA_loc = actuator["A"]
    linkB_name, posB_loc = actuator["B"]
    linkA = name_to_link(robot, linkA_name)
    linkB = name_to_link(robot, linkB_name)

    # Estados (q,qdot,qddot) para Jacobiano
    qdot = np.zeros(num_j)
    qdd = np.zeros(num_j)

    # Jacobianos en A y B
    # Nota: calculateJacobian requiere pos en coords locales del link
    Jlin_A, Jang_A = p.calculateJacobian(robot, linkA, posA_loc, q.tolist(), qdot.tolist(), qdd.tolist())[:2]
    Jlin_B, Jang_B = p.calculateJacobian(robot, linkB, posB_loc, q.tolist(), qdot.tolist(), qdd.tolist())[:2]
    Jlin_A = np.array(Jlin_A)  # 3 x n
    Jlin_B = np.array(Jlin_B)

    # Dirección de tracción (A->B) en MUNDO
    if linkA == -1:
        wposA, _, _, _, _, _ = p.getBasePositionAndOrientation(robot)
        R_A = np.eye(3)
    else:
        lsA = p.getLinkState(robot, linkA, computeForwardKinematics=True)
        wposA = np.array(lsA[0])
    if linkB == -1:
        wposB, _, _, _, _, _ = p.getBasePositionAndOrientation(robot)
    else:
        lsB = p.getLinkState(robot, linkB, computeForwardKinematics=True)
        wposB = np.array(lsB[0])

    d = (wposB - wposA)
    L = np.linalg.norm(d) + 1e-9
    u = d / L  # unit vector world A->B

    # Fuerzas en mundo (tensión)
    F_A =  force_mag * u          # aplicada en A hacia B
    F_B = -force_mag * u          # igual y opuesta en B

    # τ = J^T F (solo jacobiano lineal, torque externo 0)
    tau_A = Jlin_A.T @ F_A        # (n,)
    tau_B = Jlin_B.T @ F_B
    tau = tau_A + tau_B           # (n,)

    return tau  # pares articulares por índice de joint

def main():
    # ===== CLI args para modo ligero/headless =====
    parser = argparse.ArgumentParser(description="Probe de torque por actuador (PAM)")
    parser.add_argument("--headless", action="store_true",
                        help="Ejecuta una sola evaluación (sin sliders) y sale")
    parser.add_argument("--actuator-index", type=int, default=0,
                        help="Índice del actuador en ACTUATORS")
    parser.add_argument("--force", type=float, default=100.0,
                        help="Fuerza tensil en Newtons (A->B)")
    parser.add_argument("--joint", type=str, default=None,
                        help="Si se indica, imprime solo el torque de esta junta (p.ej. left_knee_pitch)")
    args = parser.parse_args()
    cid = p.connect(p.GUI)
    p.setAdditionalSearchPath(pybullet_data.getDataPath())
    p.resetSimulation()
    p.setGravity(0, 0, 0)
    p.setTimeStep(1.0 / 240.0)
    p.setRealTimeSimulation(0)
    p.loadURDF("plane.urdf", [0,0,0])

    urdf_path = find_urdf()
    start_pos = [0, 0, 1.5]
    start_orn = p.getQuaternionFromEuler([0, 0, 0])
    robot = p.loadURDF(urdf_path, start_pos, start_orn, useFixedBase=True,
                       flags=p.URDF_USE_SELF_COLLISION)

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

        # Calcula τ
        tau = calc_tau_from_actuator(robot, actuator, q, force_mag=Fmag)

        # Imprime resumen con signo para “pitch” (asumimos eje Y)
        os.system('cls' if os.name == 'nt' else 'clear')
        print(f"Actuador: {actuator['name']} | Fuerza = {Fmag:.1f} N (A->B)")
        print("Joint\t\t tau [N·m]\t sentido_pitch(+flex/-ext)")
        for j, name in enumerate(joint_list):
            sign_pitch = np.sign(tau[j])  # si eje es Y y convención + = flexión
            print(f"{name:24s} {tau[j]: .3f}\t\t {'+' if sign_pitch>0 else '-' if sign_pitch<0 else '0'}")

        p.stepSimulation()
        time.sleep(1.0/240.0)

    p.disconnect()

if __name__ == "__main__":
    main()