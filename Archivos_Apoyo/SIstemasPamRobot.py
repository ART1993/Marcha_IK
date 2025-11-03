import numpy as np

from Archivos_Apoyo.dinamica_pam import PAMMcKibben

def Sistema_Musculos_PAM_16(control_joint_names=None):
    pam_dict = {}
    for joint in control_joint_names:
        if "hip_roll" in joint:
            pam_dict[f"{joint}_flexor"] = PAMMcKibben(L0=0.6, r0=0.035, alpha0=np.pi/4)
            pam_dict[f"{joint}_extensor"] = PAMMcKibben(L0=0.6, r0=0.032, alpha0=np.pi/4)
        elif "hip_pitch" in joint:
            pam_dict[f"{joint}_flexor"] = PAMMcKibben(L0=0.6, r0=0.030, alpha0=np.pi/4)
            pam_dict[f"{joint}_extensor"] = PAMMcKibben(L0=0.6, r0=0.035, alpha0=np.pi/4)
        elif "knee" in joint:
            pam_dict[f"{joint}_flexor"] = PAMMcKibben(L0=0.5, r0=0.036, alpha0=np.pi/4)
            pam_dict[f"{joint}_extensor"] = PAMMcKibben(L0=0.5, r0=0.036, alpha0=np.pi/4)
        elif "ankle_roll" in joint:
            pam_dict[f"{joint}_flexor"] = PAMMcKibben(L0=0.3, r0=0.035, alpha0=np.pi/4)
            pam_dict[f"{joint}_extensor"] = PAMMcKibben(L0=0.3, r0=0.035, alpha0=np.pi/4)
        elif "ankle_pitch" in joint:
            pam_dict[f"{joint}_flexor"] = PAMMcKibben(L0=0.3, r0=0.045, alpha0=np.pi/4)
            pam_dict[f"{joint}_extensor"] = PAMMcKibben(L0=0.3, r0=0.045, alpha0=np.pi/4)
    return pam_dict

def Sistema_Musculos_PAM_20(control_joint_names=None):

    pam_dict = {}
    for joint in control_joint_names:
        if "hip_roll" in joint:
            pam_dict[f"{joint}_flexor"] = PAMMcKibben(L0=0.6, r0=0.055, alpha0=np.pi/4, max_factor_pressure=6)
            pam_dict[f"{joint}_extensor"] = PAMMcKibben(L0=0.6, r0=0.055, alpha0=np.pi/4, max_factor_pressure=6)
        elif "hip_pitch" in joint:
            pam_dict[f"{joint}_flexor"] = PAMMcKibben(L0=0.6, r0=0.055, alpha0=np.pi/4, max_factor_pressure=6)
            pam_dict[f"{joint}_extensor"] = PAMMcKibben(L0=0.6, r0=0.055, alpha0=np.pi/4, max_factor_pressure=6)
        elif "knee" in joint:
            pam_dict[f"{joint}_flexor"] = PAMMcKibben(L0=0.5, r0=0.055, alpha0=np.pi/4, max_factor_pressure=6)
            pam_dict[f"{joint}_extensor"] = PAMMcKibben(L0=0.5, r0=0.055, alpha0=np.pi/4, max_factor_pressure=6)
        elif "ankle_roll" in joint:
            pam_dict[f"{joint}_flexor"] = PAMMcKibben(L0=0.3, r0=0.040, alpha0=np.pi/4, max_factor_pressure=3)
            pam_dict[f"{joint}_extensor"] = PAMMcKibben(L0=0.3, r0=0.040, alpha0=np.pi/4, max_factor_pressure=3)
        elif "ankle_pitch" in joint:
            pam_dict[f"{joint}_flexor"] = PAMMcKibben(L0=0.3, r0=0.040, alpha0=np.pi/4, max_factor_pressure=3)
            pam_dict[f"{joint}_extensor"] = PAMMcKibben(L0=0.3, r0=0.040, alpha0=np.pi/4, max_factor_pressure=3)
    return pam_dict

def Sistema_Musculos_PAM_24(control_joint_names=None):

    pam_dict = {}
    for joint in control_joint_names:
        if "hip" in joint.lower() and 'roll' in joint.lower():
            pam_dict[f"{joint}_flexor"] = PAMMcKibben(L0=0.6, r0=0.035, alpha0=np.pi/4)
            pam_dict[f"{joint}_extensor"] = PAMMcKibben(L0=0.6, r0=0.032, alpha0=np.pi/4)
        elif"hip" in joint.lower() and "pitch" in joint.lower():
            pam_dict[f"{joint}_flexor"] = PAMMcKibben(L0=0.6, r0=0.030, alpha0=np.pi/4)
            pam_dict[f"{joint}_extensor"] = PAMMcKibben(L0=0.6, r0=0.035, alpha0=np.pi/4)
        if "hip" in joint.lower() and "yaw" in joint.lower():
            pam_dict[f"{joint}_flexor"] = PAMMcKibben(L0=0.6, r0=0.035, alpha0=np.pi/4)
            pam_dict[f"{joint}_extensor"] = PAMMcKibben(L0=0.6, r0=0.032, alpha0=np.pi/4)
        elif "knee" in joint.lower():
            pam_dict[f"{joint}_flexor"] = PAMMcKibben(L0=0.5, r0=0.046, alpha0=np.pi/4)
            pam_dict[f"{joint}_extensor"] = PAMMcKibben(L0=0.5, r0=0.046, alpha0=np.pi/4)
        elif "ankle" in joint.lower() and "roll" in joint.lower():
            pam_dict[f"{joint}_flexor"] = PAMMcKibben(L0=0.3, r0=0.055, alpha0=np.pi/4)
            pam_dict[f"{joint}_extensor"] = PAMMcKibben(L0=0.3, r0=0.055, alpha0=np.pi/4)
        elif "ankle" in joint.lower() and "pitch" in joint.lower():
            pam_dict[f"{joint}_flexor"] = PAMMcKibben(L0=0.3, r0=0.055, alpha0=np.pi/4)
            pam_dict[f"{joint}_extensor"] = PAMMcKibben(L0=0.3, r0=0.055, alpha0=np.pi/4)
    return pam_dict

def Sistema_Musculos_PAM_24(control_joint_names=None):

    pam_dict = {}
    for joint in control_joint_names:
        if "hip" in joint.lower() and 'roll' in joint.lower():
            pam_dict[f"{joint}_flexor"] = PAMMcKibben(L0=0.6, r0=0.035, alpha0=np.pi/4)
            pam_dict[f"{joint}_extensor"] = PAMMcKibben(L0=0.6, r0=0.032, alpha0=np.pi/4)
        elif"hip" in joint.lower() and "pitch" in joint.lower():
            pam_dict[f"{joint}_flexor"] = PAMMcKibben(L0=0.6, r0=0.030, alpha0=np.pi/4)
            pam_dict[f"{joint}_extensor"] = PAMMcKibben(L0=0.6, r0=0.035, alpha0=np.pi/4)
        if "hip" in joint.lower() and "yaw" in joint.lower():
            pam_dict[f"{joint}_flexor"] = PAMMcKibben(L0=0.6, r0=0.035, alpha0=np.pi/4)
            pam_dict[f"{joint}_extensor"] = PAMMcKibben(L0=0.6, r0=0.032, alpha0=np.pi/4)
        elif "knee" in joint.lower():
            pam_dict[f"{joint}_flexor"] = PAMMcKibben(L0=0.5, r0=0.046, alpha0=np.pi/4)
            pam_dict[f"{joint}_extensor"] = PAMMcKibben(L0=0.5, r0=0.046, alpha0=np.pi/4)
        elif "ankle" in joint.lower() and "roll" in joint.lower():
            pam_dict[f"{joint}_flexor"] = PAMMcKibben(L0=0.3, r0=0.055, alpha0=np.pi/4)
            pam_dict[f"{joint}_extensor"] = PAMMcKibben(L0=0.3, r0=0.055, alpha0=np.pi/4)
        elif "ankle" in joint.lower() and "pitch" in joint.lower():
            pam_dict[f"{joint}_flexor"] = PAMMcKibben(L0=0.3, r0=0.055, alpha0=np.pi/4)
            pam_dict[f"{joint}_extensor"] = PAMMcKibben(L0=0.3, r0=0.055, alpha0=np.pi/4)
    return pam_dict

def Sistema_Musculos_PAM_12(control_joint_names=None):

    pam_dict = {}
    for joint in control_joint_names:
        if "hip_pitch" in joint:
            pam_dict[f"{joint}_flexor"] = PAMMcKibben(L0=0.6, r0=0.06, alpha0=np.pi/4, max_factor_pressure=6)
            pam_dict[f"{joint}_extensor"] = PAMMcKibben(L0=0.6, r0=0.06, alpha0=np.pi/4, max_factor_pressure=6)
        elif "knee" in joint:
            pam_dict[f"{joint}_flexor"] = PAMMcKibben(L0=0.5, r0=0.06, alpha0=np.pi/4, max_factor_pressure=6)
            pam_dict[f"{joint}_extensor"] = PAMMcKibben(L0=0.5, r0=0.06, alpha0=np.pi/4, max_factor_pressure=6)
        elif "ankle_pitch" in joint:
            # Ver si 35, 40, 45 para
            pam_dict[f"{joint}_flexor"] = PAMMcKibben(L0=0.3, r0=0.040, alpha0=np.pi/4, max_factor_pressure=4)
            pam_dict[f"{joint}_extensor"] = PAMMcKibben(L0=0.3, r0=0.040, alpha0=np.pi/4, max_factor_pressure=4)
    return pam_dict

def Sistema_Musculos_PAM_12_mini(control_joint_names=None):
    """
    Mini-bípedo Sport-4bar (∼0.5 m alto):
      - Hip/Knee: L0=0.38 m, r0=0.030 m
      - Ankle:    L0=0.22 m, r0=0.0215 m
      - alpha0=45°, max_factor_pressure=5 (≈ 4 bar gauge)
    """

    pam_dict = {}
    for joint in control_joint_names:
        if "hip_pitch" in joint:
            pam_dict[f"{joint}_flexor"] = PAMMcKibben(L0=0.35, r0=0.03, alpha0=np.pi/4, max_factor_pressure=5)
            pam_dict[f"{joint}_extensor"] = PAMMcKibben(L0=0.35, r0=0.03, alpha0=np.pi/4, max_factor_pressure=5)
        elif "knee" in joint:
            pam_dict[f"{joint}_flexor"] = PAMMcKibben(L0=0.35, r0=0.03, alpha0=np.pi/4, max_factor_pressure=5)
            pam_dict[f"{joint}_extensor"] = PAMMcKibben(L0=0.35, r0=0.03, alpha0=np.pi/4, max_factor_pressure=5)
        elif "ankle_pitch" in joint:
            # Ver si 35, 40, 45 para
            pam_dict[f"{joint}_flexor"] = PAMMcKibben(L0=0.2, r0=0.021, alpha0=np.pi/4, max_factor_pressure=5)
            pam_dict[f"{joint}_extensor"] = PAMMcKibben(L0=0.2, r0=0.021, alpha0=np.pi/4, max_factor_pressure=5)
    return pam_dict


def Sistema_Musculos_PAM_12_done_2(control_joint_names=None):

    pam_dict = {}
    for joint in control_joint_names:
        if "hip_pitch" in joint:
            pam_dict[f"{joint}_flexor"] = PAMMcKibben(L0=0.6, r0=0.045, alpha0=np.pi/4)
            pam_dict[f"{joint}_extensor"] = PAMMcKibben(L0=0.6, r0=0.040, alpha0=np.pi/4)
        elif "knee" in joint:
            pam_dict[f"{joint}_flexor"] = PAMMcKibben(L0=0.5, r0=0.046, alpha0=np.pi/4)
            pam_dict[f"{joint}_extensor"] = PAMMcKibben(L0=0.5, r0=0.046, alpha0=np.pi/4)
        elif "ankle_pitch" in joint:
            # Ver si 35, 40, 45 para
            pam_dict[f"{joint}_flexor"] = PAMMcKibben(L0=0.3, r0=0.056, alpha0=np.pi/4)
            pam_dict[f"{joint}_extensor"] = PAMMcKibben(L0=0.3, r0=0.056, alpha0=np.pi/4)
    return pam_dict

def Sistema_Musculos_PAM_12_done(control_joint_names=None):

    pam_dict = {}
    for joint in control_joint_names:
        if "hip_pitch" in joint:
            pam_dict[f"{joint}_flexor"] = PAMMcKibben(L0=0.6, r0=0.030, alpha0=np.pi/4)
            pam_dict[f"{joint}_extensor"] = PAMMcKibben(L0=0.6, r0=0.035, alpha0=np.pi/4)
        elif "knee" in joint:
            pam_dict[f"{joint}_flexor"] = PAMMcKibben(L0=0.5, r0=0.046, alpha0=np.pi/4)
            pam_dict[f"{joint}_extensor"] = PAMMcKibben(L0=0.5, r0=0.046, alpha0=np.pi/4)
        elif "ankle_pitch" in joint:
            # Ver si 35, 40, 45 para
            pam_dict[f"{joint}_flexor"] = PAMMcKibben(L0=0.3, r0=0.055, alpha0=np.pi/4)
            pam_dict[f"{joint}_extensor"] = PAMMcKibben(L0=0.3, r0=0.055, alpha0=np.pi/4)
    return pam_dict

