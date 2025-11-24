import numpy as np

from Archivos_Apoyo.dinamica_pam import PAMMcKibben

def Sistema_Musculos_PAM_16(control_joint_names=None, max_pressure=5):
    pam_dict = {}
    for joint in control_joint_names:
        if "hip_roll" in joint:
            pam_dict[f"{joint}_flexor"] = PAMMcKibben(L0=0.6, r0=0.035, alpha0=np.pi/4,max_factor_pressure=max_pressure)
            pam_dict[f"{joint}_extensor"] = PAMMcKibben(L0=0.6, r0=0.032, alpha0=np.pi/4,max_factor_pressure=max_pressure)
        elif "hip_pitch" in joint:
            pam_dict[f"{joint}_flexor"] = PAMMcKibben(L0=0.6, r0=0.030, alpha0=np.pi/4,max_factor_pressure=max_pressure)
            pam_dict[f"{joint}_extensor"] = PAMMcKibben(L0=0.6, r0=0.035, alpha0=np.pi/4,max_factor_pressure=max_pressure)
        elif "knee" in joint:
            pam_dict[f"{joint}_flexor"] = PAMMcKibben(L0=0.5, r0=0.036, alpha0=np.pi/4,max_factor_pressure=max_pressure)
            pam_dict[f"{joint}_extensor"] = PAMMcKibben(L0=0.5, r0=0.036, alpha0=np.pi/4,max_factor_pressure=max_pressure)
        elif "ankle_pitch" in joint:
            pam_dict[f"{joint}_flexor"] = PAMMcKibben(L0=0.3, r0=0.045, alpha0=np.pi/4,max_factor_pressure=max_pressure)
            pam_dict[f"{joint}_extensor"] = PAMMcKibben(L0=0.3, r0=0.045, alpha0=np.pi/4,max_factor_pressure=max_pressure)
    return pam_dict

def Sistema_Musculos_blackbird(control_joint_names=None, max_pressure=5):
    pam_dict = {}
    for joint in control_joint_names:
        if "_ab_ad" in joint:
            pam_dict[f"{joint}_flexor"] = PAMMcKibben(L0=0.4, r0=0.035, alpha0=np.pi/4,max_factor_pressure=max_pressure)
            pam_dict[f"{joint}_extensor"] = PAMMcKibben(L0=0.4, r0=0.032, alpha0=np.pi/4,max_factor_pressure=max_pressure)
        elif "_yaw" in joint:
            pam_dict[f"{joint}_flexor"] = PAMMcKibben(L0=0.4, r0=0.030, alpha0=np.pi/4,max_factor_pressure=max_pressure)
            pam_dict[f"{joint}_extensor"] = PAMMcKibben(L0=0.4, r0=0.035, alpha0=np.pi/4,max_factor_pressure=max_pressure)
        elif "_upper" in joint:
            pam_dict[f"{joint}_flexor"] = PAMMcKibben(L0=0.4, r0=0.036, alpha0=np.pi/4,max_factor_pressure=max_pressure)
            pam_dict[f"{joint}_extensor"] = PAMMcKibben(L0=0.4, r0=0.036, alpha0=np.pi/4,max_factor_pressure=max_pressure)
        elif "_lower" in joint:
            pam_dict[f"{joint}_flexor"] = PAMMcKibben(L0=0.3, r0=0.035, alpha0=np.pi/4,max_factor_pressure=max_pressure)
            pam_dict[f"{joint}_extensor"] = PAMMcKibben(L0=0.3, r0=0.035, alpha0=np.pi/4,max_factor_pressure=max_pressure)
        elif "_foot" in joint:
            pam_dict[f"{joint}_flexor"] = PAMMcKibben(L0=0.2, r0=0.035, alpha0=np.pi/4,max_factor_pressure=max_pressure)
            pam_dict[f"{joint}_extensor"] = PAMMcKibben(L0=0.2, r0=0.035, alpha0=np.pi/4,max_factor_pressure=max_pressure)
    return pam_dict

def Sistema_Musculos_PAM_20(control_joint_names=None, max_pressure=5):

    pam_dict = {}
    for joint in control_joint_names:
        if "hip_roll" in joint:
            pam_dict[f"{joint}_flexor"] = PAMMcKibben(L0=0.6, r0=0.055, alpha0=np.pi/4, max_factor_pressure=max_pressure)
            pam_dict[f"{joint}_extensor"] = PAMMcKibben(L0=0.6, r0=0.055, alpha0=np.pi/4, max_factor_pressure=max_pressure)
        elif "hip_pitch" in joint:
            pam_dict[f"{joint}_flexor"] = PAMMcKibben(L0=0.6, r0=0.055, alpha0=np.pi/4, max_factor_pressure=max_pressure)
            pam_dict[f"{joint}_extensor"] = PAMMcKibben(L0=0.6, r0=0.055, alpha0=np.pi/4, max_factor_pressure=max_pressure)
        elif "knee" in joint:
            pam_dict[f"{joint}_flexor"] = PAMMcKibben(L0=0.5, r0=0.055, alpha0=np.pi/4, max_factor_pressure=max_pressure)
            pam_dict[f"{joint}_extensor"] = PAMMcKibben(L0=0.5, r0=0.055, alpha0=np.pi/4, max_factor_pressure=max_pressure)
        elif "ankle_roll" in joint:
            pam_dict[f"{joint}_flexor"] = PAMMcKibben(L0=0.3, r0=0.040, alpha0=np.pi/4, max_factor_pressure=max_pressure)
            pam_dict[f"{joint}_extensor"] = PAMMcKibben(L0=0.3, r0=0.040, alpha0=np.pi/4, max_factor_pressure=max_pressure)
        elif "ankle_pitch" in joint:
            pam_dict[f"{joint}_flexor"] = PAMMcKibben(L0=0.3, r0=0.040, alpha0=np.pi/4, max_factor_pressure=max_pressure)
            pam_dict[f"{joint}_extensor"] = PAMMcKibben(L0=0.3, r0=0.040, alpha0=np.pi/4, max_factor_pressure=max_pressure)
    return pam_dict

def Sistema_Musculos_PAM_24(control_joint_names=None, max_pressure=5):

    pam_dict = {}
    for joint in control_joint_names:
        if "hip" in joint.lower() and 'roll' in joint.lower():
            pam_dict[f"{joint}_flexor"] = PAMMcKibben(L0=0.6, r0=0.035, alpha0=np.pi/4, max_factor_pressure=max_pressure)
            pam_dict[f"{joint}_extensor"] = PAMMcKibben(L0=0.6, r0=0.032, alpha0=np.pi/4, max_factor_pressure=max_pressure)
        elif"hip" in joint.lower() and "pitch" in joint.lower():
            pam_dict[f"{joint}_flexor"] = PAMMcKibben(L0=0.6, r0=0.030, alpha0=np.pi/4, max_factor_pressure=max_pressure)
            pam_dict[f"{joint}_extensor"] = PAMMcKibben(L0=0.6, r0=0.035, alpha0=np.pi/4, max_factor_pressure=max_pressure)
        if "hip" in joint.lower() and "yaw" in joint.lower():
            pam_dict[f"{joint}_flexor"] = PAMMcKibben(L0=0.6, r0=0.035, alpha0=np.pi/4, max_factor_pressure=max_pressure)
            pam_dict[f"{joint}_extensor"] = PAMMcKibben(L0=0.6, r0=0.032, alpha0=np.pi/4, max_factor_pressure=max_pressure)
        elif "knee" in joint.lower():
            pam_dict[f"{joint}_flexor"] = PAMMcKibben(L0=0.5, r0=0.046, alpha0=np.pi/4, max_factor_pressure=max_pressure)
            pam_dict[f"{joint}_extensor"] = PAMMcKibben(L0=0.5, r0=0.046, alpha0=np.pi/4, max_factor_pressure=max_pressure)
        elif "ankle" in joint.lower() and "roll" in joint.lower():
            pam_dict[f"{joint}_flexor"] = PAMMcKibben(L0=0.3, r0=0.055, alpha0=np.pi/4, max_factor_pressure=max_pressure)
            pam_dict[f"{joint}_extensor"] = PAMMcKibben(L0=0.3, r0=0.055, alpha0=np.pi/4, max_factor_pressure=max_pressure)
        elif "ankle" in joint.lower() and "pitch" in joint.lower():
            pam_dict[f"{joint}_flexor"] = PAMMcKibben(L0=0.3, r0=0.055, alpha0=np.pi/4, max_factor_pressure=max_pressure)
            pam_dict[f"{joint}_extensor"] = PAMMcKibben(L0=0.3, r0=0.055, alpha0=np.pi/4, max_factor_pressure=max_pressure)
    return pam_dict

def Sistema_Musculos_PAM_12(control_joint_names=None, max_pressure=5):

    pam_dict = {}
    for joint in control_joint_names:
        if "hip_pitch" in joint:
            pam_dict[f"{joint}_flexor"] = PAMMcKibben(L0=0.4, r0=0.04, alpha0=np.pi/4, max_factor_pressure=max_pressure)
            pam_dict[f"{joint}_extensor"] = PAMMcKibben(L0=0.4, r0=0.04, alpha0=np.pi/4, max_factor_pressure=max_pressure)
        elif "knee" in joint:
            pam_dict[f"{joint}_flexor"] = PAMMcKibben(L0=0.4, r0=0.04, alpha0=np.pi/4, max_factor_pressure=max_pressure)
            pam_dict[f"{joint}_extensor"] = PAMMcKibben(L0=0.4, r0=0.04, alpha0=np.pi/4, max_factor_pressure=max_pressure)
        elif "ankle_pitch" in joint:
            # Ver si 35, 40, 45 para
            pam_dict[f"{joint}_flexor"] = PAMMcKibben(L0=0.2, r0=0.030, alpha0=np.pi/4, max_factor_pressure=max_pressure)
            pam_dict[f"{joint}_extensor"] = PAMMcKibben(L0=0.2, r0=0.030, alpha0=np.pi/4, max_factor_pressure=max_pressure)
    return pam_dict

# ultima vez usado: 7/11/25 
def Sistema_Musculos_PAM_12_alto_valor(control_joint_names=None):

    pam_dict = {}
    for joint in control_joint_names:
        if "hip_pitch" in joint:
            pam_dict[f"{joint}_flexor"] = PAMMcKibben(L0=0.4, r0=0.055, alpha0=np.pi/4)
            pam_dict[f"{joint}_extensor"] = PAMMcKibben(L0=0.4, r0=0.055, alpha0=np.pi/4)
        elif "knee" in joint:
            pam_dict[f"{joint}_flexor"] = PAMMcKibben(L0=0.4, r0=0.055, alpha0=np.pi/4)
            pam_dict[f"{joint}_extensor"] = PAMMcKibben(L0=0.4, r0=0.055, alpha0=np.pi/4)
        elif "ankle_pitch" in joint:
            # Ver si 35, 40, 45 para
            pam_dict[f"{joint}_flexor"] = PAMMcKibben(L0=0.2, r0=0.040, alpha0=np.pi/4)
            pam_dict[f"{joint}_extensor"] = PAMMcKibben(L0=0.2, r0=0.040, alpha0=np.pi/4)
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

