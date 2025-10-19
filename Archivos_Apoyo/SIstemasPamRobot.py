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
            pam_dict[f"{joint}_flexor"] = PAMMcKibben(L0=0.6, r0=0.035, alpha0=np.pi/4)
            pam_dict[f"{joint}_extensor"] = PAMMcKibben(L0=0.6, r0=0.032, alpha0=np.pi/4)
        elif "hip_pitch" in joint:
            pam_dict[f"{joint}_flexor"] = PAMMcKibben(L0=0.6, r0=0.030, alpha0=np.pi/4)
            pam_dict[f"{joint}_extensor"] = PAMMcKibben(L0=0.6, r0=0.035, alpha0=np.pi/4)
        elif "knee" in joint:
            pam_dict[f"{joint}_flexor"] = PAMMcKibben(L0=0.5, r0=0.036, alpha0=np.pi/4)
            pam_dict[f"{joint}_extensor"] = PAMMcKibben(L0=0.5, r0=0.036, alpha0=np.pi/4)
        elif "ankle_roll" in joint:
            pam_dict[f"{joint}_flexor"] = PAMMcKibben(L0=0.3, r0=0.055, alpha0=np.pi/4)
            pam_dict[f"{joint}_extensor"] = PAMMcKibben(L0=0.3, r0=0.055, alpha0=np.pi/4)
        elif "ankle_pitch" in joint:
            pam_dict[f"{joint}_flexor"] = PAMMcKibben(L0=0.3, r0=0.055, alpha0=np.pi/4)
            pam_dict[f"{joint}_extensor"] = PAMMcKibben(L0=0.3, r0=0.055, alpha0=np.pi/4)
    return pam_dict

