import re
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
from pathlib import Path
from typing import Dict, List, Optional, Tuple

class AngleCouplingAnalyzer:
    """
    Detecta acoplamiento de ángulos y su relación con pares (τ) usando tus CSV:
    - general_csv: columnas detalladas 'q_*', 'vel_*', 'τ_reaction_*_{x,y,z}', 'Forces_*_*'
    
    Metricas:
      * angle-angle: |corr(Δq_i, Δq_j)| y lag del pico
      * torque-angle estático: |dτ_j/dq_i| (pendiente por OLS)
      * torque-angle dinámico: pico |corr(Δq_i, τ_j)| con lag
    
    Requisitos:
      - En general_values.csv deben existir columnas 'q_<joint>_joint' y
        'τ_reaction_<joint>_joint_<axis>' (x/y/z). Elige el eje con `torque_component`.
    """

    def __init__(self,
                 general_csv: Optional[str] = None,
                 time_col: str = "t"):
        """
        torque_component: 'x'|'y'|'z' para elegir el eje del torque de reacción.
        joint_order: orden lógico de articulaciones a analizar (opcional).
        """
        pass
    
    def cargar_csv(self, path: str) -> pd.DataFrame:
        return pd.read_csv(path, header=0) if path else None

    def split_parameters_df(self, df:pd.DataFrame, startswith:str):
        """
            Separo el df general en q, v, torque y f
        """
        return df.filter(regex=f"^{startswith}")
        # self.angular_speed=self.df_general.filter(regex="^vel_")
        # self.torque=self.df_general.filter(regex="^τ_reaction_")
        # self.forces=self.df_general.filter(regex="^Forces_")
        # self.timestep_simulation=self.df_general.iloc[:,:3]

    def correlacion_entre_params(self, df:pd.DataFrame):
        return df.corr(numeric_only=True)
        
    def creacion_correlaciones(self, correlaciones, pamar_name, annot=True,
                               cell_w=0.45, cell_h=0.45,   # tamaño por celda (en pulgadas)
                                min_w=6, min_h=6,          # tamaño mínimo
                                max_w=20, max_h=20,        # tamaño máximo
                                ticksize=8, titlesize=12,  # tamaños de fuente
                                rotate_x=90, rotate_y=0,   # rotación de etiquetas
                                dpi=150):
        n_rows, n_cols = correlaciones.shape
        # tamaño basado en nº de celdas, acotado
        W = max(min_w, min(max_w, cell_w * n_cols))
        H = max(min_h, min(max_h, cell_h * n_rows))
        fig, ax = plt.subplots(figsize=(W, H), dpi=dpi)
        sns.heatmap(
            correlaciones,
            vmin=-1, vmax=1, center=0,
            cmap="vlag",        # puedes usar "coolwarm", "RdBu_r", etc.
            square=True,
            annot=annot,
            linewidths=0.3,
            cbar_kws={"shrink": 0.8},
            ax=ax
        )
        # ax.set_title(f"Matriz de correlación {pamar_name} (Pearson)")
        # plt.tight_layout()
        # plt.savefig(f"heatmap_{pamar_name}.png")
        # plt.show()
        # plt.close()
        ax.set_title(f"Matriz de correlación {pamar_name} (Pearson)", fontsize=titlesize)
        ax.tick_params(axis='x', labelsize=ticksize, rotation=rotate_x)
        ax.tick_params(axis='y', labelsize=ticksize, rotation=rotate_y)
        fig.tight_layout()

        if pamar_name:
            fig.savefig(f"heatmap_{pamar_name}.png", bbox_inches="tight", dpi=dpi)
        plt.show()
        plt.close(fig)