# -*- coding: utf-8 -*-
"""
thesis_style.py — un solo sistema gráfico para toda la tesis.

Regla semántica, heredada de la Figura 7 del Experimento 1b y mantenida en 2b:

    AZUL  #2E86C1  = la ruta espacial      (bloque espacial; posición física del dígito)
    ROJO  #E74C3C  = la ruta simbólica     (bloque simbólico; posición ordinal implicada)
    GRIS  #555555  = cantidades derivadas  (diferencias, nulos, generalización cruzada)

El color codifica la CONDICIÓN. Cuando la misma condición se mide de dos maneras
—por ejemplo EEG y mirada— se mantiene el mismo color y se cambia el ESTILO DE LÍNEA:
continua para la medida principal, discontinua para la segunda.

Así ninguna figura del capítulo introduce un color nuevo, y el lector puede leer el
color igual en los cinco experimentos.

El par azul/rojo está validado para visión con deficiencia al rojo (ΔE 19,4 en
protanopia, 30,5 en visión normal), que es la razón por la que se conserva.

Uso:
    from eeg_toolkit.thesis_style import apply_style, C
    apply_style()
    ax.plot(t, y, color=C.ESPACIAL)
"""
from __future__ import annotations
import os

__all__ = ["apply_style", "C", "STYLE_PATH"]

STYLE_PATH = os.path.join(os.path.dirname(os.path.dirname(os.path.abspath(__file__))),
                          "configs", "figure_style.mplstyle")


class C:
    """Paleta de la tesis. No añadir colores sin una razón semántica nueva."""
    ESPACIAL   = "#2E86C1"   # ruta espacial
    SIMBOLICO  = "#E74C3C"   # ruta simbólica
    DERIVADO   = "#555555"   # diferencias, nulos, generalización
    EJE        = "#333333"   # ejes, líneas de cero y de evento
    SOMBRA     = "0.93"      # banda del estímulo en pantalla
    # versiones claras, sólo para relleno de error cuando hay dos series del mismo color
    ESPACIAL_L = "#A9CCE3"
    SIMBOLICO_L= "#F5B7B1"


def apply_style(style_path: str | None = None, fallback: bool = True):
    """
    Aplica el estilo APA del artículo. Si no encuentra el archivo y `fallback`,
    aplica una versión equivalente definida aquí, para que las figuras salgan
    iguales en cualquier máquina.
    """
    import matplotlib.pyplot as plt

    path = style_path or STYLE_PATH
    if os.path.exists(path):
        plt.style.use(path)
        return path
    if not fallback:
        raise FileNotFoundError(path)
    plt.rcParams.update({
        "font.family": "sans-serif",
        "font.sans-serif": ["Arial", "Helvetica", "DejaVu Sans"],
        "font.size": 10, "axes.titlesize": 12, "axes.labelsize": 10,
        "axes.linewidth": 0.8, "axes.spines.top": True, "axes.spines.right": True,
        "xtick.labelsize": 9, "ytick.labelsize": 9,
        "xtick.major.width": 0.8, "ytick.major.width": 0.8,
        "xtick.major.size": 4, "ytick.major.size": 4,
        "xtick.direction": "in", "ytick.direction": "in",
        "legend.fontsize": 9, "legend.framealpha": 0.85,
        "lines.linewidth": 1.5,
        "figure.dpi": 150, "savefig.dpi": 600,
        "savefig.bbox": "tight", "savefig.pad_inches": 0.05,
    })
    return "(fallback interno)"
