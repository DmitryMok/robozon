"""Визуальный чекпойнт геометрии рига «3 камеры + зеркало» (Фаза 1,
[[Рассчитать геометрию 3-камерного рига и зеркала для CV (CV-пайплайн Webots)]]).

2D-схема в плоскости Y-Z: точки камер, лучи азимутов к объекту, кромки ленты, зеркало
(нижний край + наклон 23° от вертикали, физический макет пользователя, НЕ подбор),
отражённый луч методом "мнимого изображения" (точка пересечения с зеркалом = "пятно"),
подписанные углы/дистанции. Подписи — на английском (кириллица не рендерится в
headless PyQt6/offscreen).

Usage:
    python mirror_rig_viewer.py                  # интерактивное окно (PyQt6)
    python mirror_rig_viewer.py --save out.png    # headless-рендер в PNG
"""
from __future__ import annotations

import argparse
import os
import sys


def build_plot():
    import pyqtgraph as pg

    from mirror_rig_geometry import (
        BELT_HALF_WIDTH_MM,
        BOTTOM_EDGE,
        CAMERAS,
        OBJECT,
        REFLECTED,
        cam_pos,
        dist,
        dot,
        mirror_line_direction,
        mirror_normal,
        phi_from_direction,
        reflect_point,
        true_reflected_azimuth_and_distance,
    )

    pg.setConfigOption("background", "w")
    pg.setConfigOption("foreground", "k")

    d = mirror_line_direction()
    phi = phi_from_direction(d)
    n = mirror_normal(phi)
    t = dot(BOTTOM_EDGE, n)

    plot = pg.PlotWidget(title="3-camera + mirror rig — Y-Z cross-section (mm)")
    plot.setAspectLocked(True)
    plot.showGrid(x=True, y=True, alpha=0.3)
    plot.setLabel("bottom", "Y, mm")
    plot.setLabel("left", "Z, mm")
    plot.resize(1100, 950)

    from PyQt6.QtGui import QFont

    label_font = QFont("Arial")
    label_font.setPointSize(9)

    def add_text(x, y, text, color="k", anchor=(0, 1)):
        item = pg.TextItem(text, color=color, anchor=anchor)
        item.setFont(label_font)
        item.setPos(x, y)
        plot.addItem(item)

    # лента: кромки на Y=±250мм, на уровне Z=0, показать отрезком под object
    plot.plot([-BELT_HALF_WIDTH_MM, BELT_HALF_WIDTH_MM], [0, 0],
              pen=pg.mkPen("gray", width=6))
    plot.plot([-BELT_HALF_WIDTH_MM, -BELT_HALF_WIDTH_MM], [-80, 80], pen=pg.mkPen("gray", width=2))
    plot.plot([BELT_HALF_WIDTH_MM, BELT_HALF_WIDTH_MM], [-80, 80], pen=pg.mkPen("gray", width=2))
    add_text(-BELT_HALF_WIDTH_MM, -80, "belt edge", color="gray", anchor=(0, 0))
    add_text(BELT_HALF_WIDTH_MM, -80, "belt edge", color="gray", anchor=(1, 0))

    # объект
    plot.plot([OBJECT[0]], [OBJECT[1]], pen=None, symbol="o", symbolBrush="k", symbolSize=14)
    add_text(OBJECT[0], OBJECT[1], "object (belt axis)", color="k")

    # зеркало: реальный отрезок от нижнего края вверх на условную длину полотна
    PLATE_LEN = 1400.0
    top_point = (BOTTOM_EDGE[0] + PLATE_LEN * d[0], BOTTOM_EDGE[1] + PLATE_LEN * d[1])
    plot.plot([BOTTOM_EDGE[0], top_point[0]], [BOTTOM_EDGE[1], top_point[1]],
              pen=pg.mkPen("b", width=5))
    add_text(BOTTOM_EDGE[0], BOTTOM_EDGE[1],
              f"mirror bottom edge\nY={BOTTOM_EDGE[0]:.0f} Z={BOTTOM_EDGE[1]:.0f}", color="b",
              anchor=(1, 1))
    add_text(top_point[0], top_point[1], f"tilt {23.0:.0f}° from vertical", color="b",
              anchor=(1, 0))

    # нормаль (от середины полотна)
    mid = ((BOTTOM_EDGE[0] + top_point[0]) / 2, (BOTTOM_EDGE[1] + top_point[1]) / 2)
    normal_len = 400.0
    plot.plot([mid[0], mid[0] + normal_len * n[0]], [mid[1], mid[1] + normal_len * n[1]],
              pen=pg.mkPen("b", width=2, style=pg.QtCore.Qt.PenStyle.DashLine))

    colors = {"top": "r", "diag": "g", "side": "m"}
    o_img = reflect_point(OBJECT, n, t)
    for cam in CAMERAS:
        c = cam_pos(cam.angle_deg, cam.distance_mm)
        color = colors.get(cam.name, "k")
        plot.plot([c[0]], [c[1]], pen=None, symbol="s", symbolBrush=color, symbolSize=12)
        add_text(c[0], c[1], f"{cam.name}  angle={cam.angle_deg:.0f} d={cam.distance_mm:.0f}mm",
                 color=color)
        plot.plot([c[0], OBJECT[0]], [c[1], OBJECT[1]],
                  pen=pg.mkPen(color, width=1, style=pg.QtCore.Qt.PenStyle.DotLine))

        if cam.name in REFLECTED:
            path = dist(c, o_img)
            plot.plot([c[0], o_img[0]], [c[1], o_img[1]],
                      pen=pg.mkPen(color, width=1, style=pg.QtCore.Qt.PenStyle.DashLine))
            dx, dy = o_img[0] - c[0], o_img[1] - c[1]
            denom = n[0] * dx + n[1] * dy
            if abs(denom) > 1e-9:
                s = (t - dot(c, n)) / denom
                hit = (c[0] + s * dx, c[1] + s * dy)
                plot.plot([c[0], hit[0], OBJECT[0]], [c[1], hit[1], OBJECT[1]],
                          pen=pg.mkPen(color, width=2))
                v_true, _ = true_reflected_azimuth_and_distance(c, n, t, OBJECT)
                add_text(hit[0], hit[1],
                         f"{cam.name}_mirror\ntrue az={v_true:.1f} deg\npath={path:.0f}mm",
                         color=color)

    return plot


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--save", metavar="PNG_PATH", help="сохранить в PNG headless (без окна)")
    args = ap.parse_args()

    if args.save:
        os.environ.setdefault("QT_QPA_PLATFORM", "offscreen")

    from PyQt6 import QtWidgets

    app = QtWidgets.QApplication(sys.argv)
    plot = build_plot()

    if args.save:
        import pyqtgraph.exporters as exporters

        plot.resize(1300, 1100)
        exporter = exporters.ImageExporter(plot.plotItem)
        exporter.parameters()["width"] = 1300
        exporter.export(args.save)
        print(f"сохранено: {args.save}")
    else:
        plot.show()
        app.exec()


if __name__ == "__main__":
    main()
