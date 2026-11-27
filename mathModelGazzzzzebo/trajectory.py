"""Настройки маршрута. После изменения значений запустите симуляцию заново."""
import numpy as np

MODE = 'eight'  # 'eight' — восьмёрка; 'circle' — круг
RADIUS = 1.3   # метры; для восьмёрки — половина ширины
DURATION = 30.0  # секунды: больше значение — медленнее движение


def reference(t):  # положение, скорость и ускорение на маршруте
    """Заданное положение, скорость и ускорение в мировой системе координат."""
    if MODE not in ('eight', 'circle'):
        raise ValueError("MODE должен быть 'eight' или 'circle'")
    if DURATION <= 0 or RADIUS <= 0:
        raise ValueError('DURATION и RADIUS должны быть больше нуля')
    u = float(np.clip(t / DURATION, 0, 1))
    s = 10*u**3 - 15*u**4 + 6*u**5
    ds = 30*u**2 - 60*u**3 + 30*u**4
    dds = 60*u - 180*u**2 + 120*u**3
    phi = 2*np.pi*s
    w = 2*np.pi*ds/DURATION
    wd = 2*np.pi*dds/DURATION**2
    r = RADIUS
    if MODE == 'circle':
        q = np.array([r*np.sin(phi), r*(1-np.cos(phi)), 0.])
        qp = np.array([r*np.cos(phi), r*np.sin(phi), 0.])
        qpp = np.array([-r*np.sin(phi), r*np.cos(phi), 0.])
    else:
        q = np.array([r*np.sin(phi), .5*r*np.sin(2*phi), 0.])
        qp = np.array([r*np.cos(phi), r*np.cos(2*phi), 0.])
        qpp = np.array([-r*np.sin(phi), -2*r*np.sin(2*phi), 0.])
    return q, qp*w, qpp*w*w + qp*wd
