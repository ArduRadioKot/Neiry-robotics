"""Кинематика и динамика робота с меканум-колёсами.

Параметры взяты из SDF. Моменты относятся к валам колёс. Контакт вдоль
тягового направления считается идеальным. Инерция роликов и гироскопический
эффект вращения колёс вокруг горизонтальных осей здесь не учитываются.
"""
from pathlib import Path
import xml.etree.ElementTree as ET

import numpy as np
from scipy.integrate import solve_ivp

DATA = Path(__file__).resolve().parent / 'assets'


def parameters():  # читаем размеры, массы и трение из SDF
    robot = ET.parse(DATA / 'world_force.sdf').find(".//model[@name='vehicle_blue']")
    links = {link.attrib['name']: link for link in robot.findall('link')}
    names = ['front_left_wheel', 'front_right_wheel',
             'rear_left_wheel', 'rear_right_wheel']
    pos = np.array([[float(v) for v in links[n].findtext('pose').split()[:2]] for n in names])
    center = pos.mean(axis=0)
    lx = float((pos[:2, 0].mean() - pos[2:, 0].mean()) / 2)  # разницу в 1 мкм считаем округлением
    ly = float((pos[[0, 2], 1].mean() - pos[[1, 3], 1].mean()) / 2)
    r = float(links[names[0]].findtext('collision/geometry/sphere/radius'))
    masses, centers, inertias = [], [], []
    for name, link in links.items():
        masses.append(float(link.findtext('inertial/mass')))
        centers.append(np.array([float(v) for v in link.findtext('pose').split()[:2]]) - center)
        axis = 'izz' if name == 'chassis' else 'iyy'  # ось колеса повёрнута на -pi/2
        inertias.append(float(link.findtext('inertial/inertia/' + axis)))
    masses, centers = np.array(masses), np.array(centers)
    m = float(masses.sum())
    c = (masses[:, None] * centers).sum(axis=0) / m
    I0 = float(np.sum(inertias + masses * np.sum(centers**2, axis=1)))
    Iw = float(links[names[0]].findtext('inertial/inertia/izz'))
    joint = robot.find("joint[@name='front_left_wheel_joint']")
    damping = float(joint.findtext('axis/dynamics/damping'))
    friction = float(joint.findtext('axis/dynamics/friction'))
    return dict(r=r, lx=lx, ly=ly, L=lx+ly, m=m, cx=float(c[0]), cy=float(c[1]),
                I0=I0, Iw=Iw, damping=damping, friction=friction)


P = parameters()


def matrices(p=P, centered=False):  # матрицы кинематики и инерции
    r, L, m = p['r'], p['L'], p['m']
    A = np.array([[1, -1, -L], [1, 1, L], [1, 1, -L], [1, -1, L]]) / r
    cx, cy = (0., 0.) if centered else (p['cx'], p['cy'])
    M0 = np.array([[m, 0, -m*cy], [0, m, m*cx], [-m*cy, m*cx, p['I0']]])
    H = M0 + p['Iw'] * A.T @ A
    return A, M0, H


A, M0, H = matrices()


def inverse_kinematics(nu):  # скорости колёс по скорости корпуса
    return A @ np.asarray(nu)


def forward_kinematics(w):  # оцениваем скорость корпуса по скоростям колёс
    """При несогласованных скоростях колёс результат приближённый."""
    return np.linalg.lstsq(A, np.asarray(w), rcond=None)[0]


def body_to_world(nu, theta):  # переводим скорость корпуса в мировые координаты
    c, s = np.cos(theta), np.sin(theta)
    vx, vy, om = nu
    return np.array([c*vx-s*vy, s*vx+c*vy, om])


def integrate_constant_twist(q0, nu, t):  # положение при постоянной скорости корпуса
    """Формула относится к постоянной скорости, а не к моменту моторов."""
    t = np.asarray(t)
    vx, vy, om = nu
    half = om*t/2
    scale = t*np.sinc(half/np.pi)
    mid = q0[2] + half
    return np.stack([q0[0]+scale*(vx*np.cos(mid)-vy*np.sin(mid)),
                     q0[1]+scale*(vx*np.sin(mid)+vy*np.cos(mid)),
                     q0[2]+om*t], axis=-1)


def convective(nu, p=P, centered=False):  # дополнительные члены при повороте
    vx, vy, om = nu
    a, b = (0., 0.) if centered else (p['cx'], p['cy'])
    m = p['m']
    return np.array([-m*om*(vy+a*om), m*om*(vx-b*om), m*om*(a*vx+b*vy)])


def rhs(t, state, torque, p=P, centered=False, eps=1e-3):  # производная состояния
    """tanh помогает численному расчёту, но не описывает точный покой.

    При w=0 сухое трение лежит в диапазоне [-d_i, d_i]. Для расчёта
    прилипания колёс при малой тяге нужен отдельный решатель.
    """
    aa, _, hh = matrices(p, centered)
    nu = state[3:]
    w = aa @ nu
    tau = np.asarray(torque(t) if callable(torque) else torque)
    loss = p['damping']*w + p['friction']*np.tanh(w/eps)
    nudot = np.linalg.solve(hh, aa.T@(tau-loss)-convective(nu, p, centered))
    return np.r_[body_to_world(nu, state[2]), nudot]


def simulate(t, torque, state0=None, **kwargs):  # считаем движение по моментам моторов
    t = np.asarray(t)
    z0 = np.zeros(6) if state0 is None else np.asarray(state0, float)
    sol = solve_ivp(lambda tt, z: rhs(tt, z, torque, **kwargs),
                    (0., float(t[-1])), z0, t_eval=t,
                    rtol=1e-8, atol=1e-10, max_step=.01)
    if not sol.success:
        raise RuntimeError(sol.message)
    return sol.y.T


def allocate_torque(wrench):  # распределяем усилие между колёсами
    """Минимизирует сумму квадратов моментов без учёта ограничений привода."""
    return A @ np.linalg.solve(A.T @ A, np.asarray(wrench))


def limit_wheel_speeds(nu, limit):  # ограничиваем скорости всех колёс одной долей
    w = inverse_kinematics(nu)
    factor = min(1., limit/max(np.max(np.abs(w)), 1e-15))
    return np.asarray(nu)*factor, w*factor, factor


def motor_torque(U, w, R, kt, ke, gear, eta=1.):  # момент на валу колеса
    """ke задан для вала мотора, gear — отношение скоростей мотора и колеса.

    Для рекуперации и обратного потока мощности нужна отдельная модель КПД.
    """
    current = (np.asarray(U)-ke*gear*np.asarray(w))/R
    return eta*gear*kt*current

