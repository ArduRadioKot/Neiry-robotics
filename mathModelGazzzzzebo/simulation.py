"""Управление роботом в уже открытом мире Gazebo."""
import time

import numpy as np
import model
import trajectory

from gz.transport13 import Node
from gz.msgs10.double_pb2 import Double
from gz.msgs10.odometry_pb2 import Odometry


def main():  # подключаемся к открытому Gazebo и управляем колёсами
    node = Node()
    joints = [
        'front_left_wheel_joint', 'front_right_wheel_joint',
        'rear_left_wheel_joint', 'rear_right_wheel_joint',
    ]
    pubs = [node.advertise('/model/vehicle_blue/joint/' + j + '/cmd_force', Double) for j in joints]
    alloc = model.A @ np.linalg.inv(model.A.T @ model.A)  # распределение усилия по колёсам
    state = {
        'last': None, 'start_pose': None, 'start_time': None, 'last_control': -1.,
        'done': False, 'error': None, 'last_print': -1,
    }

    def send(tau):  # отправляем моменты в Gazebo
        for pub, value in zip(pubs, tau):
            msg = Double()
            msg.data = float(value)
            pub.publish(msg)

    def on_odom(m):  # получаем позу робота и считаем команды
        try:
            t = m.header.stamp.sec + m.header.stamp.nsec*1e-9
            if state['done'] or (state['last_control'] >= 0 and t-state['last_control'] < .009999):
                return
            q = m.pose.orientation
            angle = np.arctan2(2*(q.w*q.z+q.x*q.y), 1-2*(q.y*q.y+q.z*q.z))
            pose = np.array([
                m.pose.position.x - .20142725*np.cos(angle),
                m.pose.position.y - .20142725*np.sin(angle),
                angle,
            ])
            old = state['last']
            vel = np.zeros(3)
            if old is not None and t > old[0]:
                diff = pose-old[1]
                diff[2] = (diff[2]+np.pi) % (2*np.pi)-np.pi
                vel = diff/(t-old[0])
            state['last'] = (t, pose)
            state['last_control'] = t
            if state['start_pose'] is None:
                state['start_pose'] = pose.copy()
                state['start_time'] = t  # отсчёт начинается при подключении к Gazebo
            rel = t-state['start_time']
            ref, vr, ar = trajectory.reference(rel)
            ref = ref+state['start_pose']
            err = ref-pose
            err[2] = (err[2]+np.pi) % (2*np.pi)-np.pi
            co, si = np.cos(angle), np.sin(angle)
            R = np.array([[co,-si,0],[si,co,0],[0,0,1]])
            nu = R.T @ vel
            acc_world = ar+np.array([7.,7.,10.])*err+np.array([4.8,4.8,6.])*(vr-vel)
            acc_body = R.T @ acc_world + np.array([nu[2]*nu[1],-nu[2]*nu[0],0.])
            wheels = model.A @ nu
            loss = model.P['damping']*wheels + model.P['friction']*np.tanh(wheels/.03)
            wrench = model.H @ acc_body + model.convective(nu) + model.A.T @ loss
            send(np.clip(alloc @ wrench, -6., 6.))
            if int(rel)//4 > state['last_print']:
                state['last_print'] = int(rel)//4
                print(f't={rel:.1f} с: X={pose[0]:.3f}, Y={pose[1]:.3f}, курс={np.degrees(angle):.4f}°', flush=True)
            if rel >= trajectory.DURATION+2.:
                state['done'] = True
                send(np.zeros(4))
                print('Маршрут завершён.', flush=True)
        except Exception as exc:
            state['error'] = repr(exc)
            state['done'] = True
            send(np.zeros(4))

    node.subscribe(Odometry, '/model/vehicle_blue/odometry', on_odom)
    print('Жду данные из Gazebo. Откройте world.sdf и нажмите Play.', flush=True)
    try:
        while not state['done']:
            time.sleep(.1)
        if state['error']:
            raise RuntimeError(state['error'])
    except KeyboardInterrupt:
        pass
    finally:
        send(np.zeros(4))


if __name__ == '__main__':
    main()
