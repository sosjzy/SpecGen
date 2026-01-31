"""
坐标转换 - Rusinkiewicz 坐标系
用于 BRDF 参数化
"""
import numpy as np


def rotate_vector(v, axis, angle):
    """绕轴旋转向量"""
    return v * np.cos(angle) + axis * dot(axis, v) * (1 - np.cos(angle)) + cross(axis, v) * np.sin(angle)


def io_to_hd(wi, wo):
    """从入射/出射方向计算半程向量和差分向量"""
    half = normalize(*(wi + wo))
    r_h, theta_h, phi_h = xyz2sph(*half)

    bi_normal = np.tile([0.0, 1.0, 0.0], (wi.shape[1], 1)).T
    normal = np.tile([0.0, 0.0, 1.0], (wi.shape[1], 1)).T
    tmp = rotate_vector(wi, normal, -phi_h)
    diff = rotate_vector(tmp, bi_normal, -theta_h)
    return half, diff


def hd_to_io(half, diff):
    """从半程向量和差分向量计算入射/出射方向"""
    r_h, theta_h, phi_h = xyz2sph(*half)

    y_axis = np.tile([0.0, 1.0, 0.0], (half.shape[1], 1)).T
    z_axis = np.tile([0.0, 0.0, 1.0], (half.shape[1], 1)).T

    tmp = rotate_vector(diff, y_axis, theta_h)
    wi = normalize(*rotate_vector(tmp, z_axis, phi_h))
    wo = normalize(*(2 * dot(wi, half) * half - wi))
    return wi, wo


def dot(v1, v2):
    """向量点积"""
    return v1[0] * v2[0] + v1[1] * v2[1] + v1[2] * v2[2]


def cross(v1, v2):
    """向量叉积"""
    return np.cross(v1.T, v2.T).T


def xyz2sph(x, y, z):
    """笛卡尔坐标转球坐标"""
    r2_xy = x ** 2 + y ** 2
    r = np.sqrt(r2_xy + z ** 2)
    theta = np.arctan2(np.sqrt(r2_xy), z)
    phi = np.arctan2(y, x)
    return np.array([r, theta, phi])


def normalize(x, y, z):
    """归一化向量"""
    norm = np.sqrt(x ** 2 + y ** 2 + z ** 2)
    norm = np.where(norm == 0, np.inf, norm)
    return np.array([x, y, z]) / norm


def sph2xyz(r, theta, phi):
    """球坐标转笛卡尔坐标"""
    x = r * np.sin(theta) * np.cos(phi)
    y = r * np.sin(theta) * np.sin(phi)
    z = r * np.cos(theta)
    return np.array([x, y, z])


def rangles_to_rvectors(theta_h, theta_d, phi_d):
    """Rusinkiewicz 角度转向量 (假设 phi_h=0)"""
    hx = np.sin(theta_h) * np.cos(0.0)
    hy = np.sin(theta_h) * np.sin(0.0)
    hz = np.cos(theta_h)
    dx = np.sin(theta_d) * np.cos(phi_d)
    dy = np.sin(theta_d) * np.sin(phi_d)
    dz = np.cos(theta_d)
    return np.array([hx, hy, hz, dx, dy, dz])


def rvectors_to_rangles(hx, hy, hz, dx, dy, dz):
    """向量转 Rusinkiewicz 角度"""
    theta_h = np.arctan2(np.sqrt(hx ** 2 + hy ** 2), hz)
    theta_d = np.arctan2(np.sqrt(dx ** 2 + dy ** 2), dz)
    phi_d = np.arctan2(dy, dx)
    return np.array([theta_h, theta_d, phi_d])


def get_rusinkiewicz_angles(wi, wo):
    """
    给定入射方向 wi 和出射方向 wo，返回 Rusinkiewicz 坐标 (theta_h, theta_d, phi_d)
    
    Args:
        wi: 入射方向 (3,)
        wo: 出射方向 (3,)
    
    Returns:
        angles: (theta_h, theta_d, phi_d)
    """
    wi_1 = wi.reshape(3, 1)
    wo_1 = wo.reshape(3, 1)

    half, diff = io_to_hd(wi_1, wo_1)

    hx, hy, hz = half[:, 0]
    dx, dy, dz = diff[:, 0]

    angles = rvectors_to_rangles(hx, hy, hz, dx, dy, dz)
    return angles


def get_io_from_rusinkiewicz_angles(theta_h, theta_d, phi_d, phi_h=0.0):
    """
    从 Rusinkiewicz 角度生成入射/出射方向
    
    Args:
        theta_h: 半程向量极角
        theta_d: 差分向量极角
        phi_d: 差分向量方位角
        phi_h: 半程向量方位角 (默认为0)
    
    Returns:
        wi, wo: 入射和出射方向
    """
    theta_h = np.clip(theta_h, 0.0, np.pi)
    theta_d = np.clip(theta_d, 0.0, np.pi)
    phi_d = phi_d % (2.0 * np.pi)
    phi_h = phi_h % (2.0 * np.pi)

    hx, hy, hz = sph2xyz(1.0, theta_h, phi_h)
    dx, dy, dz = sph2xyz(1.0, theta_d, phi_d)

    if hz < 0:
        hx, hy, hz = -hx, -hy, -hz
    if dz < 0:
        dx, dy, dz = -dx, -dy, -dz

    half_1 = np.array([hx, hy, hz]).reshape(3, 1)
    diff_1 = np.array([dx, dy, dz]).reshape(3, 1)

    wi_1, wo_1 = hd_to_io(half_1, diff_1)

    wi = wi_1.ravel()
    wo = wo_1.ravel()
    wi /= np.linalg.norm(wi) + 1e-14
    wo /= np.linalg.norm(wo) + 1e-14

    return wi, wo
