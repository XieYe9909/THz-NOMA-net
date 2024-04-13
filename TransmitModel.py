import numpy as np
from numpy import pi
from numpy import sin, exp, sqrt, abs
from numpy.random import rand, randn
from numpy.linalg import inv, norm


class UserState:
    def __init__(self, num: int, r_max: float, random_angle: bool = True, random_fading: bool = True):
        self.num = num
        self.distance = r_max * rand(num,)

        if random_angle:
            self.theta = pi * rand(num,) - pi / 2
        else:
            self.theta = -pi / 2 + pi / (2 * num) + np.arange(num) * pi / num

        if random_fading:
            self.fading = sqrt(0.5) * (randn(num,) + 1j * randn(num,))
        else:
            self.fading = np.ones((num,))


class ChannelModel:
    __fc = 300e9
    __c = 3e8
    __lambda = __c / __fc
    __alpha_PL = 2
    __psi = 5e-3

    def __init__(self, N: int, K: int, M: int, NQ: int, RP: float, RS: float):
        self.num_antenna = N
        self.num_prim = K
        self.num_sec = M
        self.codebook_size = NQ
        self.range_prim = RP
        self.range_sec = RS

        self.state_prim = UserState(K, RP, random_angle=False, random_fading=False)
        self.state_sec = UserState(M, RS, random_angle=True, random_fading=True)

    @property
    def channel_prim(self):
        return self.channel_realize(self.state_prim)

    @property
    def channel_sec(self):
        return self.channel_realize(self.state_sec)

    @property
    def beams(self):
        return self.beamforming(self.channel_prim)

    @property
    def gain_prim(self):
        HP = self.channel_prim
        F = self.beams
        GP = np.real(np.diag(HP.conj() @ F.T))
        return GP

    @property
    def gain_sec(self):
        HS = self.channel_sec
        F = self.beams
        GS = HS.conj() @ F.T
        return GS

    def channel_realize(self, user_state: UserState):
        lamb = self.__lambda
        alpha = self.__alpha_PL
        psi = self.__psi
        N = self.num_antenna

        num = user_state.num
        r = user_state.distance
        theta = user_state.theta
        a = user_state.fading

        H = np.zeros(shape=(num, N), dtype=complex)
        for n in range(0, num):
            PL = (4 * pi / lamb) ** 2 * exp(psi * r[n]) * (r[n] ** alpha + 1)
            H[n] = a[n] * exp(-1j * pi * sin(theta[n]) * np.arange(N)) / sqrt(PL)

        return H

    def beamforming(self, H, user_state: UserState = None, analog_beamforming: bool = False):
        N = self.num_antenna
        NQ = self.codebook_size

        if analog_beamforming:
            num = user_state.num
            theta = user_state.theta

            codebook = -pi / 2 + pi / (2 * NQ) + np.arange(NQ) * pi / NQ
            abF = np.zeros(shape=(num, N), dtype=complex)
            for n in range(0, num):
                diff = theta[n] - codebook
                index = np.argmin(abs(diff))
                abF[n] = exp(-1j * pi * sin(codebook[index]) * np.arange(N)) / sqrt(N)
                codebook = np.delete(codebook, index)

            dbF = inv(abF @ H.T.conj())
            power_coef = 1 / norm(dbF, axis=1)
            dbF = np.diag(power_coef) @ dbF

            F = np.zeros(shape=(num, N), dtype=complex)
            for n in range(0, num):
                F[n] = dbF[n] @ abF
        else:
            F = inv(H @ H.T.conj()) @ H
            power_coef = 1 / norm(F, axis=1)
            F = np.diag(power_coef) @ F

        return F
