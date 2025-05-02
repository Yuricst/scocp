"""CR3BP equations of motion"""

from numba import njit, float64
import numpy as np

@njit(float64[:](float64, float64[:], float64), cache=True)
def rhs_cr3bp(t, state, mu):
    """Equation of motion in CR3BP, formulated for scipy.integrate.solve=ivp(), compatible with njit

    Args:
        t (float): time
        state (np.array): 1D array of Cartesian state, length 6
        mu (float): CR3BP parameter

    Returns:
        (np.array): 1D array of derivative of Cartesian state
    """
    # unpack positions
    x = state[0]
    y = state[1]
    z = state[2]
    # unpack velocities
    vx = state[3]
    vy = state[4]
    vz = state[5]
    # compute radii to each primary
    r1 = np.sqrt((x + mu) ** 2 + y ** 2 + z ** 2)
    r2 = np.sqrt((x - 1 + mu) ** 2 + y ** 2 + z ** 2)
    # setup vector for dX/dt
    deriv = np.zeros((6,))
    # position derivatives
    deriv[0] = vx
    deriv[1] = vy
    deriv[2] = vz
    # velocity derivatives
    deriv[3] = (
        2 * vy + x - ((1 - mu) / r1 ** 3) * (mu + x) + (mu / r2 ** 3) * (1 - mu - x)
    )
    deriv[4] = -2 * vx + y - ((1 - mu) / r1 ** 3) * y - (mu / r2 ** 3) * y
    deriv[5] = -((1 - mu) / r1 ** 3) * z - (mu / r2 ** 3) * z

    return deriv


@njit(float64[:](float64, float64[:], float64), cache=True)
def rhs_cr3bp_with_stm(t, state, mu):
    """Equation of motion in CR3BP along with its STM, compatible with njit
    Args:
        t (float): time
        state (np.array): 1D array of Cartesian state, length 6
        mu (float): CR3BP parameter

    Returns:
        (np.array): 1D array of derivative of Cartesian state
    """
    # coefficients of A matrix
    # first row
    a00 = 0
    a01 = 0
    a02 = 0
    a03 = 1
    a04 = 0
    a05 = 0
    # second row
    a10 = 0
    a11 = 0
    a12 = 0
    a13 = 0
    a14 = 1
    a15 = 0
    # third row
    a20 = 0
    a21 = 0
    a22 = 0
    a23 = 0
    a24 = 0
    a25 = 1
    # fourth row
    a33 = 0
    a34 = 2
    a35 = 0
    # fith row
    a43 = -2
    a44 = 0
    a45 = 0
    # sixth row
    a53 = 0
    a54 = 0
    a55 = 0
    # setup vector for dX/dt
    deriv = np.zeros((42,))
    # STATE
    # position derivatives
    deriv[0] = state[3]
    deriv[1] = state[4]
    deriv[2] = state[5]
    # velocitstate derivatives
    deriv[3] = (
        2 * state[4]
        + state[0]
        - ((1 - mu) / ((state[0] + mu) ** 2 + state[1] ** 2 + state[2] ** 2) ** 1.5)
        * (mu + state[0])
        + (mu / ((state[0] - 1 + mu) ** 2 + state[1] ** 2 + state[2] ** 2) ** 1.5)
        * (1 - mu - state[0])
    )
    deriv[4] = (
        -2 * state[3]
        + state[1]
        - ((1 - mu) / ((state[0] + mu) ** 2 + state[1] ** 2 + state[2] ** 2) ** 1.5)
        * state[1]
        - (mu / ((state[0] - 1 + mu) ** 2 + state[1] ** 2 + state[2] ** 2) ** 1.5)
        * state[1]
    )
    deriv[5] = (
        -((1 - mu) / ((state[0] + mu) ** 2 + state[1] ** 2 + state[2] ** 2) ** 1.5)
        * state[2]
        - (mu / ((state[0] - 1 + mu) ** 2 + state[1] ** 2 + state[2] ** 2) ** 1.5)
        * state[2]
    )

    # STATE-TRANSITION MATRIX
    # first row ...
    deriv[6] = (
        a00 * state[6]
        + a01 * state[12]
        + a02 * state[18]
        + a03 * state[24]
        + a04 * state[30]
        + a05 * state[36]
    )
    deriv[7] = (
        a00 * state[7]
        + a01 * state[13]
        + a02 * state[19]
        + a03 * state[25]
        + a04 * state[31]
        + a05 * state[37]
    )
    deriv[8] = (
        a00 * state[8]
        + a01 * state[14]
        + a02 * state[20]
        + a03 * state[26]
        + a04 * state[32]
        + a05 * state[38]
    )
    deriv[9] = (
        a00 * state[9]
        + a01 * state[15]
        + a02 * state[21]
        + a03 * state[27]
        + a04 * state[33]
        + a05 * state[39]
    )
    deriv[10] = (
        a00 * state[10]
        + a01 * state[16]
        + a02 * state[22]
        + a03 * state[28]
        + a04 * state[34]
        + a05 * state[40]
    )
    deriv[11] = (
        a00 * state[11]
        + a01 * state[17]
        + a02 * state[23]
        + a03 * state[29]
        + a04 * state[35]
        + a05 * state[41]
    )

    # second row ...
    deriv[12] = (
        a10 * state[6]
        + a11 * state[12]
        + a12 * state[18]
        + a13 * state[24]
        + a14 * state[30]
        + a15 * state[36]
    )
    deriv[13] = (
        a10 * state[7]
        + a11 * state[13]
        + a12 * state[19]
        + a13 * state[25]
        + a14 * state[31]
        + a15 * state[37]
    )
    deriv[14] = (
        a10 * state[8]
        + a11 * state[14]
        + a12 * state[20]
        + a13 * state[26]
        + a14 * state[32]
        + a15 * state[38]
    )
    deriv[15] = (
        a10 * state[9]
        + a11 * state[15]
        + a12 * state[21]
        + a13 * state[27]
        + a14 * state[33]
        + a15 * state[39]
    )
    deriv[16] = (
        a10 * state[10]
        + a11 * state[16]
        + a12 * state[22]
        + a13 * state[28]
        + a14 * state[34]
        + a15 * state[40]
    )
    deriv[17] = (
        a10 * state[11]
        + a11 * state[17]
        + a12 * state[23]
        + a13 * state[29]
        + a14 * state[35]
        + a15 * state[41]
    )

    # third row ...
    deriv[18] = (
        a20 * state[6]
        + a21 * state[12]
        + a22 * state[18]
        + a23 * state[24]
        + a24 * state[30]
        + a25 * state[36]
    )
    deriv[19] = (
        a20 * state[7]
        + a21 * state[13]
        + a22 * state[19]
        + a23 * state[25]
        + a24 * state[31]
        + a25 * state[37]
    )
    deriv[20] = (
        a20 * state[8]
        + a21 * state[14]
        + a22 * state[20]
        + a23 * state[26]
        + a24 * state[32]
        + a25 * state[38]
    )
    deriv[21] = (
        a20 * state[9]
        + a21 * state[15]
        + a22 * state[21]
        + a23 * state[27]
        + a24 * state[33]
        + a25 * state[39]
    )
    deriv[22] = (
        a20 * state[10]
        + a21 * state[16]
        + a22 * state[22]
        + a23 * state[28]
        + a24 * state[34]
        + a25 * state[40]
    )
    deriv[23] = (
        a20 * state[11]
        + a21 * state[17]
        + a22 * state[23]
        + a23 * state[29]
        + a24 * state[35]
        + a25 * state[41]
    )

    # fourth row ...
    deriv[24] = (
        (
            1
            - (1 - mu) / ((state[0] + mu) ** 2 + state[1] ** 2 + state[2] ** 2) ** 1.5
            - mu / ((state[0] - 1 + mu) ** 2 + state[1] ** 2 + state[2] ** 2) ** 1.5
            + 3
            * (
                (state[0] + mu) ** 2
                * (1 - mu)
                / ((state[0] + mu) ** 2 + state[1] ** 2 + state[2] ** 2) ** 2.5
                + (state[0] + mu - 1) ** 2
                * mu
                / ((state[0] - 1 + mu) ** 2 + state[1] ** 2 + state[2] ** 2) ** 2.5
            )
        )
        * state[6]
        + (
            3
            * state[1]
            * (
                (state[0] + mu)
                * (1 - mu)
                / ((state[0] + mu) ** 2 + state[1] ** 2 + state[2] ** 2) ** 2.5
                + (state[0] + mu - 1)
                * mu
                / ((state[0] - 1 + mu) ** 2 + state[1] ** 2 + state[2] ** 2) ** 2.5
            )
        )
        * state[12]
        + (
            3
            * state[2]
            * (
                (state[0] + mu)
                * (1 - mu)
                / ((state[0] + mu) ** 2 + state[1] ** 2 + state[2] ** 2) ** 2.5
                + (state[0] + mu - 1)
                * mu
                / ((state[0] - 1 + mu) ** 2 + state[1] ** 2 + state[2] ** 2) ** 2.5
            )
        )
        * state[18]
        + a33 * state[24]
        + a34 * state[30]
        + a35 * state[36]
    )

    deriv[25] = (
        (
            1
            - (1 - mu) / ((state[0] + mu) ** 2 + state[1] ** 2 + state[2] ** 2) ** 1.5
            - mu / ((state[0] - 1 + mu) ** 2 + state[1] ** 2 + state[2] ** 2) ** 1.5
            + 3
            * (
                (state[0] + mu) ** 2
                * (1 - mu)
                / ((state[0] + mu) ** 2 + state[1] ** 2 + state[2] ** 2) ** 2.5
                + (state[0] + mu - 1) ** 2
                * mu
                / ((state[0] - 1 + mu) ** 2 + state[1] ** 2 + state[2] ** 2) ** 2.5
            )
        )
        * state[7]
        + (
            3
            * state[1]
            * (
                (state[0] + mu)
                * (1 - mu)
                / ((state[0] + mu) ** 2 + state[1] ** 2 + state[2] ** 2) ** 2.5
                + (state[0] + mu - 1)
                * mu
                / ((state[0] - 1 + mu) ** 2 + state[1] ** 2 + state[2] ** 2) ** 2.5
            )
        )
        * state[13]
        + (
            3
            * state[2]
            * (
                (state[0] + mu)
                * (1 - mu)
                / ((state[0] + mu) ** 2 + state[1] ** 2 + state[2] ** 2) ** 2.5
                + (state[0] + mu - 1)
                * mu
                / ((state[0] - 1 + mu) ** 2 + state[1] ** 2 + state[2] ** 2) ** 2.5
            )
        )
        * state[19]
        + a33 * state[25]
        + a34 * state[31]
        + a35 * state[37]
    )

    deriv[26] = (
        (
            1
            - (1 - mu) / ((state[0] + mu) ** 2 + state[1] ** 2 + state[2] ** 2) ** 1.5
            - mu / ((state[0] - 1 + mu) ** 2 + state[1] ** 2 + state[2] ** 2) ** 1.5
            + 3
            * (
                (state[0] + mu) ** 2
                * (1 - mu)
                / ((state[0] + mu) ** 2 + state[1] ** 2 + state[2] ** 2) ** 2.5
                + (state[0] + mu - 1) ** 2
                * mu
                / ((state[0] - 1 + mu) ** 2 + state[1] ** 2 + state[2] ** 2) ** 2.5
            )
        )
        * state[8]
        + (
            3
            * state[1]
            * (
                (state[0] + mu)
                * (1 - mu)
                / ((state[0] + mu) ** 2 + state[1] ** 2 + state[2] ** 2) ** 2.5
                + (state[0] + mu - 1)
                * mu
                / ((state[0] - 1 + mu) ** 2 + state[1] ** 2 + state[2] ** 2) ** 2.5
            )
        )
        * state[14]
        + (
            3
            * state[2]
            * (
                (state[0] + mu)
                * (1 - mu)
                / ((state[0] + mu) ** 2 + state[1] ** 2 + state[2] ** 2) ** 2.5
                + (state[0] + mu - 1)
                * mu
                / ((state[0] - 1 + mu) ** 2 + state[1] ** 2 + state[2] ** 2) ** 2.5
            )
        )
        * state[20]
        + a33 * state[26]
        + a34 * state[32]
        + a35 * state[38]
    )

    deriv[27] = (
        (
            1
            - (1 - mu) / ((state[0] + mu) ** 2 + state[1] ** 2 + state[2] ** 2) ** 1.5
            - mu / ((state[0] - 1 + mu) ** 2 + state[1] ** 2 + state[2] ** 2) ** 1.5
            + 3
            * (
                (state[0] + mu) ** 2
                * (1 - mu)
                / ((state[0] + mu) ** 2 + state[1] ** 2 + state[2] ** 2) ** 2.5
                + (state[0] + mu - 1) ** 2
                * mu
                / ((state[0] - 1 + mu) ** 2 + state[1] ** 2 + state[2] ** 2) ** 2.5
            )
        )
        * state[9]
        + (
            3
            * state[1]
            * (
                (state[0] + mu)
                * (1 - mu)
                / ((state[0] + mu) ** 2 + state[1] ** 2 + state[2] ** 2) ** 2.5
                + (state[0] + mu - 1)
                * mu
                / ((state[0] - 1 + mu) ** 2 + state[1] ** 2 + state[2] ** 2) ** 2.5
            )
        )
        * state[15]
        + (
            3
            * state[2]
            * (
                (state[0] + mu)
                * (1 - mu)
                / ((state[0] + mu) ** 2 + state[1] ** 2 + state[2] ** 2) ** 2.5
                + (state[0] + mu - 1)
                * mu
                / ((state[0] - 1 + mu) ** 2 + state[1] ** 2 + state[2] ** 2) ** 2.5
            )
        )
        * state[21]
        + a33 * state[27]
        + a34 * state[33]
        + a35 * state[39]
    )

    deriv[28] = (
        (
            1
            - (1 - mu) / ((state[0] + mu) ** 2 + state[1] ** 2 + state[2] ** 2) ** 1.5
            - mu / ((state[0] - 1 + mu) ** 2 + state[1] ** 2 + state[2] ** 2) ** 1.5
            + 3
            * (
                (state[0] + mu) ** 2
                * (1 - mu)
                / ((state[0] + mu) ** 2 + state[1] ** 2 + state[2] ** 2) ** 2.5
                + (state[0] + mu - 1) ** 2
                * mu
                / ((state[0] - 1 + mu) ** 2 + state[1] ** 2 + state[2] ** 2) ** 2.5
            )
        )
        * state[10]
        + (
            3
            * state[1]
            * (
                (state[0] + mu)
                * (1 - mu)
                / ((state[0] + mu) ** 2 + state[1] ** 2 + state[2] ** 2) ** 2.5
                + (state[0] + mu - 1)
                * mu
                / ((state[0] - 1 + mu) ** 2 + state[1] ** 2 + state[2] ** 2) ** 2.5
            )
        )
        * state[16]
        + (
            3
            * state[2]
            * (
                (state[0] + mu)
                * (1 - mu)
                / ((state[0] + mu) ** 2 + state[1] ** 2 + state[2] ** 2) ** 2.5
                + (state[0] + mu - 1)
                * mu
                / ((state[0] - 1 + mu) ** 2 + state[1] ** 2 + state[2] ** 2) ** 2.5
            )
        )
        * state[22]
        + a33 * state[28]
        + a34 * state[34]
        + a35 * state[40]
    )

    deriv[29] = (
        (
            1
            - (1 - mu) / ((state[0] + mu) ** 2 + state[1] ** 2 + state[2] ** 2) ** 1.5
            - mu / ((state[0] - 1 + mu) ** 2 + state[1] ** 2 + state[2] ** 2) ** 1.5
            + 3
            * (
                (state[0] + mu) ** 2
                * (1 - mu)
                / ((state[0] + mu) ** 2 + state[1] ** 2 + state[2] ** 2) ** 2.5
                + (state[0] + mu - 1) ** 2
                * mu
                / ((state[0] - 1 + mu) ** 2 + state[1] ** 2 + state[2] ** 2) ** 2.5
            )
        )
        * state[11]
        + (
            3
            * state[1]
            * (
                (state[0] + mu)
                * (1 - mu)
                / ((state[0] + mu) ** 2 + state[1] ** 2 + state[2] ** 2) ** 2.5
                + (state[0] + mu - 1)
                * mu
                / ((state[0] - 1 + mu) ** 2 + state[1] ** 2 + state[2] ** 2) ** 2.5
            )
        )
        * state[17]
        + (
            3
            * state[2]
            * (
                (state[0] + mu)
                * (1 - mu)
                / ((state[0] + mu) ** 2 + state[1] ** 2 + state[2] ** 2) ** 2.5
                + (state[0] + mu - 1)
                * mu
                / ((state[0] - 1 + mu) ** 2 + state[1] ** 2 + state[2] ** 2) ** 2.5
            )
        )
        * state[23]
        + a33 * state[29]
        + a34 * state[35]
        + a35 * state[41]
    )

    # fifth row ...
    deriv[30] = (
        (
            3
            * state[1]
            * (
                (state[0] + mu)
                * (1 - mu)
                / ((state[0] + mu) ** 2 + state[1] ** 2 + state[2] ** 2) ** 2.5
                + (state[0] + mu - 1)
                * mu
                / ((state[0] - 1 + mu) ** 2 + state[1] ** 2 + state[2] ** 2) ** 2.5
            )
        )
        * state[6]
        + (
            1
            - (1 - mu) / ((state[0] + mu) ** 2 + state[1] ** 2 + state[2] ** 2) ** 1.5
            - mu / ((state[0] - 1 + mu) ** 2 + state[1] ** 2 + state[2] ** 2) ** 1.5
            + 3
            * state[1] ** 2
            * (
                (1 - mu) / ((state[0] + mu) ** 2 + state[1] ** 2 + state[2] ** 2) ** 2.5
                + mu / ((state[0] - 1 + mu) ** 2 + state[1] ** 2 + state[2] ** 2) ** 2.5
            )
        )
        * state[12]
        + (
            3
            * state[1]
            * state[2]
            * (
                (1 - mu) / ((state[0] + mu) ** 2 + state[1] ** 2 + state[2] ** 2) ** 2.5
                + mu / ((state[0] - 1 + mu) ** 2 + state[1] ** 2 + state[2] ** 2) ** 2.5
            )
        )
        * state[18]
        + a43 * state[24]
        + a44 * state[30]
        + a45 * state[36]
    )

    deriv[31] = (
        (
            3
            * state[1]
            * (
                (state[0] + mu)
                * (1 - mu)
                / ((state[0] + mu) ** 2 + state[1] ** 2 + state[2] ** 2) ** 2.5
                + (state[0] + mu - 1)
                * mu
                / ((state[0] - 1 + mu) ** 2 + state[1] ** 2 + state[2] ** 2) ** 2.5
            )
        )
        * state[7]
        + (
            1
            - (1 - mu) / ((state[0] + mu) ** 2 + state[1] ** 2 + state[2] ** 2) ** 1.5
            - mu / ((state[0] - 1 + mu) ** 2 + state[1] ** 2 + state[2] ** 2) ** 1.5
            + 3
            * state[1] ** 2
            * (
                (1 - mu) / ((state[0] + mu) ** 2 + state[1] ** 2 + state[2] ** 2) ** 2.5
                + mu / ((state[0] - 1 + mu) ** 2 + state[1] ** 2 + state[2] ** 2) ** 2.5
            )
        )
        * state[13]
        + (
            3
            * state[1]
            * state[2]
            * (
                (1 - mu) / ((state[0] + mu) ** 2 + state[1] ** 2 + state[2] ** 2) ** 2.5
                + mu / ((state[0] - 1 + mu) ** 2 + state[1] ** 2 + state[2] ** 2) ** 2.5
            )
        )
        * state[19]
        + a43 * state[25]
        + a44 * state[31]
        + a45 * state[37]
    )

    deriv[32] = (
        (
            3
            * state[1]
            * (
                (state[0] + mu)
                * (1 - mu)
                / ((state[0] + mu) ** 2 + state[1] ** 2 + state[2] ** 2) ** 2.5
                + (state[0] + mu - 1)
                * mu
                / ((state[0] - 1 + mu) ** 2 + state[1] ** 2 + state[2] ** 2) ** 2.5
            )
        )
        * state[8]
        + (
            1
            - (1 - mu) / ((state[0] + mu) ** 2 + state[1] ** 2 + state[2] ** 2) ** 1.5
            - mu / ((state[0] - 1 + mu) ** 2 + state[1] ** 2 + state[2] ** 2) ** 1.5
            + 3
            * state[1] ** 2
            * (
                (1 - mu) / ((state[0] + mu) ** 2 + state[1] ** 2 + state[2] ** 2) ** 2.5
                + mu / ((state[0] - 1 + mu) ** 2 + state[1] ** 2 + state[2] ** 2) ** 2.5
            )
        )
        * state[14]
        + (
            3
            * state[1]
            * state[2]
            * (
                (1 - mu) / ((state[0] + mu) ** 2 + state[1] ** 2 + state[2] ** 2) ** 2.5
                + mu / ((state[0] - 1 + mu) ** 2 + state[1] ** 2 + state[2] ** 2) ** 2.5
            )
        )
        * state[20]
        + a43 * state[26]
        + a44 * state[32]
        + a45 * state[38]
    )

    deriv[33] = (
        (
            3
            * state[1]
            * (
                (state[0] + mu)
                * (1 - mu)
                / ((state[0] + mu) ** 2 + state[1] ** 2 + state[2] ** 2) ** 2.5
                + (state[0] + mu - 1)
                * mu
                / ((state[0] - 1 + mu) ** 2 + state[1] ** 2 + state[2] ** 2) ** 2.5
            )
        )
        * state[9]
        + (
            1
            - (1 - mu) / ((state[0] + mu) ** 2 + state[1] ** 2 + state[2] ** 2) ** 1.5
            - mu / ((state[0] - 1 + mu) ** 2 + state[1] ** 2 + state[2] ** 2) ** 1.5
            + 3
            * state[1] ** 2
            * (
                (1 - mu) / ((state[0] + mu) ** 2 + state[1] ** 2 + state[2] ** 2) ** 2.5
                + mu / ((state[0] - 1 + mu) ** 2 + state[1] ** 2 + state[2] ** 2) ** 2.5
            )
        )
        * state[15]
        + (
            3
            * state[1]
            * state[2]
            * (
                (1 - mu) / ((state[0] + mu) ** 2 + state[1] ** 2 + state[2] ** 2) ** 2.5
                + mu / ((state[0] - 1 + mu) ** 2 + state[1] ** 2 + state[2] ** 2) ** 2.5
            )
        )
        * state[21]
        + a43 * state[27]
        + a44 * state[33]
        + a45 * state[39]
    )

    deriv[34] = (
        (
            3
            * state[1]
            * (
                (state[0] + mu)
                * (1 - mu)
                / ((state[0] + mu) ** 2 + state[1] ** 2 + state[2] ** 2) ** 2.5
                + (state[0] + mu - 1)
                * mu
                / ((state[0] - 1 + mu) ** 2 + state[1] ** 2 + state[2] ** 2) ** 2.5
            )
        )
        * state[10]
        + (
            1
            - (1 - mu) / ((state[0] + mu) ** 2 + state[1] ** 2 + state[2] ** 2) ** 1.5
            - mu / ((state[0] - 1 + mu) ** 2 + state[1] ** 2 + state[2] ** 2) ** 1.5
            + 3
            * state[1] ** 2
            * (
                (1 - mu) / ((state[0] + mu) ** 2 + state[1] ** 2 + state[2] ** 2) ** 2.5
                + mu / ((state[0] - 1 + mu) ** 2 + state[1] ** 2 + state[2] ** 2) ** 2.5
            )
        )
        * state[16]
        + (
            3
            * state[1]
            * state[2]
            * (
                (1 - mu) / ((state[0] + mu) ** 2 + state[1] ** 2 + state[2] ** 2) ** 2.5
                + mu / ((state[0] - 1 + mu) ** 2 + state[1] ** 2 + state[2] ** 2) ** 2.5
            )
        )
        * state[22]
        + a43 * state[28]
        + a44 * state[34]
        + a45 * state[40]
    )

    deriv[35] = (
        (
            3
            * state[1]
            * (
                (state[0] + mu)
                * (1 - mu)
                / ((state[0] + mu) ** 2 + state[1] ** 2 + state[2] ** 2) ** 2.5
                + (state[0] + mu - 1)
                * mu
                / ((state[0] - 1 + mu) ** 2 + state[1] ** 2 + state[2] ** 2) ** 2.5
            )
        )
        * state[11]
        + (
            1
            - (1 - mu) / ((state[0] + mu) ** 2 + state[1] ** 2 + state[2] ** 2) ** 1.5
            - mu / ((state[0] - 1 + mu) ** 2 + state[1] ** 2 + state[2] ** 2) ** 1.5
            + 3
            * state[1] ** 2
            * (
                (1 - mu) / ((state[0] + mu) ** 2 + state[1] ** 2 + state[2] ** 2) ** 2.5
                + mu / ((state[0] - 1 + mu) ** 2 + state[1] ** 2 + state[2] ** 2) ** 2.5
            )
        )
        * state[17]
        + (
            3
            * state[1]
            * state[2]
            * (
                (1 - mu) / ((state[0] + mu) ** 2 + state[1] ** 2 + state[2] ** 2) ** 2.5
                + mu / ((state[0] - 1 + mu) ** 2 + state[1] ** 2 + state[2] ** 2) ** 2.5
            )
        )
        * state[23]
        + a43 * state[29]
        + a44 * state[35]
        + a45 * state[41]
    )

    # sixth row ...
    deriv[36] = (
        (
            3
            * state[2]
            * (
                (state[0] + mu)
                * (1 - mu)
                / ((state[0] + mu) ** 2 + state[1] ** 2 + state[2] ** 2) ** 2.5
                + (state[0] + mu - 1)
                * mu
                / ((state[0] - 1 + mu) ** 2 + state[1] ** 2 + state[2] ** 2) ** 2.5
            )
        )
        * state[6]
        + (
            3
            * state[1]
            * state[2]
            * (
                (1 - mu) / ((state[0] + mu) ** 2 + state[1] ** 2 + state[2] ** 2) ** 2.5
                + mu / ((state[0] - 1 + mu) ** 2 + state[1] ** 2 + state[2] ** 2) ** 2.5
            )
        )
        * state[12]
        + (
            -(1 - mu) / ((state[0] + mu) ** 2 + state[1] ** 2 + state[2] ** 2) ** 1.5
            - mu / ((state[0] - 1 + mu) ** 2 + state[1] ** 2 + state[2] ** 2) ** 1.5
            + 3
            * state[2] ** 2
            * (
                (1 - mu) / ((state[0] + mu) ** 2 + state[1] ** 2 + state[2] ** 2) ** 2.5
                + mu / ((state[0] - 1 + mu) ** 2 + state[1] ** 2 + state[2] ** 2) ** 2.5
            )
        )
        * state[18]
        + a53 * state[24]
        + a54 * state[30]
        + a55 * state[36]
    )

    deriv[37] = (
        (
            3
            * state[2]
            * (
                (state[0] + mu)
                * (1 - mu)
                / ((state[0] + mu) ** 2 + state[1] ** 2 + state[2] ** 2) ** 2.5
                + (state[0] + mu - 1)
                * mu
                / ((state[0] - 1 + mu) ** 2 + state[1] ** 2 + state[2] ** 2) ** 2.5
            )
        )
        * state[7]
        + (
            3
            * state[1]
            * state[2]
            * (
                (1 - mu) / ((state[0] + mu) ** 2 + state[1] ** 2 + state[2] ** 2) ** 2.5
                + mu / ((state[0] - 1 + mu) ** 2 + state[1] ** 2 + state[2] ** 2) ** 2.5
            )
        )
        * state[13]
        + (
            -(1 - mu) / ((state[0] + mu) ** 2 + state[1] ** 2 + state[2] ** 2) ** 1.5
            - mu / ((state[0] - 1 + mu) ** 2 + state[1] ** 2 + state[2] ** 2) ** 1.5
            + 3
            * state[2] ** 2
            * (
                (1 - mu) / ((state[0] + mu) ** 2 + state[1] ** 2 + state[2] ** 2) ** 2.5
                + mu / ((state[0] - 1 + mu) ** 2 + state[1] ** 2 + state[2] ** 2) ** 2.5
            )
        )
        * state[19]
        + a53 * state[25]
        + a54 * state[31]
        + a55 * state[37]
    )

    deriv[38] = (
        (
            3
            * state[2]
            * (
                (state[0] + mu)
                * (1 - mu)
                / ((state[0] + mu) ** 2 + state[1] ** 2 + state[2] ** 2) ** 2.5
                + (state[0] + mu - 1)
                * mu
                / ((state[0] - 1 + mu) ** 2 + state[1] ** 2 + state[2] ** 2) ** 2.5
            )
        )
        * state[8]
        + (
            3
            * state[1]
            * state[2]
            * (
                (1 - mu) / ((state[0] + mu) ** 2 + state[1] ** 2 + state[2] ** 2) ** 2.5
                + mu / ((state[0] - 1 + mu) ** 2 + state[1] ** 2 + state[2] ** 2) ** 2.5
            )
        )
        * state[14]
        + (
            -(1 - mu) / ((state[0] + mu) ** 2 + state[1] ** 2 + state[2] ** 2) ** 1.5
            - mu / ((state[0] - 1 + mu) ** 2 + state[1] ** 2 + state[2] ** 2) ** 1.5
            + 3
            * state[2] ** 2
            * (
                (1 - mu) / ((state[0] + mu) ** 2 + state[1] ** 2 + state[2] ** 2) ** 2.5
                + mu / ((state[0] - 1 + mu) ** 2 + state[1] ** 2 + state[2] ** 2) ** 2.5
            )
        )
        * state[20]
        + a53 * state[26]
        + a54 * state[32]
        + a55 * state[38]
    )

    deriv[39] = (
        (
            3
            * state[2]
            * (
                (state[0] + mu)
                * (1 - mu)
                / ((state[0] + mu) ** 2 + state[1] ** 2 + state[2] ** 2) ** 2.5
                + (state[0] + mu - 1)
                * mu
                / ((state[0] - 1 + mu) ** 2 + state[1] ** 2 + state[2] ** 2) ** 2.5
            )
        )
        * state[9]
        + (
            3
            * state[1]
            * state[2]
            * (
                (1 - mu) / ((state[0] + mu) ** 2 + state[1] ** 2 + state[2] ** 2) ** 2.5
                + mu / ((state[0] - 1 + mu) ** 2 + state[1] ** 2 + state[2] ** 2) ** 2.5
            )
        )
        * state[15]
        + (
            -(1 - mu) / ((state[0] + mu) ** 2 + state[1] ** 2 + state[2] ** 2) ** 1.5
            - mu / ((state[0] - 1 + mu) ** 2 + state[1] ** 2 + state[2] ** 2) ** 1.5
            + 3
            * state[2] ** 2
            * (
                (1 - mu) / ((state[0] + mu) ** 2 + state[1] ** 2 + state[2] ** 2) ** 2.5
                + mu / ((state[0] - 1 + mu) ** 2 + state[1] ** 2 + state[2] ** 2) ** 2.5
            )
        )
        * state[21]
        + a53 * state[27]
        + a54 * state[33]
        + a55 * state[39]
    )

    deriv[40] = (
        (
            3
            * state[2]
            * (
                (state[0] + mu)
                * (1 - mu)
                / ((state[0] + mu) ** 2 + state[1] ** 2 + state[2] ** 2) ** 2.5
                + (state[0] + mu - 1)
                * mu
                / ((state[0] - 1 + mu) ** 2 + state[1] ** 2 + state[2] ** 2) ** 2.5
            )
        )
        * state[10]
        + (
            3
            * state[1]
            * state[2]
            * (
                (1 - mu) / ((state[0] + mu) ** 2 + state[1] ** 2 + state[2] ** 2) ** 2.5
                + mu / ((state[0] - 1 + mu) ** 2 + state[1] ** 2 + state[2] ** 2) ** 2.5
            )
        )
        * state[16]
        + (
            -(1 - mu) / ((state[0] + mu) ** 2 + state[1] ** 2 + state[2] ** 2) ** 1.5
            - mu / ((state[0] - 1 + mu) ** 2 + state[1] ** 2 + state[2] ** 2) ** 1.5
            + 3
            * state[2] ** 2
            * (
                (1 - mu) / ((state[0] + mu) ** 2 + state[1] ** 2 + state[2] ** 2) ** 2.5
                + mu / ((state[0] - 1 + mu) ** 2 + state[1] ** 2 + state[2] ** 2) ** 2.5
            )
        )
        * state[22]
        + a53 * state[28]
        + a54 * state[34]
        + a55 * state[40]
    )

    deriv[41] = (
        (
            3
            * state[2]
            * (
                (state[0] + mu)
                * (1 - mu)
                / ((state[0] + mu) ** 2 + state[1] ** 2 + state[2] ** 2) ** 2.5
                + (state[0] + mu - 1)
                * mu
                / ((state[0] - 1 + mu) ** 2 + state[1] ** 2 + state[2] ** 2) ** 2.5
            )
        )
        * state[11]
        + (
            3
            * state[1]
            * state[2]
            * (
                (1 - mu) / ((state[0] + mu) ** 2 + state[1] ** 2 + state[2] ** 2) ** 2.5
                + mu / ((state[0] - 1 + mu) ** 2 + state[1] ** 2 + state[2] ** 2) ** 2.5
            )
        )
        * state[17]
        + (
            -(1 - mu) / ((state[0] + mu) ** 2 + state[1] ** 2 + state[2] ** 2) ** 1.5
            - mu / ((state[0] - 1 + mu) ** 2 + state[1] ** 2 + state[2] ** 2) ** 1.5
            + 3
            * state[2] ** 2
            * (
                (1 - mu) / ((state[0] + mu) ** 2 + state[1] ** 2 + state[2] ** 2) ** 2.5
                + mu / ((state[0] - 1 + mu) ** 2 + state[1] ** 2 + state[2] ** 2) ** 2.5
            )
        )
        * state[23]
        + a53 * state[29]
        + a54 * state[35]
        + a55 * state[41]
    )

    return deriv