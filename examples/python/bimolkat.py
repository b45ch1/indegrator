from numpy import exp

def ffcn(t, x, f, p, u):

    # states

    n1   = x[0]
    n2   = x[1]
    n3   = x[2]
    n4   = x[3]
    Ckat = x[4]

    # constants

    na1 = 1.0
    na2 = 1.0
    na4 = 2.0

    # control functions

    Tc         = u[0]
    Ckat_feed  = u[1]
    a_feed     = u[2]
    b_feed     = u[3]


    # parameters (nature-given)

    kr1  = p[0] * 1.0e-2
    E    = p[1] * 60000.0
    k1   = p[2] * 0.1
    Ekat = p[3] * 40000.0
    lam  = p[4] * 0.5

    # molar masses (unit: kg/mol)

    M1 = 0.1362
    M2 = 0.09806
    M3 = M1 + M2
    M4 = 0.236

    Temp = Tc + 273.0
    Rg = 8.314
    T1 = 293.0

    # computation of reaction rates

    mR = n1*M1 + n2*M2 +n3*M3 + n4*M4

    kkat= kr1 * exp( -E/Rg  * ( 1.0/Temp - 1.0/T1 ) ) \
        + k1  * exp( -Ekat/Rg *( 1.0/Temp - 1.0/T1 ) ) \
        * Ckat

    r1 = kkat * n1 * n2 / mR

    f[0] = -r1 + a_feed
    f[1] = -r1 + b_feed
    f[2] = r1
    f[3] = 0.0
    f[4] = -lam*Ckat + Ckat_feed
