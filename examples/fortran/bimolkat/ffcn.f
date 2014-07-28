      subroutine ffcn( t, x, f, p, u)
        implicit none

        real*8 x(5), f(5), p(5), u(4),  t

        real*8 n1, n2, n3, n4
        real*8 na1, na2, na4
        real*8 fg, Temp, E , Rg , T1, Tc
        real*8 r1, mR
        real*8 kr1, kkat, Ckat, Ckat_feed, a_feed, b_feed, Ekat
        real*8 k1, lambda
        real*8 M1, M2, M3, M4
        real*8 dm

c       states

        n1   = x(1)
        n2   = x(2)
        n3   = x(3)
        n4   = x(4)
        Ckat = x(5)

c       constants

        na1 = 1.0
        na2 = 1.0
        na4 = 2.0

c       control functions

        Tc         = u(1)
        Ckat_feed  = u(2)
        a_feed     = u(3)
        b_feed     = u(4)


c       parameters (nature-given)

        kr1 = p(1) * 1.0d-2
        E = p(2) * 60000.0d+0
        k1 = p(3) * 0.10d+0
        Ekat = p(4) * 40000.0d0
        lambda = p(5) * 0.5d+0

c       molar masses (unit: kg/mol)

        M1 = 0.1362d+0
        M2 = 0.09806d+0
        M3 = M1 + M2
        M4 = 0.236d+0

        Temp = Tc + 273.0d+0
        Rg = 8.314d+0
        T1 = 293.0d+0

c       computation of reaction rates

        mR = n1*M1 + n2*M2 +n3*M3 + n4*M4

        kkat= kr1 * dexp( -E/Rg  * ( 1.0d+0/Temp - 1.0d+0/T1 ) )
     &      + k1  * dexp( -Ekat/Rg *( 1.0d+0/Temp - 1.0d+0/T1 ) )
     &      * Ckat

        r1 = kkat * n1 * n2 / mR

        f(1) = -r1 + a_feed
        f(2) = -r1 + b_feed
        f(3) = r1
        f(4) = 0.0d0
        f(5) = -lambda*Ckat + Ckat_feed


      end
