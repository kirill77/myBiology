#define _USE_MATH_DEFINES
#include <math.h>
#include <basics/myunits.h>
#include <basics/gradientDescent.h>

typedef double T;

MyUnits<T> computeAction(MyUnits<T> fOrbitRadius, MyUnits<T> fSpeed)
{
    MyUnits<T> fKineticEnergy = MyUnits<T>::electronMass() * fSpeed * fSpeed / 2;
    MyUnits<T> fPotentialEnergy = chargePotentialEnergy<T>(MyUnits<T>::electronCharge(), MyUnits<T>::electronCharge(), fOrbitRadius);
    MyUnits<T> fLagrangian = fKineticEnergy - fPotentialEnergy;
    MyUnits<T> fTime = fOrbitRadius * 2 * M_PI / fSpeed;
    MyUnits<T> fAction = fTime * fLagrangian;
    MyUnits<T> fDividedAction = fAction / MY_PLANCK_CONSTANT;
    return fDividedAction;
}

struct OrbitFunctor
{
    OrbitFunctor(MyUnits<T> fSpeed) : m_fSpeed(fSpeed) { }
    double operator ()(double fOrbitRadius)
    {
        return computeAction(MyUnits<T>(fOrbitRadius), m_fSpeed).m_value;
    }
private:
    MyUnits<T> m_fSpeed;
};
struct SpeedFunctor
{
    SpeedFunctor(MyUnits<T> fOrbitRadius) : m_fOrbitRadius(fOrbitRadius) { }
    double operator ()(double fSpeed)
    {
        return computeAction(m_fOrbitRadius, MyUnits<T>(fSpeed)).m_value;
    }
private:
    MyUnits<T> m_fOrbitRadius;
};

struct XCubeFunctor
{
    double operator ()(double fX)
    {
        return fX * fX * fX;
    }
};

int main()
{
    MyUnitsTest::test();

    //double f = searchStationaryPoint(XCubeFunctor(), 10, 128);

    // say electron whizzes around hydrogen atom in circular orbit. by varying the orbit radius and the electron speed try to find the
    // parameters with stationary action
    MyUnits<T> fCurOrbitRadius = MyUnits<T>::angstrom();
    MyUnits<T> fCurSpeed = MyUnits<double>::meter() * 2000000 / MyUnits<double>::second();
    for (NvU32 u = 0; u < 128; ++u)
    {
        fCurSpeed = MyUnits<T>(searchStationaryPoint(SpeedFunctor(fCurOrbitRadius), fCurSpeed.m_value, 128));
        fCurOrbitRadius = MyUnits<T>(searchStationaryPoint<OrbitFunctor>(OrbitFunctor(fCurSpeed), fCurOrbitRadius.m_value, 128));
    }
}
