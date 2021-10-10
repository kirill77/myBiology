#include <basics/myunits.h>

int main()
{
    MyUnitsTest::test();

#if 0
    // say electron wizzes around hydrogen atom in circular orbit
    MyUnits<double> fOrbitRadius = MyUnits<double>::angstrom();
    MyUnits<double> fSpeed = 2000000 * MyUnits<double>::meter() / MyUnits<double>::second();

    MyUnits<T> fKineticEnergy =  * fSpeed * fSpeed / 2;
    MyUnits<T> fPotentialEnergy = MyUnits<T>::dalton() * 
#endif
}
