#include "forceRandomizer.h"
#include "water.h"

void ForceRandomizer::init(PrContext<float>& c)
{
    if (m_randomDirs.size() != c.m_atoms.size())
    {
        m_randomDirs.resize(c.m_atoms.size());
    }
    double fSum = 0;
    for (NvU32 u = 0; u < m_randomDirs.size(); ++u)
    {
        // random direction on a sphere
        rtvector<float, 3> v;
        double fPhi = m_rng.generate01() * 2 * M_PI;
        v[0] = (float)sin(fPhi);
        v[1] = (float)cos(fPhi);
        v[2] = (float)m_rng.generateBetween(-1, 1);

        float fMass = c.m_atoms[u].getMass();
        m_randomDirs[u] = v * fMass; // multiply by mass to deflect all atoms by about the same extent

        // used to compute m_fAvgForce
        auto vForce = c.m_prAtoms[u].getTotalForce();
        fSum += length(vForce);
    }
    m_fAvgForce = (float)(fSum / m_randomDirs.size());
    m_fAvgForce *= 10; // fudge factor
}
void ForceRandomizer::randomize(NvU32 uAtom, PrContext<float>& c)
{
    c.m_prAtoms[uAtom].notifyForceContribution(m_randomDirs[uAtom] * m_fAvgForce);
}