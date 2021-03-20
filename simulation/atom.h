#pragma once

struct Atom : public SimLevel
{
   Atom(SimLevel *pParent, NvU32 nProtons, NvU32 nNeutrons) : m_pParent(pParent), m_nProtons(nProtons), m_nNeutrons(nNeutrons) { }

private:
   SimLevel *m_pParent = nullptr;
   NvU32 m_nProtons = 0, m_nNeutrons = 0;
};