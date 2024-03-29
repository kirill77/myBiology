#define NOMINMAX
#include <chrono>
#include <easy3d/viewer/viewer.h>
#include <easy3d/renderer/drawable_points.h>
#include <easy3d/renderer/drawable_lines.h>
#include <easy3d/renderer/renderer.h>
#include <easy3d/fileio/resources.h>
#include <easy3d/util/logging.h>
#include <easy3d/util/file_system.h>
#include <easy3d/renderer/text_renderer.h>
#include <3rd_party/glfw/include/GLFW/glfw3.h>	// for the KEYs

#include "water.h"
#include "neural/neuralTest.h"
#include "MonteCarlo/distributions.h"

using namespace easy3d;

template <class T>
inline static vec3 toVec3(const rtvector<MyUnits<T>, 3>& v)
{
    return vec3((float)v[0], (float)v[1], (float)v[2]);
}

struct MyPointsDrawable : public PointsDrawable
{
    MyPointsDrawable()
    {
        set_impostor_type(PointsDrawable::SPHERE);
        set_point_size(20);
    }
    template <class T>
    void setVertex(NvU32 index, const rtvector<MyUnits<T>, 3>& v)
    {
        if (m_points.size() <= index)
        {
            m_points.resize(index + 1);
        }
        m_points[index] = toVec3(v);
    }
    void updateVertexBuffer()
    {
        update_vertex_buffer(m_points);
    }
private:
    std::vector<vec3> m_points;
};

template <class T>
static vec3 getBoxVertex(const BBox3<MyUnits<T>>& bbox, NvU32 u)
{
    vec3 r;
    for (NvU32 uBit = 0; uBit < 3; ++uBit)
    {
        r[uBit] = (float)((u & (1 << uBit)) ? bbox.m_vMin[uBit] : bbox.m_vMax[uBit]);
    }
    return r;
}
static NvU32 countMatchingCoords(const vec3& v1, const vec3& v2)
{
    NvU32 n = 0;
    for (NvU32 u = 0; u < 3; ++u)
    {
        if (v1[u] == v2[u]) ++n;
    }
    nvAssert(n != 3);
    return n;
}

template <class T>
struct MyViewer : public Viewer
{
    MyViewer() : Viewer("water")
    {
        m_pODrawable = new MyPointsDrawable;
        m_pODrawable->set_uniform_coloring(vec4(1.0f, 0.0f, 0.0f, 1.0f));  // r, g, b, a
        m_pHDrawable = new MyPointsDrawable;
        m_pHDrawable->set_uniform_coloring(vec4(0.0f, 1.0f, 0.0f, 1.0f));  // r, g, b, a

        Viewer::add_drawable(m_pODrawable);
        Viewer::add_drawable(m_pHDrawable);

        m_pBondsDrawable = new LinesDrawable;
        Viewer::add_drawable(m_pBondsDrawable);

        Viewer::resize(3840, 2160);

        m_water.init();
    }

    virtual bool key_press_event(int key, int modifiers)
    {
        if (key == GLFW_KEY_S && modifiers == 0)
        {
            m_doSimulation = !m_doSimulation;
        }
        m_water.notifyKeyPress(key, modifiers);
        return false;
    }
    virtual bool mouse_press_event(int x, int y, int button, int modifiers)
    {
        bool bFound = false;
        vec3 vPickedPos = point_under_pixel(x, y, bFound);
        if (bFound)
        {
            double fDistSqrMin = 1e38;
            const std::vector<Atom<T>>& atoms = m_water.getSimContext().m_atoms;
            // find the closest atom to picked point
            for (NvU32 uAtom = 0; uAtom < atoms.size(); ++uAtom)
            {
                vec3 vAtomPos = toVec3(atoms[uAtom].m_vPos);
                vec3 v = vAtomPos - vPickedPos;
                double fDistSqr = dot(v, v);
                if (fDistSqr < fDistSqrMin)
                {
                    fDistSqrMin = fDistSqr;
                    m_pickedAtomIndex = uAtom;
                }
            }
        }
        else
        {
            m_pickedAtomIndex = -1;
        }
        return false;
    }

    void updateVertexBuffers()
    {
        // update vertex buffer for atom drawables
        const std::vector<Atom<T>>& atoms = m_water.getSimContext().m_atoms;
        for (NvU32 u = 0, nO = 0, nH = 0; u < atoms.size(); ++u)
        {
            const auto& atom = atoms[u];
            if (atom.m_nProtons == 8)
            {
                m_pODrawable->setVertex(nO++, atom.m_vPos);
                continue;
            }
            if (atom.m_nProtons == 1)
            {
                m_pHDrawable->setVertex(nH++, atom.m_vPos);
                continue;
            }
            nvAssert(false);
        }

        m_pODrawable->updateVertexBuffer();
        m_pHDrawable->updateVertexBuffer();

        // create drawable for bounding box (done only once)
        if (!m_pBoxDrawable)
        {
            m_pBoxDrawable = new LinesDrawable;
            const auto &bbox = m_water.getSimContext().m_bBox;

            std::vector<vec3> pBuffer;
            for (NvU32 u1 = 0; u1 < 8; ++u1)
            {
                vec3 v1 = getBoxVertex(bbox, u1);
                for (NvU32 u2 = u1 + 1; u2 < 8; ++u2)
                {
                    vec3 v2 = getBoxVertex(bbox, u2);
                    if (countMatchingCoords(v1, v2) == 2)
                    {
                        pBuffer.push_back(v1);
                        pBuffer.push_back(v2);
                    }
                }
            }
            m_pBoxDrawable->update_vertex_buffer(pBuffer);

            // Draw the lines of the bounding box in blue.
            m_pBoxDrawable->set_uniform_coloring(vec4(0.0f, 0.0f, 1.0f, 1.0f));    // r, g, b, a
            // Draw the lines with a width of 5 pixels.
            m_pBoxDrawable->set_line_width(5.0f);

            Viewer::add_drawable(m_pBoxDrawable);
        }

        // update vertex buffer for bonds drawable
        const auto& forces = m_water.getSimContext().m_forces;
        const auto& boxWrapper = m_water.getSimContext().m_bBox;
        m_pBondPoints.resize(0);
        for (NvU32 uForce = 0; uForce < forces.size(); ++uForce)
        {
            if (!forces.isValid(uForce))
                continue;
            const Force<T> &force = forces.accessForceByIndex(uForce);
#if 1
            if (!force.isCovalentBond())
                continue;
#else
            if (force.getAtom1Index() != m_pickedAtomIndex && force.getAtom2Index() != m_pickedAtomIndex)
                continue;
#endif

            const auto& atom1 = atoms[force.getAtom1Index()];
            const auto& atom2 = atoms[force.getAtom2Index()];

            vec3 atom1Pos = toVec3(atom1.m_vPos);
            vec3 atom2Pos = toVec3(atom2.m_vPos);
            vec3 vDir1 = atom1Pos - atom2Pos;
            vec3 vDir2 = toVec3(boxWrapper.computeDir(atom1.m_vPos, atom2.m_vPos));
            // if vDir1 is large - this means dir wraps around the bounding box - we have to draw two lines
            if (dot(vDir1, vDir1) > 2 * dot(vDir2, vDir2))
            {
                m_pBondPoints.push_back(atom1Pos);
                m_pBondPoints.push_back(atom1Pos - vDir2);
                m_pBondPoints.push_back(atom2Pos);
                m_pBondPoints.push_back(atom2Pos + vDir2);
            }
            else
            {
                m_pBondPoints.push_back(atom1Pos);
                m_pBondPoints.push_back(atom2Pos);
            }
        }
        m_pBondsDrawable->update_vertex_buffer(m_pBondPoints);
    }

private:
    virtual void pre_draw() override
    {
        auto curTS = std::chrono::high_resolution_clock::now();
        if (!m_bIsFirstDraw)
        {
            if (m_doSimulation)
            {
                m_water.makeTimeStep();
            }
        }

        updateVertexBuffers();

        m_prevDrawTS = curTS;
        if (m_bIsFirstDraw)
        {
            this->fit_screen();
        }
        Viewer::pre_draw();
        m_bIsFirstDraw = false;
    }
    virtual void draw() override
    {
        Viewer::draw();

        const float font_size = 40.0f + font_size_delta_;
        const float fLeftBoundary = 50.0f, fTopBoundary = 80.0f;
        float x = fLeftBoundary;
        float y = fTopBoundary;

        const int num_fonts = texter_->num_fonts();
        const float font_height = texter_->font_height(font_size);

        char sBuffer[64];

        auto& network = m_water.accessNeuralNetwork();
#if 0 // commented for now - but good code
        if (network.hasEnoughData())
        {
            std::vector<TensorRef> inputs, wantedOutputs;
            double fLossValue = network.train(64, inputs, wantedOutputs);
            sprintf_s(sBuffer, "Loss: %lf", fLossValue);
            texter_->draw(sBuffer,
                x * dpi_scaling(), y * dpi_scaling(), font_size, TextRenderer::Align(alignment_), 1, vec3(0, 0, 0),
                line_spacing_, upper_left_);
            return;
        }
#endif

        MyUnits<T> fFilteredAverageKin = m_water.getFilteredAverageKin();
        double fAverageTempC = MyUnits1<double>::toCelcius(fFilteredAverageKin);
        sprintf_s(sBuffer, "T(C): %.1lf", fAverageTempC);
        texter_->draw(sBuffer,
            x * dpi_scaling(), y * dpi_scaling(), font_size, TextRenderer::Align(alignment_), 1, vec3(0, 0, 0),
            line_spacing_, upper_left_);
        x += 200;
        double fTotalKin = fFilteredAverageKin * (double)m_water.getSimContext().m_atoms.size();
        double fPressure = MyUnits1<double>::evalPressure(fTotalKin, m_water.getSimContext().m_bBox.evalVolume());
        sprintf_s(sBuffer, "P(atm): %.1lf", MyUnits1<double>::toAtmospheres(fPressure));
        texter_->draw(sBuffer,
            x * dpi_scaling(), y * dpi_scaling(), font_size, TextRenderer::Align(alignment_), 1, vec3(0, 0, 0),
            line_spacing_, upper_left_);
        x += 200;
        sprintf_s(sBuffer, "Tstep(fs): %.4lf, nForces=%d", MyUnits1<double>::toFemtoseconds(m_water.getCurTimeStep()), (NvU32)m_water.getSimContext().m_forces.size());
        texter_->draw(sBuffer,
            x * dpi_scaling(), y * dpi_scaling(), font_size, TextRenderer::Align(alignment_), 1, vec3(0, 0, 0),
            line_spacing_, upper_left_);

        x = fLeftBoundary;
        y += 40;
        sprintf_s(sBuffer, "NN StoredSimSteps: %d", network.getNStoredSimSteps());
        texter_->draw(sBuffer,
            x * dpi_scaling(), y * dpi_scaling(), font_size, TextRenderer::Align(alignment_), 1, vec3(0, 0, 0),
            line_spacing_, upper_left_);
        y += 40;

        y = m_water.drawText(texter_, fLeftBoundary, y, dpi_scaling(), font_size);

        if (m_pickedAtomIndex != -1)
        {
            x += 400;
            sprintf_s(sBuffer, "PickedAtom: %d", m_pickedAtomIndex);
            texter_->draw(sBuffer,
                x * dpi_scaling(), y * dpi_scaling(), font_size, TextRenderer::Align(alignment_), 1, vec3(0, 0, 0),
                line_spacing_, upper_left_);
        }
    }

    bool m_bIsFirstDraw = true;
    std::chrono::high_resolution_clock::time_point m_prevDrawTS;
    MyPointsDrawable *m_pODrawable = nullptr, *m_pHDrawable = nullptr; // pointers owned by viewer
    std::vector<vec3> m_pBondPoints; // to avoid allocating every time - we just keep this array around
    LinesDrawable* m_pBoxDrawable = nullptr, *m_pBondsDrawable = nullptr;
    Water<T> m_water;
    Viewer* m_pViewer = nullptr;

    bool m_doSimulation = true;
    NvU32 m_pickedAtomIndex = -1;

    // for text rendering:
    float font_size_delta_ = -20;
    float line_spacing_ = 0;
    int alignment_ = TextRenderer::ALIGN_CENTER;
    bool upper_left_ = true;
};

NvU32 g_debugCount = 0;

int main(int argc, char** argv)
{
    DistributionsTest::test();
    MyUnitsTest::test();
    NeuralTest::test();

    // initialize logging
    logging::initialize();

    // find directory with resources
    std::string dir = file_system::executable_directory();
    for (; ; )
    {
        if (file_system::is_directory(dir + "/Easy3D"))
        {
            _chdir((dir + "/Easy3D/resources").c_str());
            break;
        }
        dir = file_system::parent_directory(dir);
    }

    // Create the default Easy3D viewer.
    // Note: a viewer must be created before creating any drawables.
    MyViewer<float> myModel;
    auto result = myModel.run();

    return result;
}