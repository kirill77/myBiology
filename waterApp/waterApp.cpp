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

using namespace easy3d;

inline static vec3 toVec3(const rtvector<MyUnits<double>, 3>& v)
{
    return vec3((float)v[0].m_value, (float)v[1].m_value, (float)v[2].m_value);
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

static vec3 getBoxVertex(const BBox3<MyUnits<double>>& bbox, NvU32 u)
{
    vec3 r;
    for (NvU32 uBit = 0; uBit < 3; ++uBit)
    {
        r[uBit] = (float)removeUnits((u & (1 << uBit)) ? bbox.m_vMin[uBit] : bbox.m_vMax[uBit]);
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

template <NvU32 LOGN>
struct MyFilter
{
    MyFilter() { memset(this, 0, sizeof(*this)); }

    void addValue(double f)
    {
        NvU32 u = (m_nValues++) & MASK;
        m_fSum -= m_fValues[u];
        m_fValues[u] = f;
        m_fSum += f;
        if (u % 1024 == 0)
        {
            resetSum();
        }
    }
    double getAverage() const
    {
        return m_nValues == 0 ? 0 : m_fSum / std::min(m_nValues, N);
    }

private:
    void resetSum()
    {
        m_fSum = m_fValues[0];
        for (NvU32 u = 1; u < N; ++u) m_fSum += m_fValues[u];
    }
    static const NvU32 N = (1 << LOGN);
    static const NvU32 MASK = N - 1;
    double m_fSum;
    double m_fValues[N];
    NvU32 m_nValues;
};

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
    }

    void updateVertexBuffers()
    {
        // update vertex buffer for atom drawables
        const auto& atoms = m_water.points();
        for (NvU32 u = 0, nO = 0, nH = 0; u < atoms.size(); ++u)
        {
            const auto& atom = atoms[u];
            if (atom.m_nProtons == 8)
            {
                m_pODrawable->setVertex(nO++, atom.m_vPos[0]);
                continue;
            }
            if (atom.m_nProtons == 1)
            {
                m_pHDrawable->setVertex(nH++, atom.m_vPos[0]);
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
            const auto &bbox = m_water.getBoundingBox();

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
        const auto& forces = m_water.getForces();
        m_pBondPoints.resize(0);
        for (NvU32 u = 0; u < forces.size(); ++u)
        {
            const auto& force = forces[u];
            // all covalent bonds are supposed to be in the beginning of the array
            if (!force.isCovalentBond())
                break;
            const auto& atom1 = atoms[force.getAtom1Index()];
            const auto& atom2 = atoms[force.getAtom2Index()];
            m_pBondPoints.push_back(toVec3(atom1.m_vPos[0]));
            m_pBondPoints.push_back(toVec3(atom2.m_vPos[0]));
        }
        m_pBondsDrawable->update_vertex_buffer(m_pBondPoints);
    }

private:
    virtual void pre_draw() override
    {
        auto curTS = std::chrono::high_resolution_clock::now();
        if (!m_bIsFirstDraw)
        {
            auto secondsElapsed = std::chrono::duration_cast<std::chrono::duration<double>>(curTS - m_prevDrawTS);
            m_water.makeTimeStep();
        }
        m_prevDrawTS = curTS;
        updateVertexBuffers();
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
        float x = 50.0f;
        float y = 80.0f;

        const int num_fonts = texter_->num_fonts();
        const float font_height = texter_->font_height(font_size);

        char sBuffer[32];
        m_fTemp.addValue(m_water.evalTemperature().toCelcius());
        double fAverageTempC = m_fTemp.getAverage();
        sprintf_s(sBuffer, "T(C): %.1lf", fAverageTempC);
        texter_->draw(sBuffer,
            x * dpi_scaling(), y * dpi_scaling(), font_size, TextRenderer::Align(alignment_), 1, vec3(0, 0, 0),
            line_spacing_, upper_left_);
        x += 200;
        m_fPressure.addValue(m_water.evalPressure().toAtmospheres());
        sprintf_s(sBuffer, "P(atm): %.1lf", m_fPressure.getAverage());
        texter_->draw(sBuffer,
            x * dpi_scaling(), y * dpi_scaling(), font_size, TextRenderer::Align(alignment_), 1, vec3(0, 0, 0),
            line_spacing_, upper_left_);
        x += 200;
        sprintf_s(sBuffer, "Tstep(fs): %.4lf", m_water.getCurTimeStep().toFemtoseconds());
        texter_->draw(sBuffer,
            x * dpi_scaling(), y * dpi_scaling(), font_size, TextRenderer::Align(alignment_), 1, vec3(0, 0, 0),
            line_spacing_, upper_left_);
    }

    bool m_bIsFirstDraw = true;
    std::chrono::high_resolution_clock::time_point m_prevDrawTS;
    MyPointsDrawable *m_pODrawable = nullptr, *m_pHDrawable = nullptr; // pointers owned by viewer
    std::vector<vec3> m_pBondPoints; // to avoid allocating every time - we just keep this array around
    LinesDrawable* m_pBoxDrawable = nullptr, *m_pBondsDrawable = nullptr;
    Water<T> m_water;
    Viewer* m_pViewer = nullptr;

    MyFilter<7> m_fTemp, m_fPressure;

    // for text rendering:
    float font_size_delta_ = -20;
    float line_spacing_ = 0;
    int alignment_ = TextRenderer::ALIGN_CENTER;
    bool upper_left_ = true;
};

int main(int argc, char** argv)
{
    DistributionsTest::test();
    MyUnitsTest::test();

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
    MyViewer<double> myModel;
    auto result = myModel.run();

    return result;
}