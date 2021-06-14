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

struct MyPointsDrawable : public PointsDrawable
{
    MyPointsDrawable()
    {
        set_impostor_type(PointsDrawable::SPHERE);
        set_point_size(20);
    }
    template <class T>
    void setVertex(NvU32 index, const rtvector<T, 3>& v)
    {
        if (m_points.size() <= index)
        {
            m_points.resize(index + 1);
        }
        m_points[index] = vec3((float)v[0], (float)v[1], (float)v[2]);
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

        Viewer::resize(3840, 2160);
    }

    void updateVertexBuffers()
    {
        const auto& points = m_water.points();
        for (NvU32 u = 0, nO = 0, nH = 0; u < points.size(); ++u)
        {
            if (points[u].m_nProtons == 8)
            {
                m_pODrawable->setVertex(nO++, removeUnits(points[u].m_vPos));
                continue;
            }
            if (points[u].m_nProtons == 1)
            {
                m_pHDrawable->setVertex(nH++, removeUnits(points[u].m_vPos));
                continue;
            }
            nvAssert(false);
        }

        m_pODrawable->updateVertexBuffer();
        m_pHDrawable->updateVertexBuffer();

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

    bool m_bIsFirstDraw = true;
    std::chrono::high_resolution_clock::time_point m_prevDrawTS;
    MyPointsDrawable *m_pODrawable = nullptr, *m_pHDrawable = nullptr; // pointers owned by viewer
    LinesDrawable* m_pBoxDrawable = nullptr;
    Water<T> m_water;
    Viewer* m_pViewer = nullptr;
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