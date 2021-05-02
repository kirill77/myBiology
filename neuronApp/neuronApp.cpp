#define NOMINMAX
#include <easy3d/viewer/viewer.h>
#include <easy3d/renderer/drawable_points.h>
#include <easy3d/renderer/renderer.h>
#include <easy3d/fileio/resources.h>
#include <easy3d/util/logging.h>
#include <easy3d/util/file_system.h>

#include "neuron.h"

using namespace easy3d;

struct MyPointsDrawable : public PointsDrawable
{
    MyPointsDrawable()
    {
        set_impostor_type(PointsDrawable::SPHERE);
        set_point_size(10);
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
    std::vector<vec3> m_points;
};

template <class T>
struct MyViewer : public Viewer
{
    MyViewer() : Viewer("neuron")
    {
        m_pKDrawable = new MyPointsDrawable;
        m_pKDrawable->set_uniform_coloring(vec4(1.0f, 0.0f, 0.0f, 1.0f));  // r, g, b, a
        m_pNaDrawable = new MyPointsDrawable;
        m_pNaDrawable->set_uniform_coloring(vec4(0.0f, 1.0f, 0.0f, 1.0f));  // r, g, b, a

        add_drawable(m_pKDrawable);
        add_drawable(m_pNaDrawable);
    }

    void updateVertexBuffers()
    {
        const auto& points = m_neuron.points();
        for (NvU32 u = 0, nK = 0, nNa = 0; u < points.size(); ++u)
        {
            if (points[u].m_flags & Neuron<T>::FLAG_K_ION)
            {
                m_pKDrawable->setVertex(nK++, points[u].m_pos);
                continue;
            }
            if (points[u].m_flags & Neuron<T>::FLAG_NA_ION)
            {
                m_pNaDrawable->setVertex(nNa++, points[u].m_pos);
                continue;
            }
        }

        m_pKDrawable->updateVertexBuffer();
        m_pNaDrawable->updateVertexBuffer();
    }

private:
    virtual void pre_draw() override
    {
        m_neuron.makeTimeStep(0.01);
        updateVertexBuffers();
        Viewer::pre_draw();
    }
    MyPointsDrawable *m_pKDrawable = nullptr, *m_pNaDrawable = nullptr; // pointers owned by viewer
    Neuron<T> m_neuron;
    Viewer* m_pViewer = nullptr;
};

int main(int argc, char** argv)
{
    DistributionsTest::test();

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