#define NOMINMAX
#include <easy3d/viewer/viewer.h>
#include <easy3d/renderer/drawable_points.h>
#include <easy3d/renderer/renderer.h>
#include <easy3d/fileio/resources.h>
#include <easy3d/util/logging.h>
#include <easy3d/util/file_system.h>

#include "neuron.h"

using namespace easy3d;

template <class T>
struct MyModel : public PointsDrawable
{
    MyModel()
    {
        const auto& points = m_neuron.points();
        for (NvU32 u = 0; u < points.size(); ++u)
        {
            auto& center = points[u].m_pos;
            addVertex(center);
        }
        update_vertex_buffer(m_points);
        set_uniform_coloring(vec4(1.0f, 0.0f, 0.0f, 1.0f));  // r, g, b, a
        set_impostor_type(PointsDrawable::SPHERE);
        set_point_size(10);
    }

private:
    void addVertex(const rtvector<T, 3>& v)
    {
        vec3 _v(v[0], v[1], v[2]);
        m_points.push_back(_v);
    }
    Neuron<T> m_neuron;
    std::vector<vec3> m_points;
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
    Viewer viewer("neuron");

    MyModel<double>* pMyModel = new MyModel<double>;
    viewer.add_drawable(pMyModel);


    auto result = viewer.run();

    return result;
}