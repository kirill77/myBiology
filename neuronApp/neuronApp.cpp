#define NOMINMAX
#include <easy3d/viewer/viewer.h>
#include <easy3d/core/surface_mesh.h>
#include <easy3d/renderer/drawable_lines.h>
#include <easy3d/renderer/renderer.h>
#include <easy3d/fileio/resources.h>
#include <easy3d/util/logging.h>
#include <easy3d/util/file_system.h>

#include "neuron.h"

using namespace easy3d;

template <class T>
struct MyModel : public SurfaceMesh
{
    MyModel()
    {
        const auto& points = m_neuron.points();
        for (NvU32 u = 0; u < points.size(); ++u)
        {
            auto& center = points[u].m_pos;

            auto nVerts = this->n_vertices();

            const T POINT_SIZE = (T)0.01;

            addVertex(center + rtvector<T, 3>({ 0,  0,  1 }) * POINT_SIZE);
            addVertex(center + rtvector<T, 3>({ 1,  0, -1 }) * POINT_SIZE);
            addVertex(center + rtvector<T, 3>({-1, -1, -1 }) * POINT_SIZE);
            addVertex(center + rtvector<T, 3>({-1,  1, -1 }) * POINT_SIZE);

            SurfaceMesh::Vertex v1(nVerts);
            SurfaceMesh::Vertex v2(nVerts + 1);
            SurfaceMesh::Vertex v3(nVerts + 2);
            SurfaceMesh::Vertex v4(nVerts + 3);

            add_triangle(v1, v2, v3);
            add_triangle(v1, v3, v4);
            add_triangle(v1, v4, v2);
            add_triangle(v2, v3, v4);
        }

        int i = 0;
        ++i;
    }

    /** prints the names of all properties to an output stream (e.g., std::cout). */
    virtual void property_stats(std::ostream& output) const override
    {
    }

private:
    void addVertex(const rtvector<T, 3>& v)
    {
        vec3 _v(v[0], v[1], v[2]);
        add_vertex(_v);
    }
    Neuron<T> m_neuron;
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
    // Load point cloud data from a file
    viewer.add_model(pMyModel, false);

    // Get the bounding box of the model. Then we defined the length of the
    // normal vectors to be 5% of the bounding box diagonal.
    const Box3& box = pMyModel->bounding_box();
    float length = box.diagonal() * 0.05f;

    // Create a drawable for rendering the normal vectors.
    auto drawable = pMyModel->renderer()->add_triangles_drawable("faces");

    // Run the viewer
    return viewer.run();
}