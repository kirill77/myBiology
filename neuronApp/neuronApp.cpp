#define NOMINMAX
#include <easy3d/viewer/viewer.h>
#include <easy3d/core/surface_mesh.h>
#include <easy3d/renderer/drawable_lines.h>
#include <easy3d/renderer/renderer.h>
#include <easy3d/fileio/resources.h>
#include <easy3d/util/logging.h>
#include <easy3d/util/file_system.h>

#include <simulation/ocTree.h>

using namespace easy3d;

struct MyModel : public Model
{
    MyModel()
    {
        m_points.push_back(vec3(0, 0, 0));
        m_points.push_back(vec3(1, 0, 0));
        m_points.push_back(vec3(1, 1, 0));
        m_points.push_back(vec3(0, 1, 0));
    }

    virtual std::vector<vec3>& points() override
    {
        return m_points;
    }
    virtual const std::vector<vec3>& points() const override
    {
        return m_points;
    }
    /** prints the names of all properties to an output stream (e.g., std::cout). */
    virtual void property_stats(std::ostream& output) const override
    {
    }

private:
    std::vector<vec3> m_points;
};

int main(int argc, char** argv)
{
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

    MyModel* pMyModel = new MyModel;
    // Load point cloud data from a file
    viewer.add_model(pMyModel, false);

    // Get the bounding box of the model. Then we defined the length of the
    // normal vectors to be 5% of the bounding box diagonal.
    const Box3& box = pMyModel->bounding_box();
    float length = box.diagonal() * 0.05f;

    // Create a drawable for rendering the normal vectors.
    auto drawable = pMyModel->renderer()->add_lines_drawable("normals");
    // Upload the data to the GPU.
    drawable->update_vertex_buffer(pMyModel->points());

    // We will draw the normal vectors in a uniform green color
    drawable->set_uniform_coloring(vec4(1.0f, 0.0f, 0.0f, 1.0f));

    // Set the line width
    drawable->set_line_width(3.0f);

    // Run the viewer
    return viewer.run();
}