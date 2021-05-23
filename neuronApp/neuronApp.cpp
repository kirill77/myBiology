#define NOMINMAX
#include <chrono>
#include <easy3d/viewer/viewer.h>
#include <easy3d/renderer/drawable_points.h>
#include <easy3d/renderer/renderer.h>
#include <easy3d/fileio/resources.h>
#include <easy3d/util/logging.h>
#include <easy3d/util/file_system.h>
#include <easy3d/renderer/text_renderer.h>
#include <3rd_party/glfw/include/GLFW/glfw3.h>	// for the KEYs

#include "neuron.h"

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
        this->resize(3840, 2160);
    }

    void updateVertexBuffers()
    {
        const auto& points = m_neuron.points();
        for (NvU32 u = 0, nK = 0, nNa = 0; u < points.size(); ++u)
        {
            if (points[u].m_flags & Neuron<T>::FLAG_K_ION)
            {
                m_pKDrawable->setVertex(nK++, removeUnits(points[u].m_vPos));
                continue;
            }
            if (points[u].m_flags & Neuron<T>::FLAG_NA_ION)
            {
                m_pNaDrawable->setVertex(nNa++, removeUnits(points[u].m_vPos));
                continue;
            }
        }

        m_pKDrawable->updateVertexBuffer();
        m_pNaDrawable->updateVertexBuffer();
    }

private:
    virtual void pre_draw() override
    {
        auto curTS = std::chrono::high_resolution_clock::now();
        if (!m_bIsFirstDraw)
        {
            auto secondsElapsed = std::chrono::duration_cast<std::chrono::duration<double>>(curTS - m_prevDrawTS);
            m_neuron.makeTimeStep(MyUnits<double>::nanoSecond());
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
    virtual void draw() const override
    {
        Viewer::draw();

        const float font_size = 28.0f + font_size_delta_;
        float x = 50.0f;
        float y = 80.0f;

        const int num_fonts = texter_->num_fonts();
        const float font_height = texter_->font_height(font_size);

        texter_->draw("hello, world",
            x * dpi_scaling(), y * dpi_scaling(), font_size, TextRenderer::Align(alignment_), 0, vec3(0, 0, 0),
            line_spacing_, upper_left_);
    }
    bool key_press_event(int key, int modifiers) override
    {
        float spacing = 0;
        bool kerning = false;
        switch (key)
        {
        case GLFW_KEY_MINUS:
                font_size_delta_ = std::max(font_size_delta_ - 1.0f, -20.0f);
                break;
        case GLFW_KEY_EQUAL:
                font_size_delta_ = std::min(font_size_delta_ + 1.0f, 250.0f);
                break;
        case GLFW_KEY_COMMA:
                spacing = texter_->character_spacing();
                texter_->set_character_spacing(std::max(spacing - 0.5f, 0.0f));
                break;
        case GLFW_KEY_PERIOD:
                spacing = texter_->character_spacing();
                texter_->set_character_spacing(std::min(spacing + 0.5f, 50.0f));
                break;
        case GLFW_KEY_DOWN:
                line_spacing_ = std::max(line_spacing_ - 0.1f, -1.0f);
                break;
        case GLFW_KEY_UP:
                line_spacing_ = std::min(line_spacing_ + 0.1f, 2.0f);
                break;
        case GLFW_KEY_L:
                alignment_ = TextRenderer::ALIGN_LEFT;
                break;
        case GLFW_KEY_C:
                alignment_ = TextRenderer::ALIGN_CENTER;
                break;
        case GLFW_KEY_R:
                alignment_ = TextRenderer::ALIGN_RIGHT;
                break;
        case GLFW_KEY_O:
                upper_left_ = !upper_left_;
                break;
        case GLFW_KEY_SPACE:
                kerning = texter_->kerning();
                texter_->set_kerning(!kerning);
                break;
        default:
                return Viewer::key_press_event(key, modifiers);
        }
        return true;
    }

    bool m_bIsFirstDraw = true;
    std::chrono::high_resolution_clock::time_point m_prevDrawTS;
    MyPointsDrawable *m_pKDrawable = nullptr, *m_pNaDrawable = nullptr; // pointers owned by viewer
    Neuron<T> m_neuron;
    Viewer* m_pViewer = nullptr;

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