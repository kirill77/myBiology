#pragma once

template <class T>
struct Tensor
{
    Tensor() { }
    void init(int n, int h, int w, int c)
    {
        nvAssert(m_pDeviceMem == nullptr && m_pHostMem == nullptr); // if already allocated - can't resize
        m_n = n;
        m_h = h;
        m_w = w;
        m_c = c;
    }
    void allocateHost()
    {

    }
    void allocateDevice()
    {

    }
    void freeHost()
    {
    }
    void freeDevice()
    {

    }

    ~Tensor()
    {
        freeHost();
        freeDevice();
    }
    int n() const { return m_n; }
    int h() const { return m_h; }
    int w() const { return m_w; }
    int c() const { return m_c; }

private:
    int m_n = -1, m_h = -1, m_w = -1, m_c = -1;
    Tensor(const Tensor& other) { } // don't want copy constructor because not clear what to do with allocated memory
    T* m_pDeviceMem = nullptr;
    T* m_pHostMem = nullptr;
};