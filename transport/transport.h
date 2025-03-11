#pragma once

#include <vector>
#include <memory>

#define TRANSPORT_LIBRARY_VERSION 1 // increment every time the interface changes

struct Connection
{
    enum Status { SUCCESS, FAILURE };

    Status sendData(void* pBytes, size_t nBytes);
    Status receiveData(void* pBytes, size_t nBytes);
};

struct Transport
{
    std::shared_ptr<Connection> waitForClient();
    std::vector<std::shared_ptr<Connection>> findAllServers();
};