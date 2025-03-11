// transport.cpp : Defines the functions for the static library.
//

#include "pch.h"
#include "framework.h"
#include "transport.h"

Connection::Status Connection::sendData(void* pBytes, size_t nBytes)
{
    return SUCCESS;
}
Connection::Status Connection::receiveData(void* pBytes, size_t nBytes)
{
    return SUCCESS;
}
std::shared_ptr<Connection> Transport::waitForClient()
{
    return std::make_shared<Connection>();
}
std::vector<std::shared_ptr<Connection>> Transport::findAllServers()
{
    std::vector<std::shared_ptr<Connection>> r;
    return r;
}