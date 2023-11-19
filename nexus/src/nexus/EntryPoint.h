#pragma once

#include "Application.h"
#include "Log.h"

extern Nexus::Application *Nexus::CreateApplication();

int main(int argc, char *argv[]) {
    Nexus::Log::Init();
    NX_CORE_WARN("Initialized log!");
    int a = 3;
    NX_INFO("Initialized log! a = {0}", a);

    Nexus::Application *application = Nexus::CreateApplication();
    application->Run();
    delete application;
}