#pragma once
#include "Application.h"

extern Nexus::Application *Nexus::CreateApplication();

int main(int argc, char *argv[]) {
    Nexus::Application *application = Nexus::CreateApplication();
    application->Run();
    delete application;
}