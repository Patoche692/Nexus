#pragma once

namespace Nexus {

    class __attribute__((visibility("default"))) Application {

    public:
        Application();
        virtual ~Application();

        void Run();
    };

    Application *CreateApplication();

}