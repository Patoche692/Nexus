#include "Nexus.h"

class NexusEditor : public Nexus::Application {
public:
    NexusEditor() {

    }

    ~NexusEditor() {

    }

};

Nexus::Application *Nexus::CreateApplication() {
    return new NexusEditor();
}