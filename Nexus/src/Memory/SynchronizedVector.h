#pragma once
#include "Memory/Host/Vector.h"
#include "Memory/Device/DeviceVector.h"

template<typename THost, typename TDevice = THost>
class SynchronizedVector
{
public:
	SynchronizedVector() = default;

	SynchronizedVector(const Vector<THost>& hostVector)
		: m_HostVector(hostVector)
	{

	}

private:
	Vector<THost> m_HostVector;
	DeviceVector<THost, TDevice> m_DeviceVector;
};