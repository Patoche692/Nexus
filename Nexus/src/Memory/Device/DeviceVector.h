#pragma once
#include "Allocators/DeviceAllocator.h"
#include "CudaMemory.h"
#include "DeviceInstance.h"

/*
 * Device vector for trivial types (does not handle copy / move constructors
 * since it is not possible to construct an object on the GPU from the host with CUDA).
 * The objects will just be copied over to the device, non trivial conversions
 * from a CPU to GPU instance must therefore be handled by the user.
 * 
 * If the specified type implements a static T* ToDevice() method
 */
template<typename T>
class DeviceVector
{
public:
	DeviceVector()
	{
		Realloc(2);
	}

	DeviceVector(size_t size, DeviceAllocator* allocator = nullptr)
		:m_Allocator(allocator)
	{
		Realloc(size);
	}

	~DeviceVector()
	{
		Clear();
		DeviceAllocator<T>::Free(m_Allocator, m_Data);
	}

	void PushBack(const T& value)
	{
		if (m_Size >= m_Capacity)
			Realloc(m_Capacity + m_Capacity / 2);

		if (is_trivially_copyable_to_device<T>::value)
			CudaMemory::Copy<T>(m_Data + m_Size, &value, 1, cudaMemcpyHostToDevice);
		else
		{
			T deviceInstance = T::ToDevice(value);
			CudaMemory::Copy<T>(m_Data + m_Size, &deviceInstance, 1, cudaMemcpyHostToDevice);
		}
		m_Size++;
	}

	void PopBack()
	{
		assert(m_Size > 0);
		m_Size--;
		if (!is_trivially_copyable_to_device<T>::value)
			T::FreeFromDevice(m_Data + m_Size);
	}

	void Clear()
	{
		if (!is_trivially_copyable_to_device<T>::value)
		{
			for (size_t i = 0; i < m_Size; i++)
				T::FreeFromDevice(m_Data + i);
		}

		m_Size = 0;
	}

	size_t Size() const { return m_Size; }

	T* Data() const { return m_Data; }

	DeviceInstance<T> operator[] (size_t index)
	{
		assert(index > 0 && index < m_Size);
		return DeviceInstance<T>(m_Data + index);
	}

private:
	Realloc(size_t newCapacity)
	{
		T* newBlock = DeviceAllocator<T>::Alloc(m_Allocator, newCapacity);

		size_t size = std::min(newCapacity, m_Size);

		CudaMemory::Copy<T>(newBlock, m_Data, size, cudaMemcpyDeviceToDevice);

		if (!is_trivially_copyable_to_device<T>::value)
		{
			for (size_t i = 0; i < size; i++)
				T::FreeFromDevice(m_Data + i);
		}

		DeviceAllocator<T>::Free(m_Allocator, m_Data);
		m_Data = newBlock;
		m_Capacity = newCapacity;
	}

private:
	T* m_Data = nullptr;
	DeviceAllocator<T>* m_Allocator = nullptr;

	size_t m_Size = 0;
	size_t m_Capacity = 0;
};
