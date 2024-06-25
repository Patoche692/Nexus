#pragma once
#include "Allocators/DeviceAllocator.h"
#include "CudaMemory.h"

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
		m_Allocator->Free(m_Data);
		//::operator delete(m_Data, m_Capacity * sizeof(T));
	}

	void PushBack(const T& value)
	{
		if (m_Size >= m_Capacity)
			Realloc(m_Capacity + m_Capacity / 2);

		CudaMemory::Copy<T>(m_Data + m_Size, &value, 1, cudaMemcpyHostToDevice);
		m_Size++;
	}

	void PopBack()
	{
		assert(m_Size > 0);
		m_Size--;
		//m_Data[--m_Size].~T();
	}

	void Clear()
	{
		//if (!std::is_trivially_destructible_v<T>)
		//{
		//	for (size_t i = 0; i < m_Size; i++)
		//		m_Data[i].~T();
		//}

		m_Size = 0;
	}

	size_t Size() const { return m_Size; }

	T* Data() const { return m_Data; }

	const T& operator[] (size_t index) const 
	{
		assert(index > 0 && index < m_Size);
		return m_Data[index]; 
	}

	T& operator[] (size_t index)
	{
		assert(index > 0 && index < m_Size);
		return m_Data[index]; 
	}

private:
	Realloc(size_t newCapacity)
	{
		T* newBlock = (T*)m_Allocator->Alloc(newCapacity * sizeof(T));

		size_t size = std::min(newCapacity, m_Size);

		CudaMemory::Copy<T>(newBlock, m_Data, size, cudaMemcpyDeviceToDevice);

		for (size_t i = 0; i < size; i++)
			m_Data[i].~T();

		m_Allocator->Free(m_Data);
		m_Data = newBlock;
		m_Capacity = newCapacity;
	}

private:
	T* m_Data = nullptr;
	DeviceAllocator* m_Allocator = nullptr;

	size_t m_Size = 0;
	size_t m_Capacity = 0;
};
