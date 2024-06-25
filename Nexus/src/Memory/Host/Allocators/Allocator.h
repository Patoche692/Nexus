#pragma once
#include <iostream>

class Allocator
{
public:
	Allocator() = default;

protected:
	virtual void* Alloc(size_t size)
	{
		return ::operator new(size);
	}
	virtual void Free(void* ptr)
	{
		::operator delete(ptr);
	}
};