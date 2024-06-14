#pragma once

class Allocator
{
public:
	Allocator() = default;

protected:
	virtual void* alloc(size_t size) = 0;
	virtual void free(void* ptr) = 0;
};