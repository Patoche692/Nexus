#pragma once

#include <vector>
#include <cuda_runtime_api.h>

class CUDAKernel
{
public:
	CUDAKernel() = default;
	CUDAKernel(void* function, dim3 gridsize, dim3 blockSize, std::vector<void*> params)
		: m_Function(function), m_GridSize(gridsize), m_BlockSize(blockSize), m_LaunchParameters(params) {}

	virtual void Init() { }
	virtual void Launch() { }

	const dim3 GetBlockSize() const { return m_BlockSize; }
	const dim3 GetGridSize() const { return m_GridSize; }
	void** GetLaunchParameters() { return m_LaunchParameters.data(); }
	void* GetFunction() const { return m_Function; }

	void SetLaunchConfiguration(dim3 GridSize, dim3 BlockSize)
	{
		m_GridSize = GridSize;
		m_BlockSize = BlockSize;
	}

	void SetLaunchParameters(const std::vector<void*>& params)
	{
		m_LaunchParameters = params;
	}

private:

	void* m_Function = nullptr;
	std::vector<void*> m_LaunchParameters;

	dim3 m_GridSize = dim3(1, 1, 1);
	dim3 m_BlockSize = dim3(1, 1, 1);
};

