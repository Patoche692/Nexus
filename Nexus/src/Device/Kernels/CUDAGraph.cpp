#include "CUDAGraph.h"

#include "Utils/Utils.h"

CUDAGraph::CUDAGraph()
{
	checkCudaErrors(cudaGraphCreate(&m_Graph, 0));
	checkCudaErrors(cudaStreamCreate(&m_Stream));
	BuildGraph();
}

CUDAGraph::~CUDAGraph()
{
	checkCudaErrors(cudaGraphExecDestroy(m_GraphExec));
	checkCudaErrors(cudaStreamDestroy(m_Stream));
	checkCudaErrors(cudaGraphDestroy(m_Graph));
}

void CUDAGraph::Reset()
{
	checkCudaErrors(cudaGraphDestroy(m_Graph));
	checkCudaErrors(cudaGraphCreate(&m_Graph, 0));
}

void CUDAGraph::BuildGraph()
{
	checkCudaErrors(cudaGraphInstantiate(&m_GraphExec, m_Graph, 0));
}

void CUDAGraph::Execute()
{
	checkCudaErrors(cudaDeviceSynchronize());
	checkCudaErrors(cudaGraphLaunch(m_GraphExec, 0));
	checkCudaErrors(cudaDeviceSynchronize());
}

void CUDAGraph::AddKernelNode(CUDAKernel& kernel)
{
	cudaGraphNode_t kernelNode;
	cudaKernelNodeParams nodeParams;
	nodeParams.blockDim = kernel.GetBlockSize();
	nodeParams.gridDim = kernel.GetGridSize();
	nodeParams.extra = nullptr;
	nodeParams.kernelParams = kernel.GetLaunchParameters();
	nodeParams.func = kernel.GetFunction();
	nodeParams.sharedMemBytes = 0;

	checkCudaErrors(cudaGraphAddKernelNode(&kernelNode, m_Graph, nullptr, 0, &nodeParams));
}
