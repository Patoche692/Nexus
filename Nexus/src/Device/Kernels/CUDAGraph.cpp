#include "CUDAGraph.h"

#include "Utils/Utils.h"

CUDAGraph::CUDAGraph()
{
	checkCudaErrors(cudaGraphCreate(&m_Graph, 0));
	checkCudaErrors(cudaStreamCreate(&m_Stream));
	checkCudaErrors(cudaGraphInstantiate(&m_GraphExec, m_Graph, 0));
}

CUDAGraph::~CUDAGraph()
{
	checkCudaErrors(cudaGraphExecDestroy(m_GraphExec));
	checkCudaErrors(cudaStreamDestroy(m_Stream));
	checkCudaErrors(cudaGraphDestroy(m_Graph));
}

void CUDAGraph::Execute()
{
	checkCudaErrors(cudaGraphLaunch(m_GraphExec, m_Stream));
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

	checkCudaErrors(cudaGraphAddKernelNode(&kernelNode, m_Graph, NULL, 0, &nodeParams));
}
