#include "OBJLoader.h"
#include <vector>

Assimp::Importer OBJLoader::m_Importer;

std::vector<Mesh> OBJLoader::LoadOBJ(const std::string& filename, AssetManager* assetManager)
{
	const aiScene* scene = m_Importer.ReadFile(filename, aiProcess_CalcTangentSpace | aiProcess_Triangulate
		| aiProcess_FlipUVs);
	
	std::vector<Mesh> meshes;

	if (!scene || scene->mFlags & AI_SCENE_FLAGS_INCOMPLETE || !scene->mRootNode)
	{
		std::cout << "OBJLoader: Error loading model " << filename << std::endl;
		return meshes;
	}

	int a = scene->mNumMeshes;
	
	for (int i = 0; i < scene->mNumMeshes; i++)
	{
		aiMesh* mesh = scene->mMeshes[i];

		std::vector<Triangle> triangles = std::vector<Triangle>(mesh->mNumFaces);
		
		for (int j = 0; j < mesh->mNumFaces; j++)
		{
			float3 pos[3] = { };
			float3 normal[3] = { };
			float2 texCoord[3] = { };

			for (int k = 0; k < 3; k++)
			{
				aiVector3D v = mesh->mVertices[3 * j + k];
				pos[k].x = v.x;
				pos[k].y = v.y;
				pos[k].z = v.z;

				if (mesh->HasNormals())
				{
					v = mesh->mNormals[3 * j + k];
					normal[k].x = v.x;
					normal[k].y = v.y;
					normal[k].z = v.z;
				}

				// We only deal with one tex coord per vertex for now
				if (mesh->HasTextureCoords(0))
				{
					v = mesh->mTextureCoords[0][3 * j + k];
					texCoord[k].x = v.x;
					texCoord[k].y = v.y;
					
				}
			}

			Triangle triangle(
				pos[0],
				pos[1],
				pos[2],
				normal[0],
				normal[1],
				normal[2],
				texCoord[0],
				texCoord[1],
				texCoord[2]
			);
			triangles[j] = triangle;
		}
		aiMaterial* material = scene->mMaterials[mesh->mMaterialIndex];

		Material newMaterial;
		newMaterial.type = Material::Type::DIFFUSE;
		int textureId = -1;

		if (material->GetTextureCount(aiTextureType_DIFFUSE) > 0)
		{
			aiString path;
			std::string fullPath;
			if (material->GetTexture(aiTextureType_DIFFUSE, 0, &path, NULL, NULL, NULL, NULL, NULL) == AI_SUCCESS)
			{
				fullPath = path.data;
				newMaterial.textureId = assetManager->AddTexture(fullPath);
			}
		}
		int materialId = assetManager->AddMaterial(newMaterial);

		BVH* bvh = new BVH(triangles);

		Mesh newMesh(bvh, materialId);

		meshes.push_back(newMesh);
	}

	std::cout << "OBJLoader: loaded model " << filename << " successfully" << std::endl;

	return meshes;
}
