#include "OBJLoader.h"
#include <vector>

Assimp::Importer OBJLoader::m_Importer;

std::vector<Mesh> OBJLoader::LoadOBJ(const std::string& path, const std::string& filename, AssetManager* assetManager)
{
	const std::string filePath = path + filename;
	const aiScene* scene = m_Importer.ReadFile(filePath, aiProcess_CalcTangentSpace | aiProcess_Triangulate
		| aiProcess_FlipUVs);
	
	std::vector<Mesh> meshes;

	if (!scene || scene->mFlags & AI_SCENE_FLAGS_INCOMPLETE || !scene->mRootNode)
	{
		std::cout << "OBJLoader: Error loading model " << filePath << std::endl;
		return meshes;
	}

	int* materialIdx = new int[scene->mNumMaterials];
	for (int i = 0; i < scene->mNumMaterials; i++)
	{
		aiMaterial* material = scene->mMaterials[i];
		Material newMaterial;

		aiColor3D diffuse(0.0f);
		material->Get(AI_MATKEY_COLOR_DIFFUSE, diffuse);
		newMaterial.diffuse = make_float3(diffuse.r, diffuse.g, diffuse.b);

		aiColor3D emission(0.0f);
		material->Get(AI_MATKEY_COLOR_EMISSIVE, emission);
		newMaterial.emissive = make_float3(emission.r, emission.g, emission.b);

		float ior = 1.45f;
		aiGetMaterialFloat(material, AI_MATKEY_REFRACTI, &ior);
		newMaterial.ior = ior;

		float shininess = 0.0f;
		if (AI_SUCCESS != aiGetMaterialFloat(material, AI_MATKEY_SHININESS, &shininess))
		{
			shininess = 20.0f;
		}
		newMaterial.roughness = clamp(1.0f - sqrt(shininess) / 30.0f, 0.0f, 1.0f);
		newMaterial.transmittance = 0.0f;

		if (material->GetTextureCount(aiTextureType_DIFFUSE) > 0)
		{
			aiString mPath;
			std::string materialPath;
			if (material->GetTexture(aiTextureType_DIFFUSE, 0, &mPath, NULL, NULL, NULL, NULL, NULL) == AI_SUCCESS)
			{
				materialPath = mPath.data;
				materialPath = path + materialPath;
				newMaterial.diffuseMapId = assetManager->AddTexture(materialPath);
			}
		}
		materialIdx[i] = assetManager->AddMaterial(newMaterial);
	}

	for (int i = 0; i < scene->mNumMeshes; i++)
	{
		aiMesh* mesh = scene->mMeshes[i];
		std::string name = mesh->mName.data;

		std::vector<Triangle> triangles = std::vector<Triangle>(mesh->mNumFaces);
		
		for (int j = 0; j < mesh->mNumFaces; j++)
		{
			float3 pos[3] = { };
			float3 normal[3] = { };
			float2 texCoord[3] = { };
			bool skipFace = false;

			for (int k = 0; k < 3; k++)
			{
				if (mesh->mFaces[j].mNumIndices != 3)
				{
					skipFace = true;
					continue;
				}
				unsigned int vertexIndex = mesh->mFaces[j].mIndices[k];

				aiVector3D v = mesh->mVertices[vertexIndex];
				pos[k].x = v.x;
				pos[k].y = v.y;
				pos[k].z = v.z;

				if (mesh->HasNormals())
				{
					v = mesh->mNormals[vertexIndex];
					normal[k].x = v.x;
					normal[k].y = v.y;
					normal[k].z = v.z;
				}

				// We only deal with one tex coord per vertex for now
				if (mesh->HasTextureCoords(0))
				{
					v = mesh->mTextureCoords[0][vertexIndex];
					texCoord[k].x = v.x;
					texCoord[k].y = v.y;
					
				}
			}
			if (skipFace)
				continue;

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

		const Mesh newMesh(name, triangles, materialIdx[mesh->mMaterialIndex]);
		meshes.push_back(newMesh);
	}

	std::cout << "OBJLoader: loaded model " << filePath << " successfully" << std::endl;

	delete[] materialIdx;

	return meshes;
}
