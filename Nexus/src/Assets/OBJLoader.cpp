#include "OBJLoader.h"
#include <vector>

Assimp::Importer OBJLoader::m_Importer;

static std::vector<Triangle> GetTrianglesFromAiMesh(const aiMesh* mesh)
{
	std::vector<Triangle> triangles(mesh->mNumFaces);

	for (int i = 0; i < mesh->mNumFaces; i++)
	{
		float3 pos[3] = { };
		float3 normal[3] = { };
		float2 texCoord[3] = { };
		bool skipFace = false;

		for (int k = 0; k < 3; k++)
		{
			if (mesh->mFaces[i].mNumIndices != 3)
			{
				std::cout << "ObjLoader: a non triangle primitive with " << mesh->mFaces[i].mNumIndices << " vertices has been discarded" << std::endl;
				skipFace = true;
				continue;
			}
			unsigned int vertexIndex = mesh->mFaces[i].mIndices[k];

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
		triangles[i] = triangle;
	}
	return triangles;
}

// Return the list of IDs of the created materials
static std::vector<int> CreateMaterialsFromAiScene(const aiScene* scene, AssetManager* assetManager, const std::string& path)
{
	std::vector<int> materialIdx(scene->mNumMaterials);

	for (int i = 0; i < scene->mNumMaterials; i++)
	{
		aiMaterial* material = scene->mMaterials[i];
		Material newMaterial;
		newMaterial.type = Material::Type::DIELECTRIC;

		aiColor3D diffuse(0.0f);
		material->Get(AI_MATKEY_COLOR_DIFFUSE, diffuse);
		newMaterial.dielectric.albedo = make_float3(diffuse.r, diffuse.g, diffuse.b);

		aiColor3D emission(0.0f);
		material->Get(AI_MATKEY_COLOR_EMISSIVE, emission);
		newMaterial.emissive = make_float3(emission.r, emission.g, emission.b);

		float ior = 1.45f;
		aiGetMaterialFloat(material, AI_MATKEY_REFRACTI, &ior);
		newMaterial.dielectric.ior = ior;

		float shininess = 0.0f;
		if (AI_SUCCESS != aiGetMaterialFloat(material, AI_MATKEY_SHININESS, &shininess))
		{
			shininess = 20.0f;
		}
		newMaterial.dielectric.roughness = clamp(1.0f - sqrt(shininess) / 31.62278f, 0.0f, 1.0f);
		newMaterial.dielectric.transmittance = 0.0f;

		if (material->GetTextureCount(aiTextureType_DIFFUSE) > 0)
		{
			aiString mPath;
			std::string materialPath;
			if (material->GetTexture(aiTextureType_DIFFUSE, 0, &mPath, NULL, NULL, NULL, NULL, NULL) == AI_SUCCESS)
			{
				materialPath = mPath.data;
				materialPath = path + materialPath;
				newMaterial.diffuseMapId = assetManager->AddTexture(materialPath, Texture::Type::DIFFUSE);
			}
		}
		if (material->GetTextureCount(aiTextureType_EMISSIVE) > 0)
		{
			aiString mPath;
			std::string materialPath;
			if (material->GetTexture(aiTextureType_EMISSIVE, 0, &mPath, NULL, NULL, NULL, NULL, NULL) == AI_SUCCESS)
			{
				materialPath = mPath.data;
				materialPath = path + materialPath;
				newMaterial.emissiveMapId = assetManager->AddTexture(materialPath, Texture::Type::EMISSIVE);
			}
		}
		materialIdx[i] = assetManager->AddMaterial(newMaterial);
	}
	return materialIdx;
}

static void GetMeshesFromAiNode(const aiScene* scene, const aiNode* node, std::vector<Mesh>& meshes, aiMatrix4x4 aiTransform, std::vector<int>& materialIds)
{
	aiTransform = node->mTransformation * aiTransform;
	for (int i = 0; i < node->mNumMeshes; i++)
	{
		aiMesh* mesh = scene->mMeshes[node->mMeshes[i]];
		std::vector<Triangle> triangles = GetTrianglesFromAiMesh(mesh);
		std::string meshName = mesh->mName.data;

		aiVector3D aiPosition, aiRotation, aiScale;
		aiTransform.Decompose(aiScale, aiRotation, aiPosition);
		float3 position = { aiPosition.x, aiPosition.y, aiPosition.z };
		float3 rotation = { Utils::ToDegrees(aiRotation.x), Utils::ToDegrees(aiRotation.y), Utils::ToDegrees(aiRotation.z) };
		float3 scale = { aiScale.x, aiScale.y, aiScale.z };

		double scaleFactor = 1.0f;
		bool result = scene->mMetaData->Get("UnitScaleFactor", scaleFactor);
		scale /= scaleFactor;
		position /= scaleFactor;

		const Mesh newMesh(meshName, triangles, materialIds[mesh->mMaterialIndex], position, rotation, scale);
		meshes.push_back(newMesh);
	}

	for (int i = 0; i < node->mNumChildren; i++)
	{
		GetMeshesFromAiNode(scene, node->mChildren[i], meshes, aiTransform, materialIds);
	}
}

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

	//double factor = 100.0f;
	//// Fix for assimp scaling FBX with a factor 100
	//scene->mMetaData->Set("UnitScaleFactor", factor);
	

	std::vector<int> materialIds = CreateMaterialsFromAiScene(scene, assetManager, path);
	GetMeshesFromAiNode(scene, scene->mRootNode, meshes, aiMatrix4x4(), materialIds);

	std::cout << "OBJLoader: loaded model " << filePath << " successfully" << std::endl;

	return meshes;
}
