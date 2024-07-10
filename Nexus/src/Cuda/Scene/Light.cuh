#pragma once
#include <cstdint>

struct D_Light
{
	enum struct Type : char
	{
		POINT_LIGHT,
		AREA_LIGHT,
		MESH_LIGHT
	};

	union
	{
		struct
		{
			uint32_t radius;
			uint32_t intensity;
		} point;

		struct
		{
			uint32_t intensity;
		} area;

		struct
		{
			uint32_t meshId;
		} mesh;
	};
};