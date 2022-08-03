#pragma once

#include <iostream>
#include <vector>
#include <limits>
#include <algorithm>
#include <GLAD/glad.h>
#include <GLFW/glfw3.h>
#include <glm/glm.hpp>
#include <glm/gtc/matrix_transform.hpp>
#include <glm/gtc/type_ptr.hpp>
#include <omp.h>

using namespace glm;

const float MAX = FLT_MAX;
const float MIN = FLT_MIN;
const float INF = std::numeric_limits<float>::infinity();

struct Light {
	vec3 position;
	vec3 emission;
	vec3 u;
	vec3 v;
	vec3 radiusAreaType;
};

//物体表面材质定义(Disney)
struct Material {
	vec3 emissive = vec3(0.0f, 0.0f, 0.0f);  //作为光源时的发光颜色
	vec3 baseColor = vec3(1.0f, 1.0f, 1.0f);
	float subsurface = 0.0f;
	float metallic = 0.0f;
	float specular = 0.0f;
	float specularTint = 0.0f;
	float roughness = 0.0f;
	float anisotropic = 0.0f;
	float sheen = 0.0f;
	float sheenTint = 0.0f;
	float clearcoat = 0.0f;
	float clearcoatGloss = 0.0f;
	float IOR = 1.0f;
	float transmission = 0.0f;
	float isTex = -1.0f;
	float texid = 0.0f;
	float lightid = 0.0f;
};

struct Triangle {
	vec3 p1, p2, p3;   //顶点坐标
	vec3 n1, n2, n3;   //顶点法线
	vec2 uv1, uv2, uv3;//UV
	Material material; //材质
};

struct Triangle_encoded {
	vec3 p1, p2, p3;    //顶点坐标
	vec3 n1, n2, n3;    //顶点法线
	vec3 uv1, uv2, uv3; //UV
	vec3 emissive;      //自发光参数
	vec3 baseColor;     //颜色
	vec3 param1;        //(subsurface, metallic, specular)
	vec3 param2;        //(specularTint, roughness, anisotropic)
	vec3 param3;        //(sheen, sheenTint, clearcoat)
	vec3 param4;        //(clearcoatGloss, IOR, transmission)
	vec3 tex;
};

//BVH树节点
struct BVHNode {
	int left, right;    //左右子树索引
	int n, index;       //叶子节点信息               
	vec3 AA, BB;        //碰撞盒
};

struct BVHNode_encoded {
	vec3 childs;        //(left, right, 保留)
	vec3 leafInfo;      //(n, index, 保留)
	vec3 AA, BB;
};

//按照三角形中心排序--比较函数
inline bool cmpx(const Triangle& t1, const Triangle& t2) {
	vec3 center1 = (t1.p1 + t1.p2 + t1.p3) / vec3(3, 3, 3);
	vec3 center2 = (t2.p1 + t2.p2 + t2.p3) / vec3(3, 3, 3);
	return center1.x < center2.x;
}
inline bool cmpy(const Triangle& t1, const Triangle& t2) {
	vec3 center1 = (t1.p1 + t1.p2 + t1.p3) / vec3(3, 3, 3);
	vec3 center2 = (t2.p1 + t2.p2 + t2.p3) / vec3(3, 3, 3);
	return center1.y < center2.y;
}
inline bool cmpz(const Triangle& t1, const Triangle& t2) {
	vec3 center1 = (t1.p1 + t1.p2 + t1.p3) / vec3(3, 3, 3);
	vec3 center2 = (t2.p1 + t2.p2 + t2.p3) / vec3(3, 3, 3);
	return center1.z < center2.z;
}

class BVH {
public:
	int nTriangles;
	int nNodes;

	std::vector<Triangle> triangles;
	std::vector<BVHNode> nodes;
	std::vector<Triangle_encoded> triangles_encoded;
	std::vector<BVHNode_encoded> nodes_encoded;

	//三角形数组
	GLuint trianglesTextureBuffer;
	GLuint tbo0;

	//BVHNode数组
	GLuint nodesTextureBuffer;
	GLuint tbo1;

public:
	BVH() {}
	BVH(std::vector<Triangle>& tr);
	//SAH优化构建BVH
	int BuildBVHwithSAH(int l, int r, int n);
	void TransformTriangleAndBVHNode();
	void CreateTBO();
};
