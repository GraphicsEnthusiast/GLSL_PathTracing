#include "BVH.h"

BVH::BVH(std::vector<Triangle>& tr) {
	triangles = tr;

	BuildBVHwithSAH(0, triangles.size() - 1, 8);
	TransformTriangleAndBVHNode();
	CreateTBO();
}

int BVH::BuildBVHwithSAH(int l, int r, int n) {
	if (l > r) {
		return 0;
	}

	nodes.push_back(BVHNode());
	int id = nodes.size() - 1;
	nodes[id].left = nodes[id].right = nodes[id].n = nodes[id].index = 0;
	nodes[id].AA = vec3(MAX);
	nodes[id].BB = vec3(MIN);

	//计算AABB
	omp_set_num_threads(32);//线程个数
#pragma omp parallel for
	for (int i = l; i <= r; i++) {
		//最小点AA
		float minx = std::min(triangles[i].p1.x, std::min(triangles[i].p2.x, triangles[i].p3.x));
		float miny = std::min(triangles[i].p1.y, std::min(triangles[i].p2.y, triangles[i].p3.y));
		float minz = std::min(triangles[i].p1.z, std::min(triangles[i].p2.z, triangles[i].p3.z));
		nodes[id].AA.x = std::min(nodes[id].AA.x, minx);
		nodes[id].AA.y = std::min(nodes[id].AA.y, miny);
		nodes[id].AA.z = std::min(nodes[id].AA.z, minz);
		//最大点BB
		float maxx = std::max(triangles[i].p1.x, std::max(triangles[i].p2.x, triangles[i].p3.x));
		float maxy = std::max(triangles[i].p1.y, std::max(triangles[i].p2.y, triangles[i].p3.y));
		float maxz = std::max(triangles[i].p1.z, std::max(triangles[i].p2.z, triangles[i].p3.z));
		nodes[id].BB.x = std::max(nodes[id].BB.x, maxx);
		nodes[id].BB.y = std::max(nodes[id].BB.y, maxy);
		nodes[id].BB.z = std::max(nodes[id].BB.z, maxz);
	}

	//不多于n个三角形，返回叶子节点
	if ((r - l + 1) <= n) {
		nodes[id].n = r - l + 1;
		nodes[id].index = l;
		return id;
	}

	//否则递归建树
	float Cost = INF;
	int Axis = 0;
	int Split = (l + r) / 2;
	omp_set_num_threads(32);//线程个数
#pragma omp parallel for
	for (int axis = 0; axis < 3; axis++) {
		//分别按x，y，z轴排序
		if (axis == 0) {
			std::sort(&triangles[0] + l, &triangles[0] + r + 1, cmpx);
		}
		if (axis == 1) {
			std::sort(&triangles[0] + l, &triangles[0] + r + 1, cmpy);
		}
		if (axis == 2) {
			std::sort(&triangles[0] + l, &triangles[0] + r + 1, cmpz);
		}

		//leftMax[i]:[l, i]中最大的xyz值
		//leftMin[i]:[l, i]中最小的xyz值
		std::vector<vec3> leftMax(r - l + 1, vec3(MIN));
		std::vector<vec3> leftMin(r - l + 1, vec3(MAX));
		//计算前缀，注意i-l以对齐到下标0
		for (int i = l; i <= r; i++) {
			Triangle& t = triangles[i];
			int bias = (i == l) ? 0 : 1;  //第一个元素特殊处理

			leftMax[i - l].x = std::max(leftMax[i - l - bias].x, std::max(t.p1.x, std::max(t.p2.x, t.p3.x)));
			leftMax[i - l].y = std::max(leftMax[i - l - bias].y, std::max(t.p1.y, std::max(t.p2.y, t.p3.y)));
			leftMax[i - l].z = std::max(leftMax[i - l - bias].z, std::max(t.p1.z, std::max(t.p2.z, t.p3.z)));

			leftMin[i - l].x = std::min(leftMin[i - l - bias].x, std::min(t.p1.x, std::min(t.p2.x, t.p3.x)));
			leftMin[i - l].y = std::min(leftMin[i - l - bias].y, std::min(t.p1.y, std::min(t.p2.y, t.p3.y)));
			leftMin[i - l].z = std::min(leftMin[i - l - bias].z, std::min(t.p1.z, std::min(t.p2.z, t.p3.z)));
		}

		//rightMax[i]:[i, r]中最大的xyz值
		//rightMin[i]:[i, r]中最小的xyz值
		std::vector<vec3> rightMax(r - l + 1, vec3(MIN));
		std::vector<vec3> rightMin(r - l + 1, vec3(MAX));
		//计算后缀，注意i-l以对齐到下标0
		for (int i = r; i >= l; i--) {
			Triangle& t = triangles[i];
			int bias = (i == r) ? 0 : 1;  //第一个元素特殊处理

			rightMax[i - l].x = std::max(rightMax[i - l + bias].x, std::max(t.p1.x, std::max(t.p2.x, t.p3.x)));
			rightMax[i - l].y = std::max(rightMax[i - l + bias].y, std::max(t.p1.y, std::max(t.p2.y, t.p3.y)));
			rightMax[i - l].z = std::max(rightMax[i - l + bias].z, std::max(t.p1.z, std::max(t.p2.z, t.p3.z)));

			rightMin[i - l].x = std::min(rightMin[i - l + bias].x, std::min(t.p1.x, std::min(t.p2.x, t.p3.x)));
			rightMin[i - l].y = std::min(rightMin[i - l + bias].y, std::min(t.p1.y, std::min(t.p2.y, t.p3.y)));
			rightMin[i - l].z = std::min(rightMin[i - l + bias].z, std::min(t.p1.z, std::min(t.p2.z, t.p3.z)));
		}

		//遍历寻找分割
		float cost = INF;
		int split = l;
		for (int i = l; i <= r - 1; i++) {
			float lenx, leny, lenz;
			//左侧[l, i]
			vec3 leftAA = leftMin[i - l];
			vec3 leftBB = leftMax[i - l];
			lenx = leftBB.x - leftAA.x;
			leny = leftBB.y - leftAA.y;
			lenz = leftBB.z - leftAA.z;
			float leftS = 2.0f * ((lenx * leny) + (lenx * lenz) + (leny * lenz));
			float leftCost = leftS * (i - l + 1);

			//右侧[i+1, r]
			vec3 rightAA = rightMin[i + 1 - l];
			vec3 rightBB = rightMax[i + 1 - l];
			lenx = rightBB.x - rightAA.x;
			leny = rightBB.y - rightAA.y;
			lenz = rightBB.z - rightAA.z;
			float rightS = 2.0f * ((lenx * leny) + (lenx * lenz) + (leny * lenz));
			float rightCost = rightS * (r - i);

			//记录每个分割的最小答案
			float totalCost = leftCost + rightCost;
			if (totalCost < cost) {
				cost = totalCost;
				split = i;
			}
		}
		//记录每个轴的最佳答案
		if (cost < Cost) {
			Cost = cost;
			Axis = axis;
			Split = split;
		}
	}

	//按最佳轴分割
	if (Axis == 0) {
		std::sort(&triangles[0] + l, &triangles[0] + r + 1, cmpx);
	}
	if (Axis == 1) {
		std::sort(&triangles[0] + l, &triangles[0] + r + 1, cmpy);
	}
	if (Axis == 2) {
		std::sort(&triangles[0] + l, &triangles[0] + r + 1, cmpz);
	}

	//递归
	int left = BuildBVHwithSAH(l, Split, n);
	int right = BuildBVHwithSAH(Split + 1, r, n);

	nodes[id].left = left;
	nodes[id].right = right;

	return id;
}

void BVH::TransformTriangleAndBVHNode() {
	//编码三角形, 材质
	nTriangles = triangles.size();
	triangles_encoded.resize(nTriangles);
	for (int i = 0; i < nTriangles; i++) {
		Triangle& t = triangles[i];
		Material& m = t.material;
		//顶点位置
		triangles_encoded[i].p1 = t.p1;
		triangles_encoded[i].p2 = t.p2;
		triangles_encoded[i].p3 = t.p3;
		//顶点法线
		triangles_encoded[i].n1 = t.n1;
		triangles_encoded[i].n2 = t.n2;
		triangles_encoded[i].n3 = t.n3;
		//UV
		triangles_encoded[i].uv1.x = t.uv1.x;
		triangles_encoded[i].uv2.x = t.uv2.x;
		triangles_encoded[i].uv3.x = t.uv3.x;
		triangles_encoded[i].uv1.y = t.uv1.y;
		triangles_encoded[i].uv2.y = t.uv2.y;
		triangles_encoded[i].uv3.y = t.uv3.y;
		//材质
		triangles_encoded[i].emissive = m.emissive;
		triangles_encoded[i].baseColor = m.baseColor;
		triangles_encoded[i].param1 = vec3(m.subsurface, m.metallic, m.specular);
		triangles_encoded[i].param2 = vec3(m.specularTint, m.roughness, m.anisotropic);
		triangles_encoded[i].param3 = vec3(m.sheen, m.sheenTint, m.clearcoat);
		triangles_encoded[i].param4 = vec3(m.clearcoatGloss, m.IOR, m.transmission);
		triangles_encoded[i].tex = vec3(m.isTex, m.texid, m.lightid);
	}

	//编码BVHNode, aabb
	nNodes = nodes.size();
	nodes_encoded.resize(nNodes);
	for (int i = 0; i < nNodes; i++) {
		nodes_encoded[i].childs = vec3(nodes[i].left, nodes[i].right, 0);
		nodes_encoded[i].leafInfo = vec3(nodes[i].n, nodes[i].index, 0);
		nodes_encoded[i].AA = nodes[i].AA;
		nodes_encoded[i].BB = nodes[i].BB;
	}
	std::cout << "BVH建立完成：共" << nodes.size() << "个节点" << std::endl;
}

void BVH::CreateTBO() {
	//三角形数组
	glGenBuffers(1, &tbo0);
	glBindBuffer(GL_TEXTURE_BUFFER, tbo0);
	glBufferData(GL_TEXTURE_BUFFER, triangles_encoded.size() * sizeof(Triangle_encoded), &triangles_encoded[0], GL_STATIC_DRAW);
	glGenTextures(1, &trianglesTextureBuffer);
	glBindTexture(GL_TEXTURE_BUFFER, trianglesTextureBuffer);
	glTexBuffer(GL_TEXTURE_BUFFER, GL_RGB32F, tbo0);

	//BVHNode数组
	glGenBuffers(1, &tbo1);
	glBindBuffer(GL_TEXTURE_BUFFER, tbo1);
	glBufferData(GL_TEXTURE_BUFFER, nodes_encoded.size() * sizeof(BVHNode_encoded), &nodes_encoded[0], GL_STATIC_DRAW);
	glGenTextures(1, &nodesTextureBuffer);
	glBindTexture(GL_TEXTURE_BUFFER, nodesTextureBuffer);
	glTexBuffer(GL_TEXTURE_BUFFER, GL_RGB32F, tbo1);
}
