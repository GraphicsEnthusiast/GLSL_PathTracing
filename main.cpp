#define STB_IMAGE_IMPLEMENTATION

#include <iostream>
#include <string>
#include <fstream>
#include <vector>
#include <sstream>
#include <iostream>
#include <algorithm>
#include <GLAD/glad.h>
#include <GLFW/glfw3.h>
#include <glm/glm.hpp>
#include <glm/gtc/matrix_transform.hpp>
#include <glm/gtc/type_ptr.hpp>

#include "Common/Shader.h"
#include "Common/BVH.h"
#include "Common/RenderPass.h"
#include "ThirdParty/HDRLoader.h"
#include "ThirdParty/stb_image.h"

using namespace std;
using namespace glm;

#pragma region 全局变量和回调函数
const int Width = 1600;
const int Height = 800;

clock_t t1, t2;
double dt, fps;
unsigned int frameCounter = 0;
unsigned int hdrCache;
int hdrResolution;

GLuint lastFrame;
GLuint hdrMap;

RenderPass pass1;
RenderPass pass2;
RenderPass pass3;

//相机参数
Camera camera;

BVH bvh;

//鼠标运动函数
double lastX = 0.0, lastY = 0.0;
bool firstMouse = true;
void mouse_callback(GLFWwindow* window, double xpos, double ypos) {
	if (firstMouse == true) {
		lastX = xpos;
		lastY = ypos;
		firstMouse = false;
	}

	frameCounter = 0;
	//调整旋转
	camera.rotatAngle += 150 * (xpos - lastX) / Width;
	camera.upAngle += 150 * (ypos - lastY) / Height;
	camera.upAngle = std::min(camera.upAngle, 89.0f);
	camera.upAngle = std::max(camera.upAngle, -89.0f);
	lastX = xpos, lastY = ypos;
}

//鼠标滚轮函数
//void scroll_callback(GLFWwindow* window, double xoffset, double yoffset) {
//
//}
//void mouseWheel(int wheel, int direction, int x, int y) {
//	frameCounter = 0;
//	r += -direction * 0.5f;
//	glutPostRedisplay();    // 重绘
//}

void ProcessInput(GLFWwindow* window) {
	if (glfwGetKey(window, GLFW_KEY_ESCAPE) == GLFW_PRESS) {
		glfwSetWindowShouldClose(window, true);
	}
	if (glfwGetMouseButton(window, GLFW_MOUSE_BUTTON_RIGHT) == GLFW_PRESS) {
		glfwSetInputMode(window, GLFW_CURSOR, GLFW_CURSOR_DISABLED);
		glfwSetCursorPosCallback(window, mouse_callback);
	}
	else {
		glfwSetInputMode(window, GLFW_CURSOR, GLFW_CURSOR_NORMAL);
		glfwSetCursorPosCallback(window, NULL);
		lastX = Width / 2;
		lastY = Height / 2;
		firstMouse = true;
	}
}
#pragma endregion

//计算HDR贴图相关缓存信息
float* CalculateHdrCache(float* HDR, int width, int height) {

	float lumSum = 0.0f;

	//初始化h行w列的概率密度pdf并统计总亮度
	std::vector<std::vector<float>> pdf(height);
	for (auto& line : pdf) line.resize(width);
	for (int i = 0; i < height; i++) {
		for (int j = 0; j < width; j++) {
			float R = HDR[3 * (i * width + j)];
			float G = HDR[3 * (i * width + j) + 1];
			float B = HDR[3 * (i * width + j) + 2];
			float lum = 0.2f * R + 0.7f * G + 0.1f * B;
			pdf[i][j] = lum;
			lumSum += lum;
		}
	}

	//概率密度归一化
	for (int i = 0; i < height; i++) {
		for (int j = 0; j < width; j++) {
			pdf[i][j] /= lumSum;
		}
	}

	//累加每一列得到x的边缘概率密度
	std::vector<float> pdf_x_margin;
	pdf_x_margin.resize(width);
	for (int j = 0; j < width; j++) {
		for (int i = 0; i < height; i++) {
			pdf_x_margin[j] += pdf[i][j];
		}
	}

	//计算x的边缘分布函数
	std::vector<float> cdf_x_margin = pdf_x_margin;
	for (int i = 1; i < width; i++) {
		cdf_x_margin[i] += cdf_x_margin[i - 1];
	}

	//计算y在X=x下的条件概率密度函数
	std::vector<std::vector<float>> pdf_y_condiciton = pdf;
	for (int j = 0; j < width; j++) {
		for (int i = 0; i < height; i++) {
			pdf_y_condiciton[i][j] /= pdf_x_margin[j];
		}
	}

	//计算y在X=x下的条件概率分布函数
	std::vector<std::vector<float>> cdf_y_condiciton = pdf_y_condiciton;
	for (int j = 0; j < width; j++) {
		for (int i = 1; i < height; i++) {
			cdf_y_condiciton[i][j] += cdf_y_condiciton[i - 1][j];
		}
	}

	//cdf_y_condiciton转置为按列存储
	//cdf_y_condiciton[i]表示y在X=i下的条件概率分布函数
	std::vector<std::vector<float>> temp = cdf_y_condiciton;
	cdf_y_condiciton = std::vector<std::vector<float>>(width);
	for (auto& line : cdf_y_condiciton) {
		line.resize(height);
	}
	for (int j = 0; j < width; j++) {
		for (int i = 0; i < height; i++) {
			cdf_y_condiciton[j][i] = temp[i][j];
		}
	}

	//穷举 xi_1, xi_2 预计算样本 xy
	//sample_x[i][j] 表示 xi_1=i/height, xi_2=j/width 时 (x,y) 中的 x
	//sample_y[i][j] 表示 xi_1=i/height, xi_2=j/width 时 (x,y) 中的 y
	//sample_p[i][j] 表示取 (i, j) 点时的概率密度
	std::vector<std::vector<float>> sample_x(height);
	for (auto& line : sample_x) {
		line.resize(width);
	}
	std::vector<std::vector<float>> sample_y(height);
	for (auto& line : sample_y) {
		line.resize(width);
	}
	std::vector<std::vector<float>> sample_p(height);
	for (auto& line : sample_p) {
		line.resize(width);
	}
	for (int j = 0; j < width; j++) {
		for (int i = 0; i < height; i++) {
			float xi_1 = float(i) / height;
			float xi_2 = float(j) / width;

			//用xi_1在cdf_x_margin中lower bound得到样本x
			int x = std::lower_bound(cdf_x_margin.begin(), cdf_x_margin.end(), xi_1) - cdf_x_margin.begin();
			//用xi_2在X=x的情况下得到样本y
			int y = std::lower_bound(cdf_y_condiciton[x].begin(), cdf_y_condiciton[x].end(), xi_2) - cdf_y_condiciton[x].begin();

			//存储纹理坐标xy和xy位置对应的概率密度
			sample_x[i][j] = float(x) / width;
			sample_y[i][j] = float(y) / height;
			sample_p[i][j] = pdf[i][j];
		}
	}

	//整合结果到纹理
	//R,G 通道存储样本(x,y)而B通道存储pdf(i, j)
	float* cache = new float[width * height * 3];
	//for (int i = 0; i < width * height * 3; i++) cache[i] = 0.0;

	for (int i = 0; i < height; i++) {
		for (int j = 0; j < width; j++) {
			cache[3 * (i * width + j)] = sample_x[i][j];        //R
			cache[3 * (i * width + j) + 1] = sample_y[i][j];    //G
			cache[3 * (i * width + j) + 2] = sample_p[i][j];    //B
		}
	}

	return cache;
}

void LoadObj(std::string filepath, std::vector<Triangle>& triangles, Material material, mat4 trans, bool smoothNormal) {

	//顶点位置，索引
	std::vector<vec3> vertices;
	std::vector<GLuint> indices;
	std::vector<vec2> UV;

	//打开文件流
	std::ifstream fin(filepath);
	std::string line;
	if (!fin.is_open()) {
		std::cout << "文件 " << filepath << " 打开失败" << std::endl;
		exit(-1);
	}

	//计算AABB盒，归一化模型大小
	float maxx = MIN;
	float maxy = MIN;
	float maxz = MIN;
	float minx = MAX;
	float miny = MAX;
	float minz = MAX;

	//按行读取
	while (std::getline(fin, line)) {
		std::istringstream sin(line);   //以一行的数据作为string stream解析并且读取
		std::string type;
		GLfloat x, y, z;
		float u, v;
		int v0, v1, v2;
		int vn0, vn1, vn2;
		int vt0, vt1, vt2;
		char slash;

		//统计斜杆数目，用不同格式读取
		int slashCnt = 0;
		for (int i = 0; i < line.length(); i++) {
			if (line[i] == '/') {
				slashCnt++;
			}
		}

		//读取obj文件
		sin >> type;
		if (type == "v") {
			sin >> x >> y >> z;
			vertices.push_back(vec3(x, y, z));
			maxx = std::max(maxx, x); maxy = std::max(maxx, y); maxz = std::max(maxx, z);
			minx = std::min(minx, x); miny = std::min(minx, y); minz = std::min(minx, z);
		}
		if (type == "vt") {
			sin >> u >> v;
			UV.push_back(vec2(u, v));
		}
		if (type == "f") {
			if (slashCnt == 6) {
				sin >> v0 >> slash >> vt0 >> slash >> vn0;
				sin >> v1 >> slash >> vt1 >> slash >> vn1;
				sin >> v2 >> slash >> vt2 >> slash >> vn2;
			}
			else if (slashCnt == 3) {
				sin >> v0 >> slash >> vt0;
				sin >> v1 >> slash >> vt1;
				sin >> v2 >> slash >> vt2;
			}
			else {
				sin >> v0 >> v1 >> v2;
			}
			indices.push_back(v0 - 1);
			indices.push_back(v1 - 1);
			indices.push_back(v2 - 1);
		}
	}

	//模型大小归一化
	float lenx = maxx - minx;
	float leny = maxy - miny;
	float lenz = maxz - minz;
	float maxaxis = std::max(lenx, std::max(leny, lenz));
	for (auto& v : vertices) {
		v.x /= maxaxis;
		v.y /= maxaxis;
		v.z /= maxaxis;
	}

	//通过矩阵进行坐标变换
	for (auto& v : vertices) {
		vec4 vv = vec4(v.x, v.y, v.z, 1);
		vv = trans * vv;
		v = vec3(vv.x, vv.y, vv.z);
	}

	//生成法线
	std::vector<vec3> normals(vertices.size(), vec3(0.0f));
	for (int i = 0; i < indices.size(); i += 3) {
		vec3 p1 = vertices[indices[i]];
		vec3 p2 = vertices[indices[i + 1]];
		vec3 p3 = vertices[indices[i + 2]];
		vec3 n = normalize(cross(p2 - p1, p3 - p1));
		normals[indices[i]] += n;
		normals[indices[i + 1]] += n;
		normals[indices[i + 2]] += n;
	}

	//构建Triangle对象数组
	int offset = triangles.size();  //增量更新
	triangles.resize(offset + indices.size() / 3);
	for (int i = 0; i < indices.size(); i += 3) {
		Triangle& t = triangles[offset + i / 3];
		// =传顶点属性
		t.p1 = vertices[indices[i]];
		t.p2 = vertices[indices[i + 1]];
		t.p3 = vertices[indices[i + 2]];

		t.uv1 = UV[indices[i]];
		t.uv2 = UV[indices[i + 1]];
		t.uv3 = UV[indices[i + 2]];

		if (!smoothNormal) {
			vec3 n = normalize(cross(t.p2 - t.p1, t.p3 - t.p1));
			t.n1 = n; t.n2 = n; t.n3 = n;
		}
		else {
			t.n1 = normalize(normals[indices[i]]);
			t.n2 = normalize(normals[indices[i + 1]]);
			t.n3 = normalize(normals[indices[i + 2]]);
		}

		//传材质
		t.material = material;
	}
}

void InitScene() {
	std::vector<Triangle> triangles;

	Material m;
	m.roughness = 0.1f;
	m.specular = 1.0f;
	m.metallic = 0.1f;
	m.clearcoat = 1.0f;
	m.clearcoatGloss = 0.0f;
	m.baseColor = vec3(0.2f, 0.85f, 0.9f);
	LoadObj("Assert/Model/Teaport.obj", triangles, m, RenderPass::GetTransformMatrix(vec3(0.0f), vec3(0.0f, -0.4f, 0.0f), vec3(1.75f)), true);

	//m.roughness = 0.2f;
	//m.specular = 0.8f;
	//m.metallic = 0.7f;
	//m.clearcoat = 0.6f;
	//m.clearcoatGloss = 0.4f;
	//m.baseColor = vec3(0.0f, 0.73f, 0.85f);
	//LoadObj("Stanford Bunny.obj", triangles, m, RenderPass::GetTransformMatrix(vec3(0.0f), vec3(1.4f, -1.5f, 2.0f), vec3(2.5f)), true);

	m.isTex = 1.0f;
	m.texid = 0.0f;
	m.specular = 0.8f;
	m.clearcoat = 0.0f;
	m.clearcoatGloss = 0.0f;
	LoadObj("Assert/Model/quad.obj", triangles, m, RenderPass::GetTransformMatrix(vec3(0.0f), vec3(0.0f, -0.5f, 0.0f), vec3(10.0f, 0.01f, 10.0f)), true);

	std::cout << "模型读取完成：共" << triangles.size() << "个三角形" << std::endl;

	//建立bvh
	bvh = BVH(triangles);

	//hdr全景图
	HDRLoaderResult hdrRes;
	bool r = HDRLoader::load("Assert/HDR/chinese_garden_2k.hdr", hdrRes);
	hdrMap = RenderPass::GetTextureRGB32F(hdrRes.width, hdrRes.height);
	glTexImage2D(GL_TEXTURE_2D, 0, GL_RGB32F, hdrRes.width, hdrRes.height, 0, GL_RGB, GL_FLOAT, hdrRes.cols);

	//hdr重要性采样cache
	std::cout << "HDR当前分辨率: " << hdrRes.width << " * " << hdrRes.height << std::endl;
	float* cache = CalculateHdrCache(hdrRes.cols, hdrRes.width, hdrRes.height);
	hdrCache = RenderPass::GetTextureRGB32F(hdrRes.width, hdrRes.height);
	glTexImage2D(GL_TEXTURE_2D, 0, GL_RGB32F, hdrRes.width, hdrRes.height, 0, GL_RGB, GL_FLOAT, cache);
	hdrResolution = hdrRes.width;

	//贴图
	GLuint albedo;
	GLuint metallic;
	GLuint roughness;
	GLuint normal;
	int x, y, n;
	unsigned char* albedoTextures = stbi_load("Assert/Texture/metal/albedo.png", &x, &y, &n, 3);
	unsigned char* metallicTextures = stbi_load("Assert/Texture/metal/metallic.png", &x, &y, &n, 3);
	unsigned char* roughnessTextures = stbi_load("Assert/Texture/metal/roughness.png", &x, &y, &n, 3);
	unsigned char* normalTextures = stbi_load("Assert/Texture/metal/normal.png", &x, &y, &n, 3);

	glGenTextures(1, &albedo);
	glActiveTexture(GL_TEXTURE5);
	glBindTexture(GL_TEXTURE_2D_ARRAY, albedo);
	glTexImage3D(GL_TEXTURE_2D_ARRAY, 0, GL_RGB8, x, y, 1, 0, GL_RGB, GL_UNSIGNED_BYTE, albedoTextures);
	glTexParameteri(GL_TEXTURE_2D_ARRAY, GL_TEXTURE_MAG_FILTER, GL_LINEAR);
	glTexParameteri(GL_TEXTURE_2D_ARRAY, GL_TEXTURE_MIN_FILTER, GL_LINEAR);
	glBindTexture(GL_TEXTURE_2D_ARRAY, 0);

	glGenTextures(1, &metallic);
	glActiveTexture(GL_TEXTURE6);
	glBindTexture(GL_TEXTURE_2D_ARRAY, metallic);
	glTexImage3D(GL_TEXTURE_2D_ARRAY, 0, GL_RGB8, x, y, 1, 0, GL_RGB, GL_UNSIGNED_BYTE, metallicTextures);
	glTexParameteri(GL_TEXTURE_2D_ARRAY, GL_TEXTURE_MAG_FILTER, GL_LINEAR);
	glTexParameteri(GL_TEXTURE_2D_ARRAY, GL_TEXTURE_MIN_FILTER, GL_LINEAR);
	glBindTexture(GL_TEXTURE_2D_ARRAY, 0);

	glGenTextures(1, &roughness);
	glActiveTexture(GL_TEXTURE7);
	glBindTexture(GL_TEXTURE_2D_ARRAY, roughness);
	glTexImage3D(GL_TEXTURE_2D_ARRAY, 0, GL_RGB8, x, y, 1, 0, GL_RGB, GL_UNSIGNED_BYTE, roughnessTextures);
	glTexParameteri(GL_TEXTURE_2D_ARRAY, GL_TEXTURE_MAG_FILTER, GL_LINEAR);
	glTexParameteri(GL_TEXTURE_2D_ARRAY, GL_TEXTURE_MIN_FILTER, GL_LINEAR);
	glBindTexture(GL_TEXTURE_2D_ARRAY, 0);

	glGenTextures(1, &normal);
	glActiveTexture(GL_TEXTURE8);
	glBindTexture(GL_TEXTURE_2D_ARRAY, normal);
	glTexImage3D(GL_TEXTURE_2D_ARRAY, 0, GL_RGB8, x, y, 1, 0, GL_RGB, GL_UNSIGNED_BYTE, normalTextures);
	glTexParameteri(GL_TEXTURE_2D_ARRAY, GL_TEXTURE_MAG_FILTER, GL_LINEAR);
	glTexParameteri(GL_TEXTURE_2D_ARRAY, GL_TEXTURE_MIN_FILTER, GL_LINEAR);
	glBindTexture(GL_TEXTURE_2D_ARRAY, 0);

	//管线配置
	Shader shader("Shader/vertex.vert", "Shader/pathtracing.frag");
	pass1.program = shader.ID;
	pass1.width = Width;
	pass1.height = Height;
	pass1.colorAttachments.push_back(RenderPass::GetTextureRGB32F(pass1.width, pass1.height));
	pass1.BindData();
	glUseProgram(pass1.program);
	glUniform1i(glGetUniformLocation(pass1.program, "nTriangles"), triangles.size());
	glUniform1i(glGetUniformLocation(pass1.program, "nNodes"), bvh.nodes.size());
	glUniform1i(glGetUniformLocation(pass1.program, "width"), pass1.width);
	glUniform1i(glGetUniformLocation(pass1.program, "height"), pass1.height);
	glUniform1i(glGetUniformLocation(pass1.program, "hdrResolution"), hdrResolution);//hdr分辨率

	glActiveTexture(GL_TEXTURE1);
	glBindTexture(GL_TEXTURE_BUFFER, bvh.trianglesTextureBuffer);
	glUniform1i(glGetUniformLocation(pass1.program, "triangles"), 1);

	glActiveTexture(GL_TEXTURE2);
	glBindTexture(GL_TEXTURE_BUFFER, bvh.nodesTextureBuffer);
	glUniform1i(glGetUniformLocation(pass1.program, "nodes"), 2);

	glActiveTexture(GL_TEXTURE3);
	glBindTexture(GL_TEXTURE_2D, hdrMap);
	glUniform1i(glGetUniformLocation(pass1.program, "hdrMap"), 3);

	glActiveTexture(GL_TEXTURE4);
	glBindTexture(GL_TEXTURE_2D, hdrCache);
	glUniform1i(glGetUniformLocation(pass1.program, "hdrCache"), 4);

	glActiveTexture(GL_TEXTURE5);
	glBindTexture(GL_TEXTURE_2D_ARRAY, albedo);
	glUniform1i(glGetUniformLocation(pass1.program, "albedoTextures"), 5);

	glActiveTexture(GL_TEXTURE6);
	glBindTexture(GL_TEXTURE_2D_ARRAY, metallic);
	glUniform1i(glGetUniformLocation(pass1.program, "metallicTextures"), 6);

	glActiveTexture(GL_TEXTURE7);
	glBindTexture(GL_TEXTURE_2D_ARRAY, roughness);
	glUniform1i(glGetUniformLocation(pass1.program, "roughnessTextures"), 7);

	glActiveTexture(GL_TEXTURE8);
	glBindTexture(GL_TEXTURE_2D_ARRAY, normal);
	glUniform1i(glGetUniformLocation(pass1.program, "normalTextures"), 8);

	glUseProgram(0);

	Shader shader2("Shader/vertex.vert", "Shader/lastframe.frag");
	pass2.program = shader2.ID;
	pass2.width = Width;
	pass2.height = Height;
	lastFrame = RenderPass::GetTextureRGB32F(pass2.width, pass2.height);
	pass2.colorAttachments.push_back(lastFrame);
	pass2.BindData();

	Shader shader3("Shader/vertex.vert", "Shader/postprocessing.frag");
	pass3.program = shader3.ID;
	pass3.width = Width;
	pass3.height = Height;
	pass3.BindData(true);
}

int main() {
#pragma region 初始化OpenGL
	glfwInit();//初始化GLFW
	glfwWindowHint(GLFW_CONTEXT_VERSION_MAJOR, 4);//OpenGL主版本号为4
	glfwWindowHint(GLFW_CONTEXT_VERSION_MINOR, 3);//OpenGL次版本号为3
	glfwWindowHint(GLFW_OPENGL_PROFILE, GLFW_OPENGL_CORE_PROFILE);//告诉GLFW我们使用的是核心模式

	GLFWwindow* window = glfwCreateWindow(Width, Height, "GLSL_PathTracing", NULL, NULL);//创建窗口
	if (window == NULL) {
		cout << "初始化窗口失败！" << endl;
		glfwTerminate();
		return -1;
	}
	glfwMakeContextCurrent(window);//将window窗口的上下文设置为当前线程的主上下文

	glfwSetInputMode(window, GLFW_CURSOR, GLFW_CURSOR_NORMAL);
	//glfwSetCursorPosCallback(window, mouse_callback);

	if (!gladLoadGLLoader((GLADloadproc)glfwGetProcAddress)) {
		cout << "初始化GLAD失败！" << endl;
		return -1;
	}

	glViewport(0, 0, Width, Height);//设置渲染视口的大小
#pragma endregion

	InitScene();

#pragma region 渲染循环
	while (!glfwWindowShouldClose(window)) {
		ProcessInput(window);

		//清空缓冲
		glClearColor(0.0f, 0.0f, 0.0f, 1.0f);
		glClear(GL_COLOR_BUFFER_BIT);

		//帧计时
		t2 = clock();
		dt = (double)(t2 - t1) / CLOCKS_PER_SEC;
		fps = 1.0 / dt;
		std::cout << "\r";
		std::cout << "FPS：" << fps << "     采样次数：" << frameCounter;
		t1 = t2;

		//相机参数eye(-3 * 根号3, 3, 0)
		vec3 eye = vec3(-sin(radians(camera.rotatAngle)) * cos(radians(camera.upAngle)), sin(radians(camera.upAngle)), cos(radians(camera.rotatAngle)) * cos(radians(camera.upAngle)));
		eye.x *= camera.r; eye.y *= camera.r; eye.z *= camera.r;
		mat4 cameraRotate = lookAt(eye, vec3(0.0f), vec3(0.0f, 1.0f, 0.0f));  //相机注视着原点
		cameraRotate = inverse(cameraRotate);   //lookat的逆矩阵将光线方向进行转换

		glUseProgram(pass1.program);
		glUniform3fv(glGetUniformLocation(pass1.program, "eye"), 1, value_ptr(eye));
		glUniformMatrix4fv(glGetUniformLocation(pass1.program, "cameraRotate"), 1, GL_FALSE, value_ptr(cameraRotate));
		glUniform1ui(glGetUniformLocation(pass1.program, "frameCounter"), frameCounter);//传计数器用作随机种子

		glActiveTexture(GL_TEXTURE0);
		glBindTexture(GL_TEXTURE_2D, lastFrame);
		glUniform1i(glGetUniformLocation(pass1.program, "lastFrame"), 0);

		//绘制
		pass1.Draw();
		pass2.Draw(pass1.colorAttachments);
		pass3.Draw(pass2.colorAttachments);

		frameCounter++;

		glfwSwapBuffers(window);
		glfwPollEvents();//检查有没有触发什么事件（比如键盘输入、鼠标移动等）、更新窗口状态，并调用对应的回调函数（可以通过回调方法手动设置）
	}
#pragma endregion

	glfwTerminate();
	return 0;
}