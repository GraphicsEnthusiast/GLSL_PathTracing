#pragma once

#include <vector>
#include <string>
#include <GLAD/glad.h>
#include <GLFW/glfw3.h>
#include <glm/glm.hpp>
#include <glm/gtc/matrix_transform.hpp>
#include <glm/gtc/type_ptr.hpp>

using namespace glm;

struct Camera {
	float upAngle = 30.0f;
	float rotatAngle = 90.0f;
	float r = 6.0f;
};

class RenderPass {
public:
	GLuint FBO = 0;
	GLuint vao, vbo;
	std::vector<GLuint> colorAttachments;
	GLuint program;
	int width = 0;
	int height = 0;

public:
	void BindData(bool finalPass = false);
	void Draw(std::vector<GLuint> texPassArray = {});
	static mat4 GetTransformMatrix(vec3 rotateCtrl, vec3 translateCtrl, vec3 scaleCtrl);//Ä£ÐÍ±ä»»¾ØÕó
	static GLuint GetTextureRGB32F(int width, int height);
};