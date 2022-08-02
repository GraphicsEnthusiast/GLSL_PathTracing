#include "RenderPass.h"

void RenderPass::BindData(bool finalPass) {
	if (!finalPass) {
		glGenFramebuffers(1, &FBO);
	}
	glBindFramebuffer(GL_FRAMEBUFFER, FBO);

	glGenBuffers(1, &vbo);
	glBindBuffer(GL_ARRAY_BUFFER, vbo);
	std::vector<vec3> square = { vec3(-1.0f, -1.0f, 0.0f), vec3(1.0f, -1.0f, 0.0f), vec3(-1.0f, 1.0f, 0.0f), vec3(1.0f, 1.0f, 0.0f), vec3(-1.0f, 1.0f, 0.0f), vec3(1.0f, -1.0f, 0.0f) };
	glBufferData(GL_ARRAY_BUFFER, sizeof(vec3) * square.size(), NULL, GL_STATIC_DRAW);
	glBufferSubData(GL_ARRAY_BUFFER, 0, sizeof(vec3) * square.size(), &square[0]);

	glGenVertexArrays(1, &vao);
	glBindVertexArray(vao);
	glEnableVertexAttribArray(0);//layout (location = 0) 
	glVertexAttribPointer(0, 3, GL_FLOAT, GL_FALSE, 0, (GLvoid*)0);
	//不是finalPass则生成帧缓冲的颜色附件
	if (!finalPass) {
		std::vector<GLuint> attachments;
		for (int i = 0; i < colorAttachments.size(); i++) {
			glBindTexture(GL_TEXTURE_2D, colorAttachments[i]);
			glFramebufferTexture2D(GL_FRAMEBUFFER, GL_COLOR_ATTACHMENT0 + i, GL_TEXTURE_2D, colorAttachments[i], 0);//将颜色纹理绑定到i号颜色附件
			attachments.push_back(GL_COLOR_ATTACHMENT0 + i);
		}
		glDrawBuffers(attachments.size(), &attachments[0]);
	}

	glBindFramebuffer(GL_FRAMEBUFFER, 0);
}

void RenderPass::Draw(std::vector<GLuint> texPassArray) {
	glUseProgram(program);
	glBindFramebuffer(GL_FRAMEBUFFER, FBO);
	glBindVertexArray(vao);
	//传上一帧的帧缓冲颜色附件
	for (int i = 0; i < texPassArray.size(); i++) {
		glActiveTexture(GL_TEXTURE0 + i);
		glBindTexture(GL_TEXTURE_2D, texPassArray[i]);
		std::string uName = "texPass" + std::to_string(i);
		glUniform1i(glGetUniformLocation(program, uName.c_str()), i);
	}
	glViewport(0, 0, width, height);
	glClear(GL_COLOR_BUFFER_BIT | GL_DEPTH_BUFFER_BIT);
	glDrawArrays(GL_TRIANGLES, 0, 6);

	glBindVertexArray(0);
	glBindFramebuffer(GL_FRAMEBUFFER, 0);
	glUseProgram(0);
}

mat4 RenderPass::GetTransformMatrix(vec3 translateCtrl, vec3 rotateCtrl, vec3 scaleCtrl) {
	mat4 unit(    //单位矩阵
		vec4(1.0f, 0.0f, 0.0f, 0.0f),
		vec4(0.0f, 1.0f, 0.0f, 0.0f),
		vec4(0.0f, 0.0f, 1.0f, 0.0f),
		vec4(0.0f, 0.0f, 0.0f, 1.0f)
	);
	mat4 scale = glm::scale(unit, scaleCtrl);
	mat4 translate = glm::translate(unit, translateCtrl);
	mat4 rotate = unit;
	rotate = glm::rotate(rotate, radians(rotateCtrl.x), vec3(1.0f, 0.0f, 0.0f));
	rotate = glm::rotate(rotate, radians(rotateCtrl.y), vec3(0.0f, 1.0f, 0.0f));
	rotate = glm::rotate(rotate, radians(rotateCtrl.z), vec3(0.0f, 0.0f, 1.0f));

	mat4 model = translate * rotate * scale;
	return model;
}

GLuint RenderPass::GetTextureRGB32F(int width, int height) {
	GLuint tex;
	glGenTextures(1, &tex);
	glBindTexture(GL_TEXTURE_2D, tex);
	glTexImage2D(GL_TEXTURE_2D, 0, GL_RGBA32F, width, height, 0, GL_RGBA, GL_FLOAT, NULL);
	glTexParameteri(GL_TEXTURE_2D, GL_TEXTURE_MIN_FILTER, GL_NEAREST);
	glTexParameteri(GL_TEXTURE_2D, GL_TEXTURE_MAG_FILTER, GL_NEAREST);
	glTexParameteri(GL_TEXTURE_2D, GL_TEXTURE_WRAP_S, GL_CLAMP_TO_EDGE);
	glTexParameteri(GL_TEXTURE_2D, GL_TEXTURE_WRAP_T, GL_CLAMP_TO_EDGE);
	return tex;
}
