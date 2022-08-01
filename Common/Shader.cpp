#include "Shader.h"
#include <iostream>
#include <fstream>
#include <SStream>
#include<GLAD/glad.h>
#include<GLFW/glfw3.h>

using namespace std;

Shader::Shader(const char* vertexPath, const char* fragmentPath) {
	ifstream vertexFile;
	ifstream fragmentFile;
	stringstream vertexSStream;
	stringstream fragmentSStream;

	vertexFile.open(vertexPath);
	fragmentFile.open(fragmentPath);

	vertexFile.exceptions(ifstream::failbit || ifstream::badbit);
	fragmentFile.exceptions(ifstream::failbit || ifstream::badbit);
	try {
		if (!vertexFile.is_open() || !fragmentFile.is_open()) {
			throw exception("����ɫ���ļ�ʧ�ܣ�");
		}

		vertexSStream << vertexFile.rdbuf();
		fragmentSStream << fragmentFile.rdbuf();

		vertexString = vertexSStream.str();
		fragmentString = fragmentSStream.str();

		vertexSource = vertexString.c_str();
		fragmentSource = fragmentString.c_str();

		unsigned int vertex;
		vertex = glCreateShader(GL_VERTEX_SHADER);
		glShaderSource(vertex, 1, &vertexSource, NULL);
		glCompileShader(vertex);
		CheckCompileErrors(vertex, "VERTEX");

		unsigned int fragment;
		fragment = glCreateShader(GL_FRAGMENT_SHADER);
		glShaderSource(fragment, 1, &fragmentSource, NULL);
		glCompileShader(fragment);
		CheckCompileErrors(fragment, "FRAGMENT");

		ID = glCreateProgram();
		glAttachShader(ID, vertex);
		glAttachShader(ID, fragment);
		glLinkProgram(ID);
		CheckCompileErrors(ID, "PROGRAM");

		glDeleteShader(vertex);
		glDeleteShader(fragment);
	}
	catch (const std::exception& ex) {
		printf(ex.what());
		cout << endl;
	}
}

Shader::Shader(const char* vertexPath, const char* fragmentPath, const char* geometryPath){
	ifstream vertexFile;
	ifstream fragmentFile;
	ifstream geometryFile;
	stringstream vertexSStream;
	stringstream fragmentSStream;
	stringstream geometrySStream;
	//���ļ�
	vertexFile.open(vertexPath);
	fragmentFile.open(fragmentPath);
	geometryFile.open(geometryPath);
	//��֤ifstream��������׳��쳣��
	vertexFile.exceptions(ifstream::failbit || ifstream::badbit);
	fragmentFile.exceptions(ifstream::failbit || ifstream::badbit);
	geometryFile.exceptions(ifstream::failbit || ifstream::badbit);

	try{
		if (!vertexFile.is_open() || !fragmentFile.is_open() || !geometryFile.is_open()){
			throw exception("open file error");
		}

		//��ȡ�ļ��������ݵ�������
		vertexSStream << vertexFile.rdbuf();
		fragmentSStream << fragmentFile.rdbuf();
		geometrySStream << geometryFile.rdbuf();

		//ת����������string
		vertexString = vertexSStream.str();
		fragmentString = fragmentSStream.str();
		geometryString = geometrySStream.str();

		vertexSource = vertexString.c_str();
		fragmentSource = fragmentString.c_str();
		geometrySource = geometryString.c_str();

		//������ɫ��
		unsigned int vertex, fragment, geometry;
		//������ɫ��
		vertex = glCreateShader(GL_VERTEX_SHADER);
		glShaderSource(vertex, 1, &vertexSource, NULL);
		glCompileShader(vertex);
		CheckCompileErrors(vertex, "VERTEX");

		//Ƭ����ɫ��
		fragment = glCreateShader(GL_FRAGMENT_SHADER);
		glShaderSource(fragment, 1, &fragmentSource, NULL);
		glCompileShader(fragment);
		CheckCompileErrors(fragment, "FRAGMENT");

		//������ɫ��
		geometry = glCreateShader(GL_GEOMETRY_SHADER);
		glShaderSource(geometry, 1, &geometrySource, NULL);
		glCompileShader(geometry);
		CheckCompileErrors(geometry, "GEOMETRY");

		//��ɫ������
		ID = glCreateProgram();
		glAttachShader(ID, vertex);
		glAttachShader(ID, fragment);
		glAttachShader(ID, geometry);
		glLinkProgram(ID);
		CheckCompileErrors(ID, "PROGRAM");

		//ɾ����ɫ���������Ѿ����ӵ����ǵĳ������ˣ��Ѿ�������Ҫ��
		glDeleteShader(vertex);
		glDeleteShader(fragment);
		glDeleteShader(geometry);
	}
	catch (const std::exception& ex){
		printf(ex.what());
		cout << endl;
	}
}

void Shader::use(){
	glUseProgram(ID);
}

void Shader::SetUniform3f(const char* paraNameString, glm::vec3 param){
	glUniform3f(glGetUniformLocation(ID, paraNameString), param.x, param.y, param.z);
}

void Shader::SetUniform1f(const char* paraNameString, float param){
	glUniform1f(glGetUniformLocation(ID, paraNameString), param);
}

void Shader::SetUniform1i(const char* paraNameString, int slot){
	glUniform1i(glGetUniformLocation(ID, paraNameString), slot);
}

void Shader::CheckCompileErrors(unsigned int id, std::string type){
	int success;
	char infolog[512];

	if (type != "PROGRAM") {
		glGetShaderiv(id, GL_COMPILE_STATUS, &success);
		if (!success) {
			glGetShaderInfoLog(id, 512, NULL, infolog);
			cout << "��ɫ���������" << infolog << endl;
		}
	}
	else {
		glGetShaderiv(id, GL_LINK_STATUS, &success);
		if (!success) {
			glGetShaderInfoLog(id, 512, NULL, infolog);
			//cout << "��ɫ�����Ӵ���" << infolog << endl;
		}
	}
}
