#version 430 core

layout (location = 0) in vec3 vPosition;  //cpu����Ķ�������

out vec3 pix;

void main() {
    gl_Position = vec4(vPosition, 1.0f);
    pix = vPosition;
}
