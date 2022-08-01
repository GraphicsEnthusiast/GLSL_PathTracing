#version 430 core

in vec3 pix;

uniform sampler2D texPass0;
uniform sampler2D texPass1;
uniform sampler2D texPass2;
uniform sampler2D texPass3;
uniform sampler2D texPass4;
uniform sampler2D texPass5;
uniform sampler2D texPass6;

void main() {
    gl_FragData[0] = vec4(texture2D(texPass0, pix.xy * 0.5f + 0.5f).rgb, 1.0f);
}
