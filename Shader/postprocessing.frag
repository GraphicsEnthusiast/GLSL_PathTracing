#version 430 core

in vec3 pix;
out vec4 fragColor;

uniform sampler2D texPass0;
uniform sampler2D texPass1;
uniform sampler2D texPass2;
uniform sampler2D texPass3;
uniform sampler2D texPass4;
uniform sampler2D texPass5;
uniform sampler2D texPass6;

vec3 ToneMapping(in vec3 c, float limit) {
    float luminance = 0.3f * c.x + 0.6f * c.y + 0.1f * c.z;
    return c * 1.0f / (1.0f + luminance / limit);
}

void main() {
    vec3 color = texture2D(texPass0, pix.xy * 0.5f + 0.5f).rgb;
    color = ToneMapping(color, 1.5f);
    color = pow(color, vec3(1.0f / 2.2f));

    fragColor = vec4(color, 1.0f);
}
