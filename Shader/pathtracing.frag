#version 430 core

in vec3 pix;

//uniform变量————————————————————————————————————
uniform uint frameCounter;
uniform int nTriangles;
uniform int nNodes;
uniform int width;
uniform int height;
uniform int hdrResolution;
uniform int numOfLights;

uniform samplerBuffer lightsTex;
uniform samplerBuffer triangles;
uniform samplerBuffer nodes;

uniform bool useEnv;
uniform sampler2D lastFrame;
uniform sampler2D hdrMap;
uniform sampler2D hdrCache;

uniform vec3 eye;
uniform mat4 cameraRotate;

uniform sampler2DArray albedoTextures;
uniform sampler2DArray metallicTextures;
uniform sampler2DArray roughnessTextures;
uniform sampler2DArray normalTextures;
//uniform变量————————————————————————————————————

//宏定义——————————————————————————————————————
#define PI 3.1415926f
#define INF 999999.0f
#define SIZE_TRIANGLE 16
#define SIZE_BVHNODE 4
//宏定义——————————————————————————————————————

//数据定义与获取——————————————————————————————————
//Triangle数据格式
struct Triangle {
    vec3 p1, p2, p3;    //顶点坐标
    vec3 n1, n2, n3;    //顶点法线
    vec2 uv1, uv2, uv3; //UV
};

//BVH树节点
struct BVHNode {
    int left;           //左子树
    int right;          //右子树
    int n;              //包含三角形数目
    int index;          //三角形索引
    vec3 AA, BB;        //碰撞盒
};

//物体表面材质定义
struct Material {
    vec3 emissive;          //作为光源时的发光颜色
    vec3 baseColor;
    float subsurface;
    float metallic;
    float specular;
    float specularTint;
    float roughness;
    float anisotropic;
    float sheen;
    float sheenTint;
    float clearcoat;
    float clearcoatGloss;
    float IOR;
    float transmission;
    float lightid;
};

//光线
struct Ray {
    vec3 origin;
    vec3 direction;
};

//光线求交结果
struct HitResult {
    bool isHit;             //是否命中
    bool isInside;          //是否从内部命中
    float distance;         //与交点的距离
    vec3 hitPoint;          //光线命中点
    vec3 normal;            //命中点法线
    vec2 uv;                //UV
    vec3 viewDir;           //击中该点的光线的方向
    Material material;      //命中点的表面材质
};

//重要性采样的返回结果
struct SampleResult {
    vec3 direction;
    float pdf;
};

struct Light { 
    vec3 position; 
    vec3 emission; 
    vec3 u; 
    vec3 v; 
    vec3 radiusAreaType; 
};

struct LightSampleRec { 
    vec3 surfacePos; 
    vec3 normal; 
    vec3 emission; 
    float pdf; 
};

//获取第i下标的三角形
Triangle GetTriangle(int i) {
    int offset = i * SIZE_TRIANGLE;
    Triangle t;

    //顶点坐标
    t.p1 = texelFetch(triangles, offset + 0).xyz;
    t.p2 = texelFetch(triangles, offset + 1).xyz;
    t.p3 = texelFetch(triangles, offset + 2).xyz;
    //法线
    t.n1 = texelFetch(triangles, offset + 3).xyz;
    t.n2 = texelFetch(triangles, offset + 4).xyz;
    t.n3 = texelFetch(triangles, offset + 5).xyz;
    //UV
    t.uv1 = texelFetch(triangles, offset + 6).xy;
    t.uv2 = texelFetch(triangles, offset + 7).xy;
    t.uv3 = texelFetch(triangles, offset + 8).xy;

    return t;
}

//获取第i下标的三角形的材质
Material GetMaterial(int i) {
    Material m;

    int offset = i * SIZE_TRIANGLE;
    vec3 param1 = texelFetch(triangles, offset + 11).xyz;
    vec3 param2 = texelFetch(triangles, offset + 12).xyz;
    vec3 param3 = texelFetch(triangles, offset + 13).xyz;
    vec3 param4 = texelFetch(triangles, offset + 14).xyz;
    
    m.emissive = texelFetch(triangles, offset + 9).xyz;
    m.baseColor = texelFetch(triangles, offset + 10).xyz;
    m.subsurface = param1.x;
    m.metallic = param1.y;
    m.specular = param1.z;
    m.specularTint = param2.x;
    m.roughness = param2.y;
    m.anisotropic = param2.z;
    m.sheen = param3.x;
    m.sheenTint = param3.y;
    m.clearcoat = param3.z;
    m.clearcoatGloss = param4.x;
    m.IOR = param4.y;
    m.transmission = param4.z;
    m.lightid = texelFetch(triangles, offset + 15).z;

    return m;
}

//获取第i下标的三角形的材质（纹理）
Material GetMaterial(int i, vec2 uv) {
    Material m;

    int offset = i * SIZE_TRIANGLE;
    vec3 param1 = texelFetch(triangles, offset + 11).xyz;
    vec3 param2 = texelFetch(triangles, offset + 12).xyz;
    vec3 param3 = texelFetch(triangles, offset + 13).xyz;
    vec3 param4 = texelFetch(triangles, offset + 14).xyz;
    
    m.emissive = texelFetch(triangles, offset + 9).xyz;
    m.baseColor = texture(albedoTextures, vec3(uv, int(texelFetch(triangles, offset + 15).y))).rgb;
    m.subsurface = param1.x;
    m.metallic = texture(metallicTextures, vec3(uv, int(texelFetch(triangles, offset + 15).y))).r;
    m.specular = param1.z;
    m.specularTint = param2.x;
    m.roughness = texture(roughnessTextures, vec3(uv, int(texelFetch(triangles, offset + 15).y))).r;
    m.anisotropic = param2.z;
    m.sheen = param3.x;
    m.sheenTint = param3.y;
    m.clearcoat = param3.z;
    m.clearcoatGloss = param4.x;
    m.IOR = param4.y;
    m.transmission = param4.z;

    return m;
}

//获取第i下标的BVHNode对象
BVHNode GetBVHNode(int i) {
    BVHNode node;

    //左右子树
    int offset = i * SIZE_BVHNODE;
    ivec3 childs = ivec3(texelFetch(nodes, offset + 0).xyz);
    ivec3 leafInfo = ivec3(texelFetch(nodes, offset + 1).xyz);
    node.left = int(childs.x);
    node.right = int(childs.y);
    node.n = int(leafInfo.x);
    node.index = int(leafInfo.y);

    //包围盒
    node.AA = texelFetch(nodes, offset + 2).xyz;
    node.BB = texelFetch(nodes, offset + 3).xyz;

    return node;
}

void GetTangent(vec3 N, inout vec3 tangent, inout vec3 bitangent) {
    vec3 helper = vec3(1.0f, 0.0f, 0.0f);
    if(abs(N.x) > 0.999f) {
        helper = vec3(0.0f, 0.0f, 1.0f);
    }
    bitangent = normalize(cross(N, helper));
    tangent = normalize(cross(N, bitangent));
}

//将向量v投影到N的法向半球
vec3 ToNormalHemisphere(vec3 v, vec3 N) {
    vec3 helper = vec3(1.0f, 0.0f, 0.0f);
    if(abs(N.x) > 0.999f) {
        helper = vec3(0.0f, 0.0f, 1.0f);
    }
    vec3 tangent = normalize(cross(N, helper));
    vec3 bitangent = normalize(cross(N, tangent));
    return v.x * tangent + v.y * bitangent + v.z * N;
}

//获取三角形中任意一点的uv
vec3 BarycentricCoord(vec3 point, vec3 v0, vec3 v1, vec3 v2) {
	vec3 ab = v1 - v0;
	vec3 ac = v2 - v0;
	vec3 ah = point - v0;

	float ab_ab = dot(ab, ab);
	float ab_ac = dot(ab, ac);
	float ac_ac = dot(ac, ac);
	float ab_ah = dot(ab, ah);
	float ac_ah = dot(ac, ah);

	float inv_denom = 1.0f / (ab_ab * ac_ac - ab_ac * ab_ac);

	float v = (ac_ac * ab_ah - ab_ac * ac_ah) * inv_denom;
	float w = (ab_ab * ac_ah - ab_ac * ab_ah) * inv_denom;
	float u = 1.0f - v - w;

	return vec3(u, v, w);
}
//数据定义与获取——————————————————————————————————

//光线求交与三角形遍历———————————————————————————————
//光线和三角形求交 
HitResult HitTriangle(int i, Triangle triangle, Ray ray) {
    HitResult res;
    res.distance = INF;
    res.isHit = false;
    res.isInside = false;

    vec3 p1 = triangle.p1;
    vec3 p2 = triangle.p2;
    vec3 p3 = triangle.p3;

    vec3 o = ray.origin;    //射线起点
    vec3 d = ray.direction;     //射线方向
    vec3 n = normalize(cross(p2 - p1, p3 - p1));    //法向量

    //从三角形背后（模型内部）击中
    if (dot(n, d) > 0.0f) {
        n = -n;   
        res.isInside = true;
    }

    //如果视线和三角形平行
    if (abs(dot(n, d)) < 0.00001f) {
        return res;
    }

    //距离
    float t = (dot(n, p1) - dot(o, n)) / dot(d, n);
    if (t < 0.0005f) {
        return res;    //如果三角形在光线背面
    }

    //交点计算
    vec3 p = o + d * t;

    //判断交点是否在三角形中
    vec3 c1 = cross(p2 - p1, p - p1);
    vec3 c2 = cross(p3 - p2, p - p2);
    vec3 c3 = cross(p1 - p3, p - p3);

    //命中，封装返回结果
    if (dot(c1, c2) > 0.0f && dot(c2, c3) > 0.0f && dot(c1, c3) > 0.0f) {
        vec3 bary = BarycentricCoord(p, triangle.p1, triangle.p2, triangle.p3);
        res.uv = triangle.uv1 * bary.x + triangle.uv2 * bary.y + triangle.uv3 * bary.z;
        res.uv.y = 1.0f - res.uv.y;
        res.isHit = true;
        res.hitPoint = p;
        res.distance = t;
        res.viewDir = d;
        
        //根据交点位置插值顶点法线
        float alpha = (-(p.x - p2.x) * (p3.y - p2.y) + (p.y - p2.y) * (p3.x - p2.x)) / (-(p1.x - p2.x) * (p3.y - p2.y) + (p1.y - p2.y) * (p3.x - p2.x) + 1e-7);
        float beta  = (-(p.x - p3.x) * (p1.y - p3.y) + (p.y - p3.y) * (p1.x - p3.x)) / (-(p2.x - p3.x) * (p1.y - p3.y) + (p2.y - p3.y) *( p1.x - p3.x) + 1e-7);
        float gama  = 1.0f - alpha - beta;
        vec3 Nsmooth = alpha * triangle.n1 + beta * triangle.n2 + gama * triangle.n3;
        Nsmooth = normalize(Nsmooth);
        vec3 n = Nsmooth;

        int offset = i * SIZE_TRIANGLE;
        if(texelFetch(triangles, offset + 15).x < 0.0f) {//没有使用贴图
            res.normal = (res.isInside) ? (-Nsmooth) : (Nsmooth);
        }
        else {//使用贴图
            //vec3 tangentNormal = texture(normalTextures, vec3(res.uv, int(texelFetch(triangles, offset + 15).y))).xyz;
            //tangentNormal = normalize(tangentNormal * 0.5f + 0.5f);
            //res.normal = ToNormalHemisphere(tangentNormal, n);

            vec3 tangentNormal = texture(normalTextures, vec3(res.uv, int(texelFetch(triangles, offset + 15).y))).xyz;
		    tangentNormal = normalize(tangentNormal * 2.0f - 1.0f);
            
		    //Orthonormal Basis
		    vec3 UpVector = abs(n.z) < 0.999f ? vec3(0.0f, 0.0f, 1.0f) : vec3(1.0f, 0.0f, 0.0f);
		    vec3 TangentX = normalize(cross(UpVector, n));
		    vec3 TangentY = cross(n, TangentX);
            
		    vec3 nrm = TangentX * tangentNormal.x + TangentY * tangentNormal.y + n * tangentNormal.z;
		    res.normal = (res.isInside) ? -nrm : nrm;
        }

    }

    return res;
}

//和aabb盒子求交，没有交点则返回-1
float HitAABB(Ray r, vec3 AA, vec3 BB) {
    vec3 invdir = 1.0f / r.direction;

    vec3 f = (BB - r.origin) * invdir;
    vec3 n = (AA - r.origin) * invdir;

    vec3 tmax = max(f, n);
    vec3 tmin = min(f, n);

    float t1 = min(tmax.x, min(tmax.y, tmax.z));
    float t0 = max(tmin.x, max(tmin.y, tmin.z));

    return (t1 >= t0) ? ((t0 > 0.0f) ? (t0) : (t1)) : (-1.0f);
}

//暴力遍历数组下标范围[l, r]求最近交点
HitResult HitArray(Ray ray, int l, int r) {
    HitResult res;
    res.isHit = false;
    res.distance = INF;
    for(int i = l; i <= r; i++) {
        Triangle triangle = GetTriangle(i);
        HitResult r = HitTriangle(i, triangle, ray);
        if(r.isHit && r.distance < res.distance) {
            res = r;
            int offset = i * SIZE_TRIANGLE;
            if(texelFetch(triangles, offset + 15).x < 0.0f) {//没有使用贴图
                res.material = GetMaterial(i);
            }
            else {
                res.material = GetMaterial(i, res.uv);//使用贴图
            }
        }
    }
    return res;
}

//遍历BVH求交
HitResult HitBVH(Ray ray) {
    HitResult res;
    res.isHit = false;
    res.distance = INF;

    //栈
    int stack[256];
    int sp = 0;

    //BVH节点下标从0开始
    stack[sp++] = 0;
    while(sp > 0) {
        int top = stack[--sp];
        BVHNode node = GetBVHNode(top);
        
        //是叶子节点，遍历三角形，求最近交点
        if(node.n > 0) {
            int L = node.index;
            int R = node.index + node.n - 1;
            HitResult r = HitArray(ray, L, R);
            if(r.isHit && r.distance < res.distance) {
                res = r;
            }
            continue;
        }
        
        //和左右盒子AABB求交
        float d1 = INF; //左盒子距离
        float d2 = INF; //右盒子距离
        if(node.left > 0) {
            BVHNode leftNode = GetBVHNode(node.left);
            d1 = HitAABB(ray, leftNode.AA, leftNode.BB);
        }
        if(node.right > 0) {
            BVHNode rightNode = GetBVHNode(node.right);
            d2 = HitAABB(ray, rightNode.AA, rightNode.BB);
        }

        //在最近的盒子中搜索
        if(d1 > 0 && d2 > 0) {
            if(d1 < d2) { //d1<d2,左边先
                stack[sp++] = node.right;
                stack[sp++] = node.left;
            } 
            else {    //d2<d1,右边先
                stack[sp++] = node.left;
                stack[sp++] = node.right;
            }
        } 
        else if(d1 > 0) {   //仅命中左边
            stack[sp++] = node.left;
        } 
        else if(d2 > 0) {   //仅命中右边
            stack[sp++] = node.right;
        }
    }

    return res;
}
//光线求交与三角形遍历———————————————————————————————

//随机数与低差异序列————————————————————————————————
//生成随机向量，依赖于frameCounter帧计数器
uint seed = uint(
    uint((pix.x * 0.5f + 0.5f) * width)  * uint(1973) + 
    uint((pix.y * 0.5f + 0.5f) * height) * uint(9277) + 
    uint(frameCounter) * uint(26699)) | uint(1);

uint wang_hash(inout uint seed) {
    seed = uint(seed ^ uint(61)) ^ uint(seed >> uint(16));
    seed *= uint(9);
    seed = seed ^ (seed >> 4);
    seed *= uint(0x27d4eb2d);
    seed = seed ^ (seed >> 15);
    return seed;
}
 
float rand() {
    return float(wang_hash(seed)) / 4294967296.0f;
}

uint seed_sync = uint(
    uint((pix.x * 0.0f + 0.5f) * width)  * uint(1973) + 
    uint((pix.y * 0.0f + 0.5f) * height) * uint(9277) + 
    uint(114514) * uint(26699)) | uint(1);

float rand_sync() {
    return float(wang_hash(seed_sync)) / 4294967296.0f;
}

//1 ~ 8 维的Sobol生成矩阵
const uint V[8 * 32] = {
    2147483648, 1073741824, 536870912, 268435456, 134217728, 67108864, 33554432, 16777216, 8388608, 4194304, 2097152, 1048576, 524288, 262144, 131072, 65536, 32768, 16384, 8192, 4096, 2048, 1024, 512, 256, 128, 64, 32, 16, 8, 4, 2, 1,
    2147483648, 3221225472, 2684354560, 4026531840, 2281701376, 3422552064, 2852126720, 4278190080, 2155872256, 3233808384, 2694840320, 4042260480, 2290614272, 3435921408, 2863267840, 4294901760, 2147516416, 3221274624, 2684395520, 4026593280, 2281736192, 3422604288, 2852170240, 4278255360, 2155905152, 3233857728, 2694881440, 4042322160, 2290649224, 3435973836, 2863311530, 4294967295,
    2147483648, 3221225472, 1610612736, 2415919104, 3892314112, 1543503872, 2382364672, 3305111552, 1753219072, 2629828608, 3999268864, 1435500544, 2154299392, 3231449088, 1626210304, 2421489664, 3900735488, 1556135936, 2388680704, 3314585600, 1751705600, 2627492864, 4008611328, 1431684352, 2147543168, 3221249216, 1610649184, 2415969680, 3892340840, 1543543964, 2382425838, 3305133397,
    2147483648, 3221225472, 536870912, 1342177280, 4160749568, 1946157056, 2717908992, 2466250752, 3632267264, 624951296, 1507852288, 3872391168, 2013790208, 3020685312, 2181169152, 3271884800, 546275328, 1363623936, 4226424832, 1977167872, 2693105664, 2437829632, 3689389568, 635137280, 1484783744, 3846176960, 2044723232, 3067084880, 2148008184, 3222012020, 537002146, 1342505107,
    2147483648, 1073741824, 536870912, 2952790016, 4160749568, 3690987520, 2046820352, 2634022912, 1518338048, 801112064, 2707423232, 4038066176, 3666345984, 1875116032, 2170683392, 1085997056, 579305472, 3016343552, 4217741312, 3719483392, 2013407232, 2617981952, 1510979072, 755882752, 2726789248, 4090085440, 3680870432, 1840435376, 2147625208, 1074478300, 537900666, 2953698205,
    2147483648, 1073741824, 1610612736, 805306368, 2818572288, 335544320, 2113929216, 3472883712, 2290089984, 3829399552, 3059744768, 1127219200, 3089629184, 4199809024, 3567124480, 1891565568, 394297344, 3988799488, 920674304, 4193267712, 2950604800, 3977188352, 3250028032, 129093376, 2231568512, 2963678272, 4281226848, 432124720, 803643432, 1633613396, 2672665246, 3170194367,
    2147483648, 3221225472, 2684354560, 3489660928, 1476395008, 2483027968, 1040187392, 3808428032, 3196059648, 599785472, 505413632, 4077912064, 1182269440, 1736704000, 2017853440, 2221342720, 3329785856, 2810494976, 3628507136, 1416089600, 2658719744, 864310272, 3863387648, 3076993792, 553150080, 272922560, 4167467040, 1148698640, 1719673080, 2009075780, 2149644390, 3222291575,
    2147483648, 1073741824, 2684354560, 1342177280, 2281701376, 1946157056, 436207616, 2566914048, 2625634304, 3208642560, 2720006144, 2098200576, 111673344, 2354315264, 3464626176, 4027383808, 2886631424, 3770826752, 1691164672, 3357462528, 1993345024, 3752330240, 873073152, 2870150400, 1700563072, 87021376, 1097028000, 1222351248, 1560027592, 2977959924, 23268898, 437609937
};

float sqr(float x) { 
    return x * x; 
}

//格林码 
uint GrayCode(uint i) {
	return i ^ (i >> 1);
}

//生成第d维度的第i个Sobol数
float Sobol(uint d, uint i) {
    uint result = 0;
    uint offset = d * 32;
    for(uint j = 0; i != 0; i >>= 1, j++) {
        if((i & 1) != 0) {
            result ^= V[j+offset];
        }
    }

    return float(result) * (1.0f / float(0xFFFFFFFFU));
}

//生成第i帧的第b次反弹需要的二维随机向量
vec2 SobolVec2(uint i, uint b) {
    float u = Sobol(b * 2, GrayCode(i));
    float v = Sobol(b * 2 + 1, GrayCode(i));
    return vec2(u, v);
}

vec2 CranleyPattersonRotation(vec2 p) {
    uint pseed = uint(
        uint((pix.x * 0.5 + 0.5) * width)  * uint(1973) + 
        uint((pix.y * 0.5 + 0.5) * height) * uint(9277) + 
        uint(114514 / 1919) * uint(26699)) | uint(1);
    
    float u = float(wang_hash(pseed)) / 4294967296.0f;
    float v = float(wang_hash(pseed)) / 4294967296.0f;

    p.x += u;
    if(p.x > 1.0f) {
        p.x -= 1.0f;
    }
    if(p.x < 0) {
        p.x += 1.0f;
    }

    p.y += v;
    if(p.y > 1) {
        p.y -= 1.0f;
    }
    if(p.y < 0) {
        p.y += 1.0f;
    }

    return p;
}
//随机数与低差异序列————————————————————————————————

//BRDF及相关计算——————————————————————————————————
float SchlickFresnel(float u) {
    float m = clamp(1.0f - u, 0.0f, 1.0f);
    float m2 = m * m;
    return m2 * m2 * m; //pow(m,5)
}

float GTR1(float NdotH, float a) {
    if (a >= 1.0f) {
        return 1.0f / PI;
    }
    float a2 = a * a;
    float t = 1.0f + (a2 - 1.0f) * NdotH * NdotH;
    return (a2 - 1.0f) / (PI * log(a2) * t);
}

float GTR2(float NdotH, float a) {
    float a2 = a * a;
    float t = 1.0f + (a2 - 1.0f) * NdotH * NdotH;
    return a2 / (PI * t * t);
}

float GTR2_aniso(float NdotH, float HdotX, float HdotY, float ax, float ay) {
    return 1.0f / (PI * ax * ay * sqr(sqr(HdotX / ax) + sqr(HdotY / ay) + NdotH * NdotH));
}

float SmithG_GGX(float NdotV, float alphaG) {
    float a = alphaG * alphaG;
    float b = NdotV * NdotV;
    return 1.0f / (NdotV + sqrt(a + b - a * b));
}

float SmithG_GGX_aniso(float NdotV, float VdotX, float VdotY, float ax, float ay) {
    return 1.0f / (NdotV + sqrt(sqr(VdotX * ax) + sqr(VdotY * ay) + sqr(NdotV)));
}

vec3 BRDF_Evaluate_aniso(vec3 V, vec3 N, vec3 L, vec3 X, vec3 Y, in Material material) {
    float NdotL = dot(N, L);
    float NdotV = dot(N, V);
    if(NdotL < 0.0f || NdotV < 0.0f) {
        return vec3(0.0f);
    }

    vec3 H = normalize(L + V);
    float NdotH = dot(N, H);
    float LdotH = dot(L, H);

    //各种颜色
    vec3 Cdlin = material.baseColor;
    float Cdlum = 0.3f * Cdlin.r + 0.6f * Cdlin.g + 0.1f * Cdlin.b;
    vec3 Ctint = (Cdlum > 0.0f) ? (Cdlin/Cdlum) : (vec3(1.0f));   
    vec3 Cspec = material.specular * mix(vec3(1.0f), Ctint, material.specularTint);
    vec3 Cspec0 = mix(0.08f * Cspec, Cdlin, material.metallic); //0°镜面反射颜色
    vec3 Csheen = mix(vec3(1.0f), Ctint, material.sheenTint);   //织物颜色

    //漫反射
    float Fd90 = 0.5f + 2.0f * LdotH * LdotH * material.roughness;
    float FL = SchlickFresnel(NdotL);
    float FV = SchlickFresnel(NdotV);
    float Fd = mix(1.0f, Fd90, FL) * mix(1.0f, Fd90, FV);

    //次表面散射
    float Fss90 = LdotH * LdotH * material.roughness;
    float Fss = mix(1.0f, Fss90, FL) * mix(1.0f, Fss90, FV);
    float ss = 1.25f * (Fss * (1.0f / (NdotL + NdotV) - 0.5f) + 0.5f);
     
    //镜面反射--各向异性
    float aspect = sqrt(1.0f - material.anisotropic * 0.9f);
    float ax = max(0.001f, sqr(material.roughness) / aspect);
    float ay = max(0.001f, sqr(material.roughness) * aspect);
    float Ds = GTR2_aniso(NdotH, dot(H, X), dot(H, Y), ax, ay);
    float FH = SchlickFresnel(LdotH);
    vec3 Fs = mix(Cspec0, vec3(1), FH);
    float Gs;
    Gs  = SmithG_GGX_aniso(NdotL, dot(L, X), dot(L, Y), ax, ay);
    Gs *= SmithG_GGX_aniso(NdotV, dot(V, X), dot(V, Y), ax, ay);

    //清漆
    float Dr = GTR1(NdotH, mix(0.1f, 0.001f, material.clearcoatGloss));
    float Fr = mix(0.04f, 1.0f, FH);
    float Gr = SmithG_GGX(NdotL, 0.25f) * SmithG_GGX(NdotV, 0.25f);

    //sheen
    vec3 Fsheen = FH * material.sheen * Csheen;
    
    vec3 diffuse = (1.0f / PI) * mix(Fd, ss, material.subsurface) * Cdlin + Fsheen;
    vec3 specular = Gs * Fs * Ds;
    vec3 clearcoat = vec3(0.25f * Gr * Fr * Dr * material.clearcoat);

    return diffuse * (1.0f - material.metallic) + specular + clearcoat;
}

vec3 BRDF_Evaluate(vec3 V, vec3 N, vec3 L, in Material material) {
    float NdotL = dot(N, L);
    float NdotV = dot(N, V);
    if(NdotL < 0.0f || NdotV < 0.0f) {
        return vec3(0.0f);
    }

    vec3 H = normalize(L + V);
    float NdotH = dot(N, H);
    float LdotH = dot(L, H);

    //各种颜色
    vec3 Cdlin = material.baseColor;
    float Cdlum = 0.3f * Cdlin.r + 0.6f * Cdlin.g + 0.1f * Cdlin.b;
    vec3 Ctint = (Cdlum > 0.0f) ? (Cdlin/Cdlum) : (vec3(1.0f));   
    vec3 Cspec = material.specular * mix(vec3(1.0f), Ctint, material.specularTint);
    vec3 Cspec0 = mix(0.08f * Cspec, Cdlin, material.metallic); //0°镜面反射颜色
    vec3 Csheen = mix(vec3(1.0f), Ctint, material.sheenTint);   //织物颜色

    //漫反射
    float Fd90 = 0.5f + 2.0f * LdotH * LdotH * material.roughness;
    float FL = SchlickFresnel(NdotL);
    float FV = SchlickFresnel(NdotV);
    float Fd = mix(1.0f, Fd90, FL) * mix(1.0f, Fd90, FV);

    //次表面散射
    float Fss90 = LdotH * LdotH * material.roughness;
    float Fss = mix(1.0f, Fss90, FL) * mix(1.0f, Fss90, FV);
    float ss = 1.25f * (Fss * (1.0f / (NdotL + NdotV) - 0.5f) + 0.5f);
     
    //镜面反射--各向同性
    float alpha = max(0.001f, sqr(material.roughness));
    float Ds = GTR2(NdotH, alpha);
    float FH = SchlickFresnel(LdotH);
    vec3 Fs = mix(Cspec0, vec3(1.0f), FH);
    float Gs = SmithG_GGX(NdotL, material.roughness);
    Gs *= SmithG_GGX(NdotV, material.roughness);

    //清漆
    float Dr = GTR1(NdotH, mix(0.1f, 0.001f, material.clearcoatGloss));
    float Fr = mix(0.04f, 1.0f, FH);
    float Gr = SmithG_GGX(NdotL, 0.25f) * SmithG_GGX(NdotV, 0.25f);

    //sheen
    vec3 Fsheen = FH * material.sheen * Csheen;
    
    vec3 diffuse = (1.0f / PI) * mix(Fd, ss, material.subsurface) * Cdlin + Fsheen;
    vec3 specular = Gs * Fs * Ds;
    vec3 clearcoat = vec3(0.25f * Gr * Fr * Dr * material.clearcoat);

    return diffuse * (1.0f - material.metallic) + specular + clearcoat;
}
//BRDF及相关计算——————————————————————————————————

//采样———————————————————————————————————————
//半球均匀采样
vec3 SampleHemisphere(float xi_1, float xi_2) {
    //xi_1 = rand(), xi_2 = rand();
    float z = xi_1;
    float r = max(0.0f, sqrt(1.0f - z*z));
    float phi = 2.0f * PI * xi_2;
    return vec3(r * cos(phi), r * sin(phi), z);
}

//余弦加权的法向半球采样
vec3 SampleCosineHemisphere(float xi_1, float xi_2, vec3 N) {
    //均匀采样xy圆盘然后投影到z半球
    float r = sqrt(xi_1);
    float theta = xi_2 * 2.0f * PI;
    float x = r * cos(theta);
    float y = r * sin(theta);
    float z = sqrt(1.0f - x * x - y * y);

    //从z半球投影到法向半球
    vec3 L = ToNormalHemisphere(vec3(x, y, z), N);
    return L;
}

//GTR2重要性采样
vec3 SampleGTR2(float xi_1, float xi_2, vec3 V, vec3 N, float alpha) {
    
    float phi_h = 2.0f * PI * xi_1;
    float sin_phi_h = sin(phi_h);
    float cos_phi_h = cos(phi_h);

    float cos_theta_h = sqrt((1.0f - xi_2)/(1.0f + (alpha*alpha - 1.0f) * xi_2));
    float sin_theta_h = sqrt(max(0.0f, 1.0f - cos_theta_h * cos_theta_h));

    //采样 "微平面" 的法向量 作为镜面反射的半角向量h 
    vec3 H = vec3(sin_theta_h * cos_phi_h, sin_theta_h * sin_phi_h, cos_theta_h);
    H = ToNormalHemisphere(H, N);   //投影到真正的法向半球

    //根据 "微法线" 计算反射光方向
    vec3 L = reflect(-V, H);

    return L;
}

//GTR1重要性采样
vec3 SampleGTR1(float xi_1, float xi_2, vec3 V, vec3 N, float alpha) {
    
    float phi_h = 2.0f * PI * xi_1;
    float sin_phi_h = sin(phi_h);
    float cos_phi_h = cos(phi_h);

    float cos_theta_h = sqrt((1.0f - pow(alpha*alpha, 1.0f - xi_2))/(1.0f - alpha * alpha));
    float sin_theta_h = sqrt(max(0.0f, 1.0f - cos_theta_h * cos_theta_h));

    //采样 "微平面" 的法向量 作为镜面反射的半角向量h 
    vec3 H = vec3(sin_theta_h * cos_phi_h, sin_theta_h * sin_phi_h, cos_theta_h);
    H = ToNormalHemisphere(H, N);   //投影到真正的法向半球

    //根据 "微法线" 计算反射光方向
    vec3 L = reflect(-V, H);

    return L;
}

//按照辐射度分布分别采样三种BRDF
vec3 SampleBRDF(float xi_1, float xi_2, float xi_3, vec3 V, vec3 N, in Material material) {
    float alpha_GTR1 = mix(0.1f, 0.001f, material.clearcoatGloss);
    float alpha_GTR2 = max(0.001f, sqr(material.roughness));
    
    //辐射度统计
    float r_diffuse = (1.0f - material.metallic);
    float r_specular = 1.0f;
    float r_clearcoat = 0.25f * material.clearcoat;
    float r_sum = r_diffuse + r_specular + r_clearcoat;

    //根据辐射度计算概率
    float p_diffuse = r_diffuse / r_sum;
    float p_specular = r_specular / r_sum;
    float p_clearcoat = r_clearcoat / r_sum;

    //按照概率采样
    float rd = xi_3;

    //漫反射
    if(rd <= p_diffuse) {
        return SampleCosineHemisphere(xi_1, xi_2, N);
    } 
    //镜面反射
    else if(p_diffuse < rd && rd <= p_diffuse + p_specular) {    
        return SampleGTR2(xi_1, xi_2, V, N, alpha_GTR2);
    } 
    //清漆
    else if(p_diffuse + p_specular < rd) {
        return SampleGTR1(xi_1, xi_2, V, N, alpha_GTR1);
    }
    return vec3(0.0f, 1.0f, 0.0f);
}

//采样预计算的HDRcache
vec3 SampleHdr(float xi_1, float xi_2) {
    vec2 xy = texture2D(hdrCache, vec2(xi_1, xi_2)).rg; //x, y
    xy.y = 1.0f - xy.y; //flip y

    //获取角度
    float phi = 2.0f * PI * (xy.x - 0.5f);    //[-pi ~ pi]
    float theta = PI * (xy.y - 0.5f);        //[-pi/2 ~ pi/2]   

    //球坐标计算方向
    vec3 L = vec3(cos(theta) * cos(phi), sin(theta), cos(theta) * sin(phi));

    return L;
}

//将三维向量v转为HDRmap的纹理坐标uv
vec2 ToSphericalCoord(vec3 v) {
    vec2 uv = vec2(atan(v.z, v.x), asin(v.y));
    uv /= vec2(2.0f * PI, PI);
    uv += 0.5f;
    uv.y = 1.0f - uv.y;
    return uv;
}

//获取HDR环境颜色
vec3 HdrColor(vec3 L) {
    vec2 uv = ToSphericalCoord(normalize(L));
    vec3 color = texture2D(hdrMap, uv).rgb;
    return color;
}

//输入光线方向L获取HDR在该位置的概率密度
//hdr分辨率为4096 x 2048 --> hdrResolution = 4096
float HdrPdf(vec3 L, int hdrResolution) {
    vec2 uv = ToSphericalCoord(normalize(L));   //方向向量转uv纹理坐标

    float pdf = texture2D(hdrCache, uv).b;      //采样概率密度
    float theta = PI * (0.5f - uv.y);            //theta范围[-pi/2 ~ pi/2]
    float sin_theta = max(sin(theta), 1e-10);

    //球坐标和图片积分域的转换系数
    float p_convert = float(hdrResolution * hdrResolution / 2.0f) / (2.0f * PI * PI * sin_theta);  
    
    return pdf * p_convert;
}

//获取BRDF在L方向上的概率密度
float BRDF_Pdf(vec3 V, vec3 N, vec3 L, in Material material) {
    float NdotL = dot(N, L);
    float NdotV = dot(N, V);
    if(NdotL < 0.0f || NdotV < 0.0f) {
        return 0.0f;
    }

    vec3 H = normalize(L + V);
    float NdotH = dot(N, H);
    float LdotH = dot(L, H);
     
    //镜面反射--各向同性
    float alpha = max(0.001f, sqr(material.roughness));
    float Ds = GTR2(NdotH, alpha); 
    float Dr = GTR1(NdotH, mix(0.1f, 0.001f, material.clearcoatGloss));   //清漆

    //分别计算三种BRDF的概率密度
    float pdf_diffuse = NdotL / PI;
    float pdf_specular = Ds * NdotH / (4.0f * dot(L, H));
    float pdf_clearcoat = Dr * NdotH / (4.0f * dot(L, H));

    //辐射度统计
    float r_diffuse = (1.0f - material.metallic);
    float r_specular = 1.0f;
    float r_clearcoat = 0.25f * material.clearcoat;
    float r_sum = r_diffuse + r_specular + r_clearcoat;

    //根据辐射度计算选择某种采样方式的概率
    float p_diffuse = r_diffuse / r_sum;
    float p_specular = r_specular / r_sum;
    float p_clearcoat = r_clearcoat / r_sum;

    //根据概率混合pdf
    float pdf = p_diffuse   * pdf_diffuse 
              + p_specular  * pdf_specular
              + p_clearcoat * pdf_clearcoat;

    pdf = max(1e-10, pdf);
    return pdf;
}

float MisMixWeight(float a, float b) {
    float t = a * a;
    return t / (b * b + t);
}

vec3 UniformSampleSphere(float u1, float u2) {
	float z = 1.0f - 2.0f * u1;
	float r = sqrt(max(0.0f, 1.0f - z * z));
	float phi = 2.0f * PI * u2;
	float x = r * cos(phi);
	float y = r * sin(phi);

	return vec3(x, y, z);
}

void SampleSphereLight(in Light light, inout LightSampleRec lightSampleRec) {
	float r1 = rand();
	float r2 = rand();

	lightSampleRec.surfacePos = light.position + UniformSampleSphere(r1, r2) * light.radiusAreaType.x;
	lightSampleRec.normal = normalize(lightSampleRec.surfacePos - light.position);
	lightSampleRec.emission = light.emission * numOfLights;
}

void SampleQuadLight(in Light light, inout LightSampleRec lightSampleRec) {
	float r1 = rand();
	float r2 = rand();

	lightSampleRec.surfacePos = light.position + light.u * r1 + light.v * r2;
	lightSampleRec.normal = normalize(cross(light.u, light.v));
	lightSampleRec.emission = light.emission * numOfLights;
}

void SampleLight(in Light light, inout LightSampleRec lightSampleRec) {
	if (int(light.radiusAreaType.z) == 0) {//Quad Light
		SampleQuadLight(light, lightSampleRec);
    }
	else {
        SampleSphereLight(light, lightSampleRec);
    }
}

float Light_Pdf(int index, in LightSampleRec lightSampleRec, HitResult hit) {
    Light light;

	//Fetch light Data
	vec3 p = texelFetch(lightsTex, index * 5 + 0).xyz;
	vec3 e = texelFetch(lightsTex, index * 5 + 1).xyz;
	vec3 u = texelFetch(lightsTex, index * 5 + 2).xyz;
	vec3 v = texelFetch(lightsTex, index * 5 + 3).xyz;
	vec3 rad = texelFetch(lightsTex, index * 5 + 4).xyz;

	light = Light(p, e, u, v, rad);

	vec3 lightDir = lightSampleRec.surfacePos - hit.hitPoint;
	float lightDist = length(lightDir);
	float lightDistSq = lightDist * lightDist;
	lightDir /= sqrt(lightDistSq);

    float pdf_light = lightDistSq / (light.radiusAreaType.y * abs(dot(lightSampleRec.normal, lightDir)));

    return pdf_light;
}
//采样———————————————————————————————————————

void DirectLight(HitResult hit, inout vec3 Lo, in vec3 history) {
    vec3 V = -hit.viewDir;
    vec3 N = hit.normal;      
    if(useEnv) {
        //HDR环境贴图重要性采样    
        Ray hdrTestRay;
        hdrTestRay.origin = hit.hitPoint;
        hdrTestRay.direction = SampleHdr(rand(), rand());
        
        //进行一次求交测试，判断是否有遮挡
        if(dot(N, hdrTestRay.direction) > 0.0f) { //如果采样方向背向点p则放弃测试，因为N dot L < 0            
            HitResult hdrHit = HitBVH(hdrTestRay);
            
            //天空光仅在没有遮挡的情况下积累亮度
            if(!hdrHit.isHit) {
                //获取采样方向L上的: 1.光照贡献, 2.环境贴图在该位置的pdf, 3.BRDF函数值, 4.BRDF在该方向的pdf
                vec3 L = hdrTestRay.direction;
                vec3 color = HdrColor(L);
                float pdf_light = HdrPdf(L, hdrResolution);
                vec3 f_r = BRDF_Evaluate(V, N, L, hit.material);
                float pdf_brdf = BRDF_Pdf(V, N, L, hit.material);
                
                //多重重要性采样
                float mis_weight = MisMixWeight(pdf_light, pdf_brdf);
                Lo += mis_weight * history * color * f_r * dot(N, L) / pdf_light;  
            }
        }
    }

    if (numOfLights > 0) {
		LightSampleRec lightSampleRec;

        //Pick a light to sample
		int index = int(rand() * numOfLights);
		Light light;

		//Fetch light Data
		vec3 p = texelFetch(lightsTex, index * 5 + 0).xyz;
		vec3 e = texelFetch(lightsTex, index * 5 + 1).xyz;
		vec3 u = texelFetch(lightsTex, index * 5 + 2).xyz;
		vec3 v = texelFetch(lightsTex, index * 5 + 3).xyz;
		vec3 rad = texelFetch(lightsTex, index * 5 + 4).xyz;

		light = Light(p, e, u, v, rad);
		SampleLight(light, lightSampleRec);

		vec3 lightDir = lightSampleRec.surfacePos - hit.hitPoint;
		float lightDist = length(lightDir);
		float lightDistSq = lightDist * lightDist;
		lightDir /= sqrt(lightDistSq);

		if (dot(lightDir, hit.normal) <= 0.0f || dot(lightDir, lightSampleRec.normal) >= 0.0f) {
            return;
        }

		Ray shadowRay;
        shadowRay.origin = hit.hitPoint;
        shadowRay.direction = lightDir;
		HitResult lightHit = HitBVH(shadowRay);

		if (!lightHit.isHit) {
            vec3 f_r = BRDF_Evaluate(V, N, lightDir, hit.material);
            float pdf_brdf = BRDF_Pdf(V, N, lightDir, hit.material);
            float pdf_light = lightDistSq / (light.radiusAreaType.y * abs(dot(lightSampleRec.normal, lightDir)));
                
            //多重重要性采样
            float mis_weight = MisMixWeight(pdf_light, pdf_brdf);
            Lo += mis_weight * history * light.emission * f_r * dot(N, lightDir) / pdf_light;  	
		}
	}
}

//路径追踪--重要性采样版本
vec3 PathTracingImportantSampling(HitResult hit, int maxBounce) {

    vec3 Lo = vec3(0.0f);      //最终的颜色
    vec3 history = vec3(1.0f); //递归积累的颜色

    for(int bounce = 0; bounce < maxBounce; bounce++) {
        vec3 V = -hit.viewDir;
        vec3 N = hit.normal;       

        //直接光照
        DirectLight(hit, Lo, history);
        
        //间接光照
        //获取3个随机数
        vec2 uv = SobolVec2(frameCounter + 1, bounce);
        uv = CranleyPattersonRotation(uv);
        float xi_1 = uv.x;
        float xi_2 = uv.y;
        float xi_3 = rand();    //xi_3是决定采样的随机数，普通rand就好

        //采样BRDF得到一个方向L
        vec3 L = SampleBRDF(xi_1, xi_2, xi_3, V, N, hit.material); 
        float NdotL = dot(N, L);
        if(NdotL <= 0.0f) {
            break;
        }

        //发射光线
        Ray randomRay;
        randomRay.origin = hit.hitPoint;
        randomRay.direction = L;
        HitResult newHit = HitBVH(randomRay);

        //获取L方向上的BRDF值和概率密度
        vec3 f_r = BRDF_Evaluate(V, N, L, hit.material);
        float pdf_brdf = BRDF_Pdf(V, N, L, hit.material);
        if(pdf_brdf <= 0.0f) {
            break;
        }

        //未命中        
        if(!newHit.isHit) {
            if(useEnv) {
                vec3 color = HdrColor(L);
                float pdf_light = HdrPdf(L, hdrResolution);            
                
                //多重重要性采样
                float mis_weight = MisMixWeight(pdf_brdf, pdf_light);   //f(a,b) = a^2 / (a^2 + b^2)
                Lo += mis_weight * history * color * f_r * NdotL / pdf_brdf;
            }
            break;
        }
        
        //命中光源积累颜色
        if(numOfLights > 0) {
            LightSampleRec lightSampleRec;
            lightSampleRec.surfacePos = newHit.hitPoint;
            lightSampleRec.normal = newHit.normal;
            float pdf_light = Light_Pdf(int(newHit.material.lightid), lightSampleRec, hit);
            
            //多重重要性采样
            float mis_weight = MisMixWeight(pdf_brdf, pdf_light);
            
            vec3 Le = newHit.material.emissive;
            Lo += mis_weight * history * Le * f_r * NdotL / pdf_brdf;
        }
        else {
            vec3 Le = newHit.material.emissive;
            Lo += history * Le * f_r * NdotL / pdf_brdf;
        }

        //递归(步进)
        hit = newHit;
        history *= f_r * NdotL / pdf_brdf;   //累积颜色
    }
    
    return Lo;
}

void main() {
    Ray ray;
    
    ray.origin = eye;
    vec2 randValue = vec2((rand() - 0.5f) / float(width), (rand() - 0.5f) / float(height));
    vec2 newpix = vec2(pix.x * float(width / height), pix.y);
    vec4 dir = cameraRotate * vec4(newpix.xy + randValue, -1.5f, 0.0f);
    ray.direction = normalize(dir.xyz);

    //primary hit
    HitResult firstHit = HitBVH(ray);
    vec3 color;
    
    if(!firstHit.isHit) {
        if(useEnv) {
            color = HdrColor(ray.direction);
        }
        else {
            color = vec3(0.0f);
        }
    } 
    else {
        int maxBounce = 3;
        vec3 Le = firstHit.material.emissive;
        vec3 Li = PathTracingImportantSampling(firstHit, maxBounce);
        color = Le + Li;
    }
    
    //和上一帧混合
    vec3 lastColor = texture2D(lastFrame, pix.xy * 0.5f + 0.5f).rgb;
    color = mix(lastColor, color, 1.0f / float(frameCounter + 1.0f));

    //输出
    gl_FragData[0] = vec4(color, 1.0f);  
}
