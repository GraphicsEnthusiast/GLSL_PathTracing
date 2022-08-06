# GLSL_PathTracing
2022 8 6更新，同时使用HDR环境光和矩形光原会白屏。

2022 8 3更新：加入了对矩形光源的重要性采样，但是效果图看起来总感觉怪怪的，不知道对不对。。。加入了OpenImageDenoise降噪。
<img width="1291" alt="rect" src="https://user-images.githubusercontent.com/75780167/182432366-9f9bba75-b29e-4e7a-886b-d5a923249925.png">
新的百度网盘链接（完整项目）：链接：https://pan.baidu.com/s/1uhLn1RTaIgdOKfTl7NRBkA 
提取码：6sw4
注意配置：
![ttt1](https://user-images.githubusercontent.com/75780167/182435714-a9417cb8-5332-4ed3-9014-77021c9a9eae.png)
![ttt2](https://user-images.githubusercontent.com/75780167/182435727-5c493eb0-f38f-471c-99fd-d8ba56d32074.png)
![ttt3](https://user-images.githubusercontent.com/75780167/182435733-693cc2dc-a658-40bd-91b6-fa7e28096f92.png)

———————————————————————————————————————————————————————————

那个兔子模型没有uv，加载时会报错，可以用建模软件（比如maya）为它生成uv后在加载，或者直接修改加载模型的代码，当没有uv时，将uv全部赋值为0。

直接将这些代码创建vs2019工程文件即可，项目太大，git发不上来。
完整项目百度网盘链接（包含Assert）：链接：https://pan.baidu.com/s/1NGE7FjKSIlAs0Fbh9zv2qQ 
提取码：2wch

主要参考：
https://blog.csdn.net/weixin_44176696/category_11195786.html
https://github.com/RobertBeckebans/OpenGL-PathTracer
https://github.com/knightcrawler25/GLSL-PathTracer

未来还会更新，更多功能正在添加中。推荐使用vcpkg配置环境（我自己64位和32位的都装了），缺glfw，glad，glm。
vcpkg教程：https://blog.csdn.net/cjmqas/article/details/79282847?spm=1001.2014.3001.5506
建议开vpn，不然容易连接超时。

![qwe](https://user-images.githubusercontent.com/75780167/182056941-5044b1b2-6001-448c-8509-591cbf911ef0.png)
![t](https://user-images.githubusercontent.com/75780167/182057003-3c2567d8-fd85-48e1-b73d-8992b527de7c.png)
![a12](https://user-images.githubusercontent.com/75780167/182057058-c9f142fb-9c9c-4646-b1e7-bf64cc7a5767.png)
<img width="1291" alt="abc" src="https://user-images.githubusercontent.com/75780167/182057102-f155e8bd-a0b9-42fa-93cc-9b6e3c1cf4bb.png">
