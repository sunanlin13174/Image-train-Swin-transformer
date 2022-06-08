# Image-train-Swin-transformer
基于Swin-transformer训练图像分类并部署web端，解释见博客：https://blog.csdn.net/hi_gril/article/details/118486070?spm=1001.2014.3001.5501
<br>由于Swin-transformer现在只支持训练ImageNet，导致用起来不方便，自己改了下代码，可用于训练自己的数据集<br>
<br>具体包含以下几个步骤：<br>
<br>1.加载预训练权重√<br>
<br>2.图片数据集准备√<br>
<br>3.训练√<br>
<br>4.推理测试√<br>
<br>5.新的数据增强调优<br>
<br>6.部署在web端√<br>
服务器端部署：运行flask_demo，<br>
客户端测试：运行client.py<br>

<br>也可以使用postman测试服务器端
