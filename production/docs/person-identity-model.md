# 人物身份预分析模型

本节点使用 OpenCV DNN CPU、YuNet 与 SFace 固定权重推理，不调用智能体。
仅识别人脸证据，不覆盖背影、无脸身体，也不承担通用物体身份识别。

SFace 官方固定来源：
https://github.com/opencv/opencv_zoo/tree/47534e27c9851bb1128ccc0102f1145e27f23f98/models/face_recognition_sface

该目录 README 声明所有文件适用 Apache-2.0；安装及分发时应保留目录的 LICENSE。
YuNet 模型目录适用 MIT。SFace 官方目录未完整说明训练数据来源，代码/权重许可不等同于对训练数据全部权利的独立核验。

SFace FP32 文件 SHA256：`0ba9fbfa01b5270c96627c4ef784da859931e02f04419c829e83484087c34e79`。
模型不提交 Git。运行时不下载，默认从当前代码树的 models 目录加载；
可用 `PERSON_IDENTITY_MODEL_PATH` 指定已有模型的绝对路径。哈希错误、模型缺失明确返回 skipped。

本开发树安装命令（默认从固定官方地址下载）：

```bash
rtk python3 /home/xy/duet-ad1-h3-max-r2v-production-candidate/scripts/install_person_identity_model.py --destination /home/xy/duet-ad1-h3-max-r2v-production-candidate/models/face_recognition_sface_2021dec.onnx
```

检测使用原分辨率并复用灰块节点阈值 0.73；关联使用镜头内时间、位置、脸部特征，跨镜头使用多帧模板与歧义门槛。至少三个不同时间的合格源采样支持才确认轨迹，同框共现禁止合并。精确九帧独立检测后以位置和特征映射，不按帧序号自动归人，也不追加为源轨迹确认票数。
输出包含模型哈希、OpenCV 版本、参数、观测框与时间；不保存 embedding 或脸部裁剪。
出现区间仅由连续采样证据支持，不填补无脸间隙。低质量、单帧或无法归并的观测保留 unresolved；不同 canonical ID 不代表已证明不同人。

余弦 0.363 来源于 OpenCV SFace 官方示例的特定评测集，并非身份概率或全平台精度保证；上线前仍需更多多人、遮挡、镜头变化样本验收。本节点不调整进程全局 OpenCV 线程设置。
