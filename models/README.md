# yamnet.tflite

- 来源：Google YAMNet（AudioSet 521 类声学分类），tflite 格式；与 bj02 TrendScout `.data/models/yamnet.tflite` 同文件
- sha256：`4d8b4a53282dc83ef04e3e7dbc4fbc98082e34e44ed798e16c3a0cdd4c584faf`（app/vocal.py 加载时校验）
- 用途：口播台词的声学验证——逐窗 Speech/Singing/Music 分，区分「真在说话」与 BGM/唱歌假转录（阈值照搬 TrendScout 盘上实测校准，见 app/vocal.py 注释）
