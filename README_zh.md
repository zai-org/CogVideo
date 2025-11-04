# CogVideo & CogVideoX

[Read this in English](./README.md)

[日本語で読む](./README_ja.md)

<div align="center">
<img src=resources/logo.svg width="50%"/>
</div>
<p align="center">
在 <a href="https://huggingface.co/spaces/THUDM/CogVideoX-5B" target="_blank"> 🤗 Huggingface Space</a> 或 <a href="https://modelscope.cn/studios/ZhipuAI/CogVideoX-5b-demo" target="_blank"> 🤖 ModelScope Space</a> 在线体验 CogVideoX-5B 模型
</p>
<p align="center">
📚 查看 <a href="https://arxiv.org/abs/2408.06072" target="_blank">论文</a> 和 <a href="https://zhipu-ai.feishu.cn/wiki/DHCjw1TrJiTyeukfc9RceoSRnCh" target="_blank">使用文档</a>
</p>
<p align="center">
    👋 加入我们的 <a href="resources/WECHAT.md" target="_blank">微信</a> 和  <a href="https://discord.gg/dCGfUsagrD" target="_blank">Discord</a>
</p>
<p align="center">
📍 前往<a href="https://chatglm.cn/video?fr=osm_cogvideox"> 清影</a> 和 <a href="https://open.bigmodel.cn/?utm_campaign=open&_channel_track_key=OWTVNma9"> API平台</a> 体验更大规模的商业版视频生成模型。
</p>

## 项目更新

- 🔥🔥 **News**: ```2025/03/24```: 我们推出了 [CogKit](https://github.com/THUDM/CogKit) 工具，这是一个微调**CogView4**, **CogVideoX** 系列的微调和推理框架，一个工具包，玩转我们的多模态生成模型。
- 🔥 **News**: ```2025/02/28```:  DDIM Inverse 已经在`CogVideoX-5B` 和 `CogVideoX1.5 -5B` 支持，查看 [here](inference/ddim_inversion.py).
- 🔥 **News**: ```2025/01/08```: 我们更新了基于`diffusers`版本模型的`Lora`微调代码，占用显存更低，详情请见[这里](finetune/README_zh.md)。
- 🔥 **News**: ```2024/11/15```: 我们发布 `CogVideoX1.5` 模型的diffusers版本，仅需调整部分参数即可沿用之前的代码。
- 🔥 **News**: ```2024/11/08```: 我们发布 `CogVideoX1.5` 模型。CogVideoX1.5 是 CogVideoX 开源模型的升级版本。
CogVideoX1.5-5B 系列模型支持 **10秒** 长度的视频和更高的分辨率，其中 `CogVideoX1.5-5B-I2V` 支持 **任意分辨率** 的视频生成，SAT代码已经更新。`diffusers`版本还在适配中。SAT版本代码前往 [这里](https://huggingface.co/THUDM/CogVideoX1.5-5B-SAT) 下载。
- 🔥**News**: ```2024/10/13```: 成本更低，单卡4090可微调 `CogVideoX-5B`
  的微调框架[cogvideox-factory](https://github.com/a-r-r-o-w/cogvideox-factory)已经推出，多种分辨率微调，欢迎使用。
- 🔥 **News**: ```2024/10/10```: 我们更新了我们的技术报告,请点击 [这里](https://arxiv.org/pdf/2408.06072)
  查看，附上了更多的训练细节和demo，关于demo，点击[这里](https://yzy-thu.github.io/CogVideoX-demo/) 查看。
- 🔥 **News**: ```2024/10/09```: 我们在飞书[技术文档](https://zhipu-ai.feishu.cn/wiki/DHCjw1TrJiTyeukfc9RceoSRnCh")
  公开CogVideoX微调指导，以进一步增加分发自由度，公开文档中所有示例可以完全复现
- 🔥 **News**: ```2024/9/19```: 我们开源 CogVideoX 系列图生视频模型 **CogVideoX-5B-I2V**
  。该模型可以将一张图像作为背景输入，结合提示词一起生成视频，具有更强的可控性。
  至此，CogVideoX系列模型已经支持文本生成视频，视频续写，图片生成视频三种任务。欢迎前往在线[体验](https://huggingface.co/spaces/THUDM/CogVideoX-5B-Space)。
- 🔥 **News**: ```2024/9/19```: CogVideoX 训练过程中用于将视频数据转换为文本描述的 Caption
  模型 [CogVLM2-Caption](https://huggingface.co/THUDM/cogvlm2-llama3-caption)
  已经开源。欢迎前往下载并使用。
- 🔥 ```2024/8/27```:  我们开源 CogVideoX 系列更大的模型 **CogVideoX-5B**
  。我们大幅度优化了模型的推理性能，推理门槛大幅降低，您可以在 `GTX 1080TI` 等早期显卡运行 **CogVideoX-2B**，在 `RTX 3060`
  等桌面端甜品卡运行 **CogVideoX-5B** 模型。 请严格按照[要求](requirements.txt)
  更新安装依赖，推理代码请查看 [cli_demo](inference/cli_demo.py)。同时，**CogVideoX-2B** 模型开源协议已经修改为**Apache 2.0 协议**。
- 🔥 ```2024/8/6```: 我们开源 **3D Causal VAE**，用于 **CogVideoX-2B**，可以几乎无损地重构视频。
- 🔥 ```2024/8/6```: 我们开源 CogVideoX 系列视频生成模型的第一个模型, **CogVideoX-2B**。
- 🌱 **Source**: ```2022/5/19```: 我们开源了 CogVideo 视频生成模型（现在你可以在 `CogVideo` 分支中看到），这是首个开源的基于
  Transformer 的大型文本生成视频模型，您可以访问 [ICLR'23 论文](https://arxiv.org/abs/2205.15868) 查看技术细节。

## 目录

跳转到指定部分：

- [快速开始](#快速开始)
  - [提示词优化](#提示词优化)
  - [SAT](#sat)
  - [Diffusers](#diffusers)
- [视频作品](#视频作品)
  - [CogVideoX-5B](#cogvideox-5b)
  - [CogVideoX-2B](#cogvideox-2b)
- [模型介绍](#模型介绍)
- [友情链接](#友情链接)
- [完整项目代码结构](#完整项目代码结构)
  - [Colab 快速使用](#colab-快速使用)
  - [inference](#inference)
  - [finetune](#finetune)
  - [sat](#sat-1)
  - [tools](#tools)
- [CogVideo(ICLR'23)](#cogvideoiclr23)
- [引用](#引用)
- [模型协议](#模型协议)

## 快速开始

### 提示词优化

在开始运行模型之前，请参考 [这里](inference/convert_demo.py) 查看我们是怎么使用GLM-4(或者同级别的其他产品，例如GPT-4)
大模型对模型进行优化的，这很重要，
由于模型是在长提示词下训练的，一个好的提示词直接影响了视频生成的质量。

### SAT

查看sat文件夹下的 [sat_demo](sat/README.md)：包含了 SAT 权重的推理代码和微调代码，推荐基于此代码进行 CogVideoX
模型结构的改进，研究者使用该代码可以更好的进行快速的迭代和开发。

### Diffusers

```
pip install -r requirements.txt
```

查看[diffusers_demo](inference/cli_demo.py)：包含对推理代码更详细的解释，包括各种关键的参数。

欲了解更多关于量化推理的细节，请参考 [diffusers-torchao](https://github.com/sayakpaul/diffusers-torchao/)。使用 Diffusers
和 TorchAO，量化推理也是可能的，这可以实现内存高效的推理，并且在某些情况下编译后速度有所提升。有关在 A100 和 H100
上使用各种设置的内存和时间基准测试的完整列表，已发布在 [diffusers-torchao](https://github.com/sayakpaul/diffusers-torchao)
上。

## 视频作品

### CogVideoX-5B

<table border="0" style="width: 100%; text-align: left; margin-top: 20px;">
  <tr>
      <td>
          <video src="https://github.com/user-attachments/assets/cf5953ea-96d3-48fd-9907-c4708752c714" width="100%" controls autoplay loop></video>
      </td>
      <td>
          <video src="https://github.com/user-attachments/assets/fe0a78e6-b669-4800-8cf0-b5f9b5145b52" width="100%" controls autoplay loop></video>
      </td>
       <td>
          <video src="https://github.com/user-attachments/assets/c182f606-8f8c-421d-b414-8487070fcfcb" width="100%" controls autoplay loop></video>
     </td>
      <td>
          <video src="https://github.com/user-attachments/assets/7db2bbce-194d-434d-a605-350254b6c298" width="100%" controls autoplay loop></video>
     </td>
  </tr>
  <tr>
      <td>
          <video src="https://github.com/user-attachments/assets/62b01046-8cab-44cc-bd45-4d965bb615ec" width="100%" controls autoplay loop></video>
      </td>
      <td>
          <video src="https://github.com/user-attachments/assets/d78e552a-4b3f-4b81-ac3f-3898079554f6" width="100%" controls autoplay loop></video>
      </td>
       <td>
          <video src="https://github.com/user-attachments/assets/30894f12-c741-44a2-9e6e-ddcacc231e5b" width="100%" controls autoplay loop></video>
     </td>
      <td>
          <video src="https://github.com/user-attachments/assets/926575ca-7150-435b-a0ff-4900a963297b" width="100%" controls autoplay loop></video>
     </td>
  </tr>
</table>

### CogVideoX-2B

<table border="0" style="width: 100%; text-align: left; margin-top: 20px;">
  <tr>
      <td>
          <video src="https://github.com/user-attachments/assets/ea3af39a-3160-4999-90ec-2f7863c5b0e9" width="100%" controls autoplay loop></video>
      </td>
      <td>
          <video src="https://github.com/user-attachments/assets/9de41efd-d4d1-4095-aeda-246dd834e91d" width="100%" controls autoplay loop></video>
      </td>
       <td>
          <video src="https://github.com/user-attachments/assets/941d6661-6a8d-4a1b-b912-59606f0b2841" width="100%" controls autoplay loop></video>
     </td>
      <td>
          <video src="https://github.com/user-attachments/assets/938529c4-91ae-4f60-b96b-3c3947fa63cb" width="100%" controls autoplay loop></video>
     </td>
  </tr>
</table>


查看画廊的对应提示词，请点击[这里](resources/galary_prompt.md)

## 模型介绍

CogVideoX是 [清影](https://chatglm.cn/video?fr=osm_cogvideox) 同源的开源版本视频生成模型。
下表展示我们提供的视频生成模型相关基础信息:

<table  style="border-collapse: collapse; width: 100%;">
  <tr>
    <th style="text-align: center;">模型名</th>
    <th style="text-align: center;">CogVideoX1.5-5B (最新)</th>
    <th style="text-align: center;">CogVideoX1.5-5B-I2V (最新)</th>
    <th style="text-align: center;">CogVideoX-2B</th>
    <th style="text-align: center;">CogVideoX-5B</th>
    <th style="text-align: center;">CogVideoX-5B-I2V </th>
  </tr>
  <tr>
    <td style="text-align: center;">发布时间</td>
    <th style="text-align: center;">2024年11月8日</th>
    <th style="text-align: center;">2024年11月8日</th>
    <th style="text-align: center;">2024年8月6日</th>
    <th style="text-align: center;">2024年8月27日</th>
    <th style="text-align: center;">2024年9月19日</th>
  </tr>
  <tr>
    <td style="text-align: center;">视频分辨率</td>
    <td colspan="1" style="text-align: center;">1360 * 768</td>
    <td colspan="1" style="text-align: center;"> Min(W, H) = 768 <br> 768 ≤ Max(W, H) ≤ 1360 <br> Max(W, H) % 16 = 0 </td>
    <td colspan="3" style="text-align: center;">720 * 480</td>
  </tr>
  <tr>
    <td style="text-align: center;">帧数</td>
    <td colspan="2" style="text-align: center;">必须为 <b>16N + 1</b> 其中 N <= 10 (默认 81)</td>
    <td colspan="3" style="text-align: center;">必须为 <b>8N + 1</b> 其中 N <= 6 (默认 49)</td>
  </tr>
  <tr>
    <td style="text-align: center;">推理精度</td>
    <td colspan="2" style="text-align: center;"><b>BF16(推荐)</b>, FP16, FP32，FP8*，INT8，不支持INT4</td>
    <td style="text-align: center;"><b>FP16*(推荐)</b>, BF16, FP32，FP8*，INT8，不支持INT4</td>
    <td colspan="2" style="text-align: center;"><b>BF16(推荐)</b>, FP16, FP32，FP8*，INT8，不支持INT4</td>
  </tr>
  <tr>
    <td style="text-align: center;">单GPU显存消耗<br></td>
    <td colspan="2" style="text-align: center;"><a href="https://github.com/THUDM/SwissArmyTransformer">SAT</a> BF16: 76GB <br><b>diffusers BF16 : 10GB起* </b><br><b>diffusers INT8(torchao): 7G起* </b></td>
    <td style="text-align: center;"><a href="https://github.com/THUDM/SwissArmyTransformer">SAT</a> FP16: 18GB <br><b>diffusers FP16: 4GB起* </b><br><b>diffusers INT8(torchao): 3.6G起*</b></td>
    <td colspan="2" style="text-align: center;"><a href="https://github.com/THUDM/SwissArmyTransformer">SAT</a> BF16: 26GB <br><b>diffusers BF16 : 5GB起* </b><br><b>diffusers INT8(torchao): 4.4G起* </b></td>
  </tr>
  <tr>
    <td style="text-align: center;">多GPU推理显存消耗</td>
    <td colspan="2" style="text-align: center;"><b>BF16: 24GB* using diffusers</b><br></td>
    <td style="text-align: center;"><b>FP16: 10GB* using diffusers</b><br></td>
    <td colspan="2" style="text-align: center;"><b>BF16: 15GB* using diffusers</b><br></td>
  </tr>
  <tr>
    <td style="text-align: center;">推理速度<br>(Step = 50, FP/BF16)</td>
    <td colspan="2" style="text-align: center;">单卡A100: ~1000秒(5秒视频)<br>单卡H100: ~550秒(5秒视频)</td>
    <td style="text-align: center;">单卡A100: ~90秒<br>单卡H100: ~45秒</td>
    <td colspan="2" style="text-align: center;">单卡A100: ~180秒<br>单卡H100: ~90秒</td>
  </tr>
  <tr>
    <td style="text-align: center;">提示词语言</td>
    <td colspan="5" style="text-align: center;">English*</td>
  </tr>
  <tr>
    <td style="text-align: center;">提示词长度上限</td>
    <td colspan="2" style="text-align: center;">224 Tokens</td>
    <td colspan="3" style="text-align: center;">226 Tokens</td>
  </tr>
  <tr>
    <td style="text-align: center;">视频长度</td>
    <td colspan="2" style="text-align: center;">5 秒 或 10 秒</td>
    <td colspan="3" style="text-align: center;">6 秒</td>
  </tr>
  <tr>
    <td style="text-align: center;">帧率</td>
    <td colspan="2" style="text-align: center;">16 帧 / 秒 </td>
    <td colspan="3" style="text-align: center;">8 帧 / 秒 </td>
  </tr>
  <tr>
    <td style="text-align: center;">位置编码</td>
    <td colspan="2" style="text-align: center;">3d_rope_pos_embed</td>
    <td style="text-align: center;">3d_sincos_pos_embed</td>
    <td style="text-align: center;">3d_rope_pos_embed</td>
    <td style="text-align: center;">3d_rope_pos_embed + learnable_pos_embed</td>
  </tr>
  <tr>
    <td style="text-align: center;">下载链接 (Diffusers)</td>
    <td style="text-align: center;"><a href="https://huggingface.co/THUDM/CogVideoX1.5-5B">🤗 HuggingFace</a><br><a href="https://modelscope.cn/models/ZhipuAI/CogVideoX1.5-5B">🤖 ModelScope</a><br><a href="https://wisemodel.cn/models/ZhipuAI/CogVideoX1.5-5B">🟣 WiseModel</a></td>
    <td style="text-align: center;"><a href="https://huggingface.co/THUDM/CogVideoX1.5-5B-I2V">🤗 HuggingFace</a><br><a href="https://modelscope.cn/models/ZhipuAI/CogVideoX1.5-5B-I2V">🤖 ModelScope</a><br><a href="https://wisemodel.cn/models/ZhipuAI/CogVideoX1.5-5B-I2V">🟣 WiseModel</a></td>
    <td style="text-align: center;"><a href="https://huggingface.co/THUDM/CogVideoX-2b">🤗 HuggingFace</a><br><a href="https://modelscope.cn/models/ZhipuAI/CogVideoX-2b">🤖 ModelScope</a><br><a href="https://wisemodel.cn/models/ZhipuAI/CogVideoX-2b">🟣 WiseModel</a></td>
    <td style="text-align: center;"><a href="https://huggingface.co/THUDM/CogVideoX-5b">🤗 HuggingFace</a><br><a href="https://modelscope.cn/models/ZhipuAI/CogVideoX-5b">🤖 ModelScope</a><br><a href="https://wisemodel.cn/models/ZhipuAI/CogVideoX-5b">🟣 WiseModel</a></td>
    <td style="text-align: center;"><a href="https://huggingface.co/THUDM/CogVideoX-5b-I2V">🤗 HuggingFace</a><br><a href="https://modelscope.cn/models/ZhipuAI/CogVideoX-5b-I2V">🤖 ModelScope</a><br><a href="https://wisemodel.cn/models/ZhipuAI/CogVideoX-5b-I2V">🟣 WiseModel</a></td>
  </tr>
  <tr>
    <td style="text-align: center;">下载链接 (SAT)</td>
    <td colspan="2" style="text-align: center;"><a href="https://huggingface.co/THUDM/CogVideoX1.5-5b-SAT">🤗 HuggingFace</a><br><a href="https://modelscope.cn/models/ZhipuAI/CogVideoX1.5-5b-SAT">🤖 ModelScope</a><br><a href="https://wisemodel.cn/models/ZhipuAI/CogVideoX1.5-5b-SAT">🟣 WiseModel</a></td>
    <td colspan="3" style="text-align: center;"><a href="./sat/README_zh.md">SAT</a></td>
  </tr>
</table>

**数据解释**

+ 使用 diffusers 库进行测试时，启用了全部`diffusers`库自带的优化，该方案未测试在非**NVIDIA A100 / H100**
  外的设备上的实际显存 / 内存占用。通常，该方案可以适配于所有 **NVIDIA 安培架构**
  以上的设备。若关闭优化，显存占用会成倍增加，峰值显存约为表格的3倍。但速度提升3-4倍左右。你可以选择性的关闭部分优化，这些优化包括:

```
pipe.enable_sequential_cpu_offload()
pipe.vae.enable_slicing()
pipe.vae.enable_tiling()
```

+ 多GPU推理时，需要关闭 `enable_sequential_cpu_offload()` 优化。
+ 使用 INT8 模型会导致推理速度降低，此举是为了满足显存较低的显卡能正常推理并保持较少的视频质量损失，推理速度大幅降低。
+ CogVideoX-2B 模型采用 `FP16` 精度训练， 搜有 CogVideoX-5B 模型采用 `BF16` 精度训练。我们推荐使用模型训练的精度进行推理。
+ [PytorchAO](https://github.com/pytorch/ao) 和 [Optimum-quanto](https://github.com/huggingface/optimum-quanto/)
  可以用于量化文本编码器、Transformer 和 VAE 模块，以降低 CogVideoX 的内存需求。这使得在免费的 T4 Colab 或更小显存的 GPU
  上运行模型成为可能！同样值得注意的是，TorchAO 量化完全兼容 `torch.compile`，这可以显著提高推理速度。在 `NVIDIA H100`
  及以上设备上必须使用 `FP8` 精度，这需要源码安装 `torch`、`torchao` Python 包。建议使用 `CUDA 12.4`。
+ 推理速度测试同样采用了上述显存优化方案，不采用显存优化的情况下，推理速度提升约10%。 只有`diffusers`版本模型支持量化。
+ 模型仅支持英语输入，其他语言可以通过大模型润色时翻译为英语。

## 友情链接

我们非常欢迎来自社区的贡献，并积极的贡献开源社区。以下作品已经对CogVideoX进行了适配，欢迎大家使用:

+ [LeMiCa](https://unicomai.github.io/LeMiCa/)：一种无需训练的高效扩散视频生成加速框架，显著提升生成帧之间的全局内容与风格一致性。
+ [RIFLEx-CogVideoX](https://github.com/thu-ml/RIFLEx)：
  RIFLEx 是一个视频长度外推的方法，只需一行代码即可将视频生成长度延伸为原先的二倍。RIFLEx 不仅支持 Training-free 的推理，也提供基于 CogVideoX 进行微调的模型，只需在原有长度视频上微调 1000 步即可大大提高长度外推能力。
+ [CogVideoX-Fun](https://github.com/aigc-apps/CogVideoX-Fun):
  CogVideoX-Fun是一个基于CogVideoX结构修改后的的pipeline，支持自由的分辨率，多种启动方式。
+ [CogStudio](https://github.com/pinokiofactory/cogstudio): CogVideo 的 Gradio Web UI单独实现仓库，支持更多功能的 Web UI。
+ [Xorbits Inference](https://github.com/xorbitsai/inference): 性能强大且功能全面的分布式推理框架，轻松一键部署你自己的模型或内置的前沿开源模型。
+ [ComfyUI-CogVideoXWrapper](https://github.com/kijai/ComfyUI-CogVideoXWrapper) 使用ComfyUI框架，将CogVideoX加入到你的工作流中。
+ [VideoSys](https://github.com/NUS-HPC-AI-Lab/VideoSys): VideoSys 提供了易用且高性能的视频生成基础设施，支持完整的管道，并持续集成最新的模型和技术。
+ [AutoDL镜像](https://www.codewithgpu.com/i/THUDM/CogVideo/CogVideoX-5b-demo): 由社区成员提供的一键部署Huggingface
  Space镜像。
+ [室内设计微调模型](https://huggingface.co/collections/bertjiazheng/koolcogvideox-66e4762f53287b7f39f8f3ba) 基于
  CogVideoX的微调模型，它专为室内设计而设计
+ [xDiT](https://github.com/xdit-project/xDiT): xDiT是一个用于在多GPU集群上对DiTs并行推理的引擎。xDiT支持实时图像和视频生成服务。
+ [CogVideoX-Interpolation](https://github.com/feizc/CogvideX-Interpolation): 基于 CogVideoX 结构修改的管道，旨在为关键帧插值生成提供更大的灵活性。
+ [DiffSynth-Studio](https://github.com/modelscope/DiffSynth-Studio): DiffSynth 工作室是一款扩散引擎。重构了架构，包括文本编码器、UNet、VAE
  等，在保持与开源社区模型兼容性的同时，提升了计算性能。该框架已经适配 CogVideoX。
+ [CogVideoX-Controlnet](https://github.com/TheDenk/cogvideox-controlnet): 一个包含 CogvideoX 模型的简单 Controlnet 模块的代码。
+ [VideoTuna](https://github.com/VideoVerses/VideoTuna)：VideoTuna 是首个集成多种 AI 视频生成模型的仓库，支持文本转视频、图像转视频、文本转图像生成。
+ [ConsisID](https://github.com/PKU-YuanGroup/ConsisID): 一种身份保持的文本到视频生成模型，基于 CogVideoX-5B，通过频率分解在生成的视频中保持面部一致性。
+ [教程](https://www.youtube.com/watch?v=5UCkMzP2VLE&ab_channel=SECourses): 一个关于在Windows和云环境中安装和优化CogVideoX1.5-5B-I2V模型的分步指南。特别感谢[FurkanGozukara](https://github.com/FurkanGozukara)的努力和支持！


## 完整项目代码结构

本开源仓库将带领开发者快速上手 **CogVideoX** 开源模型的基础调用方式、微调示例。

### Colab 快速使用

这里提供了三个能直接在免费的 Colab T4上 运行的项目

+ [CogVideoX-5B-T2V-Colab.ipynb](https://colab.research.google.com/drive/1pCe5s0bC_xuXbBlpvIH1z0kfdTLQPzCS?usp=sharing):
  CogVideoX-5B 文字生成视频 Colab 代码。
+ [CogVideoX-5B-T2V-Int8-Colab.ipynb](https://colab.research.google.com/drive/1DUffhcjrU-uz7_cpuJO3E_D4BaJT7OPa?usp=sharing):
  CogVideoX-5B 文字生成视频量化推理 Colab 代码，运行一次大约需要30分钟。
+ [CogVideoX-5B-I2V-Colab.ipynb](https://colab.research.google.com/drive/17CqYCqSwz39nZAX2YyonDxosVKUZGzcX?usp=sharing):
  CogVideoX-5B 图片生成视频 Colab 代码。
+ [CogVideoX-5B-V2V-Colab.ipynb](https://colab.research.google.com/drive/1comfGAUJnChl5NwPuO8Ox5_6WCy4kbNN?usp=sharing):
  CogVideoX-5B 视频生成视频 Colab 代码。

### inference

+ [cli_demo](inference/cli_demo.py): 更详细的推理代码讲解，常见参数的意义，在这里都会提及。
+ [cli_demo_quantization](inference/cli_demo_quantization.py):
  量化模型推理代码，可以在显存较低的设备上运行，也可以基于此代码修改，以支持运行FP8等精度的CogVideoX模型。请注意，FP8
  仅测试通过，且必须将 `torch-nightly`,`torchao`源代码安装，不建议在生产环境中使用。
+ [diffusers_vae_demo](inference/cli_vae_demo.py): 单独执行VAE的推理代码。
+ [space demo](inference/gradio_composite_demo): Huggingface Space同款的 GUI 代码，植入了插帧，超分工具。

<div style="text-align: center;">
    <img src="resources/web_demo.png" style="width: 100%; height: auto;" />
</div>

+ [convert_demo](inference/convert_demo.py): 如何将用户的输入转换成适合
  CogVideoX的长输入。因为CogVideoX是在长文本上训练的，所以我们需要把输入文本的分布通过LLM转换为和训练一致的长文本。脚本中默认使用GLM-4，也可以替换为GPT、Gemini等任意大语言模型。
+ [gradio_web_demo](inference/gradio_composite_demo/app.py): 与 Huggingface Space 完全相同的代码实现，快速部署 CogVideoX
  GUI体验。

### finetune

+ [train_cogvideox_lora](finetune/README_zh.md): diffusers版本 CogVideoX 模型微调方案和细节。

### sat

+ [sat_demo](sat/README_zh.md): 包含了 SAT 权重的推理代码和微调代码，推荐基于 CogVideoX
  模型结构进行改进，创新的研究者使用改代码以更好的进行快速的堆叠和开发。

### tools

本文件夹包含了一些工具，用于模型的转换 / Caption 等工作。

+ [convert_weight_sat2hf](tools/convert_weight_sat2hf.py): 将 SAT 模型权重转换为 Huggingface 模型权重。
+ [caption_demo](tools/caption/README_zh.md):  Caption 工具，对视频理解并用文字输出的模型。
+ [export_sat_lora_weight](tools/export_sat_lora_weight.py):  SAT微调模型导出工具，将
  SAT Lora Adapter 导出为 diffusers 格式。
+ [load_cogvideox_lora](tools/load_cogvideox_lora.py): 载入diffusers版微调Lora Adapter的工具代码。
+ [llm_flux_cogvideox](tools/llm_flux_cogvideox/llm_flux_cogvideox.py): 使用开源本地大语言模型 + Flux +
  CogVideoX实现自动化生成视频。
+ [parallel_inference_xdit](tools/parallel_inference/parallel_inference_xdit.py):
  在多个 GPU 上并行化视频生成过程，
  由[xDiT](https://github.com/xdit-project/xDiT)提供支持。
+ [cogvideox-factory](https://github.com/a-r-r-o-w/cogvideox-factory): CogVideoX低成文微调框架，适配`diffusers`
  版本模型。支持更多分辨率，单卡4090即可微调 CogVideoX-5B 。

## CogVideo(ICLR'23)

[CogVideo: Large-scale Pretraining for Text-to-Video Generation via Transformers](https://arxiv.org/abs/2205.15868)
的官方repo位于[CogVideo branch](https://github.com/THUDM/CogVideo/tree/CogVideo)。

**CogVideo可以生成高帧率视频，下面展示了一个32帧的4秒视频。**

![High-frame-rate sample](https://raw.githubusercontent.com/THUDM/CogVideo/CogVideo/assets/appendix-sample-highframerate.png)

![Intro images](https://raw.githubusercontent.com/THUDM/CogVideo/CogVideo/assets/intro-image.png)


<div align="center">
  <video src="https://github.com/user-attachments/assets/ea3af39a-3160-4999-90ec-2f7863c5b0e9" width="80%" controls autoplay></video>
</div>

CogVideo的demo网站在[https://models.aminer.cn/cogvideo](https://models.aminer.cn/cogvideo/)。您可以在这里体验文本到视频生成。
*原始输入为中文。*

## 引用

🌟 如果您发现我们的工作有所帮助，欢迎引用我们的文章，留下宝贵的stars

```
@article{yang2024cogvideox,
  title={CogVideoX: Text-to-Video Diffusion Models with An Expert Transformer},
  author={Yang, Zhuoyi and Teng, Jiayan and Zheng, Wendi and Ding, Ming and Huang, Shiyu and Xu, Jiazheng and Yang, Yuanming and Hong, Wenyi and Zhang, Xiaohan and Feng, Guanyu and others},
  journal={arXiv preprint arXiv:2408.06072},
  year={2024}
}
@article{hong2022cogvideo,
  title={CogVideo: Large-scale Pretraining for Text-to-Video Generation via Transformers},
  author={Hong, Wenyi and Ding, Ming and Zheng, Wendi and Liu, Xinghan and Tang, Jie},
  journal={arXiv preprint arXiv:2205.15868},
  year={2022}
}
```

## 模型协议

本仓库代码使用 [Apache 2.0 协议](LICENSE) 发布。

CogVideoX-2B 模型 (包括其对应的Transformers模块，VAE模块) 根据 [Apache 2.0 协议](LICENSE) 许可证发布。

CogVideoX-5B 模型 (Transformers 模块，包括图生视频，文生视频版本)
根据 [CogVideoX LICENSE](https://huggingface.co/THUDM/CogVideoX-5b/blob/main/LICENSE)
许可证发布。
