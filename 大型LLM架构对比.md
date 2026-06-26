# 大型 LLM 架构对比

## 从 DeepSeek V3 到 GLM-5：现代 LLM 架构设计纵览

**作者：Sebastian Raschka, PhD**

*最后更新：2026年4月2日（在第23节中新增了 Gemma 4）*

距离最初的 GPT 架构诞生已经过去了七年。乍一看，从 GPT-2（2019年）回顾到 DeepSeek V3 和 Llama 4（2024-2025年），你可能会惊讶于这些模型在结构上仍然如此相似。

诚然，位置嵌入从绝对位置编码演变为旋转位置编码（RoPE），多头注意力在很大程度上已被分组查询注意力取代，更高效的 SwiGLU 也取代了 GELU 等激活函数。但是，在这些微小的优化之下，我们究竟是否看到了真正突破性的变革，还是只是在打磨同样的架构基础？

通过比较 LLM 来确定影响其性能（好或不太好）的关键要素是非常困难的：数据集、训练技术和超参数差异很大，而且往往没有得到很好的记录。

然而，我认为在 2025 年，审视架构本身的结构变化，看看 LLM 开发者们正在做什么仍然非常有价值。（其中一些模型如图 1 所示。）

![图 1：本文涵盖的架构的一个子集](images/figure_01.png)

因此，在本文中，我将不再撰写关于基准性能或训练算法的内容，而是聚焦于定义当今旗舰开源模型的架构发展。

（如您可能记得，[我不久前写过关于多模态 LLM 的文章](https://magazine.sebastianraschka.com/p/understanding-multimodal-llms)；在本文中，我将聚焦于近期模型的文本能力，多模态能力的讨论留待下次。）

**提示：** 这是一篇相当全面的文章，建议您使用导航栏访问目录（只需将鼠标悬停在 Substack 页面的左侧即可）。

**可选：** 下面的视频是本文的旁白和精简版。

---

# 1. DeepSeek V3/R1

您可能已经听过不止一次，[DeepSeek R1](https://arxiv.org/abs/2501.12948) 在 2025 年 1 月发布时产生了重大影响。DeepSeek R1 是一个基于 [DeepSeek V3 架构](https://arxiv.org/abs/2412.19437) 构建的推理模型，后者于 2024 年 12 月推出。

虽然本文的重点是 2025 年发布的架构，但我认为将 DeepSeek V3 包含在内是合理的，因为它仅在 2025 年 DeepSeek R1 推出后才得到广泛的关注和采用。

如果您对 DeepSeek R1 的具体训练感兴趣，您可能也会发现我今年早些时候的文章很有用：

[**理解推理型 LLM**](https://magazine.sebastianraschka.com/p/understanding-reasoning-llms)

2025年2月5日

[阅读全文](https://magazine.sebastianraschka.com/p/understanding-reasoning-llms)

在本节中，我将重点介绍 DeepSeek V3 中引入的两项关键架构技术，它们提高了计算效率并使其区别于许多其他 LLM：

- 多头潜在注意力（MLA）
- 混合专家（MoE）

## **1.1 多头潜在注意力（MLA）**

在讨论多头潜在注意力（MLA）之前，让我们先简要回顾一些背景知识，以便理解为什么使用它。为此，让我们从分组查询注意力（GQA）开始，它已经成为近年来替代多头注意力（MHA）的新标准，具有更高的计算和参数效率。

下面简要总结一下 GQA。与 MHA 不同的是，在 MHA 中每个头都有自己的一组键和值，而为了减少内存使用，GQA 将多个头分组以共享相同的键和值投影。

例如，正如下图 2 所示，如果有 2 个键值组和 4 个注意力头，那么头 1 和头 2 可能共享一组键和值，而头 3 和头 4 共享另一组。这减少了键和值计算的总数量，从而降低了内存使用并提高了效率（根据消融研究，对建模性能没有明显影响）。

![图 2：MHA 和 GQA 之间的比较。这里，组大小为 2，其中一对键和值在 2 个查询之间共享](images/figure_02.png)

因此，GQA 的核心思想是通过在多个查询头之间共享键和值头来减少键和值头的数量。这（1）降低了模型的参数数量，（2）减少了推理期间键和值张量的内存带宽使用，因为需要从 KV 缓存中存储和检索的键和值更少。

（如果您好奇 GQA 在代码中是什么样子，请参阅我的 [GPT-2 到 Llama 3 转换指南](https://github.com/rasbt/LLMs-from-scratch/blob/main/ch05/07_gpt_to_llama/converting-llama2-to-llama3.ipynb) 中不带 KV 缓存的版本，以及我[这里](https://github.com/rasbt/LLMs-from-scratch/blob/main/pkg/llms_from_scratch/llama3.py) 的 KV 缓存版本。）

虽然 GQA 主要是对 MHA 的一种计算效率改进，但消融研究（如[原始 GQA 论文](https://arxiv.org/abs/2305.13245) 和 [Llama 2 论文](https://arxiv.org/abs/2307.09288)）表明，在 LLM 建模性能方面，它的表现可与标准 MHA 相媲美。

现在，多头潜在注意力（MLA）提供了一种不同的内存节省策略，它也与 KV 缓存特别搭配。MLA 不是像 GQA 那样共享键和值头，而是在将键和值张量存储到 KV 缓存之前将它们压缩到更低维度的空间中。

在推理时，这些压缩的张量在用于计算之前被投影回原始大小，如下图 3 所示。这增加了一个额外的矩阵乘法，但减少了内存使用。

![图 3：MLA（在 DeepSeek V3 和 R1 中使用）与常规 MHA 之间的比较](images/figure_03.png)

（顺便提一下，查询也被压缩了，但仅在训练期间，而不是推理期间。）

顺便说一句，MLA 在 DeepSeek V3 中并不是新的，因为它的 [DeepSeek-V2 前身](https://arxiv.org/abs/2405.04434) 也使用了（甚至引入了）它。此外，V2 论文中包含了一些有趣的消融研究，可以解释为什么 DeepSeek 团队选择 MLA 而不是 GQA（见下图 4）。

![图 4：来自 DeepSeek-V2 论文的注释表，https://arxiv.org/abs/2405.04434](images/figure_04.png)

如上图 4 所示，GQA 似乎表现不如 MHA，而 MLA 则提供了比 MHA 更好的建模性能，这可能是 DeepSeek 团队选择 MLA 而不是 GQA 的原因。（如果能看到 MLA 和 GQA 之间的 "每个 token 的 KV 缓存" 节省比较将会更有趣！）

在进入下一个架构组件之前总结本节，MLA 是一种巧妙的技巧，可以在减少 KV 缓存内存使用的同时，在建模性能方面甚至略胜于 MHA。

## **1.2 混合专家（MoE）**

DeepSeek 中值得强调的另一个主要架构组件是其对混合专家（MoE）层的使用。虽然 DeepSeek 并非发明了 MoE，但 MoE 今年重新兴起，我们稍后将讨论的许多架构也采用了它。

您可能已经熟悉 MoE，但快速回顾一下可能会有所帮助。

MoE 的核心思想是用多个专家层替换 transformer 块中的每个 FeedForward 模块，其中每个专家层也是一个 FeedForward 模块。这意味着我们将单个 FeedForward 块替换为多个 FeedForward 块，如下图 5 所示。

![图 5：DeepSeek V3/R1 中的混合专家（MoE）模块（右）与具有标准 FeedForward 块的 LLM（左）的比较](images/figure_05.png)

transformer 块内的 FeedForward 块（如上图中的深灰色块所示）通常包含模型总参数的大部分。（请注意，transformer 块以及 FeedForward 块在 LLM 中重复多次；在 DeepSeek V3 的情况下，重复 61 次。）

因此，将*单个* FeedForward 块替换为*多个* FeedForward 块（如 MoE 设置中所做的那样）大幅增加了模型的总参数数量。然而，关键的技巧是我们不会对每个 token 使用（"激活"）所有专家。相反，路由器仅为每个 token 选择一小部分专家。（出于时间或文章篇幅的考虑，我将在另一篇文章中更详细地介绍路由器。）

由于一次只有少数专家处于活动状态，MoE 模块通常被称为*稀疏*模块，与始终使用完整参数集的*密集*模块形成对比。然而，MoE 带来的大量总参数增加了 LLM 的容量，这意味着它可以在训练期间吸收更多知识。但稀疏性保持了推理效率，因为我们不会同时使用所有参数。

例如，DeepSeek V3 每个 MoE 模块有 256 个专家，总共有 6710 亿个参数。然而在推理过程中，一次只有 9 个专家处于活动状态（1 个共享专家加上由路由器选择的 8 个）。这意味着每个推理步骤仅使用 370 亿个参数，而不是全部 6710 亿个参数。

DeepSeek V3 的 MoE 设计的一个值得注意的特点是使用共享专家。这是一个对每个 token 始终处于活动状态的专家。这个想法并不新鲜，[DeepSeek 2024 MoE](https://arxiv.org/abs/2401.06066) 和 [2022 DeepSpeedMoE 论文](https://arxiv.org/abs/2201.05596) 中已经引入了。

![图 6：来自 "DeepSeekMoE: Towards Ultimate Expert Specialization in Mixture-of-Experts Language Models" 的注释图，https://arxiv.org/abs/2401.06066](images/figure_06.png)

拥有共享专家的好处首先在 [DeepSpeedMoE 论文](https://arxiv.org/abs/2201.05596) 中提到，他们发现与没有共享专家相比，这可以提高整体建模性能。这可能是因为常见或重复的模式不必由多个单独的专家来学习，从而为学习更专业的模式留出更多空间。

## **1.3 DeepSeek 总结**

总之，DeepSeek V3 是一个庞大的 6710 亿参数模型，在发布时优于其他开源权重模型，包括 405B Llama 3。尽管规模更大，但由于其混合专家（MoE）架构，它在推理时效率要高得多，每个 token 仅激活一小部分（仅 370 亿）参数。

另一个关键的区分特点是 DeepSeek V3 使用多头潜在注意力（MLA）而不是分组查询注意力（GQA）。MLA 和 GQA 都是标准多头注意力（MHA）的推理高效替代方案，特别是在使用 KV 缓存时。虽然 MLA 在实现上更复杂，但 DeepSeek-V2 论文中的研究表明，在建模性能方面，MLA 优于 GQA。

---

# 2. OLMo 2

非营利组织 Allen Institute for AI 推出的 OLMo 系列模型因其训练数据和代码的透明度以及相对详细的技术报告而值得注意。

虽然您可能不会在任何基准测试或排行榜的顶部找到 OLMo 模型，但它们相当干净，更重要的是，由于其透明度，它们是开发 LLM 的绝佳蓝图。

虽然 OLMo 模型因其透明度而受到欢迎，但它们的性能也相当不错。事实上，在 1 月份发布时（在 Llama 4、Gemma 3 和 Qwen 3 之前），[OLMo 2](https://arxiv.org/abs/2501.00656) 模型在计算与性能的帕累托前沿上，如下图 7 所示。

![图 7：不同 LLM 的建模基准性能（越高越好）与预训练成本（FLOPs；越低越好）。这是 OLMo 2 论文中的注释图，https://arxiv.org/abs/2501.00656](images/figure_07.png)

如前所述，我的目标是在本文中只关注 LLM 架构细节（不是训练或数据），以保持文章长度可控。那么，OLMo 2 中有趣的架构设计选择是什么？主要归结为归一化：RMSNorm 层的放置以及 QK-norm 的添加，我将在下面讨论。

另一件值得提及的事情是，OLMo 2 仍然使用传统的多头注意力（MHA），而不是 MLA 或 GQA。

## **2.1 归一化层放置**

总体而言，OLMo 2 在很大程度上遵循了原始 GPT 模型的架构，类似于其他当代 LLM。然而，有一些值得注意的偏差。让我们从归一化层开始。

与 Llama、Gemma 和大多数其他 LLM 类似，OLMo 2 从 LayerNorm 切换到 RMSNorm。

但由于 RMSNorm 已经不是什么新鲜事物（它基本上是 LayerNorm 的简化版本，具有更少的可训练参数），我将跳过 RMSNorm 与 LayerNorm 的讨论。（好奇的读者可以在我的 [GPT-2 到 Llama 转换指南](https://github.com/rasbt/LLMs-from-scratch/blob/main/ch05/07_gpt_to_llama/converting-gpt-to-llama2.ipynb) 中找到 RMSNorm 的代码实现。）

但是，值得讨论的是 RMSNorm 层的放置。原始 transformer（来自 "[Attention is all you need](https://arxiv.org/abs/1706.03762)" 论文）将 transformer 块中的两个归一化层分别放置在注意力模块和 FeedForward 模块*之后*。

这也被称为 Post-LN 或 Post-Norm。

GPT 和后来的大多数其他 LLM 将归一化层放置在注意力和 FeedForward 模块*之前*，这被称为 Pre-LN 或 Pre-Norm。下图比较了 Post-Norm 和 Pre-Norm。

![图 8：Post-Norm、Pre-Norm 和 OLMo 2 的 Post-Norm 风格的比较](images/figure_08.png)

在 [2020 年，Xiong 等人](https://arxiv.org/abs/2002.04745) 表明，Pre-LN 在初始化时产生更良好的梯度。此外，研究人员提到，Pre-LN 即使没有仔细的学习率预热也能很好地工作，而学习率预热对于 Post-LN 来说是至关重要的工具。

我之所以提到这一点，是因为 OLMo 2 采用了 Post-LN 的一种形式（但使用 RMSNorm 而不是 LayerNorm，所以我称之为 *Post-Norm*）。

在 OLMo 2 中，他们没有将归一化层放在注意力和 FeedForward 层之前，而是将它们放在之后，如上图所示。然而，请注意，与原始 transformer 架构相比，归一化层仍然在残差层（跳跃连接）内。

那么，他们为什么要改变归一化层的位置呢？原因是这样做有助于训练稳定性，如下图所示。

![图 9：Pre-Norm（如 GPT-2、Llama 3 等中使用的）与 OLMo 2 的 Post-Norm 风格之间的训练稳定性图。这是 OLMo 2 论文中的注释图，https://arxiv.org/abs/2501.00656](images/figure_09.png)

不幸的是，该图显示了归一化层重新排序与 QK-Norm 一起的结果，QK-Norm 是一个单独的概念。因此，很难判断归一化层重新排序本身贡献了多少。

## **2.2 QK-Norm**

由于上一节已经提到了 QK-norm，并且我们稍后将讨论的其他 LLM（如 Gemma 2 和 Gemma 3）也使用 QK-norm，让我们简要讨论一下它是什么。

QK-Norm 本质上是另一个 RMSNorm 层。它被放置在多头注意力（MHA）模块内，并在应用 RoPE 之前应用于查询（q）和键（k）。为了说明这一点，下面是我为 [Qwen3 从零开始实现](https://github.com/rasbt/LLMs-from-scratch/tree/main/ch05/11_qwen3) 编写的分组查询注意力（GQA）层的摘录（OLMo 中 MHA 中 QK-norm 的应用类似）：

```python
class GroupedQueryAttention(nn.Module):
    def __init__(
        self, d_in, num_heads, num_kv_groups,
        head_dim=None, qk_norm=False, dtype=None
    ):
        # ...

        if qk_norm:
            self.q_norm = RMSNorm(head_dim, eps=1e-6)
            self.k_norm = RMSNorm(head_dim, eps=1e-6)
        else:
            self.q_norm = self.k_norm = None

    def forward(self, x, mask, cos, sin):
        b, num_tokens, _ = x.shape

        # 应用投影
        queries = self.W_query(x)
        keys = self.W_key(x)
        values = self.W_value(x)

        # ...

        # 可选归一化
        if self.q_norm:
            queries = self.q_norm(queries)
        if self.k_norm:
            keys = self.k_norm(keys)

        # 应用 RoPE
        queries = apply_rope(queries, cos, sin)
        keys = apply_rope(keys, cos, sin)

        # 扩展 K 和 V 以匹配头数
        keys = keys.repeat_interleave(self.group_size, dim=1)
        values = values.repeat_interleave(self.group_size, dim=1)

        # 注意力
        attn_scores = queries @ keys.transpose(2, 3)
        # ...
```

如前所述，QK-Norm 与 Post-Norm 一起稳定了训练。请注意，QK-Norm 不是 OLMo 2 发明的，可以追溯到 [2023 年 Scaling Vision Transformers 论文](https://arxiv.org/abs/2302.05442)。

## **2.3 OLMo 2 总结**

简而言之，OLMo 2 值得注意的架构设计决策主要是 RMSNorm 的位置：RMSNorm 放在注意力和 FeedForward 模块之后而不是之前（Post-Norm 的一种风格），以及在注意力机制内对查询和键添加 RMSNorm（QK-Norm），这两者一起有助于稳定训练损失。

下图进一步比较了 OLMo 2 和 Llama 3；可以看出，除了 OLMo 2 仍然使用传统的 MHA 而不是 GQA 之外，架构在其他方面相对相似。（然而，[OLMo 2 团队在 3 个月后发布了一个使用 GQA 的 32B 变体](https://huggingface.co/allenai/OLMo-2-0325-32B-Instruct)。）

![图 10：Llama 3 和 OLMo 2 之间的架构比较](images/figure_10.png)

---

# 3. Gemma 3

Google 的 Gemma 模型一直非常出色，我认为与其他流行模型（如 Llama 系列）相比，它们一直被低估了。

Gemma 的一个区别在于其相当大的词汇量（为了更好地支持多种语言），以及更专注于 27B 大小（而不是 8B 或 70B）。但请注意，Gemma 2 也有较小的尺寸：1B、4B 和 12B。

27B 尺寸达到了一个非常好的甜蜜点：它比 8B 模型功能强大得多，但又不像 70B 模型那样资源密集，并且它可以在我的 Mac Mini 上本地运行良好。

那么，[Gemma 3](https://arxiv.org/abs/2503.19786) 还有什么有趣之处呢？正如前面讨论的，其他模型（如 DeepSeek V3/R1）使用混合专家（MoE）架构来在固定模型大小下减少推理时的内存需求。（MoE 方法也被我们稍后将讨论的几个其他模型所使用。）

Gemma 3 使用不同的"技巧"来降低计算成本，即滑动窗口注意力。

## **3.1 滑动窗口注意力**

通过滑动窗口注意力（最初在 2020 年的 [LongFormer 论文](https://arxiv.org/abs/2004.05150) 中引入，并且也已被 [Gemma 2](http://arxiv.org/abs/2408.00118) 使用），Gemma 3 团队能够大幅减少 KV 缓存中的内存需求，如下图所示。

![图 11：来自 Gemma 3 论文（https://arxiv.org/abs/2503.19786）的注释图，展示了通过滑动窗口注意力节省的 KV 缓存内存](images/figure_11.png)

那么什么是滑动窗口注意力呢？如果我们将常规自注意力视为一种*全局*注意力机制（因为每个序列元素都可以访问其他每个序列元素），那么我们可以将滑动窗口注意力视为*局部*注意力，因为这里我们限制了当前查询位置周围的上下文大小。这如下图所示。

![图 12：常规注意力（左）和滑动窗口注意力（右）之间的比较](images/figure_12.png)

请注意，滑动窗口注意力可以与多头注意力和分组查询注意力一起使用；Gemma 3 使用分组查询注意力。

如上所述，滑动窗口注意力也被称为*局部*注意力，因为局部窗口围绕当前查询位置并随之移动。相反，常规注意力是*全局*的，因为每个 token 可以访问所有其他 token。

现在，如上所述，Gemma 2 前身架构之前也使用了滑动窗口注意力。Gemma 3 的区别在于它们调整了全局（常规）和局部（滑动）注意力之间的比例。

例如，Gemma 2 使用混合注意力机制，结合滑动窗口（局部）和全局注意力，比例为 1:1。每个 token 可以关注附近的 4096 token 上下文窗口。

Gemma 2 在每隔一层使用滑动窗口注意力，而 Gemma 3 现在采用 5:1 的比例，这意味着每 5 个滑动窗口（局部）注意力层只有 1 个全局注意力层；此外，滑动窗口大小从 4096（Gemma 2）减少到仅 1024（Gemma 3）。这将模型的焦点转向更高效的、局部化的计算。

根据他们的消融研究，使用滑动窗口注意力对建模性能的影响很小，如下图所示。

![图 13：来自 Gemma 3 论文（https://arxiv.org/abs/2503.19786）的注释图，显示滑动窗口注意力对 LLM 生成的输出困惑度几乎没有影响](images/figure_13.png)

虽然滑动窗口注意力是 Gemma 3 最值得注意的架构方面，但作为前一节 OLMo 2 的后续，我也想简要讨论一下归一化层的位置。

## **3.2 Gemma 3 中归一化层的放置**

一个值得强调的、小但有趣的细节是，Gemma 3 在其分组查询注意力模块周围的 Pre-Norm 和 Post-Norm 设置中都使用 RMSNorm。

这与 Gemma 2 类似，但仍值得强调，因为它不同于（1）原始 transformer（"Attention is all you need"）中使用的 Post-Norm，（2）由 GPT-2 推广并随后在许多其他架构中使用的 Pre-Norm，以及（3）我们之前在 OLMo 2 中看到的 Post-Norm 风格。

![图 14：OLMo2 和 Gemma 3 之间的架构比较；注意 Gemma 3 中额外的归一化层](images/figure_14.png)

我认为这种归一化层的放置是一种相当直观的方法，因为它兼具 Pre-Norm 和 Post-Norm 的优点。在我看来，多一点归一化不会有坏处。在最坏的情况下，如果额外的归一化是多余的，这会通过冗余增加一点低效率。在实践中，由于 RMSNorm 在整个方案中相对便宜，这不应该产生任何明显的影响。

## **3.3 Gemma 3 总结**

Gemma 3 是一个性能良好的开源权重 LLM，在我看来，在开源圈子里有点被低估了。最有趣的部分是使用滑动窗口注意力来提高效率（将其与 MoE 结合在将来会很有趣）。

此外，Gemma 3 具有独特的归一化层放置，在注意力和 FeedForward 模块之前和之后都放置 RMSNorm 层。

## **3.4 附赠：Gemma 3n**

Gemma 3 发布几个月后，Google 分享了 [Gemma 3n](https://developers.googleblog.com/en/introducing-gemma-3n/)，这是一个为小型设备效率而优化的 Gemma 3 模型，目标是在手机上运行。

Gemma 3n 中实现更好效率的变化之一是所谓的逐层嵌入（PLE）参数层。这里的关键思想是只将模型参数的一个子集保留在 GPU 内存中。特定于 token 层的嵌入（如文本、音频和视觉模态的嵌入）然后根据需要从 CPU 或 SSD 流式传输。

下图说明了 PLE 内存节省情况，列出了标准 Gemma 3 模型的 54.4 亿个参数。这可能指的是 Gemma 3 40 亿参数变体。

![图 15：来自 Google Gemma 3n 博客（https://developers.googleblog.com/en/introducing-gemma-3n/）的注释图，说明 PLE 内存节省情况](images/figure_15.png)

54.4 亿与 40 亿参数的差异是因为 Google 有一种有趣的报告 LLM 参数数量的方法。他们经常排除嵌入参数以使模型看起来更小，除非像这种情况那样方便地将它们包含进来以使模型看起来更大。这并不是 Google 独有的，因为这种方法已成为整个领域的常见做法。

另一个有趣的技巧是 [MatFormer](https://arxiv.org/abs/2310.07707) 概念（Matryoshka Transformer 的缩写）。例如，Gemma 3n 使用一个可以切割成更小的、可独立使用的模型的共享 LLM（transformer）架构。每个切片都经过训练可以独立工作，因此在推理时，我们可以仅运行您需要的部分（而不是大型模型）。

---

# 4. Mistral Small 3.1

[Mistral Small 3.1 24B](https://mistral.ai/news/mistral-small-3-1) 于 3 月在 Gemma 3 之后不久发布，值得注意的原因是它在多个基准测试中（数学除外）优于 Gemma 3 27B，同时速度更快。

Mistral Small 3.1 相比 Gemma 3 推理延迟更低的原因可能在于其自定义分词器，以及缩小了 KV 缓存和层数。否则，它是一个标准架构，如下图所示。

![图 16：Gemma 3 27B 和 Mistral 3.1 Small 24B 之间的架构比较](images/figure_16.png)

有趣的是，早期的 Mistral 模型曾使用过滑动窗口注意力，但他们在 Mistral Small 3.1 中似乎放弃了它（如果我们考虑官方 [Model Hub 配置文件](https://huggingface.co/mistralai/Mistral-Small-3.1-24B-Instruct-2503/blob/main/config.json) 中的默认设置 `"sliding_window": null`）。此外，[模型卡片](https://huggingface.co/mistralai/Mistral-Small-3.1-24B-Instruct-2503) 也没有提到它。

因此，由于 Mistral 使用常规分组查询注意力而不是 Gemma 3 中的带滑动窗口的分组查询注意力，也许由于能够使用更多优化的代码（即 FlashAttention）而有额外的推理计算节省。例如，我推测虽然滑动窗口注意力减少了内存使用，但并不一定减少推理延迟，这正是 Mistral Small 3.1 所关注的。

---

# 5. Llama 4

本文前面关于混合专家（MoE）的广泛介绍再次得到了回报。[Llama 4](https://ai.meta.com/blog/llama-4-multimodal-intelligence/) 也采用了 MoE 方法，否则其架构遵循与 DeepSeek V3 非常相似的相对标准的架构，如下图所示。（Llama 4 包括原生多模态支持，类似于 Gemma 和 Mistral 等模型。但由于本文聚焦于语言建模，我们只关注文本模型。）

![图 17：DeepSeek V3（6710 亿参数）和 Llama 4 Maverick（4000 亿参数）之间的架构比较](images/figure_17.png)

虽然 Llama 4 Maverick 架构总体看起来与 DeepSeek V3 非常相似，但仍有一些值得强调的有趣差异。

首先，Llama 4 像其前身一样使用分组查询注意力，而 DeepSeek V3 使用我们在本文开头讨论的多头潜在注意力。现在，DeepSeek V3 和 Llama 4 Maverick 都是非常大的架构，DeepSeek V3 的总参数数量大约大 68%。然而，DeepSeek V3 有 370 亿个活动参数，比 Llama 4 Maverick（170 亿）多一倍以上。

Llama 4 Maverick 使用更经典的 MoE 设置，专家数量较少但规模较大（2 个活动专家，每个隐藏大小为 8192），而 DeepSeek V3（9 个活动专家，每个隐藏大小为 2048）。此外，DeepSeek 在每个 transformer 块（除了前 3 个）中使用 MoE 层，而 Llama 4 在每个其他 transformer 块中交替使用 MoE 和密集模块。

鉴于架构之间的许多小差异，很难确定它们对最终模型性能的确切影响。然而，主要要点是 MoE 架构在 2025 年显著流行起来。

---

# 6. Qwen3

Qwen 团队始终如一地提供高质量的开源权重 LLM。当我在 2023 年共同指导 NeurIPS LLM 效率挑战赛时，我记得获胜的顶级解决方案都是基于 Qwen2 的。

现在，Qwen3 是其尺寸类别的排行榜上另一个热门模型系列。有 7 个密集模型：0.6B、1.7B、4B、8B、14B 和 32B。还有 2 个 MoE 模型：30B-A3B 和 235B-A22B。

（顺便说一下，"Qwen3" 中缺少的空格不是拼写错误；我只是想保留 Qwen 开发者选择的原始拼写。）

## **6.1 Qwen3（密集）**

让我们先讨论密集模型架构。在撰写本文时，0.6B 模型很可能是目前存在的最小的当前一代开源权重模型。根据我的个人经验，鉴于其较小的尺寸，它的表现非常好。如果您计划在本地运行它，它具有出色的 token/秒吞吐量和低内存占用。但更重要的是，由于其较小的尺寸，它在本地训练也很容易（用于教育目的）。

因此，Qwen3 0.6B 在大多数用途中已经取代了 Llama 3 1B。下图显示了这两个架构之间的比较。

![图 18：Qwen3 0.6B 和 Llama 3 1B 之间的架构比较；请注意 Qwen3 是一个更深的架构，有更多的层，而 Llama 3 是一个更宽的架构，有更多的注意力头](images/figure_18.png)

如果您对没有外部第三方 LLM 库依赖的可读 Qwen3 实现感兴趣，我最近[从零开始实现了 Qwen3（纯 PyTorch）](https://github.com/rasbt/LLMs-from-scratch/tree/main/ch05/11_qwen3)。

上图中的计算性能数字基于我从零开始的 PyTorch 实现在 A100 GPU 上运行的结果。正如您所看到的，Qwen3 的内存占用较小，因为它的整体架构较小，但也使用了较小的隐藏层和较少的注意力头。然而，它比 Llama 3 使用更多的 transformer 块，这导致了较慢的运行速度（较低的 token/秒生成速度）。

## **6.2 Qwen3（MoE）**

如前所述，Qwen3 还有两个 MoE 变体：30B-A3B 和 235B-A22B。为什么像 Qwen3 这样的一些架构既有常规（密集）变体又有 MoE（稀疏）变体？

正如本文开头提到的，MoE 变体有助于减少大型基础模型的推理成本。根据用户的目标和约束，同时提供密集和 MoE 版本可以为用户提供灵活性。

密集模型通常更易于在各种硬件上进行微调、部署和优化。

另一方面，MoE 模型针对扩展推理进行了优化。例如，在固定推理预算下，它们可以实现更高的整体模型容量（即由于更大，训练期间的知识吸收）而不会成比例地增加推理成本。

通过发布两种类型，Qwen3 系列可以支持更广泛的用例：密集模型用于稳健性、简单性和微调，MoE 模型用于大规模高效服务。

为了总结本节，让我们看看 Qwen3 235B-A22B（请注意，A22B 代表 "220 亿活动参数"）和 DeepSeek V3 的比较，后者具有近两倍的活动参数（370 亿）。

![图 19：DeepSeek V3 和 Qwen3 235B-A22B 之间的架构比较](images/figure_19.png)

如上图所示，DeepSeek V3 和 Qwen3 235B-A22B 架构非常相似。值得注意的一点是，Qwen3 模型不再使用共享专家（早期 Qwen 模型，如 [Qwen2.5-MoE](https://qwenlm.github.io/blog/qwen2.5-max/) 确实使用了共享专家）。

不幸的是，Qwen3 团队没有透露他们放弃共享专家的原因。如果我不得不猜测的话，也许只是因为当他们将专家数量从 2（在 Qwen2.5-MoE 中）增加到 8（在 Qwen3 中）时，对于他们的设置来说，共享专家对训练稳定性来说不是必需的。然后他们能够通过使用 8 个而不是 8+1 个专家来节省额外的计算/内存成本。（然而，这并不能解释为什么 DeepSeek V3 仍然保留他们的共享专家。）

**更新。** [Junyang Lin](https://x.com/JustinLin610/status/1947364862184853626)，Qwen3 的开发者之一，回应如下：

> 当时我们没有发现共享专家有足够显著的改进，我们担心共享专家对推理的优化造成影响。说实话，这个问题没有一个明确的答案。

---

# 7. SmolLM3

[SmolLM3](https://huggingface.co/blog/smollm3) 可能没有本文涉及的其他 LLM 那样受欢迎，但我认为它仍然是一个有趣的模型，因为它在相对较小且方便的 30 亿参数模型大小下提供了非常好的建模性能，正好位于 Qwen3 1.7B 和 4B 模型之间，如下图所示。

此外，它还共享了许多训练细节，类似于 OLMo，这是罕见的并且总是值得赞赏的！

![图 20：来自 SmolLM3 公告帖子的注释图，https://huggingface.co/blog/smollm3，将 SmolLM3 胜率与 Qwen3 1.7B 和 4B 以及 Llama 3 3B 和 Gemma 3 4B 进行比较](images/figure_20.png)

如下图所示的架构比较所示，SmolLM3 架构看起来相当标准。也许最有趣的方面是它使用了 NoPE（无位置嵌入）。

![图 21：Qwen3 4B 和 SmolLM3 3B 并排的架构比较](images/figure_21.png)

## **7.1 无位置嵌入（NoPE）**

在 LLM 上下文中，NoPE 是一个较早的想法，可以追溯到 2023 年的一篇论文（[位置编码对 Transformer 长度泛化的影响](https://arxiv.org/abs/2305.19466)），它去除了显式位置信息注入（如早期 GPT 架构中的经典绝对位置嵌入层或如今的 RoPE）。

在基于 transformer 的 LLM 中，位置编码通常是必要的，因为自注意力独立于顺序处理 token。绝对位置嵌入通过添加一个将信息添加到 token 嵌入中的额外嵌入层来解决此问题。

![图 22：来自我《从零开始构建大型语言模型》一书的修改图（https://www.amazon.com/Build-Large-Language-Model-Scratch/dp/1633437167），说明绝对位置嵌入](images/figure_22.png)

另一方面，RoPE 通过相对于 token 位置旋转查询和键向量来解决这个问题。

然而，在 NoPE 层中，完全没有添加这样的位置信号：不是固定的，不是学习的，不是相对的，什么都没有。

即使没有位置嵌入，由于因果注意力掩码，模型仍然知道哪些 token 在前。此掩码防止每个 token 关注未来的 token。结果是，位于位置 _t_ 的 token 只能看到位置 _≤ t_ 的 token，从而保留了自回归顺序。

因此，虽然没有显式添加的位置信息，但模型结构中仍然隐含了方向感，并且在常规基于梯度下降的训练中，如果发现对优化目标有益，LLM 可以学习利用它。（有关更多信息，请查看 NoPE 论文的定理。）

因此，总体而言，[NoPE 论文](https://arxiv.org/abs/2305.19466) 不仅发现不需要注入位置信息，而且还发现 NoPE 具有更好的长度泛化能力，这意味着随着序列长度的增加，LLM 的回答性能下降较少，如下图所示。

![图 23：来自 NoPE 论文（https://arxiv.org/abs/2305.19466）的注释图，显示 NoPE 的长度泛化能力更好](images/figure_23.png)

请注意，上面显示的实验是使用大约 1 亿参数的相对较小的 GPT 风格模型和相对较小的上下文大小进行的。目前尚不清楚这些发现能在多大程度上推广到更大的、当代的 LLM 中。

出于这个原因，SmolLM3 团队可能只在每 4 层中"应用"了 NoPE（或者说省略了 RoPE）。

---

# 8. Kimi K2 和 Kimi K2 Thinking

[Kimi K2](https://moonshotai.github.io/Kimi-K2/) 最近在 AI 社区引起了巨大反响，因为它是一个具有令人难以置信的优秀性能的开源权重模型。根据基准测试，它与 Google 的 Gemini、Anthropic 的 Claude 和 OpenAI 的 ChatGPT 等最好的专有模型相当。

一个值得注意的方面是它在 AdamW 之上使用了相对较新的 [Muon](https://github.com/KellerJordan/Muon) 优化器的变体。据我所知，这是 Muon 首次用于任何此规模的生产模型（[之前](https://arxiv.org/abs/2502.16982)，它仅被证明可扩展到 16B）。这产生了非常好的训练损失曲线，可能帮助将这个模型推到了上述基准的顶端。

虽然人们评论说损失异常平滑（由于缺乏尖峰），但我认为它并非异常平滑（例如，请参见下图中的 OLMo 2 损失曲线；另外，梯度的 L2 范数可能是跟踪训练稳定性的更好指标）。然而，值得注意的是损失曲线的衰减程度。

但是，正如本文开头所述，训练方法是另一时间的话题。

![图 24：来自 Kimi K2 公告博客文章（https://moonshotai.github.io/Kimi-K2/）和 OLMo 2 论文（https://arxiv.org/abs/2305.19466）的注释图](images/figure_24.png)

该模型本身具有 1 万亿个参数，确实令人印象深刻。

它可能是这一代最大的 LLM（截至撰写本文时，鉴于 Llama 4 Behemoth 未发布、专有 LLM 不算、Google 的 1.6 万亿 [Switch Transformer](https://arxiv.org/abs/2101.03961) 是来自不同代的编码器-解码器架构这些约束）。

它也回到了原点，因为 Kimi K2 使用了我们在本文开头介绍的 DeepSeek V3 架构，只是他们将其做得更大，如下图所示。

![图 25.1：DeepSeek V3 和 Kimi K2 之间的架构比较](images/figure_25.png)

如上图所示，Kimi K2 基本上与 DeepSeek V3 相同，只是它在 MoE 模块中使用了更多专家，并在多头潜在注意力（MLA）模块中使用了更少的头。

Kimi K2 并非凭空出现。较早的 Kimi 1.5 模型在 [Kimi k1.5: Scaling Reinforcement Learning with LLMs 论文](https://arxiv.org/abs/2501.12599) 中讨论过，也令人印象深刻。然而，它运气不好的是，DeepSeek R1 模型论文恰好在 1 月 22 日同一天发表。此外，据我所知，Kimi 1.5 的权重从未公开分享过。

因此，Kimi K2 团队很可能从这些经验中吸取了教训，并在 DeepSeek R2 发布之前将 Kimi K2 作为开源权重模型分享。迄今为止，Kimi K2 是最令人印象深刻的开源权重模型。

**更新：** 2025 年 11 月 6 日，Kimi K2 团队还发布了他们新的"Thinking"模型变体。架构与上面的 Kimi K2 相同，只是他们将上下文大小从 128k 扩展到 256k。

根据 [Kimi 团队分享的基准](https://moonshotai.github.io/Kimi-K2/thinking.html)，该模型超过了领先的专有 LLM 的性能。（不幸的是，没有与 DeepSeek R1 的直接比较。

![图 25.2：DeepSeek R1 与 Kimi K2 Thinking 架构（上）和 Kimi K2 Thinking 基准（下）](images/figure_26.png)

---

# 9. GPT-OSS

OpenAI [发布](https://openai.com/index/introducing-gpt-oss/) 的 gpt-oss-120b 和 gpt-oss-20b，是自 2019 年 GPT-2 以来他们的第一个开源权重模型，距离我写这篇文章大约一周后。由于 OpenAI 的开源权重模型一直备受期待，我更新了这篇文章以包含它们。我将保持本节的简洁，但我已经写了另一篇更详细的文章专门介绍 gpt-oss 模型：

[**从 GPT-2 到 gpt-oss：分析架构进步**](https://magazine.sebastianraschka.com/p/from-gpt-2-to-gpt-oss-analyzing-the)

作者：Sebastian Raschka, PhD

2025年8月9日

[阅读全文](https://magazine.sebastianraschka.com/p/from-gpt-2-to-gpt-oss-analyzing-the)

在总结有趣的细节之前，让我们首先概述两个模型，gpt-oss-20b 和 gpt-oss-120b，如下图 26 所示。

![图 26：两个 gpt-oss 模型的架构概览](images/figure_27.png)

查看图 26，该架构包含我们在前面讨论的其他架构中看到的所有熟悉组件。例如，图 27 将较小的 gpt-oss 架构放在 Qwen3 30B-A3B 旁边，它也是一个活动参数数量相似的 MoE 模型（gpt-oss 有 36 亿活动参数，Qwen3 30B-A3B 有 33 亿）。

![图 27：gpt-oss 和 Qwen3 之间的架构比较](images/figure_28.png)

图 27 中未显示的一个方面是 gpt-oss 使用滑动窗口注意力（类似于 Gemma 3，但每隔一层使用，而不是使用 5:1 比例）。

### **9.1 宽度与深度**

图 27 显示 gpt-oss 和 Qwen3 使用类似的组件。但是如果我们仔细观察这两个模型，我们会发现 Qwen3 是一个更深的架构，有 48 个 transformer 块而不是 24 个。

另一方面，gpt-oss 是一个更宽的架构：

- 嵌入维度为 2880 而不是 2048

- 中间专家（前馈）投影维度也是 2880 而不是 768

还值得注意的是，gpt-oss 使用的注意力头数量是 Qwen3 的两倍，但这并不会直接增加模型的宽度。宽度由嵌入维度决定。

在固定参数数量的情况下，一种方法是否比另一种方法具有优势？根据经验，更深的模型具有更大的灵活性，但由于不稳定问题（爆炸和消失梯度）可能更难训练（RMSNorm 和跳跃连接旨在缓解这些问题）。

更宽的架构的优势在于在推理期间速度更快（具有更高的 token/秒吞吐量），因为以更高的内存成本实现了更好的并行化。

在建模性能方面，据我所知，没有好的同类比较（参数大小和数据集保持不变），除了 [Gemma 2 论文中的消融研究（表 9）](https://arxiv.org/abs/2408.00118)，该研究发现对于 90 亿参数架构，更宽的设置略优于更深的设置。在 4 个基准测试中，更宽的模型获得了 52.0 的平均分数，更深的模型获得了 50.8 的平均分数。

### **9.2 少量大型专家与许多小型专家**

如上图 27 所示，值得注意的是 gpt-oss 的专家数量惊人地少（32 而不是 128），并且每个 token 仅使用 4 个而不是 8 个活动专家。然而，每个专家比 Qwen3 中的专家大得多。

这很有趣，因为最近的趋势和发展指向更多、更小的模型是有益的。在固定总参数大小的情况下，这种变化在来自 DeepSeekMoE 论文的下图 28 中得到了很好的说明。

![图 28：来自 "DeepSeekMoE: Towards Ultimate Expert Specialization in Mixture-of-Experts Language Models" 的注释图，https://arxiv.org/abs/2401.06066](images/figure_29.png)

值得注意的是，与 DeepSeek 的模型不同，gpt-oss 和 Qwen3 都不使用共享专家。

### **9.3 注意力偏置和注意力汇聚**

gpt-oss 和 Qwen3 都使用分组查询注意力。主要区别在于，如前所述，gpt-oss 在每个第二层通过滑动窗口注意力限制上下文大小。

然而，有一个引起我注意的有趣细节。似乎 gpt-oss 在注意力权重中使用偏置单元，如下图 29 所示。

![图 29：gpt-oss 模型在注意力层中使用偏置单元](images/figure_30.png)

自从 GPT-2 时代以来，我还没有见过这些偏置单元的使用，它们通常被认为是多余的。事实上，我发现了一篇最近的论文在数学上证明这至少对于键变换（`k_proj`）来说是正确的。此外，实证结果显示有和没有偏置单元之间几乎没有差异（见下图 30）。

![图 30：来自 https://arxiv.org/pdf/2302.08626 的表，显示从头训练模型时有无偏置单元的平均测试损失](images/figure_31.png)

您可能已经注意到图 30 中代码截图中 `sinks` 的定义。在一般模型中，注意力汇聚是位于序列开头的特殊"始终被关注"的 token，以稳定注意力，这在长上下文场景中特别有用。即，如果上下文变得非常长，序列开头的这个特殊被关注的 token 仍然会被关注，它可以学习存储有关整个序列的一些一般有用信息。（我认为它最初是在 [Efficient Streaming Language Models with Attention Sinks](https://arxiv.org/abs/2309.17453) 论文中提出的。）

在 gpt-oss 实现中，*注意力汇聚* 不是输入序列中的实际 token。相反，它们是被附加到注意力分数的学习到的每头偏置对数（图 31）。目标与上述注意力汇聚相同，但不需要修改分词输入。

![图 31：gpt-oss 中注意力汇聚的使用；基于 Hugging Face 代码](images/figure_32.png)

有关 gpt-oss 的更多信息，以及它与 GPT-2 的比较，请参阅我的另一篇 gpt-oss 文章：

[**从 GPT-2 到 gpt-oss：分析架构进步**](https://magazine.sebastianraschka.com/p/from-gpt-2-to-gpt-oss-analyzing-the)

作者：Sebastian Raschka, PhD

·

2025年8月9日

[阅读全文](https://magazine.sebastianraschka.com/p/from-gpt-2-to-gpt-oss-analyzing-the)

---

# 10. Grok 2.5

在这篇文章首次上线几周后，xAI 发布了他们 2700 亿参数的 Grok 2.5 模型的权重。

我认为值得在此介绍，因为 Grok 2.5 是 xAI 去年（2024 年）的旗舰生产模型。到目前为止，我们讨论的所有模型从一开始就是作为开源权重模型发布的。例如，gpt-oss 可能不是 GPT-4 的开源权重克隆，而是专门为开源社区训练的定制模型。

通过 Grok 2.5，我们可以一窥一个真实的生产系统，即使它是去年的。

架构上，Grok 2.5 整体看起来相当标准（图 32），但有一些值得注意的细节。

![图 32：Grok 2.5 与类似大小的 Qwen3 模型对比](images/figure_33.png)

例如，Grok 2.5 使用少量的大型专家（八个），这反映了一种较旧的趋势。如前所述，更近期的设计（如 DeepSeekMoE 论文中的设计）更倾向于大量的小型专家（Qwen3 中也存在这一点）。

另一个有趣的选择是使用了等效于共享专家的东西。图 32 中左侧显示的额外 SwiGLU 模块充当一个始终在线的共享专家。它与经典的共享专家设计不完全相同，因为其中间维度加倍了，但想法是一样的。（我仍然觉得 Qwen3 省略了共享专家很有趣，看看 Qwen4 和后续模型是否会改变这种情况会很有趣。）

---

# 11. GLM-4.5

[GLM-4.5](https://arxiv.org/abs/2508.06471) 是今年另一个重要发布。

它是一个指令/推理混合模型，类似于 Qwen3，但针对函数调用和代理风格的上下文进行了更好的优化。

![图 33：来自官方 GitHub 仓库的 GLM-4.5 基准，https://github.com/zai-org/GLM-4.5](images/figure_34.png)

如图 34 所示，GLM-4.5 有两个变体。旗舰 3550 亿参数模型在 12 个基准测试中平均优于 Claude 4 Opus，仅略逊于 OpenAI 的 o3 和 xAI 的 Grok 4。还有 GLM-4.5-Air，一个更紧凑的 1060 亿参数版本，其性能仅略低于 3550 亿模型。

图 35 将 3550 亿架构与 Qwen3 进行了比较。

![图 34：GLM-4.5 与类似大小的 Qwen3 模型对比](images/figure_35.png)

设计上大体相似，但 GLM-4.5 采用了一个最初由 DeepSeek V3 引入的结构选择：3 个密集层位于混合专家（MoE）块之前。为什么？如果立即引入 MoE 路由，稀疏专家选择的不稳定性可能会干扰早期句法和语义特征提取。因此，可以说通过保持初始层为密集层，确保模型在路由决策开始塑造更高级处理之前形成稳定的低级表示。

此外，GLM-4.5 使用与 DeepSeek V3 类似的共享专家（与 Qwen3 不同）。

（有趣的是，GLM-4.5 还保留了 GPT-2 和 gpt-oss 中使用的注意力偏置机制。）

---

# 12. Qwen3-Next

2025 年 9 月 11 日，Qwen3 团队发布了 Qwen3 Next 80B-A3B（图 35），有 Instruct 和 Thinking 两种变体。虽然其设计基于之前讨论的 Qwen3 架构，但我将其作为单独的条目列在此处，以保持图编号的一致性并引起对其某些设计变更的关注。

## **12.1 专家大小和数量**

新的 Qwen3 Next 架构之所以突出，是因为尽管比之前的 235B-A22B 模型（图 35）小 3 倍，但它引入了 4 倍多的专家，甚至添加了一个共享专家。这两种设计选择（高专家数量和包含共享专家）是我在此版本之前强调的未来方向，特别是我在本文顶部链接的视频版本中。

![图 35：5 月发布的原始 Qwen3 模型（左）旁边是 9 月发布的 Qwen3 Next 模型（右）](images/figure_36.png)

## **12.2 Gated DeltaNet + Gated Attention 混合**

另一个亮点是它们用 [Gated DeltaNet](https://arxiv.org/abs/2412.06464) \+ [Gated Attention](https://arxiv.org/abs/2505.06708) 混合机制替换了常规注意力机制，这有助于在内存使用方面启用原生 262k token 上下文长度（之前的 235B-A22B 模型原生支持 32k，使用 [YaRN](https://arxiv.org/abs/2309.00071) 缩放可支持 131k）。

那么这个新注意力混合机制是如何工作的呢？相比仍为标准缩放点积注意力的分组查询注意力（GQA）（在查询头组之间共享 K/V 以减少 KV 缓存大小和内存带宽，如前所述，但其解码成本和缓存仍随序列长度增长），它们的混合机制将 _Gated DeltaNet_ 块和 _Gated Attention_ 块以 3:1 的比例混合，如图 36 所示。

![图 36：Gated DeltaNet + Gated Attention 混合机制](images/figure_37.png)

我们可以将门控注意力块视为可用于 GQA 的标准缩放点积注意力，但它在上面有一些调整。_门控注意力_ 和普通 GQA 块之间的主要区别是：

1. 输出门（sigmoid 控制，通常是逐通道的），在加回到残差之前缩放注意力结果；

2. QKNorm 的零中心 RMSNorm，而不是标准 RMSNorm；

3. 部分 RoPE（应用于维度的一个子集）。

请注意，这些本质上只是对 GQA 的稳定性更改。

Gated DeltaNet 是一个更重大的变化。在 DeltaNet 块中，q、k、v 和两个门（α、β）由线性层和带归一化的轻量卷积层产生，该层用快速权重 _[delta rule](https://arxiv.org/abs/2412.06464)_ 更新替换了注意力。

然而，权衡是 DeltaNet 提供的基于内容的检索精度低于完整注意力，这就是为什么保留一个门控注意力层。

鉴于注意力以二次方增长，添加 DeltaNet 组件是为了帮助提高内存效率。在"线性时间、无缓存"系列中，DeltaNet 块基本上是 Mamba 的替代品。Mamba 使用学习的状态空间滤波器（基本上是随时间的动态卷积）保持状态。DeltaNet 使用 α 和 β 更新的微小快速权重内存来保持状态，并使用 q 读取它，仅使用小卷积来帮助形成 q、k、v、α、β。

## **12.3 多 token 预测**

上面两个小节描述了旨在提高效率的两个设计决策。既然所有好事都成三出现，Qwen3 也在此基础上添加了另一种效率技术：[多 token 预测](https://arxiv.org/abs/2404.19737)（MTP）。

（请注意，DeepSeek V3 & V3.2，以及后来的 GLM-4.5 和 MiniMax-M2 在训练期间都使用 MTP；然而，由于它是一种训练技术，我没有在架构比较中明确讨论它。）

多 token 预测训练 LLM 在每一步预测多个未来的 token，而不是单个。这里，在每个位置 _t_，小的额外头（线性层）输出 _t+1...t+k_ 的对数，我们对这些偏移的交叉熵损失求和（在 [MTP](https://arxiv.org/abs/2404.19737) 论文中，研究人员推荐 _k=4_）。这个额外的信号加快了训练速度，并且推理可能仍保持一次生成一个 token。然而，额外的头可以用于推测性多 token 解码，这似乎是 Qwen3-Next 所做的，但细节仍然有点稀疏：

> Qwen3-Next 引入了原生多 token 预测（MTP）机制，它不仅产生具有高接受率的 MTP 模块用于推测解码，而且还增强了整体性能。此外，Qwen3-Next 专门优化了 MTP 的多步推理性能，通过保持训练和推理之间一致性的多步训练，进一步提高了实际场景中推测解码的接受率。[来源：Qwen3-Next 博客文章](https://qwen.ai/blog?id=4074cca80393150c248e508aa62983f9cb7d27cd&from=research.latest-advancements-list)

## **12.4 Qwen3-Coder-Next**

2026 年 2 月初，Qwen3 团队[分享](https://github.com/QwenLM/Qwen3-Coder/blob/main/qwen3_coder_next_tech_report.pdf) 了 80B Qwen3-Coder-Next 模型（30 亿活动参数），该模型因在编码任务上优于大得多的模型（如 DeepSeek V3.2（370 亿活动参数）和 Kimi K2.5 和 GLM-7.5（均为 320 亿活动参数））而成为头条新闻。

此外，Qwen3-Coder-Next 的 SWE-Bench Pro 性能大致与 Claude-Sonnet-4.5 相当（仅略低于 Claude-Opus-4.5），对于一个开源权重模型来说令人印象深刻！

请注意，Qwen3-Coder-Next 的底层架构与我们上面讨论的 Qwen3-Next 80B 完全相同（事实上，他们使用 Qwen3-Next 作为基础模型来训练 Qwen3-Coder-Next。由于这是一篇关于 LLM 架构的文章，训练细节不在范围内。但是，感兴趣的读者可以在 [GitHub 上的详细技术报告](https://github.com/QwenLM/Qwen3-Coder/blob/main/qwen3_coder_next_tech_report.pdf) 中找到更多信息。

---

# **13. MiniMax-M2**

最近，开源权重 LLM 开发者分享了他们为效率优化的核心架构变体。一个例子是 Qwen3-Next（见上一节），它用快速门控 DeltaNet 模块替换了一些完整注意力块。另一个例子是 DeepSeek V3.2，它使用稀疏注意力，这是一种线性注意力变体，以一些建模性能为代价换取改进的计算性能（我计划在即将发表的文章中更详细地介绍这种机制）。

现在，[MiniMax-M1](https://arxiv.org/abs/2506.13585) 属于与上述模型类似的类别，因为它使用线性注意力变体（闪电注意力），提供比常规（完整）注意力更高的效率。我最初没有介绍 MiniMax M1，因为它不像本文讨论的其他一些模型那样流行。然而，他们的新 [MiniMax-M2](https://huggingface.co/MiniMaxAI/MiniMax-M2) 发布目前被认为是最好的开源权重模型（根据基准性能），这使得它不能被忽视。

![图 37：MiniMax-M2 基准性能与其他流行的开源权重和专有 LLM 的比较。图片来自官方模型中心发布 [readme](images/figure_38.png) 文件](images/figure_44.png)

如下图概述所示，我将 MiniMax-M2 与其他解码器风格的 transformer LLM 归为一类，因为它没有使用 MiniMax-M1 中提出的高效闪电注意力变体。相反，开发者重新使用了完整注意力，可能是为了改进建模（和基准）性能。

![图 38：本文涉及的主要 LLM 时间线，旁边是一些注意力混合模型，这些模型构成了更高效的替代方案，以一些建模性能为代价换取改进的效率](images/figure_39.png)

总体而言，MiniMax-M2 令人惊讶地与 Qwen3 相似。除了更改层数、大小等之外，它整体使用相同的组件。

## 13.1 逐层 QK-Norm

也许这里值得注意的亮点是 MiniMax-M2 使用所谓的"per_layer" QK-Norm 而不是常规 QK-Norm。仔细查看 [代码](https://github.com/vllm-project/vllm/blob/main/vllm/model_executor/models/minimax_m2.py#L222C23-L222C45) 揭示了它在意力机制内是这样实现的：

```python
self.q_norm = MiniMaxText01RMSNormTP(self.head_dim * self.total_num_heads, eps=...)
self.k_norm = MiniMaxText01RMSNormTP(self.head_dim * self.total_num_kv_kv_heads, eps=...)
```

这里，`hidden_size` 等于连接的头（`num_heads * head_dim`），所以 RMSNorm 具有一个比例向量，为每个头（以及每个头维度）提供不同的参数。

因此，"`per_layer`" 意味着 RMSNorm（用于前面解释的 QK-Norm）在每个 transformer 块中定义（与常规 QK-Norm 一样），但是，此外，它不是在注意力头之间重用，而是每个注意力头都有一个唯一的 QK-Norm。

[模型配置文件](https://huggingface.co/Qwen/Qwen3-235B-A22B/blob/main/config.json) 还包括滑动窗口注意力设置（类似于第 3 节中的 Gemma 3），但是，与第 4 节中讨论的 Mistral 3.1 一样，默认情况下是禁用的。

否则，除了逐层 QK-Norm 之外，架构与 Qwen3 非常相似，如下图所示。

![图 39：Qwen3 和 MiniMax-M2 之间的比较](images/figure_40.png)

## 13.2 MoE 稀疏性

其他有趣的细节，如下图所示，包括他们不使用共享专家（类似于 Qwen3，但与 Qwen3-Next 不同）。如前所述，在我看来，共享专家是有用的，因为它们减少了其他专家之间的冗余。

此外，从上图中可以明显看出，MiniMax-M2 的"稀疏性"是 Qwen3 的两倍。即，在与 Qwen3 235B-A22B 大致相同大小的情况下，MiniMax-M2 每个 token 只有 100 亿而不是 220 亿个活动专家（也就是说，MiniMax-M2 每次推理步骤使用 4.37% 的参数，而 Qwen3 使用 9.36% 的活动 token）。

## 13.3 部分 RoPE

最后，类似于 MiniMax-M1，MiniMax-M2 在注意力模块内使用"部分" RoPE 而不是常规 RoPE 来编码位置信息。类似于常规 RoPE，旋转在应用 QK-Norm 后应用于查询和键。

这里的部分 RoPE 意味着只有每个头的前 `rotary_dim` 通道获得旋转位置编码，而剩余的 `head_dim - rotary_dim` 通道保持不变。

在官方的 M1 [README](https://github.com/MiniMax-AI/MiniMax-01) 文件中，开发者提到

> 旋转位置嵌入（RoPE）应用于一半的注意力头维度，基频为 10,000,000

我们可以这样描绘它：

```
完整 RoPE:     [r r r r r r r r]
部分 RoPE:  [r r r r — — — —]
```

在上面的概念性说明中，"r" 显示旋转（位置编码）的维度，破折号是未触及的维度。

这是什么意思？在 [M1 论文](https://arxiv.org/abs/2501.08313) 中，开发者声明

> ……在 softmax 注意力维度的一半上实现 RoPE 使得长度外推成为可能，而不会降低性能。

我的推测是，这可以防止长序列（特别是那些比训练数据集中最长文档更长的序列）"过多"的旋转。这里的理由可能是没有旋转比模型在训练中没有见过的"坏"或"过于极端"的旋转更好。

---

# **14. Kimi Linear**

最近，线性注意力机制有所复兴，以提高 LLM 的效率。

Attention Is All You Need 论文（2017 年）中介绍的注意力机制，也称为缩放点积注意力，仍然是当今 LLM 中最受欢迎的注意力变体。除了传统的多头注意力，它还以更高效的变体形式使用，如分组查询注意力、滑动窗口注意力和多头潜在注意力。

## **14.1 传统注意力和二次方成本**

原始注意力机制随序列长度呈二次方缩放：

Attention(Q,K,V)=softmax(QK⊤d)V

这是因为查询（Q）、键（K）和值（V）是 _n_×_d_ 矩阵，其中 _d_ 是嵌入维度（一个超参数），_n_ 是序列长度（即 token 数量）。

您可以在我的另一篇文章中找到有关注意力的更多详细信息：

[**理解和编码自注意力、多头注意力、因果注意力和 LLM 中的交叉注意力**](https://magazine.sebastianraschka.com/p/understanding-and-coding-self-attention)

[阅读全文](https://magazine.sebastianraschka.com/p/understanding-and-coding-self-attention)

![图 40：由于序列长度 _n_，注意力中的二次方成本示意图](images/figure_41.png)

## **14.2 线性注意力**

线性注意力变体已经存在很长时间，我记得在 2020 年代看到了大量论文。例如，我记得的最早的一篇是 2020 年的 [Transformers are RNNs: Fast Autoregressive Transformers with Linear Attention](https://arxiv.org/abs/2006.16236) 论文，其中研究人员近似了注意力机制：

Attention(Q,K,V)=softmax(QK⊤d)V≈ϕ(Q)(ϕ(K)⊤V)

这里，φ(·) 是一个核特征函数，设置为 φ(x) = elu(x) + 1。

这种近似是有效的，因为它避免了显式计算 n × n 注意力矩阵 QK^T。而不是执行所有成对的 token 交互（这需要 O(n^2d) 时间和内存）。

我不想在这些较早的尝试上花太多时间。但底线是，它们将时间和内存复杂度从 O(n^2) 降低到 O(n)，从而使长序列的注意力效率更高。

然而，它们从未真正获得过吸引力，因为它们降低了模型准确性，而且我从未真正看到这些变体之一应用于开源权重的最先进 LLM 中。

## **14.3 线性注意力的复兴**

在今年下半年，线性注意力变体有所复兴。第一个值得注意的模型是 [MiniMax-M1](https://arxiv.org/abs/2506.13585)，它具有闪电注意力，是一个 4560 亿参数的混合专家（MoE）模型，具有 460 亿活动参数，于 6 月份发布。

然后在 8 月份，Qwen3 团队推出了 Qwen3-Next，我在上面更详细地讨论了它。然后在 9 月份，DeepSeek 团队宣布了 DeepSeek V3.2。这三个模型（MiniMax-M1、Qwen3-Next、DeepSeek V3.2）将大部分或所有层中的传统二次方注意力变体替换为高效的线性变体。

有趣的是，最近出现了剧情转折，MiniMax 团队发布了他们新的 2300 亿参数 M2 模型（第 13 节中讨论），没有使用线性注意力，回到常规注意力。团队表示，线性注意力在生产 LLM 中很棘手。它似乎在常规提示下工作正常，但在推理和多轮任务中准确性较差，这不仅对常规聊天会话很重要，而且对代理应用程序也很重要。

这可能是一个转折点，线性注意力可能最终不值得追求。但是，更有意思的是。10 月份，Kimi 团队发布了他们新的 [Kimi Linear](https://arxiv.org/abs/2510.26692) 模型，具有线性注意力。

![图 41：线性注意力混合架构的概述](images/figure_42.png)

旁注：我本可以将 Qwen3-Next 和 Kimi Linear 与其他 transformer-状态空间模型（SSM）混合模型分组到概览图中。就我个人而言，我将这里讨论的其他 transformer-SSM 混合视为具有 transformer 组件的 SSM，而我将这里讨论的模型（Qwen3-Next 和 Kimi Linear）视为具有 SSM 组件的 transformer。但是，由于我在 transformer-SSM 框中列出了 IBM Granite 4.0 和 NVIDIA Nemotron Nano 2，有人可能会认为它们应该放在一个类别中。

## **14.4 Kimi Linear 与 Qwen3-Next**

Kimi Linear 与 Qwen3-Next 共享几个结构相似之处。两个模型都依赖于混合注意力策略。具体来说，它们将轻量级线性注意力与较重的完整注意力层相结合。具体来说，两者都使用 3:1 的比例，这意味着每三个使用线性 Gated DeltaNet 变体的 transformer 块，就有一个使用完整注意力的块，如下图所示。

![图 42：Qwen3-Next 和 Kimi Linear 并排](images/figure_43.png)

Gated DeltaNet 是一种线性注意力变体，灵感来自循环神经网络，包括来自 [Gated Delta Networks: Improving Mamba2 with Delta Rule](https://arxiv.org/abs/2412.06464) 论文的门控机制。从某种意义上说，Gated DeltaNet 是具有 Mamba 风格门控的 DeltaNet，而 DeltaNet 是一种线性注意力机制。由于本文的概览性质，DeltaNet 可能是未来单独文章的好主题。

请注意，上图中 Kimi Linear 部分中省略了 RoPE 框是有意为之。Kimi 在多头潜在注意力 MLA 层（全局注意力）中应用 NoPE（无位置嵌入）。正如作者所述，这使得 MLA 在推理时作为纯多查询注意力运行，并避免了为长上下文缩放重新调整 RoPE（位置偏差据称由 Kimi Delta Attention 块处理）。有关 MLA 和多查询注意力（分组查询注意力的特例）的更多信息，请参阅我的 [大型 LLM 架构对比](https://magazine.sebastianraschka.com/p/the-big-llm-architecture-comparison) 文章。

**此外，我在这里](https://sebastianraschka.com/llms-from-scratch/ch04/08_deltanet/) 写了更多关于 Gated DeltaNet 的内容。**

## **14.5 Kimi Delta Attention**

Kimi Linear 通过 Kimi Delta Attention（KDA）机制修改了 Qwen3-Next 的线性注意力机制，这本质上是 Gated DeltaNet 的一个改进版本。Qwen3-Next 应用一个标量门（每个注意力头一个值）来控制内存衰减率，Kimi Linear 替换为每个特征维度的逐通道门控。根据作者的说法，这提供了对内存的更多控制，这反过来又改善了长上下文推理。

此外，对于完整注意力层，Kimi Linear 用多头潜在注意力（MLA）替换了 Qwen3-Next 的门控注意力层（本质上是具有输出门控的标准多头注意力层）。这与我们之前在 DeepSeek V3/R1 节中讨论的相同 MLA 机制，但带有额外的门控。（回顾一下，MLA 压缩键/值空间以减少 KV 缓存大小。）

没有与 Qwen3-Next 的直接比较，但与 Gated DeltaNet 论文中的 Gated DeltaNet-H1 模型（本质上是具有滑动窗口注意力的 Gated DeltaNet）相比，Kimi Linear 在保持相同 token 生成速度的同时实现了更高的建模准确性。

![图 43：来自 Kimi Linear 论文的注释图，显示 Kimi Linear 与 GatedDeltaNet 一样快，并且比具有多头潜在注意力的架构（如 DeepSeek V3/R1）快得多，同时具有更高的基准性能](images/figure_44.png)

此外，根据 [DeepSeek-V2 论文](https://arxiv.org/abs/2405.04434) 中的消融研究，当仔细选择超参数时，MLA 与常规完整注意力相当。

Kimi Linear 在长上下文和推理基准上比 MLA 更有利的事实使得线性注意力变体再次有希望用于更大的最先进模型。话虽如此，Kimi Linear 是 480 亿参数大小，但比 Kimi K2 小 20 倍。看看 Kimi 团队是否会在即将推出的 K3 模型中采用这种方法会很有趣。

---

# 15. Olmo 3 Thinking

Allen AI [发布了他们的新 Olmo 3](https://allenai.org/blog/olmo3) 7B 和 32B 模型于 11 月 20 日发布。（官方拼写已从 OLMo 更改为 Olmo，所以我将在本节中采用此拼写。）

如前所述，Olmo 模型始终很有趣，因为它们是完全开源的。这意味着该团队还分享了[详细的训练报告](https://www.datocms-assets.com/64837/1763662397-1763646865-olmo_3_technical_report-1.pdf)、多个检查点、有关训练数据的信息等等。换句话说，Olmo 模型是完全透明的。

这一次，Olmo 套件还以额外的推理模型变体（除基础和指令模型外）形式提供，Olmo 3 的[技术报告](https://www.datocms-assets.com/64837/1763662397-1763646865-olmo_3_technical_report-1.pdf) 中有很多关于训练的有趣细节。但是，由于这是一篇关于架构比较的文章，本节仅关注 Olmo 3 的架构。

与 Olmo 3 最接近的比较模型将是 Qwen3，因为 Qwen3 系列有两个大小相似的模型，并且 Qwen3 模型的性能相似。

首先，让我们看看两者中较小的 Olmo 3 7B。

![图 44：Olmo 3 7B 和 Qwen3 8B 并排](images/figure_45.png)

正如我们可以看到的，Olmo 3 架构与 Qwen3 相对相似。然而，值得注意的是，这本质上可能是受 Olmo 2 前身的启发，而不是 Qwen3。

与 Olmo 2 类似，Olmo 3 仍然使用 post-norm 而不是 pre-norm，因为他们在 Olmo 2 论文中发现这可以稳定训练。

有趣的是，7B 模型仍然使用与 Olmo 2 类似的多头注意力。但是，为了提高效率并缩小 KV 缓存大小，他们现在使用滑动窗口注意力（例如，类似于 Gemma 3）。

接下来，让我们看看 32B 模型。

![图 45：Olmo 3 32B 和 Qwen3 32B 并排](images/figure_46.png)

总体而言，架构相同，只是规模更大。此外，比例（例如，从输入到前馈层中的中间大小的比例等）大致与 Qwen3 中的比例相匹配。

我的猜测是，由于词汇量较小，架构最初可能比 Qwen3 稍小一些，然后他们将中间大小扩展从 Qwen 3 中的 5 倍增加到 Olmo 3 中的 5.4 倍，以拥有一个 32B 模型进行直接比较。

另请注意，32B 模型使用分组查询注意力。

也许最后一个细节是，Olmo 3 对支持的 64k 上下文长度使用 YaRN 进行上下文扩展，但仅用于全局（非滑动窗口注意力）层。（[YaRN](https://arxiv.org/abs/2309.00071) 本质上是一种仔细的 RoPE 重新缩放技术，有助于在长上下文大小下更好地保持模型质量。）

在 Qwen3 中，YaRN 是可选的，用于将原生上下文从 32k token 扩展到 131k token。

如果您对其他架构细节感兴趣，我在独立笔记本中[从零开始实现了 Olmo 3](https://github.com/rasbt/LLMs-from-scratch/blob/main/ch05/13_olmo3/standalone-olmo3.ipynb)。

![图 46：Olmo 3 从零开始实现](images/figure_47.png)

---

# 16. DeepSeek V3.2

本文以 DeepSeek V3 开头，该模型于 2024 年 12 月发布。当时有过多次 DeepSeek 发布，但我在很大程度上跳过了它们，因为它们不像 DeepSeek V3 和 DeepSeek R1 那样是大型旗舰模型发布。

![图 47：自 DeepSeek V3 以来 DeepSeek 模型发布时间线。主要模型以红色显示](images/figure_48.png)

然而，DeepSeek V3.2 是一个非常重要的发布，因为在某些基准测试中它与当前的 GPT-5.1 和 Gemini 3.0 Pro 模型相当。

架构总体上与 DeepSeek V3 类似，但他们添加了稀疏注意力机制以提高效率。

![图 48：具有多头潜在和稀疏注意力的 DeepSeek 模型架构](images/figure_49.png)

我原本计划为本文写一个关于 DeepSeek V3.2 的简短部分，但它变成了超过 5000 字的篇幅，所以我将其移至单独的文章，下面是链接：

[**DeepSeek 模型从 V3 到 V3.2 的技术之旅**](https://magazine.sebastianraschka.com/p/technical-deepseek)

---

# 17. Mistral 3

2025 年 12 月 2 日，DeepSeek V3.2 发布后一天，Mistral 团队发布了他们的新 [Mistral 3](https://mistral.ai/news/mistral-3) 模型套件。这包括三个较小的密集模型（3B、8B 和 14B），称为 Ministral 3，以及他们新的 Mistral 3 Large 旗舰模型，这是一个 6750 亿参数的 MoE（具有 410 亿活动参数）。更具体地说，Mistral 3 Large 模型由以下部分组成

- 一个 6730 亿参数和 390 亿活动参数的 MoE 语言模型
- 一个 25 亿参数的视觉编码器

（由于本文关注 LLM 方面，我们将忽略本节中的视觉编码器。但我可能应该在某个时候更新我的[多模态 LLM 文章](https://magazine.sebastianraschka.com/p/understanding-multimodal-llms)。）

首先，有趣的是，这是自 2023 年 Mixtral 以来 Mistral 的第一个 MoE（在本节前面，我写道 Mistral 放弃了 MoE，DeepSeek V3 去年开启了 MoE 复兴）。

发布博客文章说，所有模型大小都提供基础、指令和推理变体，这很好。然而，他们的 6750 亿模型的推理版本尚不可用。

另一个有趣的细节是，Mistral 在这里与 NVIDIA 合作，根据他们的[公告](https://mistral.ai/news/mistral-3)优化了 Blackwell 芯片上的 token/秒吞吐量。这很好，因为这意味着 Ministral 模型在我的小型 DGX Spark 上运行速度会比可比模型快一点（我还需要测试这一点）。

除了 Mistral 3 的 token/秒速度优势外，根据质量基准，他们的较小模型 Ministral 与 Qwen3 相当。较大的旗舰模型与 DeepSeek V3.1 相当。

由于 Mistral 3 的发布仅在 DeepSeek V3.2 发布后一天，他们在文章中没有包含任何 V3.2 比较（除了 LMArena Elo 分数，其中 DeepSeek V3.2 略微领先，为 1423 vs 1418）。

不幸的是，目前不可能进行同类比较，因为 Mistral 3 Large 目前没有推理模型，而 DeepSeek V3.2 没有分享其非思考模式的基准结果，但如果您好奇，我将 DeepSeek V3.2-Thinking 数字（来自 [DeepSeek V3.2 报告](https://arxiv.org/abs/2512.02556)）与 Mistral 3 Large 基准图表叠加在一起。

![图 49：Mistral 3 Large 基准，来自 [Mistral 3 公告](images/figure_50.png)，DeepSeek V3.2 结果（来自 [DeepSeek V3.2 论文](https://arxiv.org/abs/2512.02556)）叠加在顶部](images/figure_56.png)

观察 Mistral Large 3 Instruct 模型与旁边的 DeepSeek V3.2-Thinking 模型（数字来自 DeepSeek V3.2 论文），V3.2-Thinking 模型显然要好得多。因此，我正在等待 Mistral 3 Large Thinking 发布，并期待看到更新后的图表！

所以，就目前而言，我认为由于这些优化，Mistral 3 Large 是经济高效、低延迟部署的绝佳选择。如果您想最大化答案质量，DeepSeek V3.2-Thinking 非常好。Mistral 3 Large 的另一个卖点是它还提供多模态支持（DeepSeek V3.2 仅限文本）。

顺便说一下，我在这里对 DeepSeek V3.2 的关注来自于这样一个事实：这些模型在彼此发布后一天内发布。它们的大小几乎相同，671B 和 673B，这使比较变得有趣！

不幸的是，没有包含有关模型开发的更多信息的[技术报告](https://arxiv.org/abs/2512.02556)。然而，由于它是一个开源权重模型，我们确实有[Hugging Face hub 上的模型权重可以分析](https://huggingface.co/mistralai/Mistral-Large-3-675B-Instruct-2512-NVFP4/blob/main/params.json)，但是。因此，让我们仔细看看 Mistral 3 Large。

事实证明，Mistral 3 Large 与 DeepSeek V3 和 V3.1 的架构完全相同！唯一的区别是他们将专家的大小增加了 2 倍，同时将专家数量减少了相同的因子。

![图 50：DeepSeek V3 和 Mistral 3 Large 并排](images/figure_51.png)

然而，虽然它实际上是相同的架构，但 Mistral 团队可能从头开始训练 Mistral 3，而不是从 DeepSeek V3 初始化并进一步训练，因为 Mistral 使用自己的分词器。

在 Kimi K2 之后，Mistral 3 现在是使用 DeepSeek V3 架构的第二个模型系列。然而，Kimi K2 团队将模型大小从 671B 扩展到 1 万亿，Mistral 3 团队仅更改了专家大小比例并添加了视觉编码器以支持多模态。但是，是的，为什么不呢？我认为 DeepSeek V3 是一个相当可靠的架构设计，加上它具有这些很好的 MoE 和 MLA 效率方面。那么，为什么要改变没有损坏的东西呢？这些天的很多秘密成分都在训练管道以及推理扩展策略中。

---

# 18. Nemotron 3 Nano 和 Super

本文并不是对所有 LLM 的详尽列表。为了保持可管理性，我专注于主要亮点。这里的"亮点"意味着它们要么非常受欢迎，要么表现非常好，要么具有有趣的架构组件。

话虽如此，是时候终于将 NVIDIA 的模型之一添加到此列表了。NVIDIA [刚刚发布了](https://www.google.com/search?client=safari&rls=en&q=nemotron+3&ie=UTF-8&oe=UTF-8) Nemotron 系列的最新产品 Nemotron 3，于 2025 年 12 月 15 日发布。Nemotron 的好处在于，它不仅提供开源权重和[技术报告](https://research.nvidia.com/labs/nemotron/files/NVIDIA-Nemotron-3-Nano-Technical-Report.pdf)，NVIDIA 还分享了[数据集](https://huggingface.co/nvidia/datasets?search=nemotron&p=0)和[训练代码](https://huggingface.co/datasets/nvidia/Nemotron-Pretraining-Code-v2)，类似于 Olmo 3。

根据[公告文章](https://nvidianews.nvidia.com/news/nvidia-debuts-nemotron-3-family-of-open-models)，Nemotron 3 有三种尺寸：

1. Nano（30B-A3B），
2. Super（100B），（后来更新为 120B，见 18.1 节）
3. 和 Ultra（500B）。

## 18.1 Nemotron 3 Nano

在架构上，这些模型是混合专家（MoE）Mamba-Transformer 混合架构。撰写本文时（12 月 17 日），仅 Nano 模型已作为开源权重模型发布，因此下面的讨论将重点放在它上，如下图所示。

![图 51.1：Nemotron 3 Nano 模型的轮廓，这是 Transformer-Mamba 混合模型](images/figure_52.png)

如上所示，Nemotron 3 Nano（30B-A3B）是一个 52 层混合 Mamba-Transformer 模型，将 [Mamba-2](https://arxiv.org/abs/2405.21060) 序列建模块与稀疏混合专家（MoE）前馈层交错，并仅在少数层中使用自注意力。

上图中有许多内容，但简而言之，架构组织成 13 个宏块，具有重复的 Mamba-2 → MoE 子块，以及一些分组查询注意力层。总共，如果我们将宏块和子块相乘，此架构中有 52 层。

关于 MoE 模块，每个 MoE 层包含 128 个专家，但每个 token 仅激活 1 个共享和 6 个路由专家。

Mamba-2 层本身需要一整篇文章来解释（也许是另一个时间的话题）。但就目前而言，从概念上，您可以将它们视为类似于 Qwen3-Next 和 Kimi-Linear 使用的 Gated DeltaNet 方法，我在上面介绍了。您也可以在我的其他超越标准 LLM 文章中阅读更多内容：

Gated DeltaNet 和 Mamba-2 层之间的相似之处在于，两者都使用门控状态空间更新替换标准注意力。这种状态空间风格模块背后的思想是，它维护一个运行的隐藏状态并通过学习的门混合新输入。与注意力相反，它随输入序列长度线性缩放而不是二次缩放。

这种架构真正令人兴奋的是，与类似大小的纯 transformer 架构相比，它的性能非常好，同时实现了更高的每秒 token 吞吐量。

总体而言，这是一个有趣的方向，在仅使用几个注意力层方面比 Qwen3-Next 和 Kimi-Linear 更极端。然而，transformer 架构的一个优势是其在（非常）大规模下的性能。我很好奇 Nemotron 3 Super 尤其是 Ultra 与 DeepSeek V3.2 等相比会如何。

## 18.2 Nemotron 3 Super

2026 年 3 月 11 日，NVIDIA 现在还发布了 120B Super 版本作为 [开源权重模型](https://huggingface.co/nvidia/NVIDIA-Nemotron-3-Super-120B-A12B-BF16) 在 Hugging Face Hub 上，以及一份不错的"[Super"-focused 技术报告](https://research.nvidia.com/labs/nemotron/files/NVIDIA-Nemotron-3-Super-Technical-Report.pdf)。

与 Nano 模型相比，除了扩展架构外，架构还有两个主要修改。

首先，Nemotron 3 Super 使用 [多 token 预测（MTP）](https://arxiv.org/abs/2404.19737)，这是一种在每一步训练 LLM 预测多个未来 token 而不是单个 token 的技术。

不是仅使用标准的下一个 token 目标训练模型，MTP 还训练它从同一位置预测多个未来 token 偏移。这提供了更丰富的训练信号，根据 Super 报告，这可以改善建模质量和推理效率。

![图 51.2：多 token 预测与常规下一个 token 预测的对比。（左子图灵感来自 [MTP 论文](images/figure_53.png)。）最初，MTP 仅在训练期间使用，而不是推理；因此，推理时间步（底部）显示单个下一个 token 预测](images/figure_59.png)

与 MTP 的标准使用（我在上图 51.2 中绘制的）的关键区别是 Nemotron 3 Super 不仅在训练期间使用它。

Nemotron 3 Super 明确在推理时也使用 MTP，其中共享权重 MTP 头充当原生推测解码的内部草稿模型。在生成过程中，模型可以提出候选延续，然后用主模型验证它们。这减少了推理延迟，无需单独的外部草稿模型。

由于这不完全是标准 MTP，也许更准确的是将 Nemotron 3 Super 描述为使用共享权重 MTP 进行推测解码，而不是像其他架构（如 Step 3.5 Flash，我在[这里](https://magazine.sebastianraschka.com/p/a-dream-of-spring-for-open-weight) 介绍过）那样称之为"MTP-3"。

与 Nano 的第二个主要区别是 Super 架构使用潜在专家，这意味着专家在潜在空间中操作（MoE 层的输入从 4096 维下投影到 1024 维，应用专家，然后将输出从 1024 维上投影回 4096 维。

![图 51.3：Nemotron 3 Super 120B-A12B 具有潜在 MoE 层、多 token 预测和 Mamba-2 混合注意力方法](images/figure_54.png)

在基准测试方面，Nemotron 3 Super 与 Qwen3.5 122B-A10B 和 GPT-OSS 120B 相当，但由于上述"技巧"（MTP、潜在 MoE 和混合注意力），吞吐量非常好：比 Qwen3.5 122B-A10B 快 2 倍，并且（关于 NVFP4 版本）比 GPT-OSS 120B 快 2.2 倍。

![图 51.4：Nemotron 3 Super 基准比较，来自 [Hugging Face Hub](images/figure_55.png) 页面](images/figure_61.png)

---

# 19. Xiaomi MiMo-V2-Flash

12 月 2025 年还有另一个令人印象深刻的发布。小米发布了他们最新的 Xiaomi MiMo-V2-Flash，其令人印象深刻的基准性能与 DeepSeek V3.2 相当，但参数数量只有一半，推理速度更快。它是一个 3090 亿混合专家（MoE）模型，每个 token 有 15 个活动参数。

有趣的是，它以 5:1 的比例使用滑动窗口注意力（SWA）和全局（常规）注意力，类似于 Gemma 3（见第 3 节）。但是，它使用的滑动窗口大小（128）要激进得多，比 Gemma 3（1024）小 8 倍。

![图 52：Xiaomi MiMo-V2-Flash 与具有相似基准性能的 DeepSeek V3.2 的比较](images/figure_56.png)

据我所知，这是迄今为止最大的滑动窗口注意力模型。

此外，Xiaomi 模型使用多 token 预测（MTP），如前面 12.3 节所述。

---

# 20. Arcee AI Trinity Large

自上一个 LLM 架构添加以来已经有一段时间了。1 月 27 日，Arcee AI（一家我之前没有注意到的公司）开始在其模型中心发布他们的开源权重 400B [Trinity Large](https://huggingface.co/collections/arcee-ai/trinity-large) LLM，以及两个较小的变体。

他们的旗舰大型模型是一个 400B 参数 MoE（13B 活动参数）。两个较小的变体是 Trinity Mini（26B，3B 活动参数）和 Trinity Nano（6B，1B 活动参数）。

![图 53：Trinity Large 架构概述（基于模型中心 [配置文件](images/figure_57.png)）](images/figure_63.png)

随着模型权重的发布，Arcee AI 还发布了一份不错的[技术报告](https://github.com/arcee-ai/trinity-large-tech-report)，其中包含大量详细信息。

因此，让我们仔细看看 400B 旗舰模型。下图将其与前面讨论的 GLM 4.5（第 11 节）进行比较，后者可能最相似，也相对较小。此外，Trinity 技术报告显示 Trinity Large 和 GLM-4.5 基础模型的建模性能几乎相同（我假设他们没有将其与更新的基础模型进行比较，因为许多公司现在只分享他们的微调模型。）

![图 54：Arcee AI Trinity Large 与相对相似大小（400B 对 355B）的 GLM 4.5 并排](images/figure_58.png)

但是我们可以看到，Trinity 模型中添加了几个有趣的架构组件。

首先，有像 Gemma 3、Olmo 3、Xiaomi MiMo 等中那样的交替的局部:全局（滑动窗口）注意力层。但他们没有使用 Gemma 3 和 Xiaomi 使用的常见 5:1 比例，而是选择了类似于 Olmo 3 的 3:1 比例，以及相对较大的滑动窗口大小 4096（也类似于 Olmo 3）。

除了 QK-Norm（第 2 节 Olmo 2 中介绍），他们在全局层中使用 NoPE（我们在第 7 节 SmolLM3 中讨论过 NoPE）。

他们还有一种形式的门控注意力。他们没有完整的 GatedDeltaNet（第 12 节中讨论），而是使用类似于 Qwen3-Next 中注意力机制的门控。

但他们通过在输出线性投影之前向缩放点积添加逐元素门控来修改标准注意力（如下图所示），这减少了注意力汇聚并改善了长序列泛化。此外，它还有助于训练稳定性。

![图 55：Trinity Large 在注意力机制中使用的门控机制示意图](images/figure_59.png)

您可能已经注意到在之前的 Trinity Large 架构图中使用了四个（而不是两个）RMSNorm 层。这是他们所谓的深度缩放三明治归一化，基于以前的工作，但这是我在主要架构中没有见过的。总体而言，它看起来像 Gemma 3 风格的 RMSNorm 放置，但这里的诀窍是第二个 RMSNorm（在每个块中）的增益是深度缩放的，意味着它被初始化为大约 1 / sqrt(L)（L 是总层数）。因此，在训练早期，残差更新开始很小，并随着模型学习正确的尺度而增长。

MoE 是一个类似 DeepSeek 的 MoE，具有许多小专家，但使其更粗糙，这有助于推理吞吐量（我们在 Mistral 3 Large 采用 DeepSeek V3 架构时也看到过这一点）。

最后，有一些关于训练改进的有趣细节（一种新的 MoE 负载平衡策略和另一种使用 MuOpt 优化器的策略），但由于这是一篇架构文章，这些都不在范围内。

---

# 21. GLM-5

中国新年已成为开源权重发布的可靠窗口。例如，GLM-4 和 Qwen 1.5 于 2024 年 1 月和 2 月发布，DeepSeek R1 和 Qwen 2.5 于 2025 年发布。

今年，z.AI（Zhipu AI）于 2026 年 2 月 11 日率先发布了 GLM-5，大约在 2 月 17 日农历新年前一周。

与本文前面介绍的 GLM-4.5 模型（见第 11 节，于 2025 年夏季发布）相比，其 GLM-5 后继模型大小翻倍：从 3550 亿参数增加到 7440 亿，使其进入 DeepSeek-V3.2 和 Kimi K2 之间的领域。

与 GLM-4.5 类似，GLM-5 是一个混合专家（第 1.2 节）模型，每个 token 的活动参数数量仅略有增加：GLM-5 中为 400 亿，而 GLM-4.5 中为 320 亿。

![图 56：GLM-5 和 GLM-4.5 架构并排](images/figure_60.png)

有趣的是，如上图 56 所示，GLM-5 采用了 DeepSeek 的多头潜在注意力（MLA，见第 1.1 节）以及 DeepSeek 稀疏注意力（我在 [DeepSeek V3.2](https://magazine.sebastianraschka.com/p/technical-deepseek) 文章中更详细地介绍）。这些修改的动机是减少处理长上下文时的推理成本。

除此之外，架构相对相似。大小的增加主要是由于拥有更多专家（256 而不是 160）以及略微增加层大小。例如，嵌入维度和专家大小现在为 6,144（从 5,120 上升），中间投影大小也从 1,536 略微上升到 2,048。有趣的是，层数（transformer 块）从 92 减少到 78。我假设这是为了降低推理成本并使模型更快（因为层深度无法并行化）。

我通常不在此处包含基准，因为本文专注于架构。如果我要包含训练细节和评估，这篇文章将远远超出范围和长度。话虽如此，我看到我在 2025 年 7 月包含了 GLM-4.5 基准，因此我将在此处做出另一个例外，因为基准看起来确实令人印象深刻，与所有主要旗舰 LLM 产品（GPT-5.2 超高、Gemini Pro 3 和 Claude 4.6 Opus）相当。但同样，值得强调的是基准性能并不一定等于真实世界的性能。

![图 57：GLM 架构和基准。GLM-4.7 架构与 GLM-4.5 相似。基准来自 GLM-5 发布博客文章：https://z.ai/blog/glm-5](images/figure_61.png)

---

# 22. 更多 2026 年 2 月发布：从 Kimi K2.5 到 Tiny Aya

总共，2026 年 1 月至 2 月期间有 10 个有趣的开源权重 LLM 发布：

01. Arcee AI 的 Trinity Large（2026 年 1 月 27 日）

02. Moonshot AI 的 Kimi K2.5（2026 年 1 月 27 日）

03. StepFun Step 3.5 Flash（2026 年 2 月 1 日）

04. Qwen3-Coder-Next（2026 年 2 月 3 日）

05. z.AI 的 GLM-5（2026 年 2 月 12 日）

06. MiniMax M2.5（2026 年 2 月 12 日）

07. Nanbeige 4.1 3B（2026 年 2 月 13 日）

08. Qwen 3.5（2026 年 2 月 15 日）

09. Ant Group 的 Ling 2.5 1T & Ring 2.5 1T（2026 年 2 月 16 日）

10. Cohere 的 Tiny Aya（2026 年 2 月 17 日）

11. Sarvam 30B 和 105B（2026 年 3 月 6 日）

我在上面的第 19 节和第 20 节中介绍了 Arcee AI 的 Trinity Large 和 z.AI 的 GLM-5。但是，由于 1 月至 2 月期间有很多内容要介绍，我写了一篇独立文章，其中包含有关上述 10 个架构的更多信息：

---

# 23. Gemma 4

在 3 月份 Nemotron 3 Super 发布后，该月余下的时间对于旗舰开源权重模型发布相对平静。在我等待 DeepSeek-V4 的同时，至少在 4 月，Google 给我们带来了 Gemma 4。

在架构上，Gemma 4（31B）看起来与 Gemma 3（27B）基本保持不变，如下图所示。

![图 58：Gemma 3（27B）和 Gemma 4（31B）并排](images/figure_62.png)

（请注意，Gemma 4 现在还具有多模态模型支持，但我将把图像编码器部分留待以后单独的文章；在这里，我们只关注文本部分。）

正如我们从上图中可以看到的，Gemma 4 保持了相对独特的 Pre- 和 Post-norm 设置，并且仍然相对经典，具有 5:1 混合注意力机制，结合滑动窗口（局部）层和完整注意力（全局）层。注意力机制本身也是经典的分组查询注意力（GQA）。

然而，与 Gemma 3 相比，一个容易忽视的小变化是，对于全局（完整）注意力层，他们在注意力机制中重用了键。即，他们设置 values = keys，这应该会导致进一步的 KV 缓存大小减少。

此外，Gemma 4 还使用 p-RoPE，其中只有 25% 的频率对获得位置信息。这有助于减少长上下文情况下的位置噪声。

但不要被缺乏更大架构变化所迷惑。从基准来看，Gemma 4 是从 Gemma 3 的巨大飞跃！例如，在 [AI Arena Leaderboard](https://arena.ai/leaderboard/text) 上，Gemma 4（31B）的排名与更大的 Qwen3.5-397B-A17B 模型相似。但正如我在我的模型评估文章（链接如下）中讨论的那样，竞技场分数有点问题，因为它们可以被博弈，并且偏向于人类（风格）偏好。

但是，如果我们看一些其他常见的基准，我在下面绘制的图表，我们可以看到它确实是相对于 Gemma 3 的一个非常明显的飞跃，并且与 Qwen3.5 27B 相当。

![图 59：Gemma 3 与 Gemma 4 与 Qwen3.5（数字来自 [Gemma 4](images/figure_63.png) 和 [Qwen3.5](https://huggingface.co/Qwen/Qwen3.5-27B) 模型中心页面）](images/figure_69.png)

请注意，还有一个混合专家（MoE）Gemma 4 变体，如下图所示，旁边是类似大小的 Qwen3 模型。

![图 60：Qwen3 Coder Flash 与 Gemma 4 MoE 的比较](images/figure_64.png)

如上图所示，除了 Gemma 4 使用之前讨论的独特 Pre- 和 Post-norm 放置之外，这两种方法相对相似。

在基准测试方面，Gemma 4 MoE 变体（总参数比 Gemma 4（31B）密集变体少 40 亿）的性能相对相似。

![图 61：Gemma 4 MoE（26B-A4B）仅略差于 Gemma 4（31）密集](images/figure_65.png)

如果您对这里涉及的所有架构的视觉概览感兴趣，我在[这里](https://sebastianraschka.com/llm-architecture-gallery/) 整理了一个 LLM 架构画廊。

[LLM 架构画廊](https://sebastianraschka.com/llm-architecture-gallery/)

**经过这些年，LLM 发布仍然令人兴奋，我很好奇接下来会发生什么！**

*这本杂志是一个个人激情项目，您的支持有助于保持它的活力。*

*如果您想支持我的工作，请考虑我的[《从零开始构建大型语言模型》](https://amzn.to/4fqvn0D)一书或其续集[《从零开始构建推理模型》](https://mng.bz/Nwr7)。（我相信您会从中受益匪浅；它们以您在其他地方找不到的深度解释了 LLM 的工作原理。）*

*感谢您的阅读，并支持独立研究！*

*《从零开始构建大型语言模型》现在可在[亚马逊](https://amzn.to/4fqvn0D)购买。《从零开始构建推理模型》在 Manning 的[早期访问](https://mng.bz/Nwr7)中。*

如果您阅读了这本书并有几分钟时间，我会非常感谢您的[简短评论](https://www.amazon.com/Build-Large-Language-Model-Scratch/dp/1633437167)。它对我们作者帮助很大！

**您的支持意义重大！谢谢！**

除了架构差异之外，还有一点很有趣的是知道 LLM 是在哪些文本数据上训练的。从我的角度来看，令人遗憾的是，这些信息通常不会被披露，即使是对于开源 LLM 也是如此。不只是训练数据的数量（例如 token 数量），还有数据质量，作为真正科学比较的因素。

[还有 95 条评论...](https://magazine.sebastianraschka.com/p/the-big-llm-architecture-comparison/comments)

---

## 翻译说明

本文由 Sebastian Raschka 博士撰写，原文 URL：https://magazine.sebastianraschka.com/p/the-big-llm-architecture-comparison

文章详细介绍了 2024-2026 年间发布的主要大型语言模型（LLM）的架构设计，包括 DeepSeek V3/R1、OLMo 2、Gemma 3、Mistral、Llama 4、Qwen3、SmolLM3、Kimi K2、GPT-OSS、Grok 2.5、GLM-4.5/5、Qwen3-Next、MiniMax-M2、Kimi Linear、Olmo 3、DeepSeek V3.2、Mistral 3、Nemotron 3、Xiaomi MiMo、Arcee AI Trinity、GLM-5 和 Gemma 4 等 23 个模型。

原文共有 61 个图表（Figure 1-61），本文中保留所有图片并使用 figure_XX.png 命名。

最后更新：2026年4月2日
