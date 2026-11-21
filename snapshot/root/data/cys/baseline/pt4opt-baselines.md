## Baseline 运行清单（四类实验场景，方法集合已确认）

> PT4Opt 包含四类实验场景：**ML HPO、DL HPO、peptide optimization 和 query optimization**。前两类的搜索对象是可直接数值化的超参数配置，包括连续、离散和类别变量；后两类的搜索对象分别是氨基酸序列和 query plan，属于结构化搜索空间，需要通过各场景统一初始化的 encoder/VAE 映射到连续 latent space，并将候选解码回原始结构。以下仅列最终确认要运行的外部 baseline，不含 PT4Opt 内部消融，也不再设置补充候选；但不要求每个 baseline 强行覆盖全部四类场景，其适用场景和适配方式将在后续逐项说明。同一场景内，所有方法共享相同的初始 VAE、oracle、源任务／目标任务划分、warm-start 方式、可行性约束、batch size、总评估预算和随机种子。凡使用 VAE 的方法均允许在优化过程中更新 VAE；VAE 更新后必须重新编码全部已有观测，再继续拟合 surrogate 或搜索分布。

**统一口径：** ML HPO 和 DL HPO 直接在原始的数值化／混合超参数空间中运行优化器，不使用 VAE。Peptide optimization 和 query optimization 从各自相同的初始 VAE latent space 开始比较优化后端；不在原始氨基酸序列或 query-plan 结构上直接运行 TPE、SMAC、HEBO 等数值优化器。`Latent-TPE` 指从与 LOL-BO 相同的初始 VAE 出发、在 latent space 中运行的自定义 sanity baseline，并非文献中已有的独立优化方法。

### 无模型下限

- [x] **Random Search**
  - **ML HPO / DL HPO** 直接在各任务定义的数值化／混合超参数搜索空间中随机采样候选配置。
  - **Peptide optimization / query optimization** 在各场景当前的 VAE latent space 中随机采样 latent vector，再通过对应的 decoder 还原为氨基酸序列或 query plan。

### 非 LLM・单任务优化

- [x] **[LOL-BO](baseline_papers/35_lolbo.pdf)**
  - **ML HPO / DL HPO** 不使用 VAE。将标准化后的超参数向量直接作为 GP surrogate 的输入，并在以当前最优配置为中心的自适应 trust region 内选择下一批候选；这是 LOL-BO 去掉结构表示层后的 TuRBO 风格 local GP-BO。
  - **Peptide optimization / query optimization** 运行完整 LOL-BO
- [x] **TPE / Latent-TPE**
  - **ML HPO / DL HPO** 不使用 VAE，直接在任务定义的数值化／混合超参数空间中运行标准 TPE。
  - **Peptide optimization / query optimization** 从与 LOL-BO 相同的初始 VAE 出发，在当前 latent space 中运行 TPE，再将 latent candidate 解码为氨基酸序列或 query plan。若 VAE 被更新，则重新编码全部已有观测并重新拟合 TPE 的密度模型。

### 非 LLM・传统跨任务迁移

> **共同输入前提** FSBO 和 RGPE 不会跨 ML HPO、DL HPO、peptide optimization 与 query optimization 场景共享模型。这里需要特别强调：同一场景内部也必须继续按搜索空间分组。在 ML HPO 内部，只有搜索空间 schema 完全一致的 ML tasks 才能共同训练或迁移；在 DL HPO 内部，也只有搜索空间 schema 完全一致的 DL tasks 才能共同训练或迁移。一个任务组中的源任务与目标任务必须具有相同的变量数量、每一维的参数语义、变量类型与编码、取值范围以及条件参数的激活规则；仅仅向量维度相同并不足够。不同 schema 必须分别训练 FSBO 或分别建立 RGPE source-GP 集合。对于结构化场景，这一条件由同一套 VAE encoder/decoder 和同一个 latent 坐标系保证。

- [x] **[FSBO](baseline_papers/29_fsbo.pdf)**
  - **核心迁移方式** 每个同构搜索空间任务组单独训练一个 FSBO，只汇集与目标任务具有完全相同搜索空间 schema 的所有训练任务。将任务 `t` 的历史记录视为无序的 `(x,y)` 观测集，不使用点的提议顺序、提议器身份或轨迹阶段。元训练时反复均匀采样一个训练任务，再从该任务中采样一批观测，通过最大化各任务 GP log marginal likelihood 之和，学习所有任务共享的 deep-kernel feature network、kernel 和 noise 参数；不是把不同任务的数据直接拼成一个普通 GP 数据集。
  - **ML HPO / DL HPO** ML HPO 与 DL HPO 分别训练 FSBO；进一步地，ML HPO 内部按照搜索空间 schema 对 tasks 分组，每组独立元训练一个 FSBO，DL HPO 内部也采用相同做法。组内将归一化后的数值化／混合超参数向量直接输入 deep-kernel GP，任何 schema 不同的 task 都不得加入该组。
  - **Peptide optimization / query optimization** 先用各场景共享的预训练 VAE 将氨基酸序列或 query plan 编码为 latent vector，再将其输入 FSBO 的 deep-kernel GP。FSBO 从源任务元训练到目标任务在线优化结束均冻结 VAE 的 encoder 和 decoder，不更新 VAE，也不改变 latent 坐标系；训练与 fine-tuning 只更新 FSBO 自身的 deep-kernel feature network、GP kernel 和 noise 参数。Acquisition 产生的 latent candidate 通过同一个冻结 decoder 还原为氨基酸序列或 query plan。
  - **新任务适配** 用元训练参数初始化目标任务 surrogate，并在共享 warm-start 产生的少量目标任务观测上 fine-tune deep-kernel network 与 GP kernel 参数。此后每获得一批新观测便执行少量 fine-tuning，再由 GP posterior 和 Expected Improvement 选择下一批候选；结构化场景中的 latent candidate 需经 decoder 还原后再调用 oracle。
  - **公平性控制** 不启用 FSBO 原论文额外的 evolutionary data-driven warm start，统一使用与其他方法相同的 warm-start 数据，从而只比较跨任务 surrogate 元学习带来的收益。
- [x] **[RGPE](baseline_papers/28_rgpe.pdf)**
  - **历史任务建模** 每个实验场景内，将**全部**与目标任务具有完全相同搜索空间 schema 的训练任务纳入 source-GP 候选集合，不根据 meta-feature、相似度或人工规则预先筛选 top-K 任务。对每个训练任务的无序 `(x,y)` 观测集分别拟合一个 source GP，并对各任务的 `y` 独立做零均值、单位方差标准化；所有 source GP 拟合一次后冻结。不同任务的数据不会合并进同一个 GP，也不使用轨迹顺序。搜索空间 schema 不兼容的任务不进入候选集合，这属于输入兼容性约束，不属于相关性筛选。
  - **动态 rank weighting** 所有兼容的 source GP 首先进入候选集合，随后只依据目标任务的在线观测动态决定本轮信任哪些模型。每轮在目标任务已经评估的配置上检查各 source GP 的预测排序，以 pairwise misranking 数作为 ranking loss；目标任务 GP 使用 leave-one-out 预测计算同一损失，避免用训练内预测获得不公平优势。对目标观测做 bootstrap，按每个 GP 成为最低 ranking-loss 模型的频率分配 ensemble weight；目标观测少于 3 个时对全部模型使用均匀权重。原论文的 base-model dropping 机制会在每一轮以一定概率从**当前轮** ensemble 中移除弱 source GP，以抑制大量弱模型造成的 weight dilution；这不是永久删除，下一轮计算新权重时仍重新考虑全部兼容 source GP。
  - **ML HPO / DL HPO** ML HPO 与 DL HPO 分别运行 RGPE；进一步地，ML HPO 内部按照搜索空间 schema 对 tasks 分组，每组独立建立一套 source-GP 集合，DL HPO 内部也采用相同做法。组内不使用 VAE，直接将归一化后的超参数向量作为 source GP 和 target GP 的输入，任何 schema 不同的 task 都不得进入该集合。
  - **Peptide optimization / query optimization** 使用各场景共享的预训练 VAE 将历史与当前的氨基酸序列或 query plan 编码到同一个 latent space，再分别训练 source GP 和 target GP。RGPE 全程冻结 VAE encoder 和 decoder，避免 latent 坐标变化导致所有已冻结 source GP 失效；候选在 latent space 中产生后，通过同一个冻结 decoder 还原。
  - **新任务在线优化** 使用与其他方法相同的 warm-start 数据拟合 target GP；每获得一批新观测便重新拟合 target GP、重算 ranking weights 和 base-model dropping 概率。默认运行原论文推荐的 `RGPE(TAF)`：用上述权重组合 target EI 与各 source GP 的 improvement，最大化 transfer acquisition function 后评估下一批候选。Source GP 始终不在目标任务上 fine-tune，随着目标证据增加，错误源任务的权重会降低并逐步退回主要依赖 target GP。

### 非 LLM・学习型跨任务 optimizer【先跳过】

- [x] **[OptFormer](baseline_papers/17_optformer.pdf)**
  - **模型与训练目标** 按原论文实现 T5 encoder–decoder 架构的数值序列模型，而不使用通用预训练语言模型。模型输入是一项任务的 metadata \(m\)，包括任务／指标名称、最大化或最小化方向、轨迹生成算法，以及每个变量的名称、类型、取值范围和尺度；输出是保持原始时间顺序的完整优化轨迹 \(h_t=(x_1,y_1,\ldots,x_t,y_t)\)。训练时对候选参数 token 和目标值 token 最大化自回归似然，同时学习两种分布：模仿历史行为策略的候选分布 \(\pi_{\text{prior}}(x_{t+1}\mid m,h_t)\)，以及给定候选后的目标预测分布 \(p(y\mid m,h_t,x)\)。因此它会利用轨迹的先后顺序，不得把历史数据打乱成无序点集。
  - **训练任务与搜索空间兼容性** ML HPO、DL HPO、peptide optimization 和 query optimization 分别训练一套 OptFormer，只使用对应场景的训练 tasks，禁止混入 validation／test task 的轨迹。一个场景内使用全部可用的有序训练轨迹，并在 metadata 中保留轨迹生成算法的身份；不同 optimizer 产生的轨迹可以共同训练，因为模型以算法名称作为条件。与 FSBO／RGPE 不同，OptFormer 不要求场景内各 task 具有完全相同的搜索空间 schema：每个 task 自己的变量定义会被序列化进 metadata。不过，原论文只实证了固定维数、无条件依赖的搜索空间；若任务含条件超参数或动态约束，必须加入合法取值 mask 和确定性的序列化／反序列化规则，并将其披露为适配实现。
  - **数值离散化** 对每个连续或整数变量依据该 task metadata 中声明的上下界归一化，并按原论文量化为 \(Q=1000\) 个 token；categorical／discrete 变量编码为各自候选集合中的索引。目标值先统一成“越大越好”的方向，再使用 benchmark 已提供的合法范围，或只由训练 tasks 标定且在测试前冻结的场景级范围，量化为 1000 档；不得利用目标 test task 的未来观测确定范围，同一任务的离线轨迹与在线观测也必须始终使用同一映射。
  - **ML HPO / DL HPO** 不使用 VAE，直接序列化原始数值／整数／类别超参数。ML HPO 与 DL HPO 各自训练模型；同一场景内搜索空间不同的 tasks 可以联合训练，但生成每个参数 token 时只保留该 task schema 下的合法取值，并在解码后再次执行边界、类型和条件约束检查。
  - **Peptide optimization / query optimization** 使用各场景共享的预训练 VAE，把每个氨基酸序列或 query plan 编码为定长 latent vector \(z\)，再把有序的 \((z_t,y_t)\) 轨迹按连续搜索空间序列化；模型生成 \(z\) 后通过同一个 decoder 还原结构并执行有效性、长度／相似度或 query-plan 约束检查。OptFormer 的训练和测试阶段均冻结 VAE：否则 latent 坐标系变化会使已经 token 化的历史轨迹与模型学到的条件分布失效；若要更新 VAE，必须重编码全部历史轨迹并从头重训 OptFormer，因此不纳入本 baseline。
  - **新任务在线优化** 使用与其他方法完全相同的 warm-start 观测作为初始上下文，测试期间冻结 OptFormer 权重，不做目标任务 fine-tuning。每轮先从 \(\pi_{\text{prior}}\) 采样原论文设定的 \(M=100\) 个合法且去重的候选，再由模型的目标预测分布计算 expected improvement，选择 EI 最大者进行 oracle 评估，即运行 `OptFormer(EI)`；新得到的 \((x,y)\) 按时间顺序追加到上下文后进入下一轮。若训练集含多个轨迹生成算法，测试时使用的 algorithm condition 只能依据 validation tasks 预先选定。原论文只定义 batch size = 1；若统一实验协议要求 batch size \(B>1\)，则从同一候选池中选择 EI 最高的 \(B\) 个不同候选，并明确标记为批量适配版。
  - **实现边界** 不采用 BOLT 论文中以 GPT-4o-mini 微调实现的“OptFormer”版本；该版本是 LLM-based trajectory-SFT baseline，不是这里要比较的原始非 LLM OptFormer。所有场景均固定训练轨迹集合、模型规模、训练步数与候选池大小，并单独报告结构化场景的 VAE 冻结设置和无效 decode 比例。

### LLM-based・无跨任务优化经验

- [x] **[ToSFiT](baseline_papers/11_tosfit.pdf)**：仅使用当前目标任务的反馈在线微调蛋白质 LLM，并执行 Thompson sampling；适配到统一的 oracle、生成器和约束。
- [x] **[BORA](baseline_papers/13_bora.pdf)**：在纯 BO、LLM 提议、以及 LLM 重排 BO 候选三种动作之间调度；只使用当前任务历史，不注入历史任务轨迹。

### LLM-based・跨任务轨迹迁移

- [x] **[BOLT](baseline_papers/01_bolt.pdf)**：用历史任务的 top-K 配置监督微调 LLM，由 LLM 生成 warm-start 点，随后交给 LOL-BO。
- [x] **[SemanticOpt](baseline_papers/02_semanticopt.pdf)**
  - **核心迁移方式** 四个实验场景分别训练一个 SemanticOpt LoRA adapter，训练数据就来自我们已经收集好的对应场景 train-task 优化轨迹；不重新运行优化器收集轨迹，也不产生任何新的 source-oracle 调用，validation／test task 的轨迹和目标值均不得进入训练。每条样本保留完整时间顺序，不只保留 top-K 配置，也不把轨迹转成无序 \((x,y)\) 点集。一个场景内的 tasks 可以具有不同搜索空间 schema，因为每个 task 的变量名称、类型、范围、尺度和条件规则都会写入 prompt，不要求像 FSBO／RGPE 那样按完全相同的 schema 分组。
  - **重新构建 surrogate 统计量** 已收集的轨迹通常只有 \((x_t,y_t)\)，因此需要离线回放每条 train-task 轨迹，为 SemanticOpt 补出原论文要求的预测均值和标准差。对轨迹 \(\tau=(x_1,y_1,\ldots,x_T,y_T)\) 的每个决策步 \(t\)，只使用当时已经可见的前缀 \(D_t=\{(x_i,y_i)\}_{i=1}^{t}\) 从头重新拟合 LOL-BO surrogate，再在轨迹中已经记录好的下一配置 \(x_{t+1}\) 上计算 posterior mean \(\mu_{t+1}\) 和 posterior standard deviation \(\sigma_{t+1}\)，最后才把真实 \(y_{t+1}\) 写入下一条 user message。这里不重新优化 acquisition、也不重新选择 \(x_{t+1}\)：配置来自原有轨迹，重建 surrogate 只负责补标 \(\mu\) 和 \(\sigma\)。若任务含真实 evaluation failure，则同样只用历史前缀重建 feasibility model 并补标 \(p_{\mathrm{fail}}\)；尚未观察到失败时省略该字段。
  - **每个训练样本** system message 先将模型定义为黑盒优化器并强制合法 JSON，随后把一条完整 train-task 轨迹写成多轮对话：
    - **user** 任务描述、语义背景、搜索空间、目标方向、预算和截至当前的已有观测；
    - **assistant** 轨迹中记录的下一配置 \(x_{t+1}\)，以及重建 LOL-BO surrogate 在该点给出的预测均值 \(\mu_{t+1}\)、标准差 \(\sigma_{t+1}\)；历史中出现失败后再输出 \(p_{\mathrm{fail},t+1}\)；
    - **user** 该配置已经记录好的真实 oracle 结果 \(y_{t+1}\) 或 failure；
    - 按上述 assistant–user 交互不断重复，直到这条轨迹结束。
    
    所有数值按原论文保留 4 位有效数字，训练 loss 只计算 assistant responses。共享 warm-start 作为对话的初始已有观测，不把它伪装成 LLM 生成的动作；任何 \(\mu,\sigma,p_{\mathrm{fail}}\) 都必须由相应历史前缀因果计算，不能利用 \(y_{t+1}\) 或更晚的结果。
  - **模型与训练设置** 原生复现使用 Qwen 3.5 27B，通过 Unsloth 做 assistant-only LoRA SFT：LoRA rank／alpha 均为 16，学习率 \(10^{-5}\) 并采用 cosine schedule，BF16、batch size 64、1 epoch、最大上下文 24,576 tokens。若因算力改用其他 base LLM，必须保持 SemanticOpt 的数据格式、监督字段和推理规则，并在结果中明确标为 backbone-controlled protocol reproduction；不能把公开的旧版 Llama-3.1-8B GPTOpt checkpoint 当成最新版 SemanticOpt。
  - **ML HPO / DL HPO** 不使用 VAE，assistant 直接以 JSON 输出原始连续、整数、布尔和类别超参数。ML HPO 与 DL HPO 分别训练 adapter；prompt 可以同时容纳场景内不同 schema 的 tasks，但每次生成后都必须按当前 task 的 schema 执行类型、范围、条件参数和重复配置检查。可提供数据集、模型、指标和训练预算等任务语义，但不得提供由目标 test task 未来评估推导出的信息。
  - **Peptide optimization / query optimization** 使用各场景共同初始化的 VAE，把已记录的氨基酸序列或 query plan 重新编码为定长 latent vector \(z\)，并在这些固定 latent 坐标上逐前缀重建 LOL-BO surrogate。LLM 的动作字段输出 \(z\)，decoder 再将其还原为氨基酸序列或 query plan；user observation 同时回传 \(z\)、解码后的结构、oracle 结果和合法性状态，使模型仍能利用结构语义。为保证补标的 surrogate 统计量、LoRA 学到的 latent token 和测试时 decoder 始终处于同一坐标系，SemanticOpt 在轨迹回放、SFT 和目标任务推理期间均冻结 VAE。无效 decode 按所有方法统一的约束处理规则拒绝或记为 failure，不得为 SemanticOpt 单独放宽。
  - **新任务在线优化** 为遵守统一 warm-start 协议，不采用原论文由 LLM 自行生成 5 个初始点的做法，而是把所有方法共享的 warm-start 配置及其结果直接作为初始对话历史。此后冻结 base model 和 LoRA 权重，每一步从 SemanticOpt 独立采样 2 个合法、非重复候选；模型为每个候选自报高斯预测 \((\mu,\sigma)\)，若历史已出现失败则再输出 \(p_{\mathrm{fail}}\)，否则令成功概率为 1。外部程序只计算方向正确的 \((1-p_{\mathrm{fail}})\times\mathrm{EI}\)，评估分数较高的候选，并把结果追加到对话。测试时不再拟合或调用 HEBO、LOL-BO、GP、RF 或 SVIDKL；LLM 同时充当提议策略与隐式 surrogate。
