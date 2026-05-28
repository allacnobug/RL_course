# 强化学习大作业实验方案

本项目目标是将 DQN 系列强化学习方法应用到众包任务推荐中。系统在 worker 进入平台时，只推荐一个 project。实验需要分别考虑两个优化目标：

1. 最大化参与者利益：让 worker 更容易找到感兴趣、匹配、可能获得收益的任务。
2. 最大化请求者利益：让 requester 的 project 获得更多、更高质量的提交。

本 README 给出完整实验流程、实验清单、代码结构建议、必做/选做实验，以及最终报告和汇报需要展示的结果。

组内协作请看：

```text
docs/四人分工与实验流程.md
docs/预处理代码使用说明.md
```

## 1. 实验总览

### 1.1 强化学习建模

将历史众包数据整理成按时间排序的推荐事件：

```text
某个时间 t，某个 worker 进入平台。
系统从当前 active projects 中选择一个 project 推荐给该 worker。
如果推荐结果与历史行为、worker 兴趣、提交质量、请求者收益一致，则获得较高 reward。
```

强化学习五元组：

```text
state      = worker 特征 + project 特征 + worker-project 匹配特征 + 时间特征
action     = 从 K 个候选 project 中选择一个
reward     = worker_reward 或 requester_reward
next_state = 下一个 worker 到达事件对应的推荐状态
done       = 当前时间序列 episode 结束
```

建议主实验使用固定候选集大小：

```text
K = 20
候选集 = 1 个真实提交 project + 19 个同一时间可用的负采样 active projects
```

### 1.2 主实验路线

推荐先完成以下主线：

```text
原始数据读取
-> 数据清洗
-> 构造 project / entry / worker 三张表
-> 按 entry_created_at 构造 worker 到达事件
-> 按时间划分 train / valid / test
-> 构造强化学习环境
-> 实现 DQN / Double DQN / Dueling DQN
-> 训练 worker-benefit reward 模型
-> 训练 requester-benefit reward 模型
-> baseline 对比
-> 画训练曲线和测试图
-> 写实验报告和汇报 PPT
```

## 2. 实验清单

### 2.1 数据处理实验

| 编号 | 实验/任务 | 目的 | 必做/选做 |
|---|---|---|---|
| D1 | 读取 `project_list.csv`、`worker_quality.csv`、`project/*.txt`、`entry/*.txt` | 建立基础数据表 | 必做 |
| D2 | 清洗 project 字段 | 得到 project 特征 | 必做 |
| D3 | 清洗 entry 字段 | 得到 worker 历史提交行为 | 必做 |
| D4 | 清洗 worker quality | 得到 worker 质量特征 | 必做 |
| D5 | 处理缺失值、异常值、撤回提交 | 保证训练稳定 | 必做 |
| D6 | 按时间构造 worker 到达事件 | 模拟动态众包系统 | 必做 |
| D7 | 按时间切分 train / valid / test | 避免未来信息泄漏 | 必做 |
| D8 | 统计数据分布并画图 | 报告中展示数据特征 | 必做 |
| D9 | 尝试随机切分并与时间切分比较 | 说明时间切分更严谨 | 选做 |

### 2.2 强化学习建模实验

| 编号 | 实验/任务 | 目的 | 必做/选做 |
|---|---|---|---|
| R1 | 设计 pairwise state 特征 | 表示 worker 与 project 的匹配关系 | 必做 |
| R2 | 固定动作空间 K=20 | 降低 DQN 训练难度 | 必做 |
| R3 | 设计 worker_reward | 回答“最大化参与者利益” | 必做 |
| R4 | 设计 requester_reward | 回答“最大化请求者利益” | 必做 |
| R5 | 设计综合 reward | 分析两方利益权衡 | 选做 |
| R6 | 尝试 K=10 / 20 / 50 | 分析候选集规模影响 | 选做 |

### 2.3 模型实验

| 编号 | 实验/任务 | 目的 | 必做/选做 |
|---|---|---|---|
| M1 | Random baseline | 最基础对比 | 必做 |
| M2 | Popular baseline | 与热门项目推荐比较 | 必做 |
| M3 | Deadline baseline | 与快到期任务优先规则比较 | 必做 |
| M4 | Award baseline | 与高奖励任务优先规则比较 | 必做 |
| M5 | Category-match baseline | 与基于 worker 兴趣的规则比较 | 必做 |
| M6 | Vanilla DQN | 基础 DQN 模型 | 必做 |
| M7 | Double DQN | 减少 Q 值过估计 | 必做 |
| M8 | Dueling DQN | 增强状态价值建模能力 | 必做 |
| M9 | Prioritized Replay DQN | 提升经验回放效率 | 选做 |

### 2.4 训练曲线与评估实验

| 编号 | 实验/任务 | 目的 | 必做/选做 |
|---|---|---|---|
| E1 | 训练 reward 曲线 | 展示模型是否学到策略 | 必做 |
| E2 | moving average reward 曲线 | 平滑观察收敛趋势 | 必做 |
| E3 | loss 曲线 | 展示训练稳定性 | 必做 |
| E4 | epsilon 衰减曲线 | 展示探索到利用的过程 | 必做 |
| E5 | average Q value 曲线 | 检查 Q 值是否爆炸 | 必做 |
| E6 | worker_reward / requester_reward 分开曲线 | 分析两类目标 | 必做 |
| E7 | baseline 测试结果柱状图 | 对比模型优劣 | 必做 |
| E8 | reward 设计消融 | 证明 reward 设计合理 | 必做 |
| E9 | 特征消融 | 证明关键特征有效 | 必做 |
| E10 | 不同 K 候选规模对比 | 分析动作空间影响 | 选做 |
| E11 | 不同随机种子均值和标准差 | 增强结果可信度 | 选做 |

## 3. 推荐代码结构

建议建立如下目录：

```text
RL/
  data/
    project/
    entry/
    project_list.csv
    worker_quality.csv
  src/
    preprocess.py
    build_events.py
    features.py
    env.py
    replay_buffer.py
    models.py
    train.py
    evaluate.py
    baselines.py
    plot_curves.py
    utils.py
  outputs/
    processed/
    checkpoints/
    logs/
    figures/
    tables/
  report/
    report.md
    figures/
  README.md
```

每个文件职责：

```text
preprocess.py    原始 json/csv -> project.csv, entry.csv, worker.csv
build_events.py  按时间构造 worker arrival events
features.py      构造 worker-project pair 特征
env.py           众包推荐强化学习环境
replay_buffer.py DQN 经验回放池
models.py        DQN / Double DQN / Dueling DQN 网络
train.py         训练模型并保存日志
evaluate.py      在 valid/test 上评估模型
baselines.py     规则 baseline
plot_curves.py   画训练曲线、对比图、消融图
utils.py         随机种子、归一化、日志保存等工具函数
```

## 4. 数据清洗方案

### 4.1 project 表

从 `project/project_xxx.txt` 中读取每个 project 的 JSON 字段，建议保留：

```text
project_id
category
sub_category
industry
status
start_date
deadline
entry_count
creative_count
average_score
total_awards
total_package_tips
featured
assured
private_gallery
client_feedback
watchers_count
package_name
```

派生字段：

```text
duration_hours       = deadline - start_date
award_log            = log1p(total_awards)
tips_log             = log1p(total_package_tips)
industry_id          = industry 编码
time_left_ratio(t)   = (deadline - t) / duration
project_age_ratio(t) = (t - start_date) / duration
is_active(t)         = start_date <= t <= deadline
```

缺失值处理：

```text
数值字段缺失：填 0 或训练集均值
类别字段缺失：填 unknown
时间字段缺失：丢弃该 project
deadline <= start_date：丢弃或修正后单独记录
```

### 4.2 entry 表

从 `entry/entry_projectid_offset.txt` 读取每个提交记录，建议保留：

```text
project_id
worker_id / author
entry_id
entry_number
entry_created_at
winner
finalist
withdrawn
eliminated
award_value
offer_value
tip_value
revision_count
score
is_entirely_original
```

score 处理建议：

```text
如果 revisions 非空：score = max(revision.score)
如果 revisions 为空：score = 0
normalized_score = score / 5.0
```

撤回提交处理建议：

```text
主实验：保留 withdrawn 字段，把 withdrawn 作为低质量信号。
消融实验：去掉 withdrawn=true 的 entry，观察结果变化。
```

### 4.3 worker 表

从 `worker_quality.csv` 读取：

```text
worker_id
worker_quality
```

其中 `worker_quality < 0` 的 worker 可以视作未知质量：

```text
方案 A：填训练集 worker_quality 均值
方案 B：填 0，并增加 unknown_quality 标记
```

建议主实验使用方案 B，因为未知本身也是信息。

基于历史 entry 统计 worker 动态特征：

```text
historical_submit_count
historical_avg_score
historical_win_rate
historical_finalist_rate
preferred_category
preferred_industry
category_distribution
industry_distribution
```

注意：这些统计必须只用当前时间 t 之前的数据，不能用未来数据。

## 5. 数据划分方案

必须按时间划分，避免信息泄漏。

步骤：

```text
1. 将所有 entry 按 entry_created_at 升序排列。
2. 前 70% 作为 train。
3. 中间 15% 作为 valid。
4. 最后 15% 作为 test。
```

代码入口建议：

```bash
python src/preprocess.py \
  --data_dir data \
  --output_dir outputs/processed

python src/build_events.py \
  --processed_dir outputs/processed \
  --train_ratio 0.70 \
  --valid_ratio 0.15 \
  --output_dir outputs/processed
```

报告中需要说明：

```text
随机划分会让模型看到未来 worker 偏好和未来 project 热度，因此不作为主实验。
本实验采用时间切分，更符合动态推荐场景。
```

## 6. 强化学习环境设计

### 6.1 state

每个 step 中有 K 个候选 project。对每个 candidate 构造一条 pairwise feature：

```text
worker_quality
unknown_quality
worker_historical_submit_count
worker_historical_avg_score
worker_historical_win_rate
worker_historical_finalist_rate

project_category_id
project_sub_category_id
project_industry_id
project_award_log
project_expected_answer_num
project_current_entries_so_far
project_average_score_so_far
project_time_left_ratio
project_age_ratio
project_duration_hours
project_featured
project_assured

category_match
industry_match
worker_project_history_count
worker_industry_preference_score
worker_category_preference_score
```

因此输入形状为：

```text
state.shape = [K, feature_dim]
```

### 6.2 action

动作是候选任务索引：

```text
action in {0, 1, ..., K-1}
```

每个 action 对应候选集中的一个 project。

### 6.3 transition

环境每次推荐后，进入下一个按时间排序的 worker 到达事件：

```text
s_t     = 第 t 个 worker 到达事件的候选集特征
a_t     = 推荐的 project
r_t     = 根据推荐结果计算 reward
s_{t+1} = 第 t+1 个 worker 到达事件
done    = 当前 episode 结束
```

episode 可以按固定长度切分：

```text
每 1000 个 worker 到达事件作为一个 episode
```

也可以整个 train 序列作为一个长 episode。建议主实验使用固定长度 episode，训练曲线更稳定。

## 7. 奖励函数设计

### 7.1 参与者利益 reward

目标：让 worker 获得更匹配、更可能有收益的任务。

建议定义：

```text
r_worker =
  1.0 * hit
+ 0.5 * category_match
+ 0.5 * industry_match
+ 0.5 * normalized_score
+ 1.0 * finalist
+ 2.0 * winner
+ 0.5 * log1p(award_value)
- 0.2 * withdrawn
```

其中：

```text
hit = 推荐 project 是否等于该 worker 历史真实提交 project
normalized_score = score / 5.0
```

如果没有命中：

```text
r_worker = -0.1 + 0.2 * category_match + 0.2 * industry_match
```

### 7.2 请求者利益 reward

目标：让 project 获得更多高质量提交。

建议定义：

```text
r_requester =
  1.0 * hit
+ 1.0 * worker_quality
+ 0.8 * normalized_score
+ 1.0 * finalist
+ 2.0 * winner
+ 0.3 * is_entirely_original
- 0.5 * withdrawn
```

如果推荐给高质量 worker 且 project 尚未获得足够提交，可以增加：

```text
+ 0.3 * project_need_score
```

其中：

```text
project_need_score = 1 - current_entries_so_far / expected_answer_num
```

### 7.3 综合 reward

选做，用于分析参与者和请求者利益冲突：

```text
r = alpha * r_worker + (1 - alpha) * r_requester
```

建议尝试：

```text
alpha = 0.0, 0.25, 0.5, 0.75, 1.0
```

图表展示：

```text
x 轴：平均 worker_reward
y 轴：平均 requester_reward
不同点：不同 alpha
```

## 8. 模型设计

### 8.1 DQN

输入：

```text
[K, feature_dim]
```

对每个候选 project 共享同一个 MLP，输出每个 action 的 Q 值：

```text
feature_dim -> 128 -> ReLU -> 128 -> ReLU -> 64 -> ReLU -> 1
```

最后得到：

```text
Q(s, a_1), Q(s, a_2), ..., Q(s, a_K)
```

### 8.2 Double DQN

更新目标：

```text
a* = argmax_a Q_online(s_next, a)
target = r + gamma * Q_target(s_next, a*)
```

作用：减少普通 DQN 的 Q 值过估计。

### 8.3 Dueling DQN

将 Q 值分解为：

```text
Q(s, a) = V(s) + A(s, a) - mean(A(s, a))
```

作用：当多个候选 project 差异不明显时，更好估计当前状态价值。

## 9. 训练方案

建议超参数：

```text
learning_rate = 1e-3 或 5e-4
gamma = 0.95
batch_size = 128
replay_buffer_size = 50000
min_buffer_size = 1000
epsilon_start = 1.0
epsilon_end = 0.05
epsilon_decay_steps = 30000
target_update_interval = 500
episode_length = 1000
num_episodes = 50 到 200
optimizer = Adam
```

训练命令建议：

```bash
python src/train.py \
  --model dqn \
  --reward_type worker \
  --candidate_size 20 \
  --episodes 100 \
  --output_dir outputs

python src/train.py \
  --model double_dqn \
  --reward_type worker \
  --candidate_size 20 \
  --episodes 100 \
  --output_dir outputs

python src/train.py \
  --model dueling_dqn \
  --reward_type worker \
  --candidate_size 20 \
  --episodes 100 \
  --output_dir outputs

python src/train.py \
  --model dqn \
  --reward_type requester \
  --candidate_size 20 \
  --episodes 100 \
  --output_dir outputs

python src/train.py \
  --model double_dqn \
  --reward_type requester \
  --candidate_size 20 \
  --episodes 100 \
  --output_dir outputs

python src/train.py \
  --model dueling_dqn \
  --reward_type requester \
  --candidate_size 20 \
  --episodes 100 \
  --output_dir outputs
```

每次训练保存：

```text
outputs/checkpoints/{model}_{reward_type}.pt
outputs/logs/{model}_{reward_type}.csv
```

日志字段建议：

```text
episode
step
episode_reward
moving_avg_reward
loss
epsilon
avg_q_value
worker_reward
requester_reward
hit_rate
avg_score
avg_worker_quality
```

## 10. Baseline 方案

### 10.1 Random baseline

从 K 个候选 project 中随机推荐。

必做。用于证明模型至少优于随机策略。

### 10.2 Popular baseline

推荐当前已有提交数最多或历史最热门的 project。

```text
score(project) = current_entries_so_far
```

必做。模拟平台偏向热门项目。

### 10.3 Deadline baseline

推荐最快到期的 project。

```text
score(project) = -time_left_hours
```

必做。模拟优先处理紧急项目。

### 10.4 Award baseline

推荐奖励最高的 project。

```text
score(project) = total_awards
```

必做。模拟 worker 追求收益。

### 10.5 Category-match baseline

推荐与 worker 历史偏好 category / industry 最匹配的 project。

```text
score(project) =
  worker_category_preference_score
+ worker_industry_preference_score
```

必做。模拟传统内容推荐。

## 11. 评价指标

### 11.1 参与者视角

```text
Hit@1
Category Match Rate
Industry Match Rate
Average Worker Reward
Average Award
Winner Rate
Finalist Rate
Average Recommended Project Score
```

### 11.2 请求者视角

```text
Average Requester Reward
Average Worker Quality
Average Entry Score
High-quality Submission Rate
Winner Rate
Finalist Rate
Project Coverage
Low-quality Recommendation Rate
Withdrawn Rate
```

其中：

```text
High-quality Submission = score >= 3 或 finalist/winner 为 true
Project Coverage = 被推荐过的 project 数量 / 可推荐 project 总数
```

### 11.3 系统视角

```text
Total Reward
Recommendation Diversity
Popular Project Concentration
Long-tail Project Exposure
```

## 12. 需要画的图

### 12.1 训练曲线

| 图 | 内容 | 必做/选做 |
|---|---|---|
| Figure 1 | episode reward 曲线 | 必做 |
| Figure 2 | moving average reward 曲线 | 必做 |
| Figure 3 | loss 曲线 | 必做 |
| Figure 4 | epsilon 衰减曲线 | 必做 |
| Figure 5 | average Q value 曲线 | 必做 |
| Figure 6 | worker_reward 与 requester_reward 双曲线 | 必做 |

### 12.2 测试集对比图

| 图 | 内容 | 必做/选做 |
|---|---|---|
| Figure 7 | 不同模型 Hit@1 对比 | 必做 |
| Figure 8 | 不同模型 Average Reward 对比 | 必做 |
| Figure 9 | 不同模型 Average Worker Quality 对比 | 必做 |
| Figure 10 | 不同模型 Average Entry Score 对比 | 必做 |
| Figure 11 | Baseline 与 DQN 系列综合对比 | 必做 |
| Figure 12 | 不同 reward 设计对比 | 必做 |
| Figure 13 | 不同 alpha 的利益权衡曲线 | 选做 |
| Figure 14 | 不同 K 的性能曲线 | 选做 |

## 13. 对比实验设计

### 13.1 模型对比

必做。

```text
Random
Popular
Deadline
Award
Category-match
DQN
Double DQN
Dueling DQN
```

分别在两种目标上评估：

```text
worker_reward
requester_reward
```

### 13.2 Reward 消融

必做。

参与者 reward：

```text
W1: hit
W2: hit + category_match + industry_match
W3: hit + match + score + finalist + winner + award
```

请求者 reward：

```text
R1: hit
R2: hit + worker_quality
R3: hit + worker_quality + score + finalist + winner - withdrawn
```

汇报重点：

```text
复杂 reward 是否带来更高质量推荐？
是否牺牲了 Hit@1？
```

### 13.3 特征消融

必做。

```text
Full features
- no worker_quality
- no time features
- no category/industry match features
- no historical worker behavior features
```

汇报重点：

```text
worker_quality 对 requester 目标是否重要？
时间特征是否能帮助处理动态性？
匹配特征是否能提升参与者利益？
```

### 13.4 动作空间规模实验

选做。

```text
K = 10
K = 20
K = 50
```

汇报重点：

```text
候选集越大任务越难，但更接近真实推荐场景。
```

### 13.5 多随机种子实验

选做，但如果时间允许很推荐。

```text
seed = 0, 1, 2
```

最终结果展示：

```text
mean ± std
```

## 14. 最终报告结构

建议实验报告按以下结构写：

```text
1. 引言
   - 众包任务推荐背景
   - 为什么需要强化学习
   - 本实验解决两个目标：参与者利益、请求者利益

2. 数据集与预处理
   - 数据来源和字段说明
   - project / entry / worker 表
   - 缺失值处理
   - 时间切分
   - 数据统计图

3. 强化学习问题建模
   - state
   - action
   - reward
   - transition
   - Q 函数

4. 方法
   - DQN
   - Double DQN
   - Dueling DQN
   - baseline

5. 实验设置
   - 超参数
   - 候选集采样
   - 评价指标
   - 训练细节

6. 实验结果
   - 训练曲线
   - baseline 对比
   - worker 目标结果
   - requester 目标结果

7. 消融实验
   - reward 消融
   - 特征消融
   - 可选：K 值实验

8. 分析与讨论
   - 两方利益是否冲突
   - DQN 系列模型差异
   - 哪些特征最重要
   - 动态时间建模是否有效

9. 结论
   - 最优模型
   - 主要发现
   - 局限与未来改进
```

## 15. 汇报 PPT 建议

汇报可以按 10 到 12 页组织：

```text
1. 任务背景与问题定义
2. 数据集说明
3. 数据预处理与时间序列构造
4. RL 建模：state/action/reward/transition
5. DQN 系列模型
6. Baseline 与评价指标
7. 训练曲线
8. 参与者利益实验结果
9. 请求者利益实验结果
10. 消融实验
11. 结果分析：两方利益权衡
12. 结论与未来工作
```

重点不要只讲模型公式，要讲清楚：

```text
我们如何把历史众包数据转成一个动态推荐环境。
我们为什么这样设计 reward。
DQN 相比规则 baseline 到底提升在哪里。
```

## 16. 最小可交付版本

如果时间紧，至少完成：

```text
1. 数据清洗出 project.csv / entry.csv / worker.csv
2. 构造按时间排序的 event 数据
3. 时间切分 train / valid / test
4. K=20 候选集
5. worker_reward 和 requester_reward 两套 reward
6. Random / Popular / Award / Category-match baseline
7. DQN / Double DQN / Dueling DQN
8. reward、loss、epsilon、Q value 训练曲线
9. 测试集指标表格
10. reward 消融和特征消融
```

这套最小版本已经能完整回答作业要求，并且足够支撑分组汇报。

## 17. 推荐实现顺序

建议按这个顺序推进：

```text
Day 1:
  完成 preprocess.py
  输出 project.csv / entry.csv / worker.csv
  完成基础数据统计图

Day 2:
  完成 build_events.py
  完成 features.py
  完成 env.py
  跑通 Random baseline

Day 3:
  完成 DQN / Double DQN / Dueling DQN
  跑通 worker_reward 训练
  保存训练日志

Day 4:
  跑通 requester_reward 训练
  完成 baseline 对比
  完成训练曲线和测试图

Day 5:
  完成 reward 消融和特征消融
  整理报告
  制作 PPT
```

当前已完成第一步预处理代码：

```text
src/preprocess.py
src/build_events.py
src/features.py
src/baselines.py
src/env.py
src/models.py
src/train.py
src/evaluate.py
src/plot_curves.py
docs/预处理代码使用说明.md
environment.yml
```

Conda 环境创建：

```bash
conda env create -f environment.yml
conda activate rl-course
```

运行方式：

```bash
python3 src/preprocess.py \
  --data_dir data \
  --output_dir outputs/processed

python3 src/build_events.py \
  --processed_dir outputs/processed \
  --output_dir outputs/processed \
  --candidate_size 20 \
  --train_ratio 0.70 \
  --valid_ratio 0.15 \
  --seed 42

python3 src/features.py \
  --processed_dir outputs/processed \
  --output_dir outputs/features

python3 src/baselines.py \
  --features_dir outputs/features \
  --output_dir outputs/tables \
  --split test \
  --seed 42

python3 src/env.py \
  --feature_path outputs/features/train_features.npz \
  --reward_type worker \
  --episode_length 5 \
  --steps 8 \
  --seed 42

python3 src/models.py \
  --model dqn \
  --feature_path outputs/features/train_features.npz \
  --candidate_size 20 \
  --batch_size 4

python3 src/train.py \
  --features_dir outputs/features \
  --output_dir outputs \
  --model dqn \
  --reward_type worker \
  --episodes 100 \
  --episode_length 1000 \
  --batch_size 128 \
  --buffer_size 50000 \
  --min_buffer_size 1000 \
  --target_update_interval 500 \
  --eval_interval 5 \
  --eval_max_events 20000 \
  --seed 42

python3 src/evaluate.py \
  --features_dir outputs/features \
  --checkpoint_dir outputs/checkpoints \
  --output_dir outputs/tables \
  --split test \
  --device cpu

python3 src/plot_curves.py \
  --logs_dir outputs/logs \
  --baseline_csv outputs/tables/baseline_results_test.csv \
  --eval_csv outputs/tables/eval_results_test.csv \
  --output_dir outputs/figures
```

成功运行后会生成：

```text
outputs/processed/project.csv
outputs/processed/entry.csv
outputs/processed/worker.csv
outputs/processed/industry_map.json
outputs/processed/stats.json
outputs/processed/events_train.csv
outputs/processed/events_valid.csv
outputs/processed/events_test.csv
outputs/processed/events_stats.json
outputs/features/train_features.npz
outputs/features/valid_features.npz
outputs/features/test_features.npz
outputs/features/feature_names.json
outputs/features/features_stats.json
outputs/tables/baseline_results_test.csv
outputs/tables/baseline_results_test.json
outputs/logs/{model}_{reward_type}_seed{seed}.csv
outputs/checkpoints/{model}_{reward_type}_seed{seed}_best.pt
outputs/checkpoints/{model}_{reward_type}_seed{seed}_last.pt
outputs/tables/eval_results_test.csv
outputs/figures/training/
outputs/figures/comparison/
```

## 18. 风险与注意事项

1. 不要随机划分作为主结果。随机划分会泄漏未来信息。
2. worker 历史偏好必须按时间在线统计，不能用全量数据预先统计。
3. action space 不要直接设成所有 project，DQN 很难训练。先用 K=20 候选集。
4. reward 不要只用 hit，否则更像监督学习排序；要体现 worker/requester 利益。
5. 训练曲线要保存原始 csv，方便反复画图和汇报。
6. 每个实验要固定随机种子，否则对比不稳定。
7. 汇报中要明确：这是一个基于历史日志的离线强化学习仿真实验，不是真实在线平台实验。
