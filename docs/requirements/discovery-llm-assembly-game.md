# Discovery: LLM Assembly Game

**Date**: 2026-03-23
**Status**: Validated
**Feasibility**: CAUTION

## Summary

一款面向零基础孩子的桌面组装游戏，使用飞机俯视线稿作为主题。
用户通过拖拽组件拼装不同时代的飞机，每个组件背后对应真实的 LLM
构件（Embedding、Attention、FFN 等），不同算法有不同能力值。飞机
的时代进化（双翼机 → 螺旋桨 → 喷气式 → 现代战斗机 → 隐身机）
对应 LLM 架构的渐进复杂度（Bigram → Attention → Multi-Head →
Mini-GPT → 高级组件）。组装完成后导出 JSON 配置文件，由现有 myllm
Python 代码训练模型。用户可以和训练出的模型对话，并通过任务挑战
解锁新时代的飞机和组件。

## Approach

Godot 4.4 + GDScript 做游戏前端，拖拽组装 UI。游戏导出 JSON 配置
文件，Python + PyTorch 后端（现有 myllm 代码库）读取配置执行训练。
Godot 通过 localhost HTTP（FastAPI）与 Python 后端通信，获取训练
进度和对话结果。

选择理由：
- Godot 免费开源，轻量（~30MB），契合教育项目精神
- GDScript 类 Python 语法，降低维护门槛
- 现有 myllm 的工厂模式（build_op、build_blocks、MODEL_CONFIGS）
  天然适配游戏配置编辑
- localhost HTTP 解耦游戏和训练，支持进度流式推送

## Theme Design: Aircraft Evolution

| Era | Aircraft | LLM Mapping | Unlock |
|-----|----------|-------------|--------|
| WWI | Biplane (双翼机) | Bigram — 纯 Embedding 查表 | Starting |
| WWII | Monoplane (单翼螺旋桨) | + Self-Attention | Challenge 1 |
| 1950s | Early Jet (早期喷气式) | + Multi-Head Attention + FFN | Challenge 2 |
| Modern | Fighter Jet (现代战斗机) | Mini-GPT — 多层 Transformer 堆叠 | Challenge 3 |
| Stealth | Stealth Fighter (隐身战斗机) | + Dropout, BPE 等高级组件 | Challenge 4 |

视觉风格：俯视线稿（低美术成本，干净易懂）。

## Feasibility Research

### Verdict: CAUTION

### Technical Feasibility
核心架构完全可行。现有 myllm 的工厂模式天然就是游戏配置编辑器的
后端。Godot 4.x 内置拖拽 API 成熟，JSON 双向互通零依赖。无基础
性技术障碍。`train.py` 的 `MODEL_CONFIGS` 结构可直接映射为 JSON
配置 schema。

### Stack & Ecosystem
- **Game engine**: Godot 4.4 stable
- **Language**: GDScript 2.0
- **IPC**: Localhost HTTP (FastAPI on Python, HTTPRequest on Godot)
- **Config format**: JSON (双端原生支持)
- **Python packaging**: PyInstaller --onedir (分发时捆绑)
- **Training progress**: Python HTTP endpoint, Godot Timer 轮询
- **Ecosystem health**: Godot 4.x 文档成熟，中文社区在 Bilibili
  和 godot-china.org 活跃

### Risks & Pitfalls

| Risk | Impact | Mitigation |
|------|--------|------------|
| CPU 训练等待时间 | HIGH | 强制 tiny 预设: 2层, 32维, 1000步, <60s |
| 隐喻混淆 | MEDIUM | 飞机部件名旁显示真实技术名称 (P4) |
| Python+PyTorch 打包 | MEDIUM | PyInstaller 捆绑, ~800MB; v1 可要求本地 Python |
| Windows Defender 误报 | LOW | 代码签名证书 |
| Godot↔Python IPC 缓冲 | LOW | Python flush=True, 备选文件轮询 |

### Unknowns
- CPU 训练时间基准测试 (2层 32维 1000步的实际耗时, 1h 可验证)
- Godot↔Python stdout 管道在 Windows 上的可靠性实测

## Principles

### Discovered (new — not yet in constitution)

1. **Single Skin First** — v1 只做飞机主题，验证核心循环后再加皮肤。
   Rationale: 多皮肤是美术/隐喻/测试的倍增器。

2. **Sub-60s Training** — 所有默认配置在 CPU 上训练 MUST <60 秒。
   Rationale: 孩子 15 秒失去耐心，60 秒是游戏体验极限。

3. **Era = Complexity Tier** — 飞机时代对应 LLM 复杂度层级，不可跳级。
   Rationale: 用历史时间线自然驱动难度曲线。

### Existing (from constitution v1.0.0)

- P1 Education First — 教学价值优先
- P2 Progressive Disclosure — 渐进式引入复杂度
- P3 Composable Pluggable Architecture — 可插拔可组合
- P4 Honest Metaphors — 隐喻技术准确
- P5 Runnable Output — 组装产生可运行代码
- P6 Chinese-First — 中文通俗易懂
- P7 YAGNI — 不做不需要的

## Open Questions

1. 飞机组件的具体命名映射（引擎=？机翼=？雷达=？→ 对应哪个 LLM 组件）
2. 任务挑战系统的具体设计（什么样的任务才能既好玩又能验证学习效果）
3. 是否需要一个"教学面板"在组装时显示组件的技术解释，还是纯靠游戏体验
4. PyTorch CPU-only 精简打包方案（能否把 800MB 降到更小）
5. 生成对话的质量预期管理（字符级小模型生成质量有限，如何让孩子不失望）

## Next Steps

- [ ] `/projkit.main.specify` — 基于此 discovery 创建功能规格说明书
- [ ] `/projkit.main.benchmarking` — 调研类似教育游戏产品
- [ ] `/projkit.func.mapcodebase` — 映射现有代码库，确认配置 schema
