# LF → swift 数据集转换

## 数据标准

ms-swift 对多模态数据的硬性要求：

- 图片路径必须是列表形式（`"images": ["a.png", "b.png"]`），不支持逗号拼接字符串；
- 文本中只识别裸 `<image>` 占位符，按出现顺序与 `images` 一一对应；不识别 `<image:N>` 编号形式。

不满足标准时 swift 不报错：逗号字符串读不出图片；`<image:N>` 无法识别时模板会把所有图片 token 堆到序列最前面，症状是训练首个 step 梯度范数异常偏高（纯文本数据无此症状）。因此 LF/xg 格式（`text` + 逗号拼接 `image` + `<image:N>`）的数据必须先离线转换，再进入训练。

## 用法

```bash
# 转换：在与输入相同的目录生成 swift_<原文件名>
python scripts/dataset/convert_lf_to_swift.py data/train.jsonl [更多文件...]

# 只检查不写文件；存在需要转换或无法修复的行时退出码为 1
python scripts/dataset/convert_lf_to_swift.py --check-only data/train.jsonl
```

转换规则：

- `<image:N>` 重写为 `<image>`；`images` 按占位符**出现顺序**重排，乱序编号与重复引用均保持正确对齐；
- 已合规的行原样通过（脚本幂等，可对输出重复执行）；
- 行的其他字段原样透传。

无法修复的行（占位符编号越界、有图但无占位符等）会被丢弃，行号打到 stderr；此时输出文件缺少这些行且退出码为 1，请勿直接使用缺行的输出。

## 运行时拦截

`src/torchturbo/adapters/msswift/data_check.py` 在 Swift PT 启动时（swift encode 之前）对数据集逐行执行同一套标准检查，首个违规行即抛出 ValueError 并指向本脚本。纯文本数据集与 streaming 数据集自动跳过；gate 与转换脚本共用同一个 `check_row` 实现，标准唯一。

## 边界

- 仅支持 LF/xg 格式，不泛化到其他数据来源；
- 脚本不校验图片文件是否存在，转换前请确认数据已完整落盘；
- 缺 `text` 字段或 JSON 损坏的行会让脚本以 traceback 中止，按报错行号修复后重跑。
