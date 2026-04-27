# TASK: [在此填写任务简述，如：重构 MoE 算子替换逻辑]

## 1. 环境上下文约束 (Context Limits)
- IGNORE: 忽略所有测试文件夹 `tests/` 和文档 `docs/`。
- FOCUS: 只关注 `src/operators/` 和 `kernel/` 目录。
- DONT_READ: 不要读取超过 500 行的非核心配置文件。

## 2. 核心执行步骤 (Execution Steps)
1. **分析现状**: 运行 `grep -r "fp8_index" .` 确认所有引用位置。
2. **结构检查**: 读取 `src/operators/wrapper.py`，确认 `_wrapper_logic` 的参数签名。
3. **原子修改**:
    - 修改 `register_all_custom_ops`，将 `setattr` 目标从 `partial` 改为 `torch.ops` 对象。
    - 确保 `_wrapper_logic` 内部通过判断 `_op_name` 返回正确的动态 Shape。
4. **代码校验**: 运行 `python3 -m py_compile src/operators/wrapper.py` 检查语法。
5. **验证逻辑**: 创建一个临时的 `debug_op.py`，模拟调用 `kernel.fp8_index` 并打印其 `type`。

## 3. 输出规范 (Output Specs)
- 只修改受影响的代码块，严禁重写整个文件。
- 完成后运行 `/stats` 并报告当前会话消耗的 Token。
- 如果遇到 Python 类型冲突，立即停止并询问，不要自行推测。