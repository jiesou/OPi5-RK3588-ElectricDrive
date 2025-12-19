---
mode: agent
---

# 总体要求
- 全部有 type
- 关于状态，使用 stateBus 单例和信号槽机制。dash.py 可以将 update_state 的信号作为参数传递，如
```py
ScrewTable(state_update_on=state_bus.updated)
```
- 不要 Overengineering！不要 Overengineering！不要 Overengineering！保持代码实现简短简单。如果可能，减少代码的更改。
- 重写代码时不需要考虑接口兼容性，尽量不用 TypedDict，除非要求。
- 中文注释。不用英文

