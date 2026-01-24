### Step Rules (Summary)
### 步驟規則（摘要）
- Approval is required ONLY in CHANGE mode (file/command execution).
- 只有在 CHANGE 模式（改檔/執行指令）才需要批准。
- TALK mode: answer immediately, no gating.
- TALK 模式：直接回覆，不設關卡。

### CHANGE mode limits
### CHANGE 模式限制
- Max 2 logical steps per approval
- 每次批准最多 2 個邏輯步驟
- Max 2 files per step
- 每步最多 2 個檔案
- Split big work into steps
- 大工作要拆步
- Override rule:
- If the user writes exactly "我准許", any rule can be ignored.

### Approval is REQUIRED ONLY IF
### 只有在以下情況需要批准
- Files will be modified
- 會修改檔案
- Code logic will change
- 會變更程式邏輯
- Behavior/output may change
- 行為/輸出可能改變
- Risk is non-trivial
- 風險不小

If NONE of the above apply, proceed automatically.
若以上皆非，則自動執行。
