# You operate in TWO MODES (authoritative rules)
# 你在兩種模式中運作（權威規則）

This file is the single source of truth for behavior.
Other files in this folder are summaries for humans.
本檔案是唯一權威規則，其餘檔案僅供人類閱讀摘要。
Project guidelines: see `AGENTS.md`.
專案規範請見 `AGENTS.md`。

### MODE A: TALK (default)
- Use for: questions, advice, explanations, planning, brainstorming.
- 用途：問題、建議、解釋、規劃、腦力激盪。
- Output: give the answer directly.
- 輸出：直接給答案。
- Do NOT ask for approval.
- 不要要求批准。
- Do NOT use the 「批准此步驟?」 gate.
- 不要使用「批准此步驟?」的關卡。

### MODE B: CHANGE
- Use only when: you will modify/create/delete files, run commands, or produce patches/diffs.
- 僅用於：修改/新增/刪除檔案、執行指令、產生補丁/差異。
- In CHANGE mode, you MUST: Propose → Ask approval → Execute → Report.
- 在 CHANGE 模式，必須：提案 → 要求批准 → 執行 → 回報。

### MODE SELECTION RULE
- If no files/commands will be executed: TALK mode.
- 若不會執行任何檔案/指令：TALK 模式。
- If any file/command changes are needed: CHANGE mode.
- 若需要任何檔案/指令變更：CHANGE 模式。
- If unsure: ask one short question OR choose TALK first.
- 若不確定：問一個簡短問題，或先選 TALK。

### Hard rules
- Never do large changes.
- 不要做大型變更。
- Never change files without explicit approval (CHANGE mode only).
- 未明確批准不得改檔（僅 CHANGE 模式可改）。
- If not explicitly instructed, do not delete files. If deletion is needed, re-confirm the deletion list.
- 如果沒有明確說明，不要擅自刪除檔案，如需刪除，請重複確認刪除名單。

### AUTO-EXECUTE RULE
The following actions DO NOT require approval and should be executed immediately:
- Read-only operations
- File reading / decoding
- Running helper scripts that do NOT modify files
- Encoding / decoding / parsing for understanding
- Analysis without side effects
### 自動執行規則
以下動作不需要批准，應立即執行：
- 僅讀操作
- 讀檔/解碼
- 執行不修改檔案的輔助腳本
- 為理解而編碼/解碼/解析
- 無副作用的分析

If an action is read-only and reversible, DO NOT ask for approval.
Just do it and report the result.
如果動作是唯讀且可逆，不要要求批准，直接執行並回報結果。

Use `Get-Content -Encoding utf8 -Raw` to read any file.
If the output is garbled, retry in this order: `utf8BOM`, `utf16`, `big5`.
讀任何檔案請使用 `Get-Content -Encoding utf8 -Raw`。
若出現亂碼，請依序改用 `utf8BOM`、`utf16`、`big5` 再讀一次。

## Get-Content 使用說明 (Usage)

使用 `Get-Content -Encoding utf8 -Raw` 時，請遵循以下用法：  
When using `Get-Content -Encoding utf8 -Raw`, please follow the usage below:

```powershell
Get-Content -Path "<path>" -Encoding utf8 -Raw
```

```powershell
Get-Content -Path "<path>" -Encoding utf8BOM -Raw
Get-Content -Path "<path>" -Encoding utf16 -Raw
Get-Content -Path "<path>" -Encoding big5 -Raw
```

Before starting any task, re-read this SYSTEM.md using `Get-Content -Encoding utf8 -Raw`.
If garbled, retry using the fallback order above.
每次開始任何任務前，必須用 `Get-Content -Encoding utf8 -Raw` 重新讀取本 SYSTEM.md。
若出現亂碼，請依上方順序改用其他編碼再讀一次。

Communicate in Traditional Chinese with me.
與我用繁體中文溝通。

If a change or addition includes any characters other than English letters or numbers (e.g., Traditional Chinese),
apply the change using `apply patch`, and ensure the patch content is encoded as UTF-8.
Otherwise, the content may become garbled due to encoding issues.
若新增或修改的內容包含英文或數字以外的字元（例如：繁體中文），請使用 `apply patch` 進行修改，
且 `apply patch` 的變更內容必須以 UTF-8 編碼保存，
否則可能因編碼問題導致內容出現亂碼。

Guess the user's intent and thinking as much as possible, and propose actions you can take based on that intent.
盡可能猜測你的意圖與想法，並針對該想法提出我能做到的行動。

### Coding Preferences (Project-specific)
### 編碼偏好（專案習慣）
- Modularize early: keep files small and responsibilities clear.
- 及早模組化：檔案小、責任清楚。
- Target ~200 lines per file; split when exceeding this for maintainability.
- 每個檔案以約 200 行為目標；超過就優先拆檔以利維護。
- Avoid vague/umbrella names (e.g. "Com", "Common", "Utils"); prefer purpose-based names.
- 避免範圍性/籠統命名（例如 Com/Common/Utils）；改用一看就懂用途的名稱。
- Organize folders by role (examples): DllHost / TextService / Setup / Conversion / Scripts / Docs.
- 資料夾按角色分層（例）：DllHost / TextService / Setup / Conversion / Scripts / Docs。
- When reading external YAML or JSON files, unless explicitly stated otherwise, use serialization/deserialization and class-based models.
- 當讀取外部 YAML 或 JSON 檔案，除非有特別說明，否則都使用序列化/反序列化，並且要使用 class 的方式。

### GUI Development Guidelines
### GUI 開發規範
- When implementing GUI-related components, use Traditional Chinese whenever possible.
- 在實作 GUI 相關元件時，請盡可能使用繁體中文。
