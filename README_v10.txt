AI Equity Dashboard v10

Layout:
1. 一级：市场概览（Dow / S&P 500 / Nasdaq Composite / Nasdaq 100 / Bitcoin / Gold / Brent / Copper / DXY / CSI 300）
2. 二级：AI产业链板块总览
3. 三级：板块详情（选择板块 -> 指数走势 + 成分股明细）
4. 四级：个股指标图（选择个股 -> 技术/估值/EPS/事件）

Market overview columns:
点位/价格、当日、YTD、1M、3M、6M、1Y、3Y、5Y。

Realtime logic:
- 顶部只有一个“刷新实时行情”按钮，不自动刷新。
- 按钮在 Twelve Data 已配置时始终可用。
- 美股正常交易时段刷新全部 AI 股票；非美股时段不浪费 AI 股票 quote credits。
- 市场概览按各资产交易状态使用 Twelve Data；不支持/未返回的项目回退 Yahoo。
- BTC 使用 BTC/USD；黄金 XAU/USD；Brent XBR/USD；铜 HG1；DXY 使用 DXY。
- 沪深300历史/最新数据使用 Yahoo 000300.SS fallback。
- 收盘后的美股使用 regular-session close，不使用盘后价作为主价格。
