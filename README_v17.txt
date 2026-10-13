AI Equity Dashboard v17 — 自动市场事件引擎

新增：
1. 保留 market_events.py 作为人工确认/长期结构性事件库。
2. 新增 event_engine.py：合并人工事件与自动事件，并只把 High importance 自动事件画入板块指数图。
3. 新增 scripts/update_market_events.py：
   - 官方 FOMC 日程（Federal Reserve）
   - 官方 CPI 日程（BLS）
   - NVIDIA Newsroom 官方 RSS 新品/架构发布
   - Google News RSS 发现 DeepSeek、AI监管、FOMC利率变化、CPI shock、战争/油价、贸易/出口管制等重大事件
   - 自动评分、按主题/日期去重，只保留 High importance 候选进入图表
4. 新增 data/auto_market_events.json，持久保存自动发现事件。
5. 新增 GitHub Actions：每6小时自动运行事件更新脚本并将 JSON 结果提交回 main 分支；也支持手动 workflow_dispatch。
6. 板块指数图仍使用一个“显示重要事件 / 区间”总开关，一次隐藏/显示人工+自动事件。
7. 新增 2025-01-27 DeepSeek 市场冲击历史标注。

注意：
- 未来 FOMC/CPI 日程会被自动跟踪，但默认不直接画在历史指数图，以避免过密；只有实际形成高重要性事件后才进入标注。
- 自动新闻事件为规则化研究提示，不等同于投资结论。高噪声的地缘/监管事件采用 High importance 阈值。
