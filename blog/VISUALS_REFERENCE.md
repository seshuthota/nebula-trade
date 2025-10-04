# Blog Series Visuals Reference

This document maps which visualizations are included in each blog post.

---

## Visual Assets Created

All visuals are located in `blog/visuals/`:

1. ✅ `01_master_performance_chart.png` - All models YTD comparison
2. ✅ `02_quarterly_heatmap.png` - Q1 vs Q3 performance breakdown
3. ✅ `03_sticky_bear_optimization.png` - Sticky BEAR optimization results
4. ✅ `04_risk_return_scatter.png` - Risk vs return analysis
5. ✅ `05_model_evolution_timeline.png` - Model progression v1→v5
6. ✅ `06_specialists_comparison.png` - v1 vs v2 vs v2.1 detailed comparison

---

## Post 1: Why Most Trading Bots Fail

**Visuals included:**
- **Figure 1:** `01_master_performance_chart.png`
  - Location: After "Our Approach: The Journey in Numbers" heading
  - Purpose: Show all models at a glance, establish credibility with data

- **Figure 2:** `02_quarterly_heatmap.png`
  - Location: After model comparison table
  - Purpose: Highlight regime-specific performance (bull vs bear)

**Why these visuals:**
- Post 1 is the hook - need to show impressive results upfront
- Heatmap demonstrates the specialist concept visually
- Sets up the narrative for posts 2-6

---

## Post 2: The Research Phase

**Visuals included:**
- None (intentionally)

**Why no visuals:**
- This post focuses on the technical breakthrough (VecNormalize)
- Code snippets and training curves described in text are sufficient
- Keeps focus on the debugging/problem-solving narrative
- Training curves could be added if desired, but text description works well

**Potential addition:**
- Could add before/after training loss curves if generated

---

## Post 3: v1 vs v2 Specialists

**Visuals included:**
- **Figure 1:** `06_specialists_comparison.png`
  - Location: Before "The Three Specialists: Complete Comparison" section
  - Purpose: Detailed visual of v1, v2, v2.1 across different regimes

**Why this visual:**
- Core message is "specialists beat generalists"
- Chart shows v1 dominating bulls, v2 dominating bears
- Visual proof of the training data bias impact

---

## Post 4: Failed Experiments (v3, v4, v5)

**Visuals included:**
- **Figure 1:** `05_model_evolution_timeline.png`
  - Location: In "The Training Balance Spectrum" section
  - Purpose: Show progression from v1→v5 and how complexity increased

- **Figure 2:** `04_risk_return_scatter.png`
  - Location: In "The Complexity Paradox" section
  - Purpose: Demonstrate simpler models (v1) outperformed complex ones (v4, v5)

**Why these visuals:**
- Evolution timeline shows the journey of failed experiments
- Risk-return scatter proves "complexity ≠ better" thesis
- Visual evidence that v3, v4, v5 all underperformed

---

## Post 5: The Ensemble Implementation

**Visuals included:**
- None (intentionally)

**Why no visuals:**
- This post is architecture-focused (code and diagrams in text)
- The whipsaw problem is best explained through switching logs (text format)
- Could add architecture diagram if created separately

**Potential additions:**
- Ensemble architecture diagram (if created with draw.io)
- Regime switching timeline visualization

---

## Post 6: The Sticky BEAR

**Visuals included:**
- **Figure 1:** `03_sticky_bear_optimization.png`
  - Location: After showing the 3-config comparison results
  - Purpose: Dramatic visualization of Q3 improvement (-4.86% → +0.73%)

**Why this visual:**
- This is the climax - the visual needs to show the dramatic improvement
- Chart demonstrates the power of one parameter change
- Before/after comparison is most impactful as a visual

---

## Visual Distribution Summary

| Post | # Visuals | Charts Used | Purpose |
|------|-----------|-------------|---------|
| Post 1 | 2 | Master chart, Quarterly heatmap | Hook readers with results |
| Post 2 | 0 | — | Focus on technical narrative |
| Post 3 | 1 | Specialists comparison | Prove specialist thesis |
| Post 4 | 2 | Evolution timeline, Risk-return | Show failed experiments |
| Post 5 | 0 | — | Architecture focus |
| Post 6 | 1 | Sticky BEAR optimization | Show final breakthrough |

**Total visuals used:** 6/6 ✅

---

## Markdown Image Syntax Used

All images use relative paths from the post location:

```markdown
![Description](../visuals/filename.png)
*Figure N: Caption text*
```

This works because:
- Posts are in `blog/posts/`
- Visuals are in `blog/visuals/`
- `../visuals/` navigates from posts to visuals directory

---

## For Substack Publication

When publishing to Substack:

1. **Upload images** to Substack's media library
2. **Replace paths** in markdown:
   - From: `![Description](../visuals/01_master_performance_chart.png)`
   - To: Substack's image embed or hosted URL

Or use Substack's drag-and-drop interface to add images at the marked locations.

---

## Visual Quality

All charts are:
- ✅ High resolution (300 DPI)
- ✅ Professional styling (seaborn theme)
- ✅ Clear annotations and labels
- ✅ Consistent color scheme across all visuals
- ✅ Publication-ready quality

Generated using: `blog/scripts/generate_charts.py`

---

**Last Updated:** [Current Date]
**Status:** All visuals integrated into blog posts ✅
