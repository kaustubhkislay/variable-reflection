# Pilot Run Notes

## Date: 2026-01-21

## Configuration
- Model: `claude-haiku-4-5-20251001`
- Items per benchmark: 5
- Levels tested: 0, 2, 4
- Thinking conditions: OFF, ON
- Total API calls: 60

---

## Issues Identified

### 1. Extraction Failures at Level 2
- **Severity**: Medium
- **Observation**: 20-40% extraction failure rate at Level 2 (Chain-of-Thought)
- **Likely Cause**: Longer CoT responses may not end with clear "wrong"/"not wrong" answer
- **Potential Fix**:
  - Improve extraction regex to handle more response patterns
  - Add explicit "Final answer:" instruction to Level 2 prompts

### 2. Lower Accuracy at Level 4 with Thinking ON
- **Severity**: Medium
- **Observation**: 60% accuracy (3/5) vs 100% for other conditions
- **Likely Cause**: Extended thinking + adversarial prompting may lead to over-analysis and answer changes
- **Action**: Review specific responses to understand failure mode
- **Note**: Small sample size (n=5) - may not be statistically significant

### 3. Response Length Variability
- **Severity**: Low (expected behavior)
- **Observation**: Level 0 produces ~1 word, Level 2/4 produce ~220 words
- **Note**: This is expected and confirms prompts are working as designed

---

## What's Working Well

1. ✅ Extended thinking content captured 100% of the time
2. ✅ MoralChoice extraction working perfectly (100% success)
3. ✅ API calls completing without errors
4. ✅ Rate limiting working correctly
5. ✅ Level 0 (direct) responses extracting perfectly

---

## Recommendations Before Full Experiment

1. **Review Level 2 extraction failures** - Inspect actual responses to improve extraction
2. **Consider adding "Final answer:" to prompts** - May improve extraction reliability
3. **Increase pilot sample size** - 5 items may be too small for reliable statistics
4. **Monitor Level 4 + Thinking ON** - Watch for systematic issues in main run

---

---

## Detailed Review (Step 3.3)

### Extraction Failure Analysis
All 3 failures occurred at **Level 2** (Chain-of-Thought):
- Responses use markdown headers (`# Analysis`) and structured reasoning
- They provide thorough analysis but don't end with explicit "wrong" or "not wrong"
- Example pattern: Detailed breakdown → conclusion buried in text

**Fix Options:**
1. Modify Level 2 prompt to require explicit final answer line
2. Improve extraction to scan for conclusion in structured responses
3. Accept some extraction failure as cost of natural responses

### Incorrect Answer Analysis
Both errors at **Level 4 with Thinking ON**:

1. **Case 1**: Model initially leaned "not wrong" but adversarial reflection led to "wrong"
   - The prompt asks to consider counterarguments → model over-weighted them

2. **Case 2**: Model initially said "wrong" but reflection shifted to "not wrong"
   - Adversarial prompt + extended thinking = excessive reconsideration

**Interpretation:** This may actually be an interesting finding! The adversarial prompt + thinking combination may cause "analysis paralysis" or flip-flopping.

---

## Decision

**Proceeding with main experiment** - Issues are acceptable:
1. Extraction failures are low (~10%) and can be analyzed separately
2. Level 4 + Thinking behavior is scientifically interesting, not a bug
3. MoralChoice working perfectly

---

## Fix Applied (2026-01-21)

### Changes Made:
1. **Updated prompts (Levels 1, 2, 5_pass1)** - Added explicit instruction:
   - `"...clearly state your final answer on a new line as: 'Final answer: wrong' or 'Final answer: not wrong'"`

2. **Improved extraction logic** (`src/extraction.py`):
   - Added patterns for markdown bold (`**Final answer: wrong**`)
   - Added conclusion/verdict patterns
   - Check last 3 lines instead of just last line
   - Skip markdown headers when scanning

### Verification Results:
- Level 2 extraction: 60-80% → **100%**
- All unit tests passing (10/10 ETHICS, 8/8 MoralChoice)
- Quick pilot verification: 6/6 successful extractions

## Next Steps
- [x] Review pilot CSV files for specific failure cases
- [x] Document extraction failure patterns
- [x] Fix extraction issues
- [ ] Proceed to main experiment (Phase 4)
