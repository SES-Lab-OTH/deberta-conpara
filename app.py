"""
DeBERTa-ConPara: public demo.

A research artifact. The output is supporting evidence for a human
judgement, never a verdict on its own.

Run locally:   python3 app.py
On Spaces:     this file is the entry point.

DESIGN NOTE
-----------
This reads as a measuring instrument rather than a verdict machine, because
that is what the measurements support. The central graphic plots the logit
margin on a diverging scale and DRAWS THE UNRELIABLE ZONE onto it: around
|margin| < 4 the error rate rises sharply, so that region is shaded rather
than described in a footnote. A percentage gauge would hide exactly the
thing a reader most needs to see.

MODEL
-----
Released checkpoint (rawguard.pt): DeBERTa-v3-large without the feature
branch, trained on raw text, Unicode normalisation at inference only.
RAID hidden test: AUROC 99.61, TPR@5% FPR 99.01, TPR@1% FPR 96.57.
Fixed-threshold balanced accuracy: HC3-QA 99.69, HC3-SI 83.50, MAGE 96.23,
M4 98.27. Paper: arXiv:2610.00883 (AACL-IJCNLP 2026).
The margin statistics below were measured on the development checkpoint
that preceded the release.

WHY MARGINS AND NOT PROBABILITIES
---------------------------------
98.5% of this model's outputs sit at |margin| > 4, where a float32 softmax
returns 0.999998 or 0.000002. In an earlier submission that saturation tied
606,000 of 672,000 scores at exactly 1.0 and destroyed the ranking the
leaderboard metrics depend on. The margin is the same quantity before that
rounding. Band boundaries are measured error rates over 6,000 validation
rows:  >=12 -> 0.36%,  >=8 -> 1.48%,  >=4 -> 3.00%,  <4 -> much higher.

SENTENCE VIEW
-------------
Two measures per sentence, both shown because they disagree on ~34%:
    in context   document margin minus the margin without that sentence
    alone        the sentence scored by itself
On spliced passages with real sentence-level ground truth (200 sentences,
25 passages), in-context located the boundary at AUROC 0.818 against 0.731
for alone, so ordering and shading follow in-context. Both sit far below
the 0.9956 a whole passage reaches: this locates, it does not decide.
"""
import os
import re
import sys
import numpy as np
import torch
import torch.nn as nn
import gradio as gr
from transformers import DebertaV2Tokenizer, DebertaV2Model
from sklearn.preprocessing import RobustScaler
from huggingface_hub import hf_hub_download

sys.path.insert(0, os.path.join(os.path.dirname(__file__), "src"))
from unicode_preprocessing_v2 import unicode_normalize

# --------------------------------------------------------------- config ---
REPO_ID   = os.environ.get("CONPARA_REPO", "SES-Lab-OTH/deberta-conpara")
CKPT_FILE = os.environ.get("CONPARA_CKPT", "rawguard.pt")
BACKBONE  = "microsoft/deberta-v3-large"
HIDDEN    = 1024
MAX_LEN   = 512
DEVICE    = torch.device("cuda" if torch.cuda.is_available() else "cpu")

MIN_RELIABLE_WORDS = 75
MIN_WORDS          = 25
MAX_SENTENCES      = 5
MIN_SENT_WORDS     = 10
TRUNC_WORDS        = 380   # ~512 tokens; past this the model reads nothing
SCALE_LIMIT        = 20.0


# ---------------------------------------------------------------- model ---
class ConParaDetector(nn.Module):
    """Released configuration: no feature branch. CLS -> Linear(1024,512)
    -> GELU -> Dropout(0.1) -> Linear(512,2); scored by the logit margin."""
    def __init__(self):
        super().__init__()
        self.encoder = DebertaV2Model.from_pretrained(BACKBONE)
        hs = HIDDEN
        self.classifier = nn.Sequential(
            nn.Linear(hs, hs // 2), nn.GELU(), nn.Dropout(0.1),
            nn.Linear(hs // 2, 2))

    def forward(self, input_ids, attention_mask):
        enc = self.encoder(input_ids=input_ids, attention_mask=attention_mask)
        return self.classifier(enc.last_hidden_state[:, 0, :])


print("loading checkpoint ...")
_ckpt_path = hf_hub_download(repo_id=REPO_ID, filename=CKPT_FILE)
_ck = torch.load(_ckpt_path, map_location="cpu", weights_only=False)
VERSION = str(_ck.get("version", "unknown"))
THRESHOLD = float(_ck.get("threshold_margin", 0.0))   # stored operating point

MODEL = ConParaDetector().to(DEVICE)
MODEL.load_state_dict(_ck["model_state_dict"], strict=True)
MODEL.eval()

TOKENIZER = DebertaV2Tokenizer.from_pretrained(BACKBONE)
print(f"ready: {VERSION}, no feature branch, threshold {THRESHOLD:.3f}, device {DEVICE}")


# ------------------------------------------------------------ sentences ---
# Naive re.split(r'[.!?]') breaks on "Dr.", "U.S.", "p.m.", "3.14" and on
# ellipses. Python lookbehind must be fixed-width, so a variable-length
# abbreviation list cannot live there: protect those dots, split, restore.
_ABBR = ['Dr', 'Mr', 'Mrs', 'Ms', 'Prof', 'St', 'Jr', 'Sr', 'vs', 'etc',
         'Inc', 'Ltd', 'Co', 'Corp', 'Fig', 'No', 'Vol', 'approx', 'al',
         'cf', 'Ave', 'Rd', 'Blvd', 'Mt', 'Gen', 'Sen', 'Rep', 'Gov',
         'Jan', 'Feb', 'Mar', 'Apr', 'Jun', 'Jul', 'Aug', 'Sep', 'Oct',
         'Nov', 'Dec']
_DOT = '\x00'
_SPLIT = re.compile(r'([.!?]+["\')\]]?)\s+')


def _protect(t):
    t = re.sub(r'\b(' + '|'.join(_ABBR) + r')\.', r'\1' + _DOT, t)
    t = re.sub(r'\b([A-Z])\.', r'\1' + _DOT, t)
    t = re.sub(r'(\d)\.(\d)', r'\1' + _DOT + r'\2', t)
    t = re.sub(r'\b([a-zA-Z])\.([a-zA-Z])\.', r'\1' + _DOT + r'\2' + _DOT, t)
    return t.replace('...', _DOT * 3)


def split_sentences(text):
    t = re.sub(r'\s+', ' ', str(text or '')).strip()
    parts = _SPLIT.split(_protect(t))
    out, buf = [], ''
    for i, seg in enumerate(parts):
        if not seg:
            continue
        if i % 2 == 1:
            out.append((buf + seg).strip())
            buf = ''
        else:
            buf = seg
    if buf.strip():
        out.append(buf.strip())
    return [s.replace(_DOT, '.') for s in out if s.strip()]


def esc(s):
    return (str(s).replace("&", "&amp;").replace("<", "&lt;")
            .replace(">", "&gt;"))


# ------------------------------------------------------------ inference ---
def margins(texts):
    """Logit margins. margin = logit[AI] - logit[human]; positive favours
    machine. Deliberately not softmaxed -- see the module docstring."""
    clean = [unicode_normalize(t) for t in texts]
    enc = TOKENIZER(clean, max_length=MAX_LEN, truncation=True,
                    padding=True, return_tensors="pt")
    with torch.no_grad():
        lg = MODEL(enc["input_ids"].to(DEVICE),
                   enc["attention_mask"].to(DEVICE))
    lg = lg.float()
    return (lg[:, 1] - lg[:, 0]).cpu().numpy()


def reading(m):
    a = abs(m)
    if a < 4:
        return ("No usable reading", "void", "below the reliable range",
                "most of this system's errors sit here")
    if a >= 12:
        conf, err = "high", "0.4% error at this strength"
    elif a >= 8:
        conf, err = "moderate", "1.5% error at this strength"
    else:
        conf, err = "low", "3% error at this strength"
    if m > 0:
        return (("Strongly machine-like" if a >= 12 else "Machine-like"),
                "machine", conf, err)
    return (("Strongly human-like" if a >= 12 else "Human-like"),
            "human", conf, err)


def pos(m):
    return float(np.clip((m + SCALE_LIMIT) / (2 * SCALE_LIMIT) * 100, 1, 99))


def fmt_tick(v):
    return "0" if v == 0 else f"{v:+d}"


def scale_html(m):
    """The signature graphic: the scale carries its own dead band, so the
    unreliable region is visible rather than described underneath.

    Built from positioned divs, not SVG. An SVG stretched with
    preserveAspectRatio="none" distorts its own tick labels into unreadable
    smears -- checked by rendering it.
    """
    x = pos(m)
    dl, dr = pos(-4), pos(4)
    ticks = "".join(
        f'<span class="cp-tick" style="left:{pos(v)}%"></span>'
        f'<span class="cp-ticklab" style="left:{pos(v)}%">{fmt_tick(v)}</span>'
        for v in (-16, -8, 0, 8, 16))
    return (f'<div class="cp-gauge" role="img" aria-label="reading '
            f'{m:+.2f}, where negative reads human-like and positive reads '
            f'machine-like">'
            f'<div class="cp-track">'
            f'<span class="cp-dead" style="left:{dl}%;width:{dr - dl}%">'
            f'</span>'
            f'<span class="cp-dot" style="left:{x}%"></span>'
            f'</div>'
            f'<div class="cp-ticks">{ticks}</div>'
            f'</div>')


READOUT = """
<div class="cp-read cp-{cls}">
  <div class="cp-head">
    <span class="cp-label">{label}</span>
    <span class="cp-margin">{m:+.2f}</span>
  </div>
  {svg}
  <div class="cp-axis"><span>human-like</span>
    <span class="cp-deadlab">no usable reading</span>
    <span>machine-like</span></div>
  <dl class="cp-facts">
    <div><dt>strength</dt><dd>{conf}</dd></div>
    <div><dt>measured</dt><dd>{err}</dd></div>
    <div><dt>length</dt><dd>{words} words{trunc}</dd></div>
  </dl>
</div>
"""

NOTE_SHORT = """
<div class="cp-note cp-note-warn">
  <span class="cp-note-k">short text</span>
  {words} words. Error rate on this model is 9.3% below 50 words against
  0.5% above 300. Treat the result above as weak.
</div>
"""

NOTE_ALWAYS = """
<div class="cp-note">
  <span class="cp-note-k">evidence, not a finding</span>
  The judgement stays with you. This system marks 13&ndash;67% of formal
  academic prose as machine-like depending on domain, so a machine-like
  reading is not evidence of misconduct and must not be used as such.
</div>
"""

SENT_INTRO = """
<div class="cp-sent-intro">
  <div class="cp-sent-title">Which sentences move the result</div>
  <p>Each sentence below was removed from the text and the whole thing read
  again. The bar shows which way the result moved and by how much &mdash;
  that is what the sentence was contributing.</p>
  <p class="cp-sent-warn">A single sentence is much harder to read than a
  whole text: AUROC 0.82 against 0.9956. Use this to find where to look,
  not to judge a sentence.</p>
</div>
"""


def sent_row(sent, effect, alone, mx, repeats=1):
    """One sentence. `effect` is how far the reading moves when this
    sentence is taken out; `alone` is the sentence read by itself."""
    toward_machine = effect > 0
    tone = "machine" if toward_machine else "human"
    word = "machine-like" if toward_machine else "human-like"
    width = min(100.0, abs(effect) / mx * 100.0)
    side = "right" if toward_machine else "left"
    alone_word = "machine-like" if alone > 0 else "human-like"
    diff = "" if toward_machine == (alone > 0) else " cp-div"
    rep = (f'<span class="cp-s-rep">&nbsp;appears {repeats}&times;</span>'
           if repeats > 1 else '')
    return (f'<li class="cp-s">'
            f'<div class="cp-s-text">{esc(sent)}{rep}</div>'
            f'<div class="cp-s-foot">'
            f'<span class="cp-s-mini"><span class="cp-s-mini-fill '
            f'cp-bar-{tone}" style="width:{width:.0f}%;float:{side}">'
            f'</span></span>'
            f'<span class="cp-s-say">makes the text read '
            f'<b class="cp-{tone}-t">{word}</b> '
            f'<span class="cp-s-num">{abs(effect):.1f}</span></span>'
            f'<span class="cp-s-alone{diff}">on its own: {alone_word}</span>'
            f'</div></li>')


def sentence_rows(text, doc_margin):
    """True leave-one-out on the text that was actually read.

    Three things this gets right that earlier versions did not:
      * the baseline is the margin of the SAME text the sentence is removed
        from. Rejoining a subset and comparing against the full document
        made every effect nearly identical -- a constant offset, not a
        per-sentence signal.
      * only sentences inside the first ~512 tokens are considered.
        Removing a sentence the model never reads cannot change anything.
      * repeated sentences are collapsed and ALL their occurrences removed
        together. Degenerate machine text often repeats a sentence
        verbatim; removing one copy of four measures almost nothing and
        printing four identical rows measures it four times.
    """
    all_sents = split_sentences(text)
    window, seen = [], 0
    for sn in all_sents:
        n = len(sn.split())
        if seen + n > TRUNC_WORDS:
            break
        seen += n
        if n >= MIN_SENT_WORDS:
            window.append(sn)
    if len(window) < 2:
        return ('<div class="cp-note"><span class="cp-note-k">nothing to '
                'break down</span>This text has fewer than two full '
                'sentences the model can read.</div>')

    counts = {}
    for sn in window:
        counts[sn] = counts.get(sn, 0) + 1
    uniq = list(counts)
    if len(uniq) < 2:
        return ('<div class="cp-note"><span class="cp-note-k">nothing to '
                'break down</span>This text repeats a single sentence, so '
                'there is nothing to compare.</div>')

    cands = (uniq if len(uniq) <= 12 else
             sorted(uniq, key=lambda t: -len(t.split()))[:12])

    # remove EVERY occurrence, so a sentence repeated four times is
    # measured for what it actually contributes
    loo = margins([text.replace(sn, " ") for sn in cands])
    effect = doc_margin - loo
    alone = margins(cands)

    order = np.argsort(-np.abs(effect))[:MAX_SENTENCES]
    mx = max(float(np.abs(effect[order]).max()), 1e-6)
    rows = "".join(sent_row(cands[i], float(effect[i]), float(alone[i]),
                            mx, counts[cands[i]]) for i in order)

    # when nothing moves the reading much, say so: heavily repetitive text
    # is held up by everything at once rather than by any one sentence
    flat = ('<div class="cp-sent-flat">No single sentence changes the '
            'reading much here &mdash; the strongest is worth '
            f'{mx:.1f} against a reading of {abs(doc_margin):.1f}. Texts '
            'that repeat themselves tend to look like this.</div>'
            if mx < 0.05 * max(abs(doc_margin), 1) else '')

    dupes = sum(1 for t in uniq if counts[t] > 1)
    foot = (f'<div class="cp-sent-foot">showing {len(order)} of '
            f'{len(uniq)} distinct sentences the model reads'
            + (f' &middot; {dupes} repeated verbatim' if dupes else '')
            + '</div>')
    return SENT_INTRO + flat + '<ol class="cp-slist">' + rows + '</ol>' + foot


def analyse(text, do_sentences):
    if not text or not text.strip():
        return "", "", ""
    words = len(text.split())
    if words < MIN_WORDS:
        return ('<div class="cp-note cp-note-warn">'
                '<span class="cp-note-k">too short to read</span>'
                f'{words} words. This needs at least {MIN_WORDS} before a '
                'reading means anything.</div>'), "", ""

    m = float(margins([text])[0])
    label, cls, conf, err = reading(m)
    trunc = ", first 512 tokens read" if words > 380 else ""
    head = READOUT.format(label=label, cls=cls, m=m, svg=scale_html(m),
                          conf=conf, err=err, words=words, trunc=trunc)
    notes = (NOTE_SHORT.format(words=words)
             if words < MIN_RELIABLE_WORDS else "") + NOTE_ALWAYS
    sents = sentence_rows(text, m) if do_sentences else ""
    return head, notes, sents


# ------------------------------------------------------------------ UI ----
CSS = """
@import url('https://fonts.googleapis.com/css2?family=IBM+Plex+Sans:wght@400;500;600&family=IBM+Plex+Mono:wght@400;500&display=swap');

:root{
  --ink:#16181D; --paper:#F7F7F5; --rule:#DCDCD6; --soft:#6E6E66;
  --human:#2D6B5F; --machine:#A63D2F; --void:#8A8A82;
  --sans:'IBM Plex Sans',system-ui,-apple-system,sans-serif;
  --mono:'IBM Plex Mono',ui-monospace,SFMono-Regular,monospace;
}
.gradio-container{max-width:820px!important;margin:auto;
  font-family:var(--sans)!important;}

.cp-mast h1{font-size:26px;font-weight:600;letter-spacing:-.02em;
  margin:0 0 3px;color:var(--ink);}
.cp-mast .cp-sub{font-family:var(--mono);font-size:11px;
  letter-spacing:.09em;text-transform:uppercase;color:var(--soft);
  margin:0 0 13px;}
.cp-mast p:not(.cp-sub){font-size:14px;line-height:1.62;color:#3C3E44;
  margin:0;max-width:62ch;}

.cp-read{border:1px solid var(--rule);border-radius:2px;background:#fff;
  padding:20px 22px 15px;margin-top:6px;border-left-width:3px;
  border-left-style:solid;}
.cp-read.cp-machine{border-left-color:var(--machine);}
.cp-read.cp-human{border-left-color:var(--human);}
.cp-read.cp-void{border-left-color:var(--void);}
.cp-head{display:flex;justify-content:space-between;align-items:baseline;
  gap:16px;margin-bottom:12px;}
.cp-label{font-size:19px;font-weight:600;letter-spacing:-.01em;}
.cp-read.cp-machine .cp-label{color:var(--machine);}
.cp-read.cp-human .cp-label{color:var(--human);}
.cp-read.cp-void .cp-label{color:var(--void);}
.cp-margin{font-family:var(--mono);font-size:21px;font-weight:500;
  color:var(--ink);font-variant-numeric:tabular-nums;}

.cp-gauge{margin:2px 0 6px;}
.cp-track{position:relative;height:10px;border-radius:1px;
  background:linear-gradient(90deg,
    rgba(45,107,95,.55) 0%, rgba(45,107,95,.30) 26%,
    rgba(45,107,95,.10) 42%, rgba(166,61,47,.10) 58%,
    rgba(166,61,47,.30) 74%, rgba(166,61,47,.55) 100%);}
.cp-dead{position:absolute;top:0;height:100%;background:#E4E4DE;
  box-shadow:inset 0 0 0 1px rgba(0,0,0,.05);}
.cp-dot{position:absolute;top:-1px;width:12px;height:12px;
  border-radius:50%;background:var(--ink);transform:translateX(-6px);
  box-shadow:0 0 0 2.5px #fff;}
.cp-read.cp-machine .cp-dot{background:var(--machine);}
.cp-read.cp-human .cp-dot{background:var(--human);}
.cp-read.cp-void .cp-dot{background:var(--void);}
.cp-ticks{position:relative;height:15px;margin-top:5px;}
.cp-tick{position:absolute;top:-5px;width:1px;height:4px;background:#C6C6BE;}
.cp-ticklab{position:absolute;top:1px;transform:translateX(-50%);
  font-family:var(--mono);font-size:9.5px;color:var(--soft);
  font-variant-numeric:tabular-nums;}
.cp-axis{display:flex;justify-content:space-between;font-family:var(--mono);
  font-size:9.5px;letter-spacing:.06em;text-transform:uppercase;
  color:var(--soft);margin:2px 0 15px;}
.cp-deadlab{color:#ABABA3;}

.cp-facts{display:flex;gap:28px;margin:0;padding:13px 0 0;
  border-top:1px solid var(--rule);flex-wrap:wrap;}
.cp-facts>div{display:flex;flex-direction:column;gap:3px;}
.cp-facts dt{font-family:var(--mono);font-size:9px;letter-spacing:.09em;
  text-transform:uppercase;color:var(--soft);margin:0;}
.cp-facts dd{font-size:13px;color:var(--ink);margin:0;}

.cp-note{font-size:13px;line-height:1.62;color:#3C3E44;background:#fff;
  border:1px solid var(--rule);border-radius:2px;padding:12px 15px;
  margin-top:8px;}
.cp-note-warn{border-color:#E3CBA9;background:#FDFAF4;}
.cp-note-k{display:block;font-family:var(--mono);font-size:9px;
  letter-spacing:.1em;text-transform:uppercase;color:var(--soft);
  margin-bottom:4px;}

.cp-sent-intro{margin-top:24px;padding-top:17px;
  border-top:1px solid var(--rule);}
.cp-sent-title{font-size:15px;font-weight:600;margin-bottom:7px;
  color:var(--ink);}
.cp-sent-intro p{font-size:13px;line-height:1.62;color:#3C3E44;
  margin:0 0 7px;max-width:64ch;}
.cp-sent-intro p.cp-sent-warn{color:var(--soft);font-size:12px;}
.cp-slist{list-style:none;margin:13px 0 0;padding:0;}
.cp-s{border:1px solid var(--rule);border-radius:2px;background:#fff;
  padding:13px 15px 11px;margin-bottom:7px;}
.cp-s-text{font-size:13.5px;line-height:1.58;color:var(--ink);}
.cp-s-foot{display:flex;align-items:center;gap:11px;margin-top:9px;
  flex-wrap:wrap;}
.cp-s-mini{flex:0 0 84px;height:5px;background:#EDEDE8;border-radius:1px;
  overflow:hidden;}
.cp-s-mini-fill{display:block;height:100%;}
.cp-bar-machine{background:var(--machine);}
.cp-bar-human{background:var(--human);}
.cp-s-say{font-size:12px;color:#3C3E44;}
.cp-machine-t{color:var(--machine);font-weight:600;}
.cp-human-t{color:var(--human);font-weight:600;}
.cp-s-num{font-family:var(--mono);font-size:11px;color:var(--soft);
  font-variant-numeric:tabular-nums;margin-left:3px;}
.cp-s-alone{font-size:11.5px;color:var(--soft);margin-left:auto;}
.cp-s-alone.cp-div{color:#93661E;}
.cp-s-rep{font-family:var(--mono);font-size:10px;color:var(--soft);
  margin-left:8px;white-space:nowrap;}
.cp-sent-flat{font-size:12.5px;line-height:1.6;color:#3C3E44;
  background:#FDFAF4;border:1px solid #E3CBA9;border-radius:2px;
  padding:11px 14px;margin-top:12px;}
.cp-sent-foot{font-family:var(--mono);font-size:10px;color:var(--soft);
  margin-top:10px;}

@media (prefers-reduced-motion:no-preference){
  .cp-dot{transition:left .45s cubic-bezier(.4,0,.2,1);}
}
@media (max-width:600px){
  .cp-facts{gap:18px;} .cp-margin{font-size:18px;}
  .cp-label{font-size:17px;}
}
footer{display:none!important;}
"""

MAST = """
<div class="cp-mast">
<h1>DeBERTa-ConPara</h1>
<p class="cp-sub">machine-text detection &middot; research artifact</p>
<p>Paste some text. You get a reading on a scale, with the uncertainty of
that reading marked on the scale rather than hidden beneath it. Nothing
you submit is stored.</p>
</div>
"""

try:
    from examples import EXAMPLES
except Exception:
    EXAMPLES = [
        ["The mitochondrion is a double-membrane-bound organelle found in "
         "most eukaryotic organisms. Mitochondria generate most of the "
         "cell's supply of adenosine triphosphate, used as a source of "
         "chemical energy. They were first discovered by Albert von "
         "Kolliker in 1857, though their function remained unclear for "
         "several decades afterwards."],
    ]

with gr.Blocks(title="DeBERTa-ConPara") as demo:
    gr.HTML(MAST)

    inp = gr.Textbox(lines=10, label="Text",
                     placeholder="Paste at least 75 words. Longer text gives "
                                 "a firmer reading.")
    sent_cb = gr.Checkbox(
        value=False, label="Show which sentences drove this",
        info="Re-reads the text once per sentence. A few seconds on GPU, up to a minute on the free CPU tier.")
    with gr.Row():
        btn = gr.Button("Analyse", variant="primary", scale=2)
        clr = gr.Button("Clear", scale=1)

    out_head = gr.HTML()
    out_note = gr.HTML()
    out_sent = gr.HTML()

    gr.Examples(EXAMPLES, inputs=inp, label="Examples from the test set")

    with gr.Accordion("What this measures, and where it fails", open=False):
        gr.Markdown("""
**The number.** The reading is a logit margin: the model's score for
machine minus its score for human. Negative reads human, positive reads
machine, and distance from zero is the strength of the reading. A
percentage is not shown because the model is saturated: on validation
data 98.5% of passages sit far enough from the boundary that a probability rounds to
0.000 or 1.000 and stops distinguishing a moderate reading from an extreme
one.

**The shaded band.** Around |margin| < 4 the measured error rate rises
sharply, so that region is marked as giving no usable reading. Outside it,
error rates over 6,000 validation passages are 3% beyond ±4, 1.5% beyond
±8, and 0.4% beyond ±12.

**Pipeline.** Attack-aware Unicode normalisation, which neutralises
homoglyph, zero-width and whitespace perturbation before tokenisation,
followed by DeBERTa-v3-large read through its [CLS] representation. The
model was trained on raw, unnormalised text, so it has seen the attacks;
normalising only at inference is what the paper identifies as the best
placement. A handcrafted-feature branch was evaluated and left out: it was
inert in distribution and harmful outside it.

**Benchmarks.** RAID hidden test (official leaderboard): 99.61 AUROC,
99.01% TPR at 5% FPR, 96.57% TPR at 1% FPR. Balanced accuracy at one fixed
threshold: 99.69% HC3-QA, 83.50% HC3-SI, 96.23% MAGE, 98.27% M4.

*Note.* The error rates quoted for the shaded band, for short passages and
for the sentence view were measured on the development checkpoint that
preceded the released model; the benchmarks above are the released
model's own.

**Where it fails.**

- *Short passages.* 9.3% error below 50 words against 0.5% above 300.
  Below about 30 words there is very little to read.
- *Formal academic prose.* 13–67% false positives depending on domain.
  Structured, impersonal writing reads as machine-like to this and to
  every detector tested.
- *Paraphrase.* Preprocessing does not defend against semantic rewriting;
  what robustness exists comes from the model and is partial.
- *English only.* No claim is made about any other language.

**Sentence readings.** On spliced passages with real sentence-level ground
truth, the in-context measure located the human/machine boundary at AUROC
0.818 and the isolated score at 0.731, both far below the 0.9956 a whole
passage reaches. Aggregating sentence readings back into a passage verdict
recovers only 0.884, which is why the passage is read whole and the
sentence view is offered for locating rather than deciding.

**Not a misconduct detector.** A machine-like reading is weak
circumstantial signal warranting human review. It is not a finding and
should never be treated as one.

Paper: [arXiv:2610.00883](https://arxiv.org/abs/2610.00883) (AACL-IJCNLP 2026) ·
Code and evaluation: [github.com/SES-Lab-OTH/deberta-conpara](https://github.com/SES-Lab-OTH/deberta-conpara) ·
Smart Embedded Systems Lab, OTH Regensburg
        """)

    btn.click(analyse, inputs=[inp, sent_cb],
              outputs=[out_head, out_note, out_sent])
    inp.submit(analyse, inputs=[inp, sent_cb],
               outputs=[out_head, out_note, out_sent])
    clr.click(lambda: ("", "", "", ""),
              outputs=[inp, out_head, out_note, out_sent])

if __name__ == "__main__":
    demo.launch(css=CSS,
                theme=gr.themes.Soft(primary_hue="slate",
                                     neutral_hue="stone"))
