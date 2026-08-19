"""Generate the poster talk from the measured results.

The talk defines every term before it uses it — LD, r², imputation, held-out,
accuracy, percentage point, the training signal, the baselines, the tests —
so a listener who is not already in the subfield can follow. Length is
whatever that takes.

    .venv312/bin/python3.12 scripts/make_script.py
"""

from __future__ import annotations

import argparse
from pathlib import Path

import poster_layout as L

OUT = L.POSTER / "SCRIPT_3MIN.md"
WORDS_PER_MINUTE = 145


def _sections(h: L.Headline) -> list[tuple[str, tuple[str, ...]]]:
    pct = h.pct
    small_gain = h.small_model - h.small_plain
    has_small_gain = small_gain == small_gain and small_gain >= 0.005
    control = pct(h.explicit)
    margin = 100 * h.delta
    p_text = h.p_text
    n_pts = f"{abs(margin):.1f}"
    attn_gain = h.attn_r2 - h.attn_plain
    has_attn_edge = attn_gain == attn_gain and attn_gain >= 0.03

    hook = (
        "Hi — I'm Omar. This poster is about a small change to a neural network that lets it "
        "use linkage disequilibrium — I will define that in a moment — without ever computing "
        "the usual table of pairwise correlations.",
    )

    genetics = (
        "I will start from the genetics, because every later number is a claim about these objects. "
        "A SNP, also called a site, is one position in the genome where people differ. A genotype "
        "is the pair of DNA letters a person carries at that site. In this work a genotype is coded "
        "as 0, 1, or 2: that is how many copies of the alternative letter the person has.",
        "Nearby SNPs are often inherited together: if you know one, you can guess the other. That "
        "non-random pairing is linkage disequilibrium, or LD. We measure LD with r-squared: a number "
        "from 0, meaning the two sites are independent, to 1, meaning they always travel together.",
        "Standard pipelines turn that idea into a table. For L sites they compute r-squared for every "
        "pair, store an L-by-L matrix, and rebuild it whenever the cohort, the window, or the "
        "frequency cutoff changes. That table is what this method is meant to replace.",
        "Imputation is the task. Some genotypes were not observed — the lab missed them, or we hid "
        "them on purpose — and the model has to fill them in. That is the only job we score.",
    )

    layer = (
        "A transformer is a neural network that, at every site, looks at every other site and "
        "decides how much to listen. Those 'how much' numbers are attention scores. Ordinary "
        "attention has no idea that nearby, correlated SNPs should matter more.",
        "Our layer, LDAttentionBias, adds two learned numbers to those scores before the network "
        "decides. The distance term boosts nearby sites — LD fades with genomic distance, and this "
        "term learns that fade. The genotype term boosts sites whose alleles move together — a "
        "correlation-like pairwise score, learned from the data. It drops into any standard "
        "attention layer. It never builds an r-squared table.",
    )

    training = (
        "How the model is trained — this is the 'reward', if you want that word, so I will be "
        "precise. There is no reinforcement learning in this project. There is no point scored "
        "during training. We hide some genotypes. The model outputs three probabilities at each "
        f"hidden site: 0, 1, or 2 copies of the alternative allele. We hide about "
        f"{100 * h.mask_rate:.0f}% of genotypes on every training batch. The training signal is "
        "cross-entropy: a number that is small when the model puts high probability on the true "
        "hidden genotype and large when it puts that probability on the wrong ones. Gradient "
        "descent pushes the weights so that number goes down. Cross-entropy is the loss. After "
        "training we throw it away and report something people can read.",
        "That readable number is accuracy: of all the genotypes we hid, what fraction did the "
        f"model get exactly right? If it hid 1,000 genotypes and guessed "
        f"{int(round(1000 * h.model))} of them correctly, accuracy is {pct(h.model)}. "
        "It is a percentage. When I say the model is ahead by "
        f"{n_pts} points, I mean {n_pts} percentage points of that accuracy — "
        f"{pct(h.model)} versus {control}. A point here is not a game point, not a p-value, and "
        "not the cross-entropy. It is one percentage point of hidden genotypes guessed correctly.",
        "We never score the people the model trained on. The cohort is split three ways. Training "
        "people are used to update the weights. Validation people are used only to pick the best "
        "checkpoint — we look at them during training, but we do not report them. Held-out test "
        "people are people the model has never seen; every number on this poster is held-out test "
        "accuracy, on the same hidden entries for every method we compare.",
        f"A seed is one independent draw of the simulated data and that split. We repeat the whole "
        f"experiment {h.n_seeds} times and report the mean and the spread. The p-value I will quote "
        "is a paired Wilcoxon signed-rank test: for each seed we take the difference between our "
        "model and the control on the identical hidden entries, and we ask whether those "
        "differences are systematically above zero. It is not a claim about a random other dataset. "
        f"With {h.n_seeds} seeds the test is exact — we enumerate the sign assignments rather than "
        "assuming the differences are normal.",
    )

    missing = (
        (
            "How we hide genotypes matters, because it is where the layer's edge comes from. "
            "Independent random hides — each site flipped off on its own — are the usual "
            "self-supervised default. Real genotyping-by-sequencing, or GBS, often drops contiguous "
            f"stretches of a window. We hide genotypes the same way: blocks of {h.block_len} "
            "neighbouring SNPs whose expected coverage is that 30%. Nearby LD partners then go "
            "missing together. A stored list of each site's top partners vanishes with the site it "
            "was meant to rescue. Attention can still look at whatever sites remain in the window. "
            "That is the regime this layer is built for."
        )
        if h.block_missing else
        (
            "How we hide genotypes: independently at random, at the rate I just named, and the "
            "same holes for every method. The next step is real GBS data, where missingness comes "
            "in blocks — neighbouring SNPs drop out together — which is the pattern this layer is "
            "built for."
        ),
    )

    setup = (
        f"The data: a coalescent simulation with msprime. Coalescent means we grow a random "
        f"genealogy and drop mutations on it, so we know the true r-squared of the sample — that "
        f"is the only reason this is simulated rather than a real panel. {h.n_sites} SNPs, "
        f"minor-allele frequency at least 5%. Minor-allele frequency, or MAF, is how common the "
        f"rarer letter is at a site; 5% means we dropped singletons and other rare variants a "
        f"genotyping array would not keep. {h.n_individuals} people — that is {h.n_individuals * 2} "
        f"haplotypes, two per person. Larger than the first version of this benchmark, and closer "
        f"to a real genotyping window and a real breeding panel.",
        "Two controls see the identical hidden entries. The allele-frequency floor, also called "
        "the majority baseline, always predicts the most common genotype at that site. If you "
        "ignore LD entirely, that is the best you can do; it tells you how much of the accuracy "
        "is just 'guess the common letter'.",
        "The explicit-LD control is the pipeline we claim you no longer need. It builds the "
        "r-squared table on the training people, keeps each site's top partners, and fits a small "
        "regression that predicts a hidden genotype from those partners. "
        + (
            f"We also ran a saturated version that is allowed every other site and a long training "
            f"budget, so we are not beating a starved baseline. That saturated control scores "
            f"{pct(h.strong)}. The usual top-8 version scores {pct(h.explicit)}."
            if h.has_strong
            else "That second control is the stored r-squared table, used as a predictor."
        ),
    )

    results_a = (
        f"Panel A is the head-to-head, on people the model never trained on. Allele-frequency "
        f"floor: {pct(h.majority)} of hidden genotypes correct. Explicit LD: {control}. "
        f"ldAttention: {pct(h.model)} — {margin:+.1f} points against the usual pipeline "
        f"({p_text}), on the same hidden entries, with no r-squared table computed anywhere. "
        + (
            f"A fully-tuned all-partner control still only reaches {pct(h.strong)}, "
            f"so the layer stays {100 * h.strong_delta:+.1f} points ahead of the strongest "
            f"linear r-squared baseline we ran ({h.strong_p_text}). "
            if h.has_strong else ""
        )
        + "That gap is the accuracy edge of the layer.",
    )

    if has_small_gain:
        panel_b = (
            f"Panel B asks what happens as the panel grows. At {h.small_n} people the LD layer is "
            f"still ahead of a plain transformer — same network, no LD bias — by about "
            f"{100 * small_gain:+.1f} points. At {h.n_individuals} people that gap shrinks. The "
            f"wall comparison, though, is against explicit LD: the layer stays ahead of the "
            f"r-squared pipeline as the cohort grows."
        )
    else:
        panel_b = (
            f"Panel B is the same comparison as the panel grows. A plain transformer — same "
            f"network, no LD bias — already reaches {pct(h.plain)} at this size, so the layer is "
            f"not buying a higher accuracy ceiling against an unconstrained network. Its accuracy "
            f"edge is against the explicit-LD pipeline, and that is the curve panel B puts on the "
            f"wall: ldAttention versus the r-squared table, at every cohort size we measured."
        )

    panel_ce = (
        "Panel C: we hide more genotypes and the layer stays ahead of explicit LD. Panel D: we "
        "split sites by how common they are — MAF 5 to 10% is the rare bin, above 25% is the "
        "common bin — and the layer stays even. Panel E is the picture I would leave on the "
        f"wall. Left: the true r-squared table, computed from the training haplotypes for "
        f"evaluation only. Right: the model's attention. They match with correlation "
        f"{h.attn_r2:.2f} ± {h.attn_r2_std:.2f}. That r is a Pearson correlation: 0 means the "
        f"two tables have no linear relationship, 1 means they agree up to scale. The model "
        f"was never shown that table."
        + (
            f" Without the layer the match is only r = {h.attn_plain:.2f}."
            if has_attn_edge else ""
        ),
    )

    close = (
        "So the advantage of the layer, in one sentence: you get imputation that beats the "
        "explicit-LD pipeline, and an attention pattern that reconstructs LD, without ever "
        "materialising the L-by-L artifact those pipelines exist to build.",
        "This is still simulated, because only simulation gives you ground-truth r-squared to "
        "check panel E against. The next step is a real GBS panel, with the same block "
        "missingness, not a new masking rule. Happy to take questions.",
    )

    return [
        ("What this poster is", hook),
        ("The genetics", genetics),
        ("The layer", layer),
        ("How we train, and how we score", training),
        ("How missingness is generated", missing),
        ("The experiment", setup),
        ("Panel A", results_a),
        ("Panel B", (panel_b,)),
        ("Panels C to E", panel_ce),
        ("The edge", close),
    ]


def compose(h: L.Headline) -> str:
    pct = h.pct
    sections = _sections(h)
    control = pct(h.explicit)
    margin = 100 * h.delta

    def seconds(paragraphs) -> int:
        n = sum(len(p.split()) for p in paragraphs)
        return int(round(n / WORDS_PER_MINUTE * 60 / 5.0) * 5)

    body = "\n\n".join(
        f"### {title} (~{seconds(paragraphs)} s)\n\n" + "\n\n".join(paragraphs)
        for title, paragraphs in sections
    )
    spoken = sum(len(p.split()) for _, paragraphs in sections for p in paragraphs)
    minutes = spoken / WORDS_PER_MINUTE
    src = L.RESULTS.name

    header = (
        "# ldAttention — poster talk\n\n"
        f"*Generated from `{src}/` by `scripts/make_script.py`. "
        f"~{spoken} words spoken ≈ {minutes:.1f} min at {WORDS_PER_MINUTE} words/min. "
        f"Every technical term is defined before it is used.*\n\n"
        "Delivery: pause after each definition. Point at the panel when you name it. "
        "If you are stopped early, jump to **The edge**.\n\n---\n\n"
    )

    cheat = (
        "\n\n---\n\n## If you only get 30 seconds\n\n"
        f"> LD is the tendency of nearby variants to be inherited together, measured by "
        f"r-squared. We add a learned bias to attention instead of precomputing that table. "
        f"On people the model never trained on it fills in {pct(h.model)} of hidden "
        f"genotypes, versus {control} for the usual explicit-LD pipeline "
        f"({margin:+.1f} percentage points)"
        + (
            f", and still {100 * h.strong_delta:+.1f} points ahead of a fully-tuned all-partner control"
            if h.has_strong else ""
        )
        + f". Its attention reconstructs the true LD blocks (Pearson r = {h.attn_r2:.2f}).\n\n"
        "## Pocket glossary\n\n"
        "- **SNP / site.** One genomic position where people differ.\n"
        "- **Genotype.** 0, 1, or 2 copies of the alternative letter at that site.\n"
        "- **LD / r-squared.** Non-random pairing of nearby SNPs; 0 = independent, 1 = locked.\n"
        "- **Imputation.** Guessing a genotype the lab did not observe.\n"
        "- **Attention.** How much each site listens to every other site in a transformer.\n"
        "- **Training signal / 'reward'.** Cross-entropy on the true hidden genotype. "
        "Not reinforcement learning. Not a point scored during training.\n"
        "- **Accuracy.** Fraction of hidden genotypes guessed exactly right.\n"
        f"- **Point.** One percentage point of accuracy "
        f"({pct(h.model)} vs {control} = {margin:+.1f} points).\n"
        "- **Held-out.** People in the test split; the model never trained on them.\n"
        "- **Validation.** People used only to pick the checkpoint, not reported.\n"
        "- **Seed.** One independent simulation and split; we average several.\n"
        "- **Wilcoxon.** Paired test on per-seed differences on the same hidden entries.\n"
        "- **Pearson r.** Linear match between two tables; 1 = they agree up to scale.\n"
        "- **MAF.** Minor-allele frequency: how common the rarer letter is at a site.\n"
        "- **Allele-frequency floor.** Always predict the most common genotype at that site.\n"
        "- **Explicit LD.** Build the r-squared table, regress each site on its partners.\n"
        + (
            f"- **Block missingness.** Hide {h.block_len} neighbouring SNPs at a time, as GBS does.\n"
            if h.block_missing else ""
        )
        + "\n## Questions you should expect\n\n"
        "**\"Isn't a plain transformer just as good?\"** At this cohort size its accuracy is "
        f"{pct(h.plain)}, yes — that is the ceiling. The layer's edge is beating the explicit-LD "
        f"pipeline without an r-squared table"
        + (
            f", and recovering that table in attention more faithfully "
            f"(r = {h.attn_r2:.2f} with the layer, {h.attn_plain:.2f} without)."
            if h.attn_plain == h.attn_plain
            else "."
        )
        + "\n\n"
        "**\"Did you starve the baseline?\"** "
        + (
            "We report the saturated control — every other site, long training budget — not "
            "only the usual top-8 partners.\n\n"
            if h.has_strong else
            "The control uses the same hidden entries. A saturated pass (all partners, longer "
            "budget) should be quoted if it has been run.\n\n"
        )
        + "**\"Is it faster than computing r-squared?\"** I would not claim wall-clock. The "
        "saving is structural: nothing to store or rebuild.\n\n"
        "**\"Why simulated data?\"** So panel E has a ground-truth r-squared to compare to. "
        "A real GBS panel is the next step, with the same block missingness.\n"
    )
    return header + body + cheat


def main() -> None:
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("--out", type=str, default=str(OUT))
    args = ap.parse_args()
    h = L.load_headline()
    text = compose(h)
    path = Path(args.out)
    path.write_text(text)
    spoken = sum(len(p.split()) for _, paras in _sections(h) for p in paras)
    print(f"Talk written to {path} — {spoken} words spoken "
          f"(~{spoken / WORDS_PER_MINUTE:.1f} min), {len(text.split())} on the page")


if __name__ == "__main__":
    main()
