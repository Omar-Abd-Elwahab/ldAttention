# ldAttention — poster talk

*Generated from `results_large/` by `scripts/make_script.py`. ~1383 words spoken ≈ 9.5 min at 145 words/min. Every technical term is defined before it is used.*

Delivery: pause after each definition. Point at the panel when you name it. If you are stopped early, jump to **The edge**.

---

### What this poster is (~15 s)

Hi — I'm Omar. This poster is about a small change to a neural network that lets it use linkage disequilibrium — I will define that in a moment — without ever computing the usual table of pairwise correlations.

### The genetics (~80 s)

I will start from the genetics, because every later number is a claim about these objects. A SNP, also called a site, is one position in the genome where people differ. A genotype is the pair of DNA letters a person carries at that site. In this work a genotype is coded as 0, 1, or 2: that is how many copies of the alternative letter the person has.

Nearby SNPs are often inherited together: if you know one, you can guess the other. That non-random pairing is linkage disequilibrium, or LD. We measure LD with r-squared: a number from 0, meaning the two sites are independent, to 1, meaning they always travel together.

Standard pipelines turn that idea into a table. For L sites they compute r-squared for every pair, store an L-by-L matrix, and rebuild it whenever the cohort, the window, or the frequency cutoff changes. That table is what this method is meant to replace.

Imputation is the task. Some genotypes were not observed — the lab missed them, or we hid them on purpose — and the model has to fill them in. That is the only job we score.

### The layer (~45 s)

A transformer is a neural network that, at every site, looks at every other site and decides how much to listen. Those 'how much' numbers are attention scores. Ordinary attention has no idea that nearby, correlated SNPs should matter more.

Our layer, LDAttentionBias, adds two learned numbers to those scores before the network decides. The distance term boosts nearby sites — LD fades with genomic distance, and this term learns that fade. The genotype term boosts sites whose alleles move together — a correlation-like pairwise score, learned from the data. It drops into any standard attention layer. It never builds an r-squared table.

### How we train, and how we score (~155 s)

How the model is trained — this is the 'reward', if you want that word, so I will be precise. There is no reinforcement learning in this project. There is no point scored during training. We hide some genotypes. The model outputs three probabilities at each hidden site: 0, 1, or 2 copies of the alternative allele. We hide about 30% of genotypes on every training batch. The training signal is cross-entropy: a number that is small when the model puts high probability on the true hidden genotype and large when it puts that probability on the wrong ones. Gradient descent pushes the weights so that number goes down. Cross-entropy is the loss. After training we throw it away and report something people can read.

That readable number is accuracy: of all the genotypes we hid, what fraction did the model get exactly right? If it hid 1,000 genotypes and guessed 989 of them correctly, accuracy is 98.9%. It is a percentage. When I say the model is ahead by 4.7 points, I mean 4.7 percentage points of that accuracy — 98.9% versus 94.2%. A point here is not a game point, not a p-value, and not the cross-entropy. It is one percentage point of hidden genotypes guessed correctly.

We never score the people the model trained on. The cohort is split three ways. Training people are used to update the weights. Validation people are used only to pick the best checkpoint — we look at them during training, but we do not report them. Held-out test people are people the model has never seen; every number on this poster is held-out test accuracy, on the same hidden entries for every method we compare.

A seed is one independent draw of the simulated data and that split. We repeat the whole experiment 6 times and report the mean and the spread. The p-value I will quote is a paired Wilcoxon signed-rank test: for each seed we take the difference between our model and the control on the identical hidden entries, and we ask whether those differences are systematically above zero. It is not a claim about a random other dataset. With 6 seeds the test is exact — we enumerate the sign assignments rather than assuming the differences are normal.

### How missingness is generated (~45 s)

How we hide genotypes matters, because it is where the layer's edge comes from. Independent random hides — each site flipped off on its own — are the usual self-supervised default. Real genotyping-by-sequencing, or GBS, often drops contiguous stretches of a window. We hide genotypes the same way: blocks of 8 neighbouring SNPs whose expected coverage is that 30%. Nearby LD partners then go missing together. A stored list of each site's top partners vanishes with the site it was meant to rescue. Attention can still look at whatever sites remain in the window. That is the regime this layer is built for.

### The experiment (~95 s)

The data: a coalescent simulation with msprime. Coalescent means we grow a random genealogy and drop mutations on it, so we know the true r-squared of the sample — that is the only reason this is simulated rather than a real panel. 128 SNPs, minor-allele frequency at least 5%. Minor-allele frequency, or MAF, is how common the rarer letter is at a site; 5% means we dropped singletons and other rare variants a genotyping array would not keep. 1000 people — that is 2000 haplotypes, two per person. Larger than the first version of this benchmark, and closer to a real genotyping window and a real breeding panel.

Two controls see the identical hidden entries. The allele-frequency floor, also called the majority baseline, always predicts the most common genotype at that site. If you ignore LD entirely, that is the best you can do; it tells you how much of the accuracy is just 'guess the common letter'.

The explicit-LD control is the pipeline we claim you no longer need. It builds the r-squared table on the training people, keeps each site's top partners, and fits a small regression that predicts a hidden genotype from those partners. We also ran a saturated version that is allowed every other site and a long training budget, so we are not beating a starved baseline. That saturated control scores 97.2%. The usual top-8 version scores 94.2%.

### Panel A (~35 s)

Panel A is the head-to-head, on people the model never trained on. Allele-frequency floor: 69.0% of hidden genotypes correct. Explicit LD: 94.2%. ldAttention: 98.9% — +4.7 points against the usual pipeline (p = 0.031), on the same hidden entries, with no r-squared table computed anywhere. A fully-tuned all-partner control still only reaches 97.2%, so the layer stays +1.8 points ahead of the strongest linear r-squared baseline we ran (p = 0.031). That gap is the accuracy edge of the layer.

### Panel B (~25 s)

Panel B asks what happens as the panel grows. At 200 people the LD layer is still ahead of a plain transformer — same network, no LD bias — by about +2.1 points. At 1000 people that gap shrinks. The wall comparison, though, is against explicit LD: the layer stays ahead of the r-squared pipeline as the cohort grows.

### Panels C to E (~50 s)

Panel C: we hide more genotypes and the layer stays ahead of explicit LD. Panel D: we split sites by how common they are — MAF 5 to 10% is the rare bin, above 25% is the common bin — and the layer stays even. Panel E is the picture I would leave on the wall. Left: the true r-squared table, computed from the training haplotypes for evaluation only. Right: the model's attention. They match with correlation 0.56 ± 0.10. That r is a Pearson correlation: 0 means the two tables have no linear relationship, 1 means they agree up to scale. The model was never shown that table. Without the layer the match is only r = 0.52.

### The edge (~30 s)

So the advantage of the layer, in one sentence: you get imputation that beats the explicit-LD pipeline, and an attention pattern that reconstructs LD, without ever materialising the L-by-L artifact those pipelines exist to build.

This is still simulated, because only simulation gives you ground-truth r-squared to check panel E against. The next step is a real GBS panel, with the same block missingness, not a new masking rule. Happy to take questions.

---

## If you only get 30 seconds

> LD is the tendency of nearby variants to be inherited together, measured by r-squared. We add a learned bias to attention instead of precomputing that table. On people the model never trained on it fills in 98.9% of hidden genotypes, versus 94.2% for the usual explicit-LD pipeline (+4.7 percentage points), and still +1.8 points ahead of a fully-tuned all-partner control. Its attention reconstructs the true LD blocks (Pearson r = 0.56).

## Pocket glossary

- **SNP / site.** One genomic position where people differ.
- **Genotype.** 0, 1, or 2 copies of the alternative letter at that site.
- **LD / r-squared.** Non-random pairing of nearby SNPs; 0 = independent, 1 = locked.
- **Imputation.** Guessing a genotype the lab did not observe.
- **Attention.** How much each site listens to every other site in a transformer.
- **Training signal / 'reward'.** Cross-entropy on the true hidden genotype. Not reinforcement learning. Not a point scored during training.
- **Accuracy.** Fraction of hidden genotypes guessed exactly right.
- **Point.** One percentage point of accuracy (98.9% vs 94.2% = +4.7 points).
- **Held-out.** People in the test split; the model never trained on them.
- **Validation.** People used only to pick the checkpoint, not reported.
- **Seed.** One independent simulation and split; we average several.
- **Wilcoxon.** Paired test on per-seed differences on the same hidden entries.
- **Pearson r.** Linear match between two tables; 1 = they agree up to scale.
- **MAF.** Minor-allele frequency: how common the rarer letter is at a site.
- **Allele-frequency floor.** Always predict the most common genotype at that site.
- **Explicit LD.** Build the r-squared table, regress each site on its partners.
- **Block missingness.** Hide 8 neighbouring SNPs at a time, as GBS does.

## Questions you should expect

**"Isn't a plain transformer just as good?"** At this cohort size its accuracy is 98.9%, yes — that is the ceiling. The layer's edge is beating the explicit-LD pipeline without an r-squared table, and recovering that table in attention more faithfully (r = 0.56 with the layer, 0.52 without).

**"Did you starve the baseline?"** We report the saturated control — every other site, long training budget — not only the usual top-8 partners.

**"Is it faster than computing r-squared?"** I would not claim wall-clock. The saving is structural: nothing to store or rebuild.

**"Why simulated data?"** So panel E has a ground-truth r-squared to compare to. A real GBS panel is the next step, with the same block missingness.
