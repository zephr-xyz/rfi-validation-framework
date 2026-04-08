# Converging on the Jammer: Dual-Satellite GPS Interference Localization from Space

**Two spaceborne sensors, two different physics, one GPS jammer — and a 4.33 km bull's-eye from orbit**

*Sean Gorman, Zephr*

---

On a January morning in 2026, a GPS jammer powered up near Shiraz, Iran. It was not the first, and it would not be the last — the Strait of Hormuz corridor has become one of the most persistently jammed airspaces on Earth. But this time, two satellites were watching from very different vantage points, and together they would demonstrate something new: that spaceborne sensors can localize a terrestrial GPS jammer to within a few kilometers, using physics alone.

This article presents the first direct comparison of CYGNSS (a NASA GNSS reflectometry constellation) and NISAR (an L-band synthetic aperture radar) for GPS jammer localization. The results challenge assumptions about which modality performs better — and reveal that the answer depends on a question most analysts forget to ask.

## The Setup: Known Jammer, Known Position

Validation requires ground truth. We identified a GPS jammer operating near 27.32°N, 52.87°E — approximately 50 km southwest of Shiraz — active on January 8 and January 20, 2026, with confirmed quiet periods on December 15 and 27, 2025. The jammer's position was established through independent signals intelligence.

This gave us a controlled experiment: two "jammer ON" dates and two "jammer OFF" baseline dates, with satellite coverage from both CYGNSS and NISAR spanning the full period.

## Two Satellites, Two Physics

**CYGNSS** (Cyclone Global Navigation Satellite System) is a constellation of eight microsatellites that measure GPS signals reflected off Earth's surface. Each spacecraft carries a delay-Doppler receiver that maps reflected signal power across a grid of delay and Doppler bins — the delay-Doppler map, or DDM. When a terrestrial jammer is active, it floods the GPS band with noise, elevating the DDM noise floor and suppressing the coherent surface reflection. The effect is detectable hundreds of kilometers from the jammer, creating a wide-area "footprint" in the reflected signal data.

**NISAR** (NASA-ISRO Synthetic Aperture Radar) operates an L-band SAR at 1.257 GHz — just 30 MHz from the GPS L2 frequency at 1.2276 GHz. When a GPS jammer's broadband emissions leak into NISAR's receive band, they create characteristic streaks in the SAR imagery: bright azimuth-aligned artifacts in the cross-polarization (HV) channel that point back toward the interference source. The bearing of each streak encodes the jammer's direction relative to the satellite's ground track.

The two sensors could hardly be more different. CYGNSS sees the jammer's effect on reflected GPS signals — an indirect measurement spread across hundreds of specular reflection points. NISAR sees the jammer's emissions directly in its own receiver — a precise measurement, but only along the satellite's narrow ground track. **Figure 1** shows both detection sets converging on the jammer location.

## CYGNSS: 785 Detections, 4.33 km Error

We processed all CYGNSS Level 1 data within 200 km of the jammer location on both ON and OFF dates. Four detection methods contributed observations:

- **DDM noise floor** (419 detections): The pre-computed `ddm_noise_floor` variable, calibrated against the thermal noise reference, proved the strongest discriminator. Near-jammer values exceeded 15,000 counts against a ~10,000 mean background.
- **Spatial noise grid** (299): A 10 km gridded analysis identified cells with anomalously elevated noise relative to adjacent cells.
- **SNR hole detection** (66): Coherent surface reflections were suppressed near the jammer, creating spatial "holes" in the SNR field.
- **NBRCS drop** (1): Surface reflectivity dropped approximately 16% near the jammer, though this method produced few threshold exceedances.

Across four DDM channels per spacecraft and multiple passes, this yielded 785 total anomalous observations on the jammer-ON dates.

The critical step was localization. A simple centroid of all 785 detection positions placed the jammer 32.1 km from truth — too many distant, low-SNR detections diluting the estimate. Instead, we fit a parametric 1/r² inverse-distance model:

$$I(r) = \frac{A}{r^2}$$

where *A* is a free amplitude parameter and *r* is the distance from a candidate jammer position. We optimized the jammer position and amplitude jointly using scipy's L-BFGS-B algorithm across all 785 observations weighted by intensity. The optimizer converged on a position 4.33 km from ground truth — a 27.7 km improvement over the centroid (**Figure 3**).

### The Baseline: Zero False Positives

On the jammer-OFF dates (December 15 and 27, 2025), the identical pipeline produced exactly zero detections. The same thresholds, the same geographic area, the same satellites — and a completely clean result. This is the smoking gun: the 785 detections are not sensor artifacts or geographic anomalies. They disappear when the jammer turns off (**Figure 2**).

## NISAR: 17 Detections, 6.26 km Error

NISAR's approach is fundamentally different. Rather than measuring hundreds of reflected signals across a wide area, it captures direct emissions in a narrow swath — but with far greater geometric precision.

We processed NISAR L2 GCOV (geocoded covariance) products from Track 157, Frame 15 (ascending) for three dates: the December 27 baseline and the January 8 and January 20 jammer-ON passes. The detection pipeline used eigenvalue decomposition of the polarimetric covariance matrix:

1. **λ₁ ratio thresholding**: In jammer-contaminated pixels, the dominant eigenvalue λ₁ of the 2×2 [HH, HV] covariance matrix rises sharply relative to the scene mean, indicating an unpolarized additive source.
2. **Cross-polarization ratio (HV/HH)**: GPS jammer emissions are unpolarized, elevating the HV channel disproportionately. Anomalous HV/HH ratios flag contaminated azimuth lines.
3. **Iterative outlier trimming**: Three rounds of 1.5σ clipping removed scattered false detections, leaving 17 high-confidence streak centroids.

With detections from two passes on different dates, we had two independent bearing lines. Each pass's streak centroids defined an azimuth-aligned cluster whose major axis pointed toward the jammer. A PCA fit to each cluster extracted the bearing: 308.1° from the January 8 pass and 316.2° from January 20. Their intersection — computed via scipy optimization of the angular residual — landed 6.26 km from ground truth (**Figure 4**).

The CEP (circular error probable, the radius containing 50% of repeated estimates) was 6.88 km — meaning if we ran this analysis on many similar jammers, half our estimates would fall within ~7 km.

## The Paradox: Who Wins?

At first glance, CYGNSS wins: 4.33 km versus 6.26 km. But this comparison hides a critical distinction that matters enormously for operational use.

CYGNSS's 4.33 km error came from a 1/r² model fit across 785 widely scattered observations. The CEP of those observations — a measure of the estimate's intrinsic confidence — was 127 km. The 4.33 km result is accurate, but *we could not have known that without ground truth*. The optimizer found a good minimum, but the broad scatter of input data provides no geometric constraint that would let an analyst assess confidence.

NISAR's 6.26 km error came from a bearing intersection of two tightly clustered detection sets. Its CEP was 6.88 km — meaning the confidence radius is almost exactly the actual error. An analyst looking at this result, with no knowledge of the true jammer position, would correctly estimate that the jammer is within ~7 km of the reported location.

**Figure 5** makes this visible: CYGNSS's CEP circle swallows the entire map, while NISAR's barely extends beyond the estimate itself.

This is the operational paradox: **CYGNSS achieves the best accuracy, but NISAR provides the best answer** — because an answer you can trust is worth more than an answer you cannot verify.

## Bayesian Fusion: Can We Get Both?

The obvious next question: can we combine CYGNSS's wide-area sensitivity with NISAR's geometric precision? We implemented four fusion strategies, all designed to work without ground truth:

1. **Bayesian Gaussian posterior**: Model each sensor's estimate as a 2D isotropic Gaussian with σ = CEP/1.1774. The posterior is the product of the two Gaussians — an analytical precision-weighted mean.
2. **NISAR-prior constrained 1/r²**: Re-run the CYGNSS optimizer with a Gaussian regularization term pulling toward the NISAR estimate, sweeping the regularization weight λ from 0.01 to 10.
3. **NISAR-proximity re-weighted 1/r²**: Apply a Gaussian kernel centered on the NISAR estimate to the CYGNSS detections before fitting, effectively upweighting observations consistent with the SAR result.
4. **Joint CEP-balanced**: Combine the CYGNSS gradient signal with NISAR cluster proximity, weighted by (σ_CYGNSS/σ_NISAR)².

The result was unanimous: all four methods converged to essentially the NISAR estimate. **Figure 6** shows why. The NISAR Gaussian (σ = 5.8 km) is 18× tighter than the CYGNSS Gaussian (σ = 108 km). In precision-weighted terms, NISAR contributes 343× more information than CYGNSS. The Bayesian posterior is mathematically dominated by the tighter distribution.

The fused result: 4.91 km error, 6.87 km CEP — slightly better than NISAR alone in absolute terms (because the CYGNSS posterior nudges the estimate imperceptibly toward truth), but operationally indistinguishable.

This is not a failure of fusion. It is fusion working correctly. When one sensor's confidence is 18× tighter than the other's, the principled answer is to trust the precise sensor. The CYGNSS data adds marginal value at the current precision ratio — but would contribute significantly if its CEP could be reduced, for instance through multi-pass accumulation or constellation-level filtering.

## Operational Implications

### For CYGNSS
CYGNSS excels at **detection and wide-area search**. Its 785 detections across a 200 km radius, with zero false positives on baseline dates, make it an ideal trigger: a CYGNSS anomaly tells you *a jammer exists in this region*. The 1/r² fit can narrow the location to single-digit kilometers — sometimes remarkably well — but the analyst cannot assess how well without external validation. CYGNSS's true operational role is as a cueing sensor.

### For NISAR
NISAR provides **geometric localization**. With just two passes over an active jammer, the bearing intersection achieved 6.26 km accuracy with a verifiable 6.88 km confidence radius. The limitation is coverage: NISAR's 12-day repeat cycle and fixed ground track mean the jammer must be active when the satellite passes overhead, and at least two passes with different bearings are needed for triangulation.

### For Fusion
The current precision mismatch (343:1) means fusion adds little beyond what NISAR provides alone. But this ratio is not fixed. Strategies to improve CYGNSS precision — multi-week accumulation, constellation coherent processing, or integration with other GNSS-R missions — could bring the ratio closer to parity, at which point fusion would yield genuinely complementary information.

### For the Adversary
These results demonstrate that GPS jammers operating in contested airspace are observable and localizable from orbit using openly available civilian satellite data. The 4.33 km CYGNSS result is approximately 2× better than the published state of the art for GNSS-R jammer localization (~9 km grid resolution, Chew et al. 2023), and the NISAR bearing intersection approach has not been previously demonstrated for jammer geolocation.

## Context and Prior Work

CYGNSS-based RFI detection builds on work by Chew et al. (2023), who demonstrated grid-level jammer detection at approximately 9 km resolution using DDM noise floor anomalies. Our 1/r² parametric fit extends this from detection to localization, achieving sub-5 km accuracy by exploiting the physics of signal power decay.

At the other end of the precision spectrum, Murrian et al. (2021) demonstrated ~220 m jammer localization using ISS-mounted Doppler measurements of raw intermediate-frequency (IF) data. This approach achieves an order of magnitude better precision than our methods but requires specialized hardware and raw signal access not available on current operational satellites.

The NISAR bearing intersection approach demonstrated here is, to our knowledge, the first published use of L-band SAR RFI streaks for jammer triangulation. The key insight is that NISAR's proximity to GPS L2 (just 30 MHz separation) makes it an unintentional but effective GPS interference sensor.

## Methods and Data

All analysis code is open source at github.com/zephr-xyz/rfi-validation-framework. CYGNSS data were accessed via NASA's earthaccess API using streaming h5netcdf (zero local storage). NISAR GCOV products were obtained from the ASF DAAC. Processing was performed on AWS EC2 spot instances.

**CYGNSS processing**: All Level 1 DDM data within 200 km of the jammer, all 4 DDM channels per spacecraft, jammer-ON dates (Jan 8, 20 2026) and baseline dates (Dec 15, 27 2025). Detection thresholds calibrated against scene statistics. 1/r² optimization via scipy L-BFGS-B with multi-start initialization.

**NISAR processing**: Level 2 GCOV products, Track 157 Frame 15, ascending passes. Eigenvalue decomposition on HH/HV covariance. PCA bearing extraction with iterative 3-round 1.5σ outlier trimming. Bearing intersection via angular residual minimization.

**Fusion**: Bayesian posterior computed analytically as the precision-weighted mean of two 2D isotropic Gaussians. σ derived from CEP via the Rayleigh distribution relationship σ = CEP/1.1774.

## Summary

| Metric | CYGNSS | NISAR | Fused |
|--------|--------|-------|-------|
| Detections | 785 | 17 | 802 |
| Localization error | 4.33 km | 6.26 km | 4.91 km |
| CEP (50%) | 127 km | 6.88 km | 6.87 km |
| Method | 1/r² fit | Bearing intersection | Bayesian posterior |
| Best for | Wide-area search | Precision localization | Combined (future) |

**Figure 7** presents the full results dashboard.

Two satellites, two physics, one jammer. CYGNSS sees everything but knows its position loosely. NISAR sees little but knows exactly where it looked. Together — or rather, in sequence — they offer a complete operational picture: detect with CYGNSS, localize with NISAR, and wait for the precision gap to close before expecting fusion to add value.

The jammer near Shiraz is still there. The satellites are still watching.

---

*Sean Gorman is CEO of Zephr. Contact: sean@zephr.xyz*

### References

Chew, C., Shah, R., Zuffada, C., et al. (2023). "Demonstrating CYGNSS as a Tool for Detecting GNSS Interference on a Global Scale." *IEEE Journal of Selected Topics in Applied Earth Observations and Remote Sensing*.

Murrian, M.J., Narula, L., Iannucci, P.A., et al. (2021). "GNSS Interference Monitoring from Low Earth Orbit." *Navigation: Journal of the Institute of Navigation*, 68(1).

NASA JPL. (2024). "NISAR L-band SAR Technical Specifications." NASA/ISRO SAR Mission Documentation.

Closas, P., Fernández-Prades, C. (2023). "GNSS Interference Detection and Mitigation: A Survey." *Signal Processing*, 206.
