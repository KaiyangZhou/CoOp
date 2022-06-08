import os
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt


save_dir = "main_curves"
if not os.path.exists(save_dir):
    os.makedirs(save_dir)

path = "Results.xlsx"  # this is the excel file containing the results (like the one we released)
file = pd.read_excel(path, sheet_name="imcls_fewshot")

datasets = [
    "OxfordPets", "Flowers102", "FGVCAircraft", "DTD",
    "EuroSAT", "StanfordCars", "Food101", "SUN397",
    "Caltech101", "UCF101", "ImageNet"
]

shots = [1, 2, 4, 8, 16]

COLORS = {
    "zs": "C4",
    "linear": "C4",
    "ours_v16_end": "C0",
    "ours_v16_mid": "C2",
    "ours_v16_end_csc": "C1",
    "ours_v16_mid_csc": "C3"
}
MS = 3
ALPHA = 1
plt.rcParams.update({"font.size": 12})

average = {
    "zs": 0.,
    "ours_v16_end": np.array([0., 0., 0., 0., 0.]),
    "ours_v16_mid": np.array([0., 0., 0., 0., 0.]),
    "ours_v16_end_csc": np.array([0., 0., 0., 0., 0.]),
    "ours_v16_mid_csc": np.array([0., 0., 0., 0., 0.]),
    "linear": np.array([0., 0., 0., 0., 0.])
}

for dataset in datasets:
    print(f"Processing {dataset} ...")

    zs = file[dataset][0]

    ours_v16_end = file[dataset][2:7]
    ours_v16_end = [float(num) for num in ours_v16_end]

    ours_v16_mid = file[dataset][7:12]
    ours_v16_mid = [float(num) for num in ours_v16_mid]

    ours_v16_end_csc = file[dataset][12:17]
    ours_v16_end_csc = [float(num) for num in ours_v16_end_csc]

    ours_v16_mid_csc = file[dataset][17:22]
    ours_v16_mid_csc = [float(num) for num in ours_v16_mid_csc]

    linear = file[dataset][22:27]
    linear = [float(num) for num in linear]

    average["zs"] += zs
    average["ours_v16_end"] += np.array(ours_v16_end)
    average["ours_v16_mid"] += np.array(ours_v16_mid)
    average["ours_v16_end_csc"] += np.array(ours_v16_end_csc)
    average["ours_v16_mid_csc"] += np.array(ours_v16_mid_csc)
    average["linear"] += np.array(linear)

    # Plot
    values = [zs]
    values += linear
    values += ours_v16_end
    values += ours_v16_mid
    values += ours_v16_end_csc
    values += ours_v16_mid_csc
    val_min, val_max = min(values), max(values)
    diff = val_max - val_min
    val_bot = val_min - diff*0.05
    val_top = val_max + diff*0.05

    fig, ax = plt.subplots()
    ax.set_facecolor("#EBEBEB")

    ax.set_xticks([0] + shots)
    ax.set_xticklabels([0] + shots)
    ax.set_xlabel("Number of labeled training examples per class")
    ax.set_ylabel("Score (%)")
    ax.grid(axis="x", color="white", linewidth=1)
    ax.axhline(zs, color="white", linewidth=1)
    ax.set_title(dataset)
    ax.set_ylim(val_bot, val_top)

    ax.plot(
        0, zs,
        marker="*",
        markersize=MS*1.5,
        color=COLORS["zs"],
        alpha=ALPHA
    )
    ax.plot(
        shots, ours_v16_end,
        marker="o",
        markersize=MS,
        color=COLORS["ours_v16_end"],
        label="CLIP + CoOp ($M\!=\!16$, end)",
        alpha=ALPHA
    )
    ax.plot(
        shots, ours_v16_mid,
        marker="o",
        markersize=MS,
        color=COLORS["ours_v16_mid"],
        label="CLIP + CoOp ($M\!=\!16$, mid)",
        alpha=ALPHA
    )
    ax.plot(
        shots, ours_v16_end_csc,
        marker="o",
        markersize=MS,
        color=COLORS["ours_v16_end_csc"],
        label="CLIP + CoOp ($M\!=\!16$, end, CSC)",
        alpha=ALPHA
    )
    ax.plot(
        shots, ours_v16_mid_csc,
        marker="o",
        markersize=MS,
        color=COLORS["ours_v16_mid_csc"],
        label="CLIP + CoOp ($M\!=\!16$, mid, CSC)",
        alpha=ALPHA
    )
    ax.plot(
        shots, linear,
        marker="o",
        markersize=MS,
        color=COLORS["linear"],
        label="Linear probe CLIP",
        linestyle="dotted",
        alpha=ALPHA
    )

    ax.text(-0.5, zs-diff*0.11, "Zero-shot\nCLIP", color=COLORS["zs"])
    ax.legend(loc="lower right")

    fig.savefig(f"{save_dir}/{dataset}.pdf", bbox_inches="tight")


# Plot
average = {k: v/len(datasets) for k, v in average.items()}
zs = average["zs"]
linear = list(average["linear"])
ours_v16_end = list(average["ours_v16_end"])
ours_v16_mid = list(average["ours_v16_mid"])
ours_v16_end_csc = list(average["ours_v16_end_csc"])
ours_v16_mid_csc = list(average["ours_v16_mid_csc"])

values = [zs]
values += linear
values += ours_v16_end
values += ours_v16_mid
values += ours_v16_end_csc
values += ours_v16_mid_csc
val_min, val_max = min(values), max(values)
diff = val_max - val_min
val_bot = val_min - diff*0.05
val_top = val_max + diff*0.05

fig, ax = plt.subplots()
ax.set_facecolor("#EBEBEB")

ax.set_xticks([0] + shots)
ax.set_xticklabels([0] + shots)
ax.set_xlabel("Number of labeled training examples per class")
ax.set_ylabel("Score (%)")
ax.grid(axis="x", color="white", linewidth=1)
ax.axhline(zs, color="white", linewidth=1)
ax.set_title("Average over 11 datasets", fontweight="bold")
ax.set_ylim(val_bot, val_top)

ax.plot(
    0, zs,
    marker="*",
    markersize=MS*1.5,
    color=COLORS["zs"],
    alpha=ALPHA
)
ax.plot(
    shots, ours_v16_end,
    marker="o",
    markersize=MS,
    color=COLORS["ours_v16_end"],
    label="CLIP + CoOp ($M\!=\!16$, end)",
    alpha=ALPHA
)
ax.plot(
    shots, ours_v16_mid,
    marker="o",
    markersize=MS,
    color=COLORS["ours_v16_mid"],
    label="CLIP + CoOp ($M\!=\!16$, mid)",
    alpha=ALPHA
)
ax.plot(
    shots, ours_v16_end_csc,
    marker="o",
    markersize=MS,
    color=COLORS["ours_v16_end_csc"],
    label="CLIP + CoOp ($M\!=\!16$, end, CSC)",
    alpha=ALPHA
)
ax.plot(
    shots, ours_v16_mid_csc,
    marker="o",
    markersize=MS,
    color=COLORS["ours_v16_mid_csc"],
    label="CLIP + CoOp ($M\!=\!16$, mid, CSC)",
    alpha=ALPHA
)
ax.plot(
    shots, linear,
    marker="o",
    markersize=MS,
    color=COLORS["linear"],
    label="Linear probe CLIP",
    linestyle="dotted",
    alpha=ALPHA
)

ax.text(-0.5, zs-diff*0.11, "Zero-shot\nCLIP", color=COLORS["zs"])
ax.legend(loc="lower right")

fig.savefig(f"{save_dir}/average.pdf", bbox_inches="tight")
