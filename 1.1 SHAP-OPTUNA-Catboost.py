
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import matplotlib
import os ,re
import warnings
import optuna
from statsmodels.nonparametric.smoothers_lowess import lowess
import shap
from sklearn.metrics import r2_score, mean_squared_error, mean_absolute_error
import matplotlib.colors as mcolors
from catboost import CatBoostRegressor
from sklearn.model_selection import train_test_split, cross_val_score
import matplotlib.gridspec as gridspec
import pickle

##====0. figure background setting===============
matplotlib.use('TkAgg')
plt.rcParams['font.serif'] = ['Times New Roman']
plt.rcParams['axes.unicode_minus'] = False # Configure matplotlib to display negative signs correctly
warnings.filterwarnings("ignore", category=RuntimeWarning)
# ========== General Utility Functions ===============================================
def apply_plot_styles(ax, tick_width=1.5, tick_length=6, spine_width=1.5):
    # Set the thickness of all four picture frames
    for spine in ax.spines.values():
        spine.set_linewidth(spine_width)
    ax.tick_params(axis='both', which='major', direction='in',
                   width=tick_width, length=tick_length)
def sanitize_filename(name): # clean invalid characters from filenames.
    return re.sub(r'[\\/*?:"<>|]', '_', name) # 使用正则表达式将Windows文件名中的非法字符替换为下划线
def bootstrap_lowess_ci(x, y, n_boot=200, frac=0.5, ci_level=0.95):
    """ Calculate confidence intervals for lowess smoothing using the bootstrap method """
    if len(x) < 10: return None, None
    boot_lines = [] # Initialize a list
    x_range = np.linspace(x.min(), x.max(), 100) #
    for _ in range(n_boot): #
        sample_indices = np.random.choice(len(x), len(x), replace=True)
        x_sample, y_sample = x.iloc[sample_indices], y[sample_indices]
        sorted_indices = np.argsort(x_sample)
        x_sorted, y_sorted = x_sample.iloc[sorted_indices].values, y_sample[sorted_indices]
        if len(np.unique(x_sorted)) < 2: continue
        smoothed = lowess(y_sorted, x_sorted, frac=frac)
        interp_func = np.interp(x_range, smoothed[:, 0], smoothed[:, 1])
        boot_lines.append(interp_func)
    if not boot_lines: return None, None
    sorted_indices_orig = np.argsort(x)
    x_sorted_orig, y_sorted_orig = x.iloc[sorted_indices_orig].values, y[sorted_indices_orig]
    main_smoothed = lowess(y_sorted_orig, x_sorted_orig, frac=frac)
    boot_lines_arr = np.array(boot_lines)
    alpha = (1 - ci_level) / 2 # Calculate the alpha value corresponding to the confidence level
    lower_bound, upper_bound = np.quantile(boot_lines_arr, alpha, axis=0), np.quantile(boot_lines_arr, 1 - alpha,axis=0)
    return main_smoothed, (x_range, lower_bound, upper_bound)
def find_and_plot_crossings(ax, x_curve, y_curve, color, base_y_offset=0.9):
    """
    Define a function to find and plot the intersection points of a curve with the y=0 line (threshold).
    """
    sign_changes = np.where(np.diff(np.sign(y_curve)))[0]
    for i, k in enumerate(sign_changes):
        x1, y1, x2, y2 = x_curve[k], y_curve[k], x_curve[k + 1], y_curve[k + 1]
        if (y2 - y1) == 0: continue
        x_root = x1 - y1 * (x2 - x1) / (y2 - y1)
        ax.axvline(x=x_root, color=color, linestyle='--', linewidth=1)
        y_text_position = ax.get_ylim()[1] * (base_y_offset - (i % 2) * 0.1)
        ax.text(x_root, y_text_position, f' {x_root:.2f} ', color='white', backgroundcolor=color, ha='center',
                va='center', fontsize=9, bbox=dict(facecolor=color, edgecolor='none', pad=1))
def find_roots(x_curve, y_curve):
    """  Find all points where a curve intersects the y-axis (i.e., the roots of the equation)  """
    roots = []
    sign_changes = np.where(np.diff(np.sign(y_curve)))[0]
    for k in sign_changes:
        x1, y1, x2, y2 = x_curve[k], y_curve[k], x_curve[k + 1], y_curve[k + 1]
        if (y2 - y1) == 0: continue
        x_root = x1 - y1 * (x2 - x1) / (y2 - y1)
        roots.append(x_root)
    return roots
def plot_shap_dependence(feature_name, x_values, shap_values_for_feature, save_folder, custom_annotation=None):
    # the SHAP dependency map for a single feature
    print(f"  -> 正在绘制特征: {feature_name} ...")
    fig_dep, ax1 = plt.subplots(figsize=(8, 6), dpi=150)
    ax2 = ax1.twinx()
    ax2.patch.set_alpha(0)
    counts, bin_edges = np.histogram(x_values, bins=30)
    bin_centers = (bin_edges[:-1] + bin_edges[1:]) / 2
    bin_width = bin_edges[1] - bin_edges[0]
    ax1.bar(bin_centers, counts, width=bin_width * 0.6, align='center', color='#4B0082', alpha=0.3, label='Distribution')
    ax1.set_ylabel('Distribution', fontsize=12)
    ax1.set_ylim(0, counts.max() * 1.1)
    ax2.scatter(x_values, shap_values_for_feature, alpha=0.3, s=25, color='#00008B', label='Sample', zorder=2)
    if len(x_values) > 1:
        main_fit, ci_data = bootstrap_lowess_ci(x_values, shap_values_for_feature, frac=0.3)
        if main_fit is not None and ci_data is not None:
            ax2.plot(main_fit[:, 0], main_fit[:, 1], color='#9400D3', lw=2, label='LOWESS Fit', zorder=4)
            ax2.fill_between(ci_data[0], ci_data[1], ci_data[2], color='#D3D3D3', alpha=0.15, label='95%CI')
            ax2.axhline(0, color='black', linestyle='--', lw=1, zorder=1)
            find_and_plot_crossings(ax2, main_fit[:, 0], main_fit[:, 1], 'black')
    ax2.set_ylabel('SHAP value', fontsize=12)
    y_max = np.abs(shap_values_for_feature).max() * 1.15
    if y_max < 1e-6: y_max = 1
    ax2.set_ylim(-y_max, y_max)
    ax1.set_xlabel(f'{feature_name}', fontsize=12)
    if custom_annotation and isinstance(custom_annotation, dict):
        text = custom_annotation.get('text', '')
        x_pos = custom_annotation.get('x', 0.95)
        y_pos = custom_annotation.get('y', 0.95)
        props = {'ha': custom_annotation.get('ha', 'right'), 'va': custom_annotation.get('va', 'top'),
                 'fontsize': custom_annotation.get('fontsize', 12), 'color': custom_annotation.get('color', 'darkred'),
                 'fontweight': custom_annotation.get('fontweight', 'bold')}
        ax1.text(x_pos, y_pos, text, transform=ax1.transAxes, **props)
    h1, l1 = ax1.get_legend_handles_labels()
    h2, l2 = ax2.get_legend_handles_labels()
    ax2.legend(h2 + h1, l2 + l1, loc='upper right', fontsize=10)
    apply_plot_styles(ax1)
    apply_plot_styles(ax2)
    sanitized_feature_name = sanitize_filename(feature_name)
    output_filename = f"dependence_plot_{sanitized_feature_name}.png"
    full_path = os.path.join(save_folder, output_filename)
    fig_dep.savefig(full_path, dpi=200, bbox_inches='tight')
    plt.close(fig_dep)
def apply_spine_style(axes, linewidth=0.5):
    """a uniform border style"""
    for ax in axes:
        for spine in ax.spines.values():
            spine.set_visible(True)
            spine.set_linewidth(linewidth)
def plot_advanced_interaction_NEW(primary_feature_name, interacting_feature_name, x_values,
                                  interaction_feature_values, shap_interaction_slice,ax=None,subplot_idx=None):
    plt.rcParams['font.family'] = 'Times New Roman'
    plt.rcParams['mathtext.fontset'] = 'stix'

    if ax is None:
        fig, ax = plt.subplots(figsize=(8, 6), dpi=150)
        create_new_fig = True
    else:
        create_new_fig = False
        fig = ax.get_figure()

    cmap = mcolors.LinearSegmentedColormap.from_list("custom_cmap",
                                                     ["#1D610D", "#69AA34", "#E2ECC2", "#FDE2EB", "#DE689C", "#9D014D"])
    points = ax.scatter(x_values, shap_interaction_slice, c=interaction_feature_values,
                        cmap=cmap, alpha=1, s=15,  label='sample')
    median_val = interaction_feature_values.median()
    low_mask, high_mask = interaction_feature_values <= median_val, interaction_feature_values > median_val
    groups = {
        'low': {'mask': low_mask, 'color': 'blue', 'offset': 0.9,
                'label': f' {interacting_feature_name} ≤ {median_val:.2f}'},
        'high': {'mask': high_mask, 'color': 'red', 'offset': 0.8,
                 'label': f' {interacting_feature_name} > {median_val:.2f}'}
    }

    fits, roots = {}, {}
    for name, info in groups.items():
        x_group, shap_group = x_values[info['mask']], shap_interaction_slice[info['mask']]
        if len(x_group) < 10:
            continue
        main_fit, ci_data = bootstrap_lowess_ci(x_group, shap_group)
        if main_fit is not None and ci_data is not None:
            ax.plot(main_fit[:, 0], main_fit[:, 1], color=f'dark{info["color"]}', lw=2,
                    label=info['label'],zorder=2)
            ax.fill_between(ci_data[0], ci_data[1], ci_data[2], color=info['color'], zorder=2,alpha=0.15)
            fits[name] = main_fit
            roots[name] = find_roots(main_fit[:, 0], main_fit[:, 1])

    if 'low' in roots and 'high' in roots:
        tolerance = (x_values.max() - x_values.min()) * 0.05
        for r_low in roots['low']:
            for r_high in roots['high']:
                if abs(r_low - r_high) < tolerance:
                    avg_root = (r_low + r_high) / 2
                    ax.axvline(x=avg_root, color='black', linestyle='--', linewidth=1)
                    ax.text(avg_root, ax.get_ylim()[1] * 0.9, f' {avg_root:.2f} ', color='black',
                            ha='center', va='center', fontsize=8,
                            fontfamily='Times New Roman')

    ax.set_xlabel(f'{primary_feature_name}', fontsize=12, fontfamily='Times New Roman')
    ax.set_ylabel(f'SHAP Interaction Value', fontsize=12, fontfamily='Times New Roman')
    ax.axhline(0, color='black', linestyle='--', lw=1, zorder=0)

    y_max_abs = np.abs(shap_interaction_slice).max() * 1.1
    ax.set_ylim(-y_max_abs if y_max_abs > 1e-6 else -1, y_max_abs if y_max_abs > 1e-6 else 1)

    ax.legend(loc='best', fontsize=11, prop={'family': 'Times New Roman'})

    if not create_new_fig:
        ax_pos = ax.get_position()

        cbar_width = 0.008
        cbar_pad = 0.0035

        cbar_left = ax_pos.x1 + cbar_pad
        cbar_bottom = ax_pos.y0
        cbar_height = ax_pos.height

        cbar_ax = fig.add_axes([cbar_left, cbar_bottom, cbar_width, cbar_height])
        cbar = fig.colorbar(points, cax=cbar_ax)
        cbar.set_label(f"Value of {interacting_feature_name}", size=12, fontfamily='Times New Roman')

        for label in cbar_ax.get_yticklabels():
            label.set_fontfamily('Times New Roman')
            label.set_fontsize(12)
    else:
        cbar = plt.colorbar(points, ax=ax)
        cbar.set_label(f"Value of {interacting_feature_name}", size=12, fontfamily='Times New Roman')
    apply_plot_styles(ax)

    for label in ax.get_xticklabels() + ax.get_yticklabels():
        label.set_fontfamily('Times New Roman')
        label.set_fontsize(12)

    return fig, ax

##=====1. input data and output path============
target_column_name = 'OBS LH'
output_main_folder = r'F:\LHSH\XLUcode\Xgboost0701\20250915\SHAP'

print("--- start ---")
print("load data")
df0 = pd.read_excel('LH_site.xlsx',engine='openpyxl', index_col=0)
df = df0[df0['time'] <= '2020-01-01']
X = df0.iloc[:,4:18]
y= df0.iloc[:,-1]
feature_names = X.columns.tolist()

# ===== 2. Dataset Splitting, Hyperparameter Search, and Model Training (using the CatBoost model) =====
# split the dataset into training and testing sets.
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)
print("\nCatBoost model search good parameters...")
# Define the Optuna objective function, using R² as the objective
def objective(trial):
    param = {
        "iterations": trial.suggest_int("iterations", 100, 150),
        "depth": trial.suggest_int("depth", 4, 10),
        "learning_rate": trial.suggest_float("learning_rate", 0.01, 0.3, log=True),
        "l2_leaf_reg": trial.suggest_float("l2_leaf_reg", 1, 10),
        "random_strength": trial.suggest_float("random_strength", 0.1, 10),
        "bagging_temperature": trial.suggest_float("bagging_temperature", 0, 1),
        "min_data_in_leaf": trial.suggest_int("min_data_in_leaf", 1, 20),
        "verbose": 0,
        "loss_function": "RMSE",
        "random_state": 42,
    }
    model = CatBoostRegressor(**param)
    scores = cross_val_score(model, X_train, y_train, cv=5, scoring="r2")
    return scores.mean()

# Parameter Optimization Using Optuna
study = optuna.create_study(direction="maximize")  # maximize R²
study.optimize(objective, n_trials=30)

# print optimal parameters
print("Best trial:")
print(study.best_trial.params)

best_params = study.best_trial.params
model = CatBoostRegressor(**best_params)
model.fit(X_train, y_train)
y_train_pred = model.predict(X_train)
y_test_pred = model.predict(X_test)
#Calculate the metrics for the training set
r2_train = r2_score(y_train, y_train_pred)
rmse_train = np.sqrt(mean_squared_error(y_train, y_train_pred))
mae_train = mean_absolute_error(y_train, y_train_pred)
# Calculate the metrics for the testing set
r2_test = r2_score(y_test, y_test_pred)
rmse_test = np.sqrt(mean_squared_error(y_test, y_test_pred))
mae_test = mean_absolute_error(y_test, y_test_pred)
print(f"\ntrain: R2={r2_train:.4f}, RMSE={rmse_train:.4f}, MAE={mae_train:.4f}")
print(f"test: R2={r2_test:.4f}, RMSE={rmse_test:.4f}, MAE={mae_test:.4f}")

#=======3.0 Use the trained model for prediction=========================
y_initial_pred = model.predict(X)
# save the output
initial_results = pd.DataFrame({
    'Initial_X_Index': X.index,  #
    'Predicted_y': y_initial_pred
})
# Save results to an Excel file
initial_results.to_excel(r'OPTUNA_CAT_LH_Initial_Predictions.xlsx', index=False)

# save the trained model
with open("CatBoost_best_model_SH.pkl", "wb") as f:
    pickle.dump(model, f) # 保存模型

# ===== 4. SHAP analysis (based on X_test)   =============================
explainer = shap.Explainer(model) # Create a SHAP explainer using the best trained model
shap_values_obj = explainer(X_test)
shap_values = shap_values_obj.values
print("SHAP value finish")

shap_interaction_values = explainer.shap_interaction_values(X_test)
print("SHAP interaction finish")
mean_shap = np.abs(shap_values).mean(axis=0)

shap_df = pd.DataFrame({
    "feature": feature_names,
    "mean_shap": mean_shap
}).sort_values("mean_shap", ascending=False)
sorted_features = shap_df["feature"].values

X_test_sorted = X_test[sorted_features]
orig_index = [feature_names.index(f) for f in sorted_features]
shap_values_sorted = shap_values[:, orig_index]
shap_interaction_values_sorted = shap_interaction_values[:, orig_index][:, :, orig_index]

# ===== 4.1. SHAP Overview Plot==================================================
print("\nDraw SHAP summuary plot ...")
fig = plt.figure(figsize=(12,5), dpi=300)
plt.rcParams['font.family'] = 'Times New Roman'
plt.rcParams['font.size'] = 10
ax_sw = fig.add_axes([0.45, 0.11, 0.59, 0.77])
ax_bar = ax_sw.twiny()
ax_bar.set_zorder(0)
ax_sw.set_zorder(1)
ax_sw.patch.set_alpha(0)
y_pos = np.arange(len(sorted_features))[::-1]
bars = ax_bar.barh(y=y_pos, width=shap_df["mean_shap"].values, height=0.6, color="#6495ED", alpha=0.5, edgecolor="none",zorder=0)

# Calculate the SHAP value percentage for each feature
total_shap = shap_df["mean_shap"].sum()
shap_percentages = (shap_df["mean_shap"].values / total_shap) * 100

xlim_bar = shap_df["mean_shap"].values.max() * 1.15

for i, (bar, mean_val, percentage) in enumerate(zip(bars, shap_df["mean_shap"].values, shap_percentages)):
    width = bar.get_width()
    ax_bar.text(0.01 + xlim_bar * 0.01, bar.get_y() + bar.get_height()/2,
                f'{mean_val:.3f} ({percentage:.1f}%)',
                va='center', ha='left', fontsize=10, fontfamily='Times New Roman')

ax_bar.set_xlim(0, xlim_bar)
xticks_bar = np.linspace(0, xlim_bar, 5)
ax_bar.set_xticks(xticks_bar)
ax_bar.set_xticklabels([f"{x:.2f}" for x in xticks_bar])
ax_bar.set_xlabel("Mean (|SHAP| value)", fontsize=10)
ax_bar.set_yticks(y_pos)
max_abs_shap = np.abs(shap_values_sorted).max()
xlim_sw = max_abs_shap * 1.1
ax_sw.set_xlim(-xlim_sw, xlim_sw)
sw_xticks = np.linspace(-xlim_sw, xlim_sw, 5)
ax_sw.set_xticks(sw_xticks)
ax_sw.set_xticklabels([f"{x:.2f}" for x in sw_xticks])

expl_main = shap.Explanation(
    values=shap_values_sorted,
    data=X_test_sorted.values,
    feature_names=list(sorted_features),
    base_values=shap_values_obj.base_values[0]
)

shap.plots.beeswarm(expl_main, max_display=len(sorted_features), ax=ax_sw, show=False,
                        color='RdBu_r',plot_size=None)
ax_sw.set_xlabel("SHAP value (impact on model output LH)", fontsize=9)
ax_sw.set_yticks(y_pos)
ax_sw.set_yticklabels(sorted_features, fontsize=10)
ax_sw.tick_params(axis='y', length=4)
# ===== Remove the image frames at the top and right side. =====
border_width = 0.5
for ax in [ax_sw, ax_bar]:
    for spine in ['top', 'right']:
        ax.spines[spine].set_visible(True)
        ax.spines[spine].set_linewidth(border_width)

# =======================================
apply_plot_styles(ax_sw)
apply_plot_styles(ax_bar)
ax_bar.tick_params(axis='y', length=0)

combined_image_path = os.path.join(output_main_folder, 'LH_combined_shap_summary_plot.png')
plt.savefig(combined_image_path, dpi=208, bbox_inches='tight')



# ========4.2 Overview of SHAP Dependency Graph==============

print("SHAP dependance plot is strating! ")
top_6_features = shap_df['feature'].head(6).tolist()
fig=plt.figure(figsize=(10, 8))
plt.rcParams['font.family'] = 'Times New Roman'
plt.rcParams['font.size'] = 11
aesthetic_params = {
    'suptitle_size': 11,
    'ax_label_size': 10,
    'tick_label_size': 11,
    'legend_size':11,
    'cbar_label_size': 11,
    'summary_cbar_width': 0.015,
    'summary_cbar_height_shrink': 1.0,
    'summary_cbar_pad': 0.01,
    'dep_cbar_width': 0.005,
    'dep_cbar_height_shrink': 1.0,
    'dep_cbar_pad': 0.005,
    'dep_cbar_tick_length': 1,
    'grid_wspace': 0.15,
    'grid_hspace': 0.30,
    'global_cbar_bottom_pad': 0.12
}
gs = gridspec.GridSpec(3, 2,wspace=aesthetic_params['grid_wspace'],hspace=aesthetic_params['grid_hspace'])  #
cmap = plt.get_cmap("viridis")

axes_scatter = []
for i in range(3):
    for j in range(2):
        axes_scatter.append(fig.add_subplot(gs[i, j]))

scatter_plots = []

# Iteratively draw each dependency graph
for i, feature in enumerate(top_6_features):
    ax = axes_scatter[i]
    feature_idx = X_test.columns.get_loc(feature)
    x_data = X_test[feature]
    y_data = shap_values_obj.values[:, feature_idx]
    color_data = y_test
    # =====Scatter Plot====
    scatter = ax.scatter(x_data, y_data, c=color_data, cmap=cmap, s=20, alpha=0.8)
    scatter_plots.append(scatter)
    if len(y_data) > 1:
        main_fit, ci_data = bootstrap_lowess_ci(x_data, y_data, frac=0.3)
        if main_fit is not None and ci_data is not None:
            ax.plot(main_fit[:, 0], main_fit[:, 1], color='#9400D3', lw=2, label='LOWESS Fit', zorder=4)
            ax.fill_between(ci_data[0], ci_data[1], ci_data[2], color='#D3D3D3', alpha=0.65, label='95%CI',zorder=2)
            ax.axhline(0, color='black', linestyle='--', lw=1, zorder=1)
            find_and_plot_crossings(ax, main_fit[:, 0], main_fit[:, 1], 'black')

    ax.set_xlabel(feature, fontsize=aesthetic_params['ax_label_size'], labelpad=2)
    ax.set_ylabel(f"SHAP value of {feature}", fontsize=9, labelpad=-3)

    label_letter = chr(97 + i)  #65 is the ASCII code for ‘A’, generating A, B, C...
    ax.set_title(f'({label_letter})',
                 fontsize=12, fontfamily='Times New Roman', pad=5, loc='left')
    ax.tick_params(axis='both', which='major', labelsize=aesthetic_params['tick_label_size'])


if scatter_plots:
    fig.tight_layout()
    fig.subplots_adjust(bottom=aesthetic_params['global_cbar_bottom_pad'])  # 为颜色条留出底部空间

    cax_global = fig.add_axes([
        0.15,
        0.04,
        0.7,
        0.015
    ])
    cbar_global = fig.colorbar(scatter_plots[0], cax=cax_global, orientation='horizontal')
    cbar_global.set_label(target_column_name, fontsize=aesthetic_params['ax_label_size'],labelpad=1)
    cbar_global.outline.set_visible(False)

    cbar_global.ax.tick_params(
        axis='x',
        length=aesthetic_params['dep_cbar_tick_length'],
        labelsize=aesthetic_params['tick_label_size']
    )
interaction_summary_plot_path = os.path.join(output_main_folder, 'LH_shap_dependence_summary_plot.png')
plt.savefig(interaction_summary_plot_path, dpi=300, bbox_inches='tight')


# ============4.3 SHAP Interaction Overview Diagram==============================
output_folder_interactions = os.path.join(output_main_folder, 'interaction_values_per_feature_LH')
os.makedirs(output_folder_interactions, exist_ok=True)
for i, primary_feature_name in enumerate(sorted_features):
    interaction_data = {}
    for j, secondary_feature_name in enumerate(sorted_features):
        if i == j: continue
        column_name = f"interaction_{sanitize_filename(primary_feature_name)}_vs_{sanitize_filename(secondary_feature_name)}"
        interaction_values_for_pair = shap_interaction_values_sorted[:, i, j]
        interaction_data[column_name] = interaction_values_for_pair
    feature_interaction_df = pd.DataFrame(interaction_data)
    csv_filename = f"interactions_for_{sanitize_filename(primary_feature_name)}.csv"
    full_path = os.path.join(output_folder_interactions, csv_filename)
    feature_interaction_df.to_csv(full_path, index=False, encoding='utf-8-sig')

##=======================

if len(sorted_features) > 0:
    first_feature_name = sorted_features[0]
    first_feature_csv_path = os.path.join(output_folder_interactions,
                                          f"interactions_for_{sanitize_filename(first_feature_name)}.csv")
    if os.path.exists(first_feature_csv_path):
        preview_df = pd.read_csv(first_feature_csv_path)
        print(preview_df.head())

# Set MERRA2 as the primary feature
primary_feature_name = "LH_JRA55"

fig, axes = plt.subplots(3, 3, figsize=(15, 12))
plt.subplots_adjust(wspace=0.38, hspace=0.28)
axes = axes.flatten()
interacting_features = [feat for feat in sorted_features if feat != primary_feature_name]
n_interactions = min(9, len(interacting_features))

for idx in range(n_interactions):
    interacting_feature_name = interacting_features[idx]

    x_values = X_test_sorted[primary_feature_name]
    interaction_feature_values = X_test_sorted[interacting_feature_name]

    i_index = np.where(sorted_features == primary_feature_name)[0][0]
    j_index = np.where(sorted_features == interacting_feature_name)[0][0]
    shap_interaction_slice = shap_interaction_values_sorted[:, i_index, j_index] * 2


    ax = axes[idx]
    plot_advanced_interaction_NEW(
        primary_feature_name=primary_feature_name,
        interacting_feature_name=interacting_feature_name,
        x_values=x_values,
        interaction_feature_values=interaction_feature_values,
        shap_interaction_slice=shap_interaction_slice,
        ax=ax,
        subplot_idx=idx
    )

    subplot_letter = chr(97 + idx)
    ax.set_title(f'({subplot_letter}) {primary_feature_name} vs {interacting_feature_name}',
                 fontsize=12, fontfamily='Times New Roman', pad=10,loc='left')
# Save the entire graphic
output_filename = os.path.join(output_main_folder,
                              f"{primary_feature_name}_interactions_grid.png")
plt.savefig(output_filename, dpi=200, bbox_inches='tight')
plt.close()

