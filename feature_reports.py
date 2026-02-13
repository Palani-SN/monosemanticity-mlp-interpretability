import os
import json
import torch
import numpy as np
import pandas as pd
from torch.utils.data import DataLoader, ConcatDataset
from mlp.mlp_definition import InterpretabilityMLP
from sae.sae_definition import SparseAutoencoder
from dataset.data_loader import load_excel_to_dataloader
from openpyxl.styles import Alignment

def save_styled_excel(df, filename="circuit_trace_detailed.xlsx"):
    # 1. Create the Excel writer object
    with pd.ExcelWriter(filename, engine='openpyxl') as writer:
        df.to_excel(writer, index=False, sheet_name='Detailed Trace')
        
        # 2. Access the openpyxl worksheet object
        workbook = writer.book
        worksheet = workbook['Detailed Trace']
        
        # 3. Define the center alignment style
        center_alignment = Alignment(horizontal='center', vertical='center')
        
        # 4. Iterate through all rows and columns to apply alignment
        # We start from row 1 to include headers
        for row in worksheet.iter_rows(min_row=1, max_row=worksheet.max_row, 
                                       min_col=1, max_col=worksheet.max_column):
            for cell in row:
                cell.alignment = center_alignment

        # 5. Optional: Auto-adjust column width for better readability
        for column_cells in worksheet.columns:
            length = max(len(str(cell.value)) for cell in column_cells)
            worksheet.column_dimensions[column_cells[0].column_letter].width = length + 2

    print(f"Clean, centered report saved to {filename}")

def prep_raw_data(mlp_path, sae_path, excel_files):
    # 1. Load Models
    mlp = InterpretabilityMLP()
    mlp.load_state_dict(torch.load(mlp_path))
    mlp.eval()
    
    sae = SparseAutoencoder(input_dim=512, dict_size=2048)
    sae.load_state_dict(torch.load(sae_path))
    sae.eval()

    # 2. Combine all datasets (Train, Val, Test)
    all_data = []
    for file in excel_files:
        # Re-using your existing loader logic
        loader = load_excel_to_dataloader(file, batch_size=64)
        all_data.append(loader.dataset)
    
    combined_loader = DataLoader(ConcatDataset(all_data), batch_size=64, shuffle=False)

    results = []
    total_count = 0
    print("Tracing logic flow through the entire circuit...")

    with torch.no_grad():
        for batch_x, batch_y in combined_loader:
            actual_batch = mlp(batch_x)
            acts = mlp.activations['layer2']
            _, features = sae(acts)
            
            for i in range(batch_x.size(0)):
                total_count += 1
                x = batch_x[i].cpu().numpy()
                y = batch_y[i].cpu().numpy()

                # 1. Correct Indexing Logic
                idx1 = int(x[4])
                idx2 = int(x[9])
                
                # 2. Extract specific values pointed to by the indices
                # Col 1 is x[0:4], Col 2 is x[5:9]
                val1 = int(x[idx1])
                val2 = int(x[5 + idx2])
                
                # 3. Scalar extraction for this specific sample
                expected = int(round(y.item()))
                # Index into the batch result before calling .item()
                actual_val = actual_batch[i]
                
                # # 4. Activation Analysis
                # top_neuron = torch.argmax(acts[i]).item()
                # top_feat = torch.argmax(features[i]).item()
                # feat_act = torch.max(features[i]).item()

                # --- Extraction: Top 10 Neurons & Top 10 Features ---
                # We capture both the ID and the raw activation magnitude
                top_n_vals, top_n_ids = torch.topk(acts[i], 10)
                top_f_vals, top_f_ids = torch.topk(features[i], 10)
                
                # Convert to CPU once to speed up the inner loop
                n_ids = top_n_ids.tolist()
                n_vls = top_n_vals.tolist()
                f_ids = top_f_ids.tolist()
                f_vls = top_f_vals.tolist()

                # --- Row-per-Rank Expansion ---
                # This creates a 'Long Form' dataset perfect for SQL-like analysis
                for rank in range(10):

                    results.append({
                        "sample_id": total_count,
                        "idx_combination": f"{idx1}-{idx2}", # MUST be this name
                        "val_combination": f"{val1}-{val2}", # MUST be this name
                        
                        # Inputs
                        "inputs": x.tolist(),
                        "indices": (idx1, idx2),
                        "values": (val1, val2),

                        # Circuit Rank (1 = strongest)
                        "rank": rank + 1,
                        
                        # Neuron Data
                        "neuron_id": n_ids[rank],
                        "neuron_act": round(n_vls[rank], 6),
                        
                        # SAE Feature Data
                        "feature_id": f_ids[rank],
                        "feature_act": round(f_vls[rank], 6),

                        # Outputs
                        "outputs": y.tolist(),

                        # Expected vs Actual
                        "expected": expected,
                        "actual": f"{actual_val.item():.4f}",
                        "delta": f"{abs(actual_val.item() - expected):.4f}",
                        "equals": int(round(actual_val.item())) == expected,
                    })

    return pd.DataFrame(results)


def generate_logic_heatmap_data(df):
    # 1. Clean and Calculate
    df['neuron_act'] = pd.to_numeric(df['neuron_act'], errors='coerce')
    df['feature_act'] = pd.to_numeric(df['feature_act'], errors='coerce')
    df['co_act_energy'] = df['neuron_act'] * df['feature_act']
    
    # Extract Val1 for Y-axis (assuming "val1-val2" format)
    df['val1'] = df['val_combination'].apply(lambda x: str(x).split('-')[0])

    # 2. Pivot Table: We'll use the SUM of energy for the color scale
    # This reflects the "Total Logic Power" being applied to that specific problem
    logic_matrix = df.pivot_table(
        index='val1', 
        columns='idx_combination', 
        values='co_act_energy',
        aggfunc='sum' 
    ).fillna(0)

    # 3. Generate Unique Top 10 Circuit Metadata
    hover_metadata = {}
    
    # Group by logic cell coordinates
    grouped = df.groupby(['val1', 'idx_combination'])
    
    for (v1, idx_c), group in grouped:
        # AGGREGATION STEP: Sum energy for identical Neuron-Feature pairs
        # This removes duplicates and ranks by cumulative strength
        unique_circuits = group.groupby(['neuron_id', 'feature_id'])['co_act_energy'].sum().reset_index()
        unique_circuits = unique_circuits.sort_values('co_act_energy', ascending=False).head(10)
        
        circuit_list = []
        for _, row in unique_circuits.iterrows():
            circuit_list.append({
                "n_id": row['neuron_id'],
                "f_id": row['feature_id'],
                "energy_sum": round(float(row['co_act_energy']), 4)
            })
            
        hover_metadata[f"{idx_c}_{v1}"] = circuit_list

    return logic_matrix, hover_metadata

def generate_logic_heatmap_html(matrix_df, metadata, filename="logic_circuit_map.html"):
    indices = matrix_df.columns.tolist()
    values = matrix_df.index.tolist()
    
    data_points = []
    for y, val_cat in enumerate(values):
        for x, idx_cat in enumerate(indices):
            total_energy = float(matrix_df.iloc[y, x])
            data_points.append([x, y, round(total_energy, 4)])

    html_content = f"""
    <!DOCTYPE html>
    <html>
    <head>
        <title>Logic Circuit Heatmap</title>
        <script src="https://code.highcharts.com/highcharts.js"></script>
        <script src="https://code.highcharts.com/modules/heatmap.js"></script>
        <style>
            body {{ font-family: 'Inter', -apple-system, sans-serif; background: #fdfdfd; padding: 40px; }}
            .container-wrapper {{ 
                background: white; padding: 25px; border-radius: 15px;
                box-shadow: 0 12px 40px rgba(0,0,0,0.08); width: fit-content; margin: auto;
                border: 1px solid #eee;
            }}
            #container {{ width: 1000px; height: 650px; }}
        </style>
    </head>
    <body>
        <div class="container-wrapper">
            <h2 style="text-align: center; color: #1a1a1a; margin-bottom: 5px;">Input Logic Density Map</h2>
            <p style="text-align: center; color: #777; margin-bottom: 25px;">
                Color intensity = <b>Total Cumulative Energy</b> (Sum of N_act * F_act)
            </p>
            <div id="container"></div>
        </div>

        <script>
        const metadata = {json.dumps(metadata)};

        Highcharts.chart('container', {{
            chart: {{ type: 'heatmap', backgroundColor: '#ffffff' }},
            title: {{ text: null }},
            xAxis: {{
                categories: {json.dumps(indices)},
                title: {{ text: 'Index Combination', style: {{ color: '#444', fontWeight: 'bold' }} }},
                labels: {{ style: {{ color: '#666' }} }}
            }},
            yAxis: {{
                categories: {json.dumps(values)},
                title: {{ text: 'Input Value (V1)', style: {{ color: '#444', fontWeight: 'bold' }} }},
                reversed: true
            }},
            colorAxis: {{
                stops: [
                    [0, '#ffffff'],       // Pure white for zero
                    [0.05, '#f1f8ff'],    // Soft start
                    [0.2, '#bbdefb'],     // Light Blue
                    [0.5, '#42a5f5'],     // Mid Blue
                    [1, '#0d47a1']        // Deep "Ground Truth" Blue
                ],
                min: 0
            }},
            tooltip: {{
                useHTML: true,
                backgroundColor: '#ffffff',
                borderWidth: 1,
                borderColor: '#ccc',
                shadow: true,
                formatter: function () {{
                    const key = this.series.xAxis.categories[this.point.x] + '_' + this.series.yAxis.categories[this.point.y];
                    const circuits = metadata[key] || [];
                    
                    let html = '<div style="padding:8px; min-width:220px">';
                    html += '<b style="color:#333">Top Unique Circuits (Summed)</b><br/>';
                    html += '<table style="width:100%; border-collapse: collapse; margin-top:8px; font-size:12px;">';
                    html += '<tr style="border-bottom: 2px solid #eee; text-align:left"><th>Pair</th><th style="text-align:right">Total Energy</th></tr>';
                    
                    circuits.forEach(c => {{
                        html += `<tr style="border-bottom: 1px solid #f9f9f9">
                            <td style="padding:4px 0">${{c.n_id}} ↔ ${{c.f_id}}</td>
                            <td style="padding:4px 0; text-align:right"><b>${{c.energy_sum}}</b></td>
                        </tr>`;
                    }});
                    html += '</table></div>';
                    return html;
                }}
            }},
            series: [{{
                name: 'Cumulative Energy',
                data: {json.dumps(data_points)},
                borderWidth: 1,
                borderColor: '#ffffff',
                dataLabels: {{
                    enabled: true,
                    color: 'contrast',
                    style: {{ textOutline: 'none', fontSize: '11px', fontWeight: 'bold', textShadow: 'none' }},
                    formatter: function() {{ return this.point.value > 0 ? this.point.value.toFixed(1) : ''; }}
                }}
            }}]
        }});
        </script>
    </body>
    </html>
    """
    # Open with explicit utf-8 encoding to support logic symbols like ↔
    with open(filename, "w", encoding="utf-8") as f: 
        f.write(html_content)
    return os.path.abspath(filename)

def generate_sankey_diagram(df, top_k=500):
    df = df.copy()
    # 1. Calculate Energy
    df['energy'] = pd.to_numeric(df['neuron_act']) * pd.to_numeric(df['feature_act'])
    
    # 2. Strict Level Prefixing (To keep 5 columns)
    df['L1'] = "(Idx) " + df['idx_combination'].astype(str)
    df['L2'] = "(Val) " + df['val_combination'].astype(str)
    df['L3'] = "(Neu) " + df['neuron_id'].astype(str)
    df['L4'] = "(Feat) " + df['feature_id'].astype(str)
    df['L5'] = "(Out) " + df['expected'].astype(str)
    
    flow_steps = [('L1', 'L2'), ('L2', 'L3'), ('L3', 'L4'), ('L4', 'L5')]
    rows = []
    
    # SCALING FACTOR: Boost small energy values so lines appear thick
    # Adjust this multiplier (e.g., 100 or 1000) based on your data range
    scale_factor = 100 

    # Increase the height dynamically based on the number of unique nodes
    # If top_k is large, we want a taller canvas.
    canvas_height = 1500  # Increased from 850 for a "Deep-Dive" view

    for src_col, tgt_col in flow_steps:
        step_data = df.groupby([src_col, tgt_col])['energy'].sum().reset_index()
        step_data.columns = ['source', 'target', 'value']
        
        # Take the most important signals
        top_step = step_data.nlargest(top_k // 4, 'value')
        
        for _, row in top_step.iterrows():
            if row['value'] > 0:
                # Multiply by scale_factor to ensure visual thickness
                weighted_val = float(row['value']) * scale_factor
                rows.append([row['source'], row['target'], round(weighted_val, 4)])

    # 3. HTML with Enhanced Sharpness
    html_content = f"""
    <!DOCTYPE html>
    <html>
    <head>
        <script type="text/javascript" src="https://www.gstatic.com/charts/loader.js"></script>
        <style>
            body {{ font-family: 'Inter', sans-serif; background: #ffffff; padding: 30px; }}
            #sankey_main {{ width: 100%; height: {canvas_height}px; min-width: 1400px; }}
            .header {{ text-align: center; margin-bottom: 20px; }}
        </style>
    </head>
    <body>
        <div class="header">
            <h2 style="color: #2c3e50;">Structural Interpretability: High-Fidelity Flow</h2>
            <p style="color: #7f8c8d;">Energy values scaled x{scale_factor} for visual clarity</p>
        </div>
        <div id="sankey_main"></div>

        <script type="text/javascript">
          google.charts.load('current', {{'packages':['sankey']}});
          google.charts.setOnLoadCallback(drawChart);

          function drawChart() {{
            var data = new google.visualization.DataTable();
            data.addColumn('string', 'From');
            data.addColumn('string', 'To');
            data.addColumn('number', 'Weight');
            data.addRows({json.dumps(rows)});

            // Using a Sharper, Vibrant Palette
            var colors = ['#1f77b4', '#ff7f0e', '#2ca02c', '#d62728', '#9467bd', 
                          '#8c564b', '#e377c2', '#7f7f7f', '#bcbd22', '#17becf'];

            var options = {{
              height: {canvas_height},
              sankey: {{
                iterations: 0, // Forces strict column segregation
                node: {{
                  colors: colors,
                  label: {{ fontSize: 10, color: '#1a1a1a', bold: true }},
                  width: 30,           // Thicker nodes
                  labelPadding: 25,    // More space for labels
                  interactivity: true
                }},
                link: {{
                  colorMode: 'gradient',
                  fillOpacity: 0.5,     // BOOSTED OPACITY for "Sharp" look
                  colors: colors
                }}
              }}
            }};

            var chart = new google.visualization.Sankey(document.getElementById('sankey_main'));
            chart.draw(data, options);
          }}
        </script>
    </body>
    </html>
    """

    filename = "uhd_bold_sankey.html"
    with open(filename, "w", encoding="utf-8") as f:
        f.write(html_content)
    return os.path.abspath(filename)

def generate_stacked_norm_dist(df, top_k=20):
    """
    Generates a Highcharts Bell Curve plot.
    If top_k is None, it plots all available unique Neuron-Feature pairs.
    """
    # 1. Prepare Numeric Data
    df['neuron_act'] = pd.to_numeric(df['neuron_act'], errors='coerce')
    df['feature_act'] = pd.to_numeric(df['feature_act'], errors='coerce')
    df['energy'] = df['neuron_act'] * df['feature_act']
    
    # 2. Statistical Aggregation
    stats = df.groupby(['neuron_id', 'feature_id'])['energy'].agg(['mean', 'std', 'count']).reset_index()
    
    # Filter for validity (std requires > 1 sample, count > 2 for a meaningful curve)
    stats = stats[stats['count'] > 2].dropna()
    
    # 3. Handle top_k logic
    if top_k is not None:
        stats['impact'] = stats['mean'] * stats['count']
        plot_stats = stats.nlargest(top_k, 'impact')
    else:
        plot_stats = stats.sort_values('mean', ascending=False)

    # 4. Curve Generation Logic
    series_data = []
    # Determine global X-axis bounds to keep all curves in frame
    if not plot_stats.empty:
        low_bound = plot_stats['mean'].min() - (plot_stats['std'].max() * 4)
        high_bound = plot_stats['mean'].max() + (plot_stats['std'].max() * 4)
        x_range = np.linspace(low_bound, high_bound, 300)

        for _, row in plot_stats.iterrows():
            mu = row['mean']
            sigma = row['std']
            
            # Normal Distribution PDF formula
            y_values = (1 / (sigma * np.sqrt(2 * np.pi))) * np.exp(-0.5 * ((x_range - mu) / sigma)**2)
            
            curve_points = [[round(x, 4), round(y, 6)] for x, y in zip(x_range, y_values)]
            
            series_data.append({
                "name": f"{row['neuron_id']} ↔ {row['feature_id']}",
                "data": curve_points,
                "mu": round(mu, 4),
                "sigma": round(sigma, 4),
                "count": int(row['count']),
                "turboThreshold": 0 # Handle large datasets
            })

    # 5. HTML Template
    html_content = f"""
    <!DOCTYPE html>
    <html>
    <head>
        <title>Circuit Stability Distributions</title>
        <script src="https://code.highcharts.com/highcharts.js"></script>
        <style>
            body {{ font-family: 'Inter', -apple-system, sans-serif; background: #fcfcfc; padding: 20px; }}
            .container {{ background: white; padding: 20px; border-radius: 12px; border: 1px solid #eee; box-shadow: 0 4px 20px rgba(0,0,0,0.05); }}
            #chart-div {{ width: 100%; height: 750px; }}
        </style>
    </head>
    <body>
        <div class="container">
            <h2 style="text-align: center; color: #1a1a1a; margin-bottom: 5px;">Circuit Reliability Analysis</h2>
            <p style="text-align: center; color: #777; margin-bottom: 25px;">
                Circuit Reliability = <b>Mu & Sigma</b> (From list of N_act * F_act)
            </p>
            <div id="chart-div"></div>
        </div>

        <script>
        Highcharts.chart('chart-div', {{
            chart: {{ 
                type: 'areaspline', 
                backgroundColor: '#ffffff',
                zoomType: 'x',
                resetZoomButton: {{ position: {{ align: 'right', verticalAlign: 'top' }} }}
            }},
            title: {{ 
                text: null, 
                style: {{ fontWeight: 'bold' }} }},
            xAxis: {{ 
                title: {{ text: 'Activation Energy' }},
                gridLineWidth: 1,
                gridLineColor: '#f0f0f0'
            }},
            yAxis: {{ 
                title: {{ text: 'Probability Density' }},
                gridLineColor: '#f0f0f0'
            }},
            legend: {{
                layout: 'horizontal',
                align: 'center',
                verticalAlign: 'bottom',
                itemStyle: {{ fontSize: '9px', fontWeight: '400' }}
            }},
            tooltip: {{
                shared: false,      /* ONLY one curve at a time */
                followPointer: true,
                useHTML: true,
                backgroundColor: 'rgba(255, 255, 255, 0.9)',
                headerFormat: '<small style="color:{{series.color}}">●</small> <b>{{series.name}}</b><br/>',
                pointFormat: `
                    Density: <b>{{point.y:.5f}}</b><br/>
                    Mean (μ): <b>{{series.options.mu}}</b><br/>
                    Std Dev (σ): <b>{{series.options.sigma}}</b><br/>
                    Samples: <b>{{series.options.count}}</b>`
            }},
            plotOptions: {{
                areaspline: {{
                    fillOpacity: 0.1,
                    lineWidth: 2,
                    stickyTracking: false, /* Tooltip only triggers on direct hover */
                    marker: {{ enabled: false }},
                    states: {{ hover: {{ lineWidth: 4, fillOpacity: 0.3 }} }}
                }}
            }},
            series: {json.dumps(series_data)}
        }});
        </script>
    </body>
    </html>
    """

    path = "circuit_bell_curves.html"
    with open(path, "w", encoding="utf-8") as f:
        f.write(html_content)
    
    return os.path.abspath(path)


if __name__ == "__main__":

    files = ["dataset/mlp_train.xlsx", "dataset/mlp_val.xlsx", "dataset/mlp_test.xlsx"]

    df = prep_raw_data("mlp/perfect_mlp.pth", "sae/sae_model.pth", files)

    save_styled_excel(df)

    # --- Usage Example ---
    matrix_df, metadata = generate_logic_heatmap_data(df)

    filepath = generate_logic_heatmap_html(matrix_df, metadata)
    print(f"Logic heatmap saved to: {filepath}")

    filepath = generate_stacked_norm_dist(df)
    print(f"Stacked norm dist saved to: {filepath}")

    filepath = generate_sankey_diagram(df)
    print(f"Sankey diagram saved to: {filepath}")

    # # --- Usage Example ---
    # matrix_df, metadata = generate_correlation_matrix(df)

    # filepath = generate_correlation_heatmap(matrix_df, metadata)
    # print(f"Correlation heatmap saved to: {filepath}")