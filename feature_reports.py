import torch
import numpy as np
import pandas as pd
import json
from torch.utils.data import DataLoader, ConcatDataset
from mlp.mlp_definition import InterpretabilityMLP
from sae.sae_definition import SparseAutoencoder
from dataset.data_loader import load_excel_to_dataloader


def generate_full_circuit_dashboard(mlp_path, sae_path, excel_files):
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
    print("Tracing logic flow through the entire circuit...")

    with torch.no_grad():
        for batch_x, batch_y in combined_loader:
            actual_batch = mlp(batch_x)
            acts = mlp.activations['layer2']
            _, features = sae(acts)
            
            for i in range(batch_x.size(0)):
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
                
                # 4. Activation Analysis
                top_neuron = torch.argmax(acts[i]).item()
                top_feat = torch.argmax(features[i]).item()
                feat_act = torch.max(features[i]).item()

                results.append({
                    "inputs": x.tolist(),
                    "indices": (idx1, idx2),
                    "values": (val1, val2),
                    "neuron": f"N#{top_neuron}",
                    "feature": f"F#{top_feat}",
                    "outputs": y.tolist(),
                    "expected": expected,
                    "actual": f"{actual_val.item():.4f}",
                    "delta": f"{abs(actual_val.item() - expected):.4f}",
                    # Logic check: rounded actual vs integer expected
                    "equals": int(round(actual_val.item())) == expected,
                    "activation": feat_act,
                    # For Sankey mapping later
                    "source": f"Idx({idx1},{idx2})",
                    "target": f"Res:{expected}"
                })

    df = pd.DataFrame(results)
    df.to_excel("circuit_flows.xlsx", index=True)
    # 3. Aggregate flows for Sankey
    f1 = df.groupby(['indices', 'neuron']).size().reset_index(name='v').values.tolist()
    f2 = df.groupby(['neuron', 'feature']).size().reset_index(name='v').values.tolist()
    f3 = df.groupby(['feature', 'expected']).size().reset_index(name='v').values.tolist()

    sankey_data = f1 + f2 + f3

    # 4. Bell Curve Data for Top Features
    dist_series = []
    top_5_features = df['feature'].value_counts().head(5).index.tolist()
    
    highcharts_series = []
    for i, f_id in enumerate(top_5_features):
        acts = df[df['feature'] == f_id]['activation'].tolist()
        # Create the pair of series (the bell curve and the hidden scatter data)
        curve = {
            "name": f_id,
            "type": "bellcurve",
            "xAxis": 1,
            "yAxis": 1,
            "baseSeries": f"s{i}",
            "zIndex": -1
        }
        scatter = {
            "id": f"s{i}",
            "type": "scatter",
            "data": acts,
            "visible": False
        }
        highcharts_series.append(json.dumps(curve))
        highcharts_series.append(json.dumps(scatter))

    # Join the series into a JS-ready string
    js_series_string = ",\n".join(highcharts_series)

    # 5. Generate HTML
    html_template = f"""
    <!DOCTYPE html>
    <html>
    <head>
        <title>Circuit Dashboard</title>
        <script src="https://code.highcharts.com/highcharts.js"></script>
        <script src="https://code.highcharts.com/modules/sankey.js"></script>
        <script src="https://code.highcharts.com/modules/histogram-bellcurve.js"></script>
        <link href="https://cdn.jsdelivr.net/npm/bootstrap@5.3.0/dist/css/bootstrap.min.css" rel="stylesheet">
        <style>
            body {{ background: #f4f7f6; font-family: sans-serif; }}
            .card {{ border-radius: 15px; border:none; margin-bottom: 20px; }}
            .header-box {{ background: #2c3e50; color: white; padding: 2rem; border-radius: 0 0 20px 20px; }}
        </style>
    </head>
    <body>
        <div class="header-box text-center mb-5 shadow">
            <h1>Deterministic Circuit Map</h1>
            <p>Tracing Indices &rarr; Neurons &rarr; SAE Features &rarr; Result (0-9)</p>
        </div>

        <div class="container-fluid px-5">
            <div class="row">
                <div class="col-lg-8">
                    <div class="card shadow p-4">
                        <h4>Sankey Information Flow</h4>
                        <div id="sankey-container" style="height: 750px;"></div>
                    </div>
                </div>
                <div class="col-lg-4">
                    <div class="card shadow p-4">
                        <h4>Feature Activation Density</h4>
                        <div id="bell-container" style="height: 750px;"></div>
                    </div>
                </div>
            </div>
        </div>

        <script>
        Highcharts.chart('sankey-container', {{
            title: {{ text: '' }},
            series: [{{
                keys: ['from', 'to', 'weight'],
                data: {json.dumps(sankey_data)},
                type: 'sankey',
                name: 'Circuit Flow'
            }}]
        }});

        Highcharts.chart('bell-container', {{
            title: {{ text: '' }},
            xAxis: [{{ title: {{ text: 'Raw Strength' }} }}, {{ title: {{ text: 'Distribution' }}, opposite: true }}],
            yAxis: [{{ title: {{ text: 'Count' }} }}, {{ title: {{ text: 'Density' }}, opposite: true }}],
            series: [{js_series_string}]
        }});
        </script>
    </body>
    </html>
    """
    
    with open("mechanistic_map.html", "w") as f:
        f.write(html_template)
    print("Success: mechanistic_map.html generated.")

if __name__ == "__main__":
    files = ["dataset/mlp_train.xlsx", "dataset/mlp_val.xlsx", "dataset/mlp_test.xlsx"]
    generate_full_circuit_dashboard("mlp/perfect_mlp.pth", "sae/sae_model.pth", files)